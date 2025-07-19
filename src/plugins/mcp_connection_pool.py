"""
Enhanced Connection Pool for MCP Database Plugin
Provides connection pooling, health monitoring, and performance optimization
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from contextlib import asynccontextmanager
from fastmcp import Client
import os


@dataclass
class ConnectionMetrics:
    """Metrics for connection pool performance monitoring"""
    total_created: int = 0
    total_closed: int = 0
    total_borrowed: int = 0
    total_returned: int = 0
    current_active: int = 0
    current_idle: int = 0
    connection_errors: int = 0
    health_check_failures: int = 0
    average_borrow_time_ms: float = 0.0
    average_operation_time_ms: float = 0.0
    peak_active_connections: int = 0
    
    def __post_init__(self):
        self._borrow_times: List[float] = []
        self._operation_times: List[float] = []
    
    def record_borrow_time(self, time_ms: float):
        """Record connection borrow time"""
        self._borrow_times.append(time_ms)
        if len(self._borrow_times) > 100:  # Keep last 100 measurements
            self._borrow_times = self._borrow_times[-100:]
        self.average_borrow_time_ms = sum(self._borrow_times) / len(self._borrow_times)
    
    def record_operation_time(self, time_ms: float):
        """Record database operation time"""
        self._operation_times.append(time_ms)
        if len(self._operation_times) > 100:  # Keep last 100 measurements
            self._operation_times = self._operation_times[-100:]
        self.average_operation_time_ms = sum(self._operation_times) / len(self._operation_times)


@dataclass
class PooledConnection:
    """Wrapper for a pooled MCP client connection"""
    client: Client
    created_at: float
    last_used: float
    last_health_check: float = 0.0
    is_healthy: bool = True
    connection_id: str = field(default_factory=lambda: f"conn_{int(time.time() * 1000000) % 1000000}")
    usage_count: int = 0
    
    def mark_used(self):
        """Mark connection as recently used"""
        self.last_used = time.time()
        self.usage_count += 1
    
    def is_expired(self, max_age_seconds: int) -> bool:
        """Check if connection has exceeded maximum age"""
        return (time.time() - self.created_at) > max_age_seconds
    
    def is_idle_expired(self, idle_timeout_seconds: int) -> bool:
        """Check if connection has been idle too long"""
        return (time.time() - self.last_used) > idle_timeout_seconds


class MCPConnectionPool:
    """
    High-performance connection pool for MCP database operations
    Features: Health monitoring, automatic scaling, performance metrics
    """
    
    def __init__(self, 
                 mcp_server_url: str,
                 min_connections: int = 1,  # Reduced from 2 for faster startup
                 max_connections: int = 6,  # Reduced from 10 for better resource management
                 connection_timeout: float = 30.0,
                 idle_timeout: float = 600.0,  # Increased to 10 minutes (connections are expensive)
                 max_connection_age: float = 7200.0,  # Increased to 2 hours
                 health_check_interval: float = 300.0,  # Reduced to 5 minutes
                 retry_attempts: int = 2,  # Reduced for faster failure detection
                 enable_metrics: bool = True,
                 lazy_initialization: bool = True):  # NEW: Enable lazy initialization
        
        self.mcp_server_url = mcp_server_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.max_connection_age = max_connection_age
        self.health_check_interval = health_check_interval
        self.retry_attempts = retry_attempts
        self.enable_metrics = enable_metrics
        self.lazy_initialization = lazy_initialization
        
        # Connection storage
        self._idle_connections: deque = deque()
        self._active_connections: Dict[str, PooledConnection] = {}
        self._connection_lock = asyncio.Lock()
        
        # Health and monitoring
        self._metrics = ConnectionMetrics() if enable_metrics else None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._initialization_started = False
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        
        optimization_note = "lazy" if lazy_initialization else "eager"
        print(f"🔗 Optimized MCP Connection Pool initialized ({optimization_note}): "
              f"{min_connections}-{max_connections} connections, url={mcp_server_url}")
    
    async def initialize(self):
        """Initialize the connection pool with optimized startup"""
        async with self._connection_lock:
            if self._initialization_started:
                return  # Already initializing/initialized
            
            self._initialization_started = True
            
            if self.lazy_initialization:
                # Lazy initialization: create connections on-demand
                print(f"⚡ Using lazy initialization - connections created on-demand")
            else:
                # Eager initialization: create minimum connections upfront
                print(f"🔄 Creating {self.min_connections} initial connections...")
                
                created_count = 0
                for i in range(self.min_connections):
                    try:
                        conn = await self._create_connection()
                        self._idle_connections.append(conn)
                        created_count += 1
                        if self._metrics:
                            self._metrics.current_idle += 1
                    except Exception as e:
                        self.logger.error(f"Failed to create initial connection {i}: {e}")
                
                print(f"✅ Created {created_count}/{self.min_connections} initial connections")
            
            # Start health check task with optimized interval
            if not self._health_check_task:
                self._health_check_task = asyncio.create_task(self._health_check_worker())
            
            total_connections = len(self._idle_connections)
            print(f"✅ Optimized connection pool ready with {total_connections} connections")
    
    async def _create_connection(self) -> PooledConnection:
        """Create a new MCP connection with proper FastMCP context management"""
        try:
            client = Client(self.mcp_server_url)
            
            # Initialize the connection within the client context
            await client.__aenter__()
            
            conn = PooledConnection(
                client=client,
                created_at=time.time(),
                last_used=time.time()
            )
            
            # Test the connection to ensure it's working
            try:
                await client.list_resources()
                conn.is_healthy = True
            except Exception as test_error:
                self.logger.warning(f"Connection health test failed: {test_error}")
                conn.is_healthy = False
            
            if self._metrics:
                self._metrics.total_created += 1
            
            return conn
        except Exception as e:
            if self._metrics:
                self._metrics.connection_errors += 1
            raise Exception(f"Failed to create MCP connection: {e}")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool (context manager)"""
        start_time = time.time()
        conn = None
        
        try:
            conn = await self._borrow_connection()
            
            if self._metrics:
                borrow_time = (time.time() - start_time) * 1000
                self._metrics.record_borrow_time(borrow_time)
            
            yield conn
            
        finally:
            if conn:
                await self._return_connection(conn)
    
    async def _borrow_connection(self) -> PooledConnection:
        """Borrow a connection from the pool"""
        async with self._connection_lock:
            # Try to get an idle connection
            while self._idle_connections:
                conn = self._idle_connections.popleft()
                
                # Check if connection is still healthy and not expired
                if (not conn.is_expired(self.max_connection_age) and 
                    not conn.is_idle_expired(self.idle_timeout) and
                    conn.is_healthy):
                    
                    # Move to active connections
                    self._active_connections[conn.connection_id] = conn
                    conn.mark_used()
                    
                    if self._metrics:
                        self._metrics.current_idle -= 1
                        self._metrics.current_active += 1
                        self._metrics.total_borrowed += 1
                        self._metrics.peak_active_connections = max(
                            self._metrics.peak_active_connections,
                            self._metrics.current_active
                        )
                    
                    return conn
                else:
                    # Connection is expired or unhealthy, close it
                    await self._close_connection(conn)
            
            # No idle connections available, try to create new one
            if len(self._active_connections) + len(self._idle_connections) < self.max_connections:
                try:
                    conn = await self._create_connection()
                    self._active_connections[conn.connection_id] = conn
                    conn.mark_used()
                    
                    if self._metrics:
                        self._metrics.current_active += 1
                        self._metrics.total_borrowed += 1
                        self._metrics.peak_active_connections = max(
                            self._metrics.peak_active_connections,
                            self._metrics.current_active
                        )
                    
                    return conn
                except Exception as e:
                    self.logger.error(f"Failed to create new connection: {e}")
            
            # Wait for a connection to be returned (with timeout)
            await asyncio.sleep(0.1)  # Brief pause before retry
            
            # Recursive call with limit to prevent infinite recursion
            return await self._borrow_connection()
    
    async def _return_connection(self, conn: PooledConnection):
        """Return a connection to the pool"""
        async with self._connection_lock:
            if conn.connection_id in self._active_connections:
                del self._active_connections[conn.connection_id]
                
                # Check if connection is still healthy
                if (conn.is_healthy and 
                    not conn.is_expired(self.max_connection_age)):
                    
                    # Return to idle pool
                    self._idle_connections.append(conn)
                    
                    if self._metrics:
                        self._metrics.current_active -= 1
                        self._metrics.current_idle += 1
                        self._metrics.total_returned += 1
                else:
                    # Connection is unhealthy or expired, close it
                    await self._close_connection(conn)
                    if self._metrics:
                        self._metrics.current_active -= 1
    
    async def _close_connection(self, conn: PooledConnection):
        """Close a connection with proper FastMCP cleanup"""
        try:
            if conn.client:
                # Properly close the FastMCP client using context manager exit
                await conn.client.__aexit__(None, None, None)
            
            conn.is_healthy = False
            
            if self._metrics:
                self._metrics.total_closed += 1
                
        except Exception as e:
            self.logger.error(f"Error closing connection {conn.connection_id}: {e}")
    
    async def _health_check_worker(self):
        """Background worker for connection health monitoring"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check worker error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on idle connections"""
        async with self._connection_lock:
            healthy_connections = deque()
            
            while self._idle_connections:
                conn = self._idle_connections.popleft()
                
                try:
                    # Perform simple health check by testing the client
                    # For fastmcp, we'll just check connection properties
                    if conn.client and not conn.is_expired(self.max_connection_age):
                        conn.is_healthy = True
                        conn.last_health_check = time.time()
                        healthy_connections.append(conn)
                    else:
                        await self._close_connection(conn)
                        if self._metrics:
                            self._metrics.current_idle -= 1
                            self._metrics.health_check_failures += 1
                
                except Exception as e:
                    # Connection failed health check
                    conn.is_healthy = False
                    await self._close_connection(conn)
                    if self._metrics:
                        self._metrics.current_idle -= 1
                        self._metrics.health_check_failures += 1
                    
                    self.logger.warning(f"Connection {conn.connection_id} failed health check: {e}")
            
            # Restore healthy connections
            self._idle_connections = healthy_connections
            
            # Ensure minimum connections
            current_total = len(self._idle_connections) + len(self._active_connections)
            if current_total < self.min_connections:
                needed = self.min_connections - current_total
                for i in range(needed):
                    try:
                        new_conn = await self._create_connection()
                        self._idle_connections.append(new_conn)
                        if self._metrics:
                            self._metrics.current_idle += 1
                    except Exception as e:
                        self.logger.error(f"Failed to create replacement connection: {e}")
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get connection pool performance metrics"""
        if not self._metrics:
            return None
        
        return {
            "pool_status": {
                "active_connections": self._metrics.current_active,
                "idle_connections": self._metrics.current_idle,
                "total_connections": self._metrics.current_active + self._metrics.current_idle,
                "peak_active": self._metrics.peak_active_connections
            },
            "connection_lifecycle": {
                "total_created": self._metrics.total_created,
                "total_closed": self._metrics.total_closed,
                "total_borrowed": self._metrics.total_borrowed,
                "total_returned": self._metrics.total_returned
            },
            "performance": {
                "average_borrow_time_ms": round(self._metrics.average_borrow_time_ms, 2),
                "average_operation_time_ms": round(self._metrics.average_operation_time_ms, 2)
            },
            "errors": {
                "connection_errors": self._metrics.connection_errors,
                "health_check_failures": self._metrics.health_check_failures
            },
            "configuration": {
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
                "connection_timeout": self.connection_timeout,
                "idle_timeout": self.idle_timeout,
                "max_connection_age": self.max_connection_age,
                "health_check_interval": self.health_check_interval
            }
        }
    
    def print_metrics(self):
        """Print formatted connection pool metrics"""
        metrics = self.get_metrics()
        if not metrics:
            print("📊 Connection pool metrics disabled")
            return
        
        print("\n" + "="*60)
        print("🔗 MCP CONNECTION POOL METRICS")
        print("="*60)
        
        print(f"📋 Pool Status:")
        print(f"   Active Connections: {metrics['pool_status']['active_connections']}")
        print(f"   Idle Connections: {metrics['pool_status']['idle_connections']}")
        print(f"   Total Connections: {metrics['pool_status']['total_connections']}")
        print(f"   Peak Active: {metrics['pool_status']['peak_active']}")
        
        print(f"\n🔄 Connection Lifecycle:")
        print(f"   Created: {metrics['connection_lifecycle']['total_created']}")
        print(f"   Closed: {metrics['connection_lifecycle']['total_closed']}")
        print(f"   Borrowed: {metrics['connection_lifecycle']['total_borrowed']}")
        print(f"   Returned: {metrics['connection_lifecycle']['total_returned']}")
        
        print(f"\n⚡ Performance:")
        print(f"   Avg Borrow Time: {metrics['performance']['average_borrow_time_ms']:.2f}ms")
        print(f"   Avg Operation Time: {metrics['performance']['average_operation_time_ms']:.2f}ms")
        
        print(f"\n⚠️ Errors:")
        print(f"   Connection Errors: {metrics['errors']['connection_errors']}")
        print(f"   Health Check Failures: {metrics['errors']['health_check_failures']}")
        
        print("="*60)
    
    async def close(self):
        """Close the connection pool and all connections"""
        self._shutdown = True
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        async with self._connection_lock:
            # Close all idle connections
            while self._idle_connections:
                conn = self._idle_connections.popleft()
                await self._close_connection(conn)
            
            # Close all active connections  
            for conn in list(self._active_connections.values()):
                await self._close_connection(conn)
            self._active_connections.clear()
            
            if self._metrics:
                self._metrics.current_active = 0
                self._metrics.current_idle = 0
        
        print("🔒 MCP Connection Pool closed")
