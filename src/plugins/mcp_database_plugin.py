import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from pydantic import Field
from fastmcp import Client
from .mcp_connection_pool import MCPConnectionPool


class MCPDatabasePlugin(KernelBaseModel):
    """
    Enhanced MCP Database Plugin with Connection Pooling
    Provides high-performance database operations through managed connection pool
    Features: Connection pooling, health monitoring, performance metrics, retry logic
    """
    
    mcp_server_url: str = Field(description="MCP server URL")
    connection_pool: Optional[MCPConnectionPool] = Field(default=None, exclude=True)
    enable_connection_pooling: bool = Field(default=True, description="Enable connection pooling")
    enable_performance_tracking: bool = Field(default=True, description="Enable performance metrics")
    
    def __init__(self, mcp_server_url: str, enable_pooling: bool = True, **kwargs):
        super().__init__(mcp_server_url=mcp_server_url, enable_connection_pooling=enable_pooling, **kwargs)
        
        if enable_pooling:
            # Initialize connection pool with configuration from environment
            pool_config = {
                'min_connections': int(os.getenv('MCP_POOL_MIN_CONNECTIONS', '2')),
                'max_connections': int(os.getenv('MCP_POOL_MAX_CONNECTIONS', '8')),
                'connection_timeout': float(os.getenv('MCP_POOL_CONNECTION_TIMEOUT', '30.0')),
                'idle_timeout': float(os.getenv('MCP_POOL_IDLE_TIMEOUT', '300.0')),
                'max_connection_age': float(os.getenv('MCP_POOL_MAX_AGE', '3600.0')),
                'health_check_interval': float(os.getenv('MCP_POOL_HEALTH_INTERVAL', '60.0')),
                'retry_attempts': int(os.getenv('MCP_POOL_RETRY_ATTEMPTS', '3')),
                'enable_metrics': self.enable_performance_tracking
            }
            
            self.connection_pool = MCPConnectionPool(mcp_server_url, **pool_config)
            print(f"🔗 Enhanced MCP Database Plugin initialized with connection pooling")
        else:
            # Fallback to single client mode
            self.client = Client(mcp_server_url)
            print(f"🔌 MCP Database Plugin initialized without pooling (legacy mode)")
    
    async def initialize(self):
        """Initialize the plugin and connection pool"""
        if self.connection_pool:
            await self.connection_pool.initialize()
    
    async def _execute_with_pool(self, operation_name: str, tool_name: str, params: dict = None) -> str:
        """Execute MCP operation using connection pool with retry logic"""
        if not self.connection_pool:
            raise ValueError("Connection pool not initialized")
        
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.connection_pool.retry_attempts):
            try:
                async with self.connection_pool.get_connection() as conn:
                    # Execute the MCP tool call
                    async with conn.client:
                        if params:
                            result = await conn.client.call_tool(tool_name, params)
                        else:
                            result = await conn.client.call_tool(tool_name)
                    
                    # Record performance metrics
                    if self.connection_pool._metrics:
                        operation_time = (time.time() - start_time) * 1000
                        self.connection_pool._metrics.record_operation_time(operation_time)
                    
                    return str(result)
            
            except Exception as e:
                last_exception = e
                if attempt < self.connection_pool.retry_attempts - 1:
                    wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    print(f"⚠️ {operation_name} attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {e}")
        
        # All attempts failed
        if self.connection_pool._metrics:
            self.connection_pool._metrics.connection_errors += 1
        
        raise Exception(f"{operation_name} failed after {self.connection_pool.retry_attempts} attempts: {last_exception}")
    
    async def _execute_legacy(self, operation_name: str, tool_name: str, params: dict = None) -> str:
        """Execute MCP operation using legacy single client"""
        try:
            async with self.client:
                if params:
                    result = await self.client.call_tool(tool_name, params)
                else:
                    result = await self.client.call_tool(tool_name)
                return str(result)
        except Exception as e:
            raise Exception(f"{operation_name} failed: {e}")
    
    async def _execute_operation(self, operation_name: str, tool_name: str, params: dict = None) -> str:
        """Execute MCP operation with appropriate method based on configuration"""
        if self.enable_connection_pooling and self.connection_pool:
            return await self._execute_with_pool(operation_name, tool_name, params)
        else:
            return await self._execute_legacy(operation_name, tool_name, params)
    
    @kernel_function(
        description="List all available tables in the database with connection pooling",
        name="list_database_tables"
    )
    async def list_tables(self) -> str:
        """List all tables in the database using pooled connections"""
        return await self._execute_operation("list_tables", "list_tables")
    
    @kernel_function(
        description="Get detailed schema information for a specific table with connection pooling",
        name="describe_table_schema"
    )
    async def describe_table(self, table_name: str) -> str:
        """Get table structure and schema information using pooled connections"""
        return await self._execute_operation(
            "describe_table", 
            "describe_table", 
            {"table_name": table_name}
        )
    
    @kernel_function(
        description="Execute a SELECT query and return results with connection pooling (automatically adds dev. schema prefix)",
        name="execute_sql_query"
    )
    async def read_data(self, query: str, limit: int = None) -> str:
        """Execute SELECT queries to read data from the database using pooled connections"""
        # Use environment variable for default limit if not specified
        if limit is None:
            limit = int(os.getenv('DEFAULT_QUERY_LIMIT', '100'))
        
        # Add dev. schema prefix to table names if not present
        modified_query = self._add_schema_prefix(query)
        
        return await self._execute_operation(
            "execute_sql_query",
            "read_data", 
            {"query": modified_query, "limit": limit}
        )
    
    @kernel_function(
        description="Get comprehensive database information and connection status with connection pooling",
        name="get_database_info"
    )
    async def database_info(self) -> str:
        """Get database connection information using pooled connections"""
        return await self._execute_operation("database_info", "database_info")
    
    def _add_schema_prefix(self, query: str) -> str:
        """Add dev. schema prefix to table names in the query"""
        table_names = ["cliente", "cliente_cedi", "mercado", "producto", "segmentacion", "tiempo"]
        modified_query = query
        
        for table in table_names:
            # Replace table name with dev.table_name if not already prefixed
            if f"dev.{table}" not in modified_query.lower():
                # Use word boundaries to avoid partial matches
                import re
                pattern = r'\b' + re.escape(table) + r'\b'
                modified_query = re.sub(pattern, f"dev.{table}", modified_query, flags=re.IGNORECASE)
        
        return modified_query
    
    # Performance and monitoring methods
    def get_pool_metrics(self) -> Optional[Dict[str, Any]]:
        """Get connection pool performance metrics"""
        if self.connection_pool:
            return self.connection_pool.get_metrics()
        return None
    
    def print_pool_metrics(self):
        """Print connection pool performance metrics"""
        if self.connection_pool:
            self.connection_pool.print_metrics()
        else:
            print("📊 Connection pooling disabled - no metrics available")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            "connection_pooling_enabled": self.enable_connection_pooling,
            "performance_tracking_enabled": self.enable_performance_tracking,
            "mcp_server_url": self.mcp_server_url
        }
        
        if self.connection_pool:
            metrics = self.connection_pool.get_metrics()
            if metrics:
                summary.update({
                    "pool_efficiency": {
                        "connection_utilization": (
                            metrics['pool_status']['active_connections'] / 
                            max(metrics['configuration']['max_connections'], 1)
                        ) * 100,
                        "average_borrow_time": metrics['performance']['average_borrow_time_ms'],
                        "average_operation_time": metrics['performance']['average_operation_time_ms'],
                        "error_rate": (
                            metrics['errors']['connection_errors'] / 
                            max(metrics['connection_lifecycle']['total_borrowed'], 1)
                        ) * 100
                    },
                    "pool_status": metrics['pool_status']
                })
        
        return summary
    
    # Enhanced context manager methods for batch operations
    async def __aenter__(self):
        """Enhanced context manager entry for batch operations"""
        if self.connection_pool:
            # Use connection pool for batch operations
            self._batch_connection = await self.connection_pool._borrow_connection()
            return self
        else:
            # Legacy mode
            await self.client.__aenter__()
            return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Enhanced context manager exit for batch operations"""
        if self.connection_pool and hasattr(self, '_batch_connection'):
            # Return connection to pool
            await self.connection_pool._return_connection(self._batch_connection)
            delattr(self, '_batch_connection')
        elif hasattr(self.client, '__aexit__'):
            # Legacy mode
            return await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    # Enhanced raw methods for batch operations
    async def _list_tables_raw(self) -> str:
        """List tables - optimized for batch operations"""
        try:
            if hasattr(self, '_batch_connection'):
                # Use pooled connection in batch mode
                async with self._batch_connection.client:
                    result = await self._batch_connection.client.call_tool("list_tables")
                    return str(result)
            else:
                # Standard operation
                return await self.list_tables()
        except Exception as e:
            return f"Error listing tables: {str(e)}"
    
    async def _describe_table_raw(self, table_name: str) -> str:
        """Describe table - optimized for batch operations"""
        try:
            if hasattr(self, '_batch_connection'):
                # Use pooled connection in batch mode
                async with self._batch_connection.client:
                    result = await self._batch_connection.client.call_tool(
                        "describe_table", {"table_name": table_name}
                    )
                    return str(result)
            else:
                # Standard operation
                return await self.describe_table(table_name)
        except Exception as e:
            return f"Error describing table {table_name}: {str(e)}"
    
    async def _database_info_raw(self) -> str:
        """Get database info - optimized for batch operations"""
        try:
            if hasattr(self, '_batch_connection'):
                # Use pooled connection in batch mode
                async with self._batch_connection.client:
                    result = await self._batch_connection.client.call_tool("database_info")
                    return str(result)
            else:
                # Standard operation
                return await self.database_info()
        except Exception as e:
            return f"Error getting database info: {str(e)}"
    
    async def close(self):
        """Close the plugin and all connections"""
        if self.connection_pool:
            await self.connection_pool.close()
            print("🔒 Enhanced MCP Database Plugin closed")
        elif hasattr(self, 'client') and self.client:
            # Legacy mode cleanup
            self.client = None
            print("🔒 MCP Database Plugin closed (legacy mode)")
    
    def __del__(self):
        """Cleanup when plugin is garbage collected"""
        if hasattr(self, 'connection_pool') and self.connection_pool:
            # Note: Can't call async method in __del__, but the pool will cleanup automatically
            pass
