"""
Optimized base agent implementation with unified interface
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import semantic_kernel as sk

try:
    # Try relative imports first (when used as module)
    from .agent_interface import AgentMessage, AgentProtocol, WorkflowStage, WorkflowConfig
    from ..services.performance_monitor import track_async_performance, perf_monitor
except ImportError:
    # Fall back to absolute imports (when used directly)
    from agents.agent_interface import AgentMessage, AgentProtocol, WorkflowStage, WorkflowConfig
    from services.performance_monitor import track_async_performance, perf_monitor


class OptimizedBaseAgent(AgentProtocol, ABC):
    """
    Optimized base agent that implements the unified interface while maintaining
    backward compatibility with existing agent functionality
    """
    
    def __init__(self, agent_name: str, capabilities: List[str], kernel: sk.Kernel = None):
        self._agent_name = agent_name
        self.capabilities = capabilities
        self.kernel = kernel
        self._initialized = False
        self._agent_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Performance tracking
        self.performance_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "average_response_time": 0.0,
            "cache_hits": 0
        }
        self._execution_count = 0
        self._total_execution_time = 0.0
        
        # Caching for repeated operations
        self._cache: Dict[str, Any] = {}
    
    @property
    def agent_name(self) -> str:
        """Return the agent name"""
        return getattr(self, '_agent_name', 'OptimizedAgent')
        self._cache_enabled = True
        self._max_cache_size = 100
    
    @property
    def agent_id(self) -> str:
        return self._agent_id
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def enable_cache(self, max_size: int = 100):
        """Enable caching with specified max size"""
        self._cache_enabled = True
        self._max_cache_size = max_size
    
    def disable_cache(self):
        """Disable caching"""
        self._cache_enabled = False
        self.clear_cache()
    
    def clear_cache(self):
        """Clear agent cache"""
        self._cache.clear()
    
    def _get_cache_key(self, message: AgentMessage) -> str:
        """Generate cache key for message"""
        return f"{message.message_type}_{hash(str(message.content))}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache"""
        if not self._cache_enabled:
            return None
        return self._cache.get(cache_key)
    
    def _store_in_cache(self, cache_key: str, result: Any):
        """Store result in cache"""
        if not self._cache_enabled:
            return
        
        # Implement LRU-like behavior
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result
    
    @track_async_performance("agent_initialization")
    async def initialize(self, config: WorkflowConfig) -> bool:
        """
        Initialize the agent with configuration
        
        Args:
            config: Workflow configuration
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Perform agent-specific initialization
            await self._initialize_agent(config)
            self._initialized = True
            
            # Record initialization metric
            perf_monitor.record_manual(f"{self.agent_name}_initialization", 0.1, True)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize {self.agent_name}: {str(e)}")
            perf_monitor.record_manual(f"{self.agent_name}_initialization", 0.1, False)
            return False
    
    @abstractmethod
    async def _initialize_agent(self, config: WorkflowConfig):
        """Agent-specific initialization logic"""
        pass
    
    @track_async_performance("agent_message_processing")
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message using the unified interface
        
        Args:
            message: Incoming agent message
            
        Returns:
            AgentMessage: Response message
        """
        start_time = datetime.now()
        
        try:
            # Check if agent is initialized
            if not self._initialized:
                return AgentMessage(
                    message_type="error",
                    content={"error": f"Agent {self.agent_name} not initialized"},
                    metadata={"agent_id": self.agent_id, "timestamp": datetime.now().isoformat()}
                )
            
            # Check cache first
            cache_key = self._get_cache_key(message)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                # Add cache hit metadata
                cached_result.metadata = cached_result.metadata or {}
                cached_result.metadata["cache_hit"] = True
                return cached_result
            
            # Process the message
            response = await self._process_message_impl(message)
            
            # Store in cache
            self._store_in_cache(cache_key, response)
            
            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._execution_count += 1
            self._total_execution_time += execution_time
            
            # Add metadata
            response.metadata = response.metadata or {}
            response.metadata.update({
                "agent_id": self.agent_id,
                "execution_time": execution_time,
                "execution_count": self._execution_count,
                "cache_hit": False
            })
            
            return response
            
        except Exception as e:
            error_response = AgentMessage(
                message_type="error",
                content={"error": str(e), "agent": self.agent_name},
                metadata={
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(e).__name__
                }
            )
            return error_response
    
    @abstractmethod
    async def _process_message_impl(self, message: AgentMessage) -> AgentMessage:
        """Agent-specific message processing logic"""
        pass
    
    async def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        return self.capabilities.copy()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status and performance metrics"""
        avg_execution_time = (
            self._total_execution_time / self._execution_count 
            if self._execution_count > 0 else 0.0
        )
        
        return {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "initialized": self._initialized,
            "capabilities": self.capabilities,
            "execution_count": self._execution_count,
            "total_execution_time": round(self._total_execution_time, 3),
            "avg_execution_time": round(avg_execution_time, 3),
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size
        }
    
    # Backward compatibility methods for existing agent interface
    @track_async_performance("legacy_agent_process")
    async def process(self, user_input: str, session_id: str = None, 
                     conversation_context: str = None, **kwargs) -> Dict[str, Any]:
        """
        Legacy process method for backward compatibility
        Converts to new interface and back
        """
        # Convert legacy call to new interface
        message = AgentMessage(
            message_type="user_query",
            content={
                "user_input": user_input,
                "session_id": session_id,
                "conversation_context": conversation_context,
                **kwargs
            },
            metadata={"source": "legacy_interface"}
        )
        
        # Process using new interface
        response = await self.process_message(message)
        
        # Convert back to legacy format
        if response.message_type == "error":
            return {
                "success": False,
                "error": response.content.get("error", "Unknown error"),
                "agent": self.agent_name
            }
        
        # Extract legacy response format
        result = response.content.copy()
        result["success"] = True
        result["agent"] = self.agent_name
        
        # Add metadata as separate fields
        if response.metadata:
            result.update({f"meta_{k}": v for k, v in response.metadata.items()})
        
        return result
    
    def _create_success_response(self, content: Dict[str, Any], 
                               metadata: Dict[str, Any] = None) -> AgentMessage:
        """Helper to create success response"""
        return AgentMessage(
            message_type="success",
            content=content,
            metadata=metadata or {}
        )
    
    def _create_error_response(self, error: str, 
                             metadata: Dict[str, Any] = None) -> AgentMessage:
        """Helper to create error response"""
        return AgentMessage(
            message_type="error",
            content={"error": error},
            metadata=metadata or {}
        )


class AgentPool:
    """
    Pool for managing multiple optimized agents with load balancing
    """
    
    def __init__(self):
        self.agents: Dict[str, List[OptimizedBaseAgent]] = {}
        self.round_robin_counters: Dict[str, int] = {}
    
    def add_agent(self, agent: OptimizedBaseAgent):
        """Add agent to pool"""
        agent_type = agent.agent_name
        if agent_type not in self.agents:
            self.agents[agent_type] = []
            self.round_robin_counters[agent_type] = 0
        
        self.agents[agent_type].append(agent)
    
    def get_agent(self, agent_type: str) -> Optional[OptimizedBaseAgent]:
        """Get agent using round-robin load balancing"""
        if agent_type not in self.agents or not self.agents[agent_type]:
            return None
        
        # Round-robin selection
        agents_list = self.agents[agent_type]
        counter = self.round_robin_counters[agent_type]
        agent = agents_list[counter % len(agents_list)]
        
        self.round_robin_counters[agent_type] = (counter + 1) % len(agents_list)
        
        return agent
    
    async def process_message_with_load_balancing(self, agent_type: str, 
                                                message: AgentMessage) -> AgentMessage:
        """Process message with automatic load balancing"""
        agent = self.get_agent(agent_type)
        if not agent:
            return AgentMessage(
                message_type="error",
                content={"error": f"No agents available for type: {agent_type}"},
                metadata={"requested_type": agent_type}
            )
        
        return await agent.process_message(message)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get status of all agents in pool"""
        status = {}
        for agent_type, agents_list in self.agents.items():
            status[agent_type] = {
                "count": len(agents_list),
                "agents": [agent.agent_id for agent in agents_list]
            }
        return status


# Global agent pool instance
agent_pool = AgentPool()
