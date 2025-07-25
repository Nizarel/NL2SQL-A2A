"""
Unified Agent Communication Interface
Provides standardized message format and protocol for inter-agent communication
"""

from typing import Dict, Any, Protocol, Optional
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AgentOperation(Enum):
    """Standard agent operations"""
    ANALYZE_SCHEMA = "analyze_schema"
    GENERATE_SQL = "generate_sql"
    EXECUTE_SQL = "execute_sql"
    SUMMARIZE = "summarize"
    ANALYZE_CONTEXT = "analyze_context"


class WorkflowStage(Enum):
    """Workflow execution stages"""
    INITIALIZATION = "initialization"
    CONTEXT_ANALYSIS = "context_analysis"
    SCHEMA_ANALYSIS = "schema_analysis"
    SQL_GENERATION = "sql_generation"
    SQL_EXECUTION = "sql_execution"
    RESULT_SUMMARIZATION = "result_summarization"
    FINALIZATION = "finalization"


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution"""
    parallel_execution: bool = True
    max_parallel_agents: int = 3
    enable_caching: bool = True
    performance_monitoring: bool = True
    timeout_seconds: int = 120
    retry_attempts: int = 2
    workflow_stages: list = field(default_factory=lambda: [
        WorkflowStage.CONTEXT_ANALYSIS,
        WorkflowStage.SCHEMA_ANALYSIS,
        WorkflowStage.SQL_GENERATION,
        WorkflowStage.SQL_EXECUTION,
        WorkflowStage.RESULT_SUMMARIZATION
    ])


@dataclass
class AgentMessage:
    """Standardized message format for inter-agent communication"""
    message_type: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_id: str = field(default_factory=lambda: f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "message_type": self.message_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create from dictionary"""
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        return cls(
            message_type=data["message_type"],
            content=data["content"],
            metadata=data.get("metadata"),
            timestamp=timestamp,
            message_id=data.get("message_id", f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        )


@dataclass
class AgentResponse:
    """Standardized response format from agents"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[float] = None
    agent_name: Optional[str] = None
    
    def to_message(self, operation: str, target_agent: str = "orchestrator") -> AgentMessage:
        """Convert response to message format"""
        return AgentMessage(
            agent_name=self.agent_name or "unknown",
            operation=f"{operation}_response",
            payload=self.to_dict(),
            context={"target_agent": target_agent}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
            "agent_name": self.agent_name
        }


class AgentProtocol(Protocol):
    """Protocol defining the interface all agents must implement"""
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming message and return standardized response"""
        ...
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities and supported operations"""
        ...
    
    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return agent name"""
        ...


class BaseAgentAdapter:
    """
    Base adapter to help existing agents conform to the new protocol
    without breaking existing functionality
    """
    
    def __init__(self, legacy_agent, agent_name: str):
        self.legacy_agent = legacy_agent
        self._agent_name = agent_name
        self._capabilities = self._detect_capabilities()
    
    @property
    def agent_name(self) -> str:
        return self._agent_name
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return detected capabilities"""
        return self._capabilities
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect capabilities from legacy agent"""
        capabilities = {
            "operations": [],
            "input_formats": ["dict"],
            "output_formats": ["dict"],
            "supports_async": hasattr(self.legacy_agent, 'process') and callable(self.legacy_agent.process)
        }
        
        # Detect supported operations based on agent type
        agent_type = self._agent_name.lower()
        if "schema" in agent_type:
            capabilities["operations"] = [AgentOperation.ANALYZE_SCHEMA.value]
        elif "sql" in agent_type or "generator" in agent_type:
            capabilities["operations"] = [AgentOperation.GENERATE_SQL.value]
        elif "executor" in agent_type:
            capabilities["operations"] = [AgentOperation.EXECUTE_SQL.value]
        elif "summar" in agent_type:
            capabilities["operations"] = [AgentOperation.SUMMARIZE.value]
        
        return capabilities
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process message using legacy agent interface
        This maintains backward compatibility
        """
        start_time = datetime.now()
        
        try:
            # Convert message to legacy format
            legacy_input = self._convert_message_to_legacy_format(message)
            
            # Call legacy agent
            if hasattr(self.legacy_agent, 'process') and callable(self.legacy_agent.process):
                result = await self.legacy_agent.process(legacy_input)
            else:
                raise NotImplementedError(f"Agent {self._agent_name} does not support process method")
            
            # Convert legacy result to new format
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AgentResponse(
                success=result.get("success", False),
                data=result.get("data", {}),
                error=result.get("error"),
                metadata=result.get("metadata", {}),
                processing_time_ms=processing_time,
                agent_name=self._agent_name
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return AgentResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time,
                agent_name=self._agent_name
            )
    
    def _convert_message_to_legacy_format(self, message: AgentMessage) -> Dict[str, Any]:
        """Convert new message format to legacy input format"""
        # Start with the payload as base
        legacy_input = message.payload.copy()
        
        # Add context if available
        if message.context:
            legacy_input.update(message.context)
        
        # Ensure required fields for legacy agents
        if "question" not in legacy_input and "query" in legacy_input:
            legacy_input["question"] = legacy_input["query"]
        
        return legacy_input


def create_agent_adapter(legacy_agent, agent_name: str) -> BaseAgentAdapter:
    """
    Factory function to create agent adapters
    """
    return BaseAgentAdapter(legacy_agent, agent_name)
