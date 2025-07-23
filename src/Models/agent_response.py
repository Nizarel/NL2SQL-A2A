from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
import uuid


class FormattedResults(BaseModel):
    """Enhanced formatted results for both agent responses and conversations"""
    success: bool = True  # Renamed from 'status' for clarity
    headers: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    total_rows: int = 0
    chart_recommendations: List[str] = Field(default_factory=list)  # For UI guidance
    
    # Legacy support
    @property
    def status(self) -> str:
        """Legacy property for backward compatibility"""
        return "success" if self.success else "error"


class LogTokens(BaseModel):
    """Token usage tracking for agent responses"""
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_tokens: int = 0


class AgentResponse(BaseModel):
    """Enhanced agent response with conversation-ready fields"""
    agent_type: str
    response: str
    success: bool = True
    error_message: Optional[str] = None
    processing_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    formatted_results: Optional[FormattedResults] = Field(None, description="Formatted results for UI or downstream processing")
    
    # Enhanced business-focused fields for conversations
    executive_summary: Optional[str] = Field(None, description="Business-focused summary")
    key_insights: List[str] = Field(default_factory=list, description="Key business insights")
    recommendations: List[str] = Field(default_factory=list, description="Business recommendations")
    confidence_level: str = Field("medium", description="Response confidence: low, medium, high")
    
    # Legacy fields (maintained for backward compatibility)
    summary: Optional[Any] = Field(None, description="Legacy summary field")
    insights: Optional[Any] = Field(None, description="Legacy insights field") 
    
    # Performance tracking
    tokens: Optional[LogTokens] = None
    execution_time_ms: Optional[int] = None
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        if 'timestamp' in data and isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data


class Session(BaseModel):
    """Session model for Cosmos DB storage with hierarchical partitioning"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_name: Optional[str] = None
    is_active: bool = True
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Use session_id as the document id for Cosmos DB"""
        return self.session_id
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        for field in ['created_at', 'last_activity']:
            if field in data and data[field] is not None:
                if isinstance(data[field], datetime):
                    data[field] = data[field].isoformat()
        return data


class Message(BaseModel):
    """Message model for chat conversations with hierarchical partitioning"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    role: str  # "user" or "assistant"
    content: Union[str, Dict[str, Any]]  # Support both string and structured content
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Use message_id as the document id for Cosmos DB"""
        return self.message_id
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        if 'timestamp' in data and isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data


class CacheItem(BaseModel):
    """Cache item model for vector embeddings and other cached data"""
    key: str
    value: str  # JSON serialized value
    embedding: Optional[List[float]] = None  # Vector embedding array for /embedding path
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    @property
    def id(self) -> str:
        """Use key as the document id for Cosmos DB"""
        return self.key
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        for field in ['created_at', 'expires_at']:
            if field in data and data[field] is not None:
                if isinstance(data[field], datetime):
                    data[field] = data[field].isoformat()
        return data


class UserSession(BaseModel):
    """User session model for Cosmos DB storage"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    total_turns: int = 0
    is_active: bool = True
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Use session_id as the document id for Cosmos DB"""
        return self.session_id
    
    @property
    def partition_key(self) -> str:
        """Simple partition key for backward compatibility"""
        return self.user_id
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        for field in ['start_time', 'last_activity', 'end_time']:
            if field in data and data[field] is not None:
                if isinstance(data[field], datetime):
                    data[field] = data[field].isoformat()
        return data

class ChatLogEntry(BaseModel):
    """Enhanced chat log entry model with hierarchical partitioning support"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    user_input: str
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_turn: int = 1
    vector_cache_hit: bool = False
    processing_time_ms: Optional[int] = None
    
    @property
    def partition_key(self) -> str:
        """Hierarchical partition key for better Cosmos DB distribution"""
        return f"{self.user_id}/{self.session_id}"
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        # Convert datetime objects to ISO format strings
        if 'timestamp' in data and isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        for response in data.get('agent_responses', []):
            if 'timestamp' in response and isinstance(response['timestamp'], datetime):
                response['timestamp'] = response['timestamp'].isoformat()
        return data

    class Config:
        model_config = ConfigDict(
            json_encoders={
                datetime: lambda v: v.isoformat()
            }
        )


# ======================================
# Conversation-Specific Models
# ======================================

class ConversationPerformance(BaseModel):
    """High-level performance metrics for conversations"""
    processing_time_ms: int = 0
    cache_hits: List[str] = Field(default_factory=list)
    success: bool = True
    query_complexity: str = "simple"  # simple, medium, complex
    cache_efficiency: float = 0.0


class ConversationMetadata(BaseModel):
    """Metadata for conversation categorization and tracking"""
    conversation_type: str = "general_analytics"
    result_quality: str = "medium"  # low, medium, high
    user_satisfaction: Optional[int] = None  # 1-5 rating (can be set later)
    follow_up_suggested: bool = False
    conversation_turn: int = 1
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ConversationLog(BaseModel):
    """Complete conversation log entry - business focused only"""
    id: str = Field(default_factory=lambda: f"conversation_{uuid.uuid4()}")
    user_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # User interaction
    user_input: str
    
    # Business results (using existing FormattedResults)
    formatted_results: Optional[FormattedResults] = None
    
    # AI response (using enhanced AgentResponse)
    agent_response: Optional[AgentResponse] = None
    
    # Performance summary
    performance: ConversationPerformance = Field(default_factory=ConversationPerformance)
    
    # Conversation metadata
    metadata: ConversationMetadata = Field(default_factory=ConversationMetadata)
    
    # Document type for Cosmos DB
    type: str = "nl2sql_conversation"
    
    @property
    def partition_key(self) -> str:
        """Hierarchical partition key for Cosmos DB"""
        return f"{self.user_id}/{self.session_id}"
    
    def to_cosmos_dict(self) -> Dict[str, Any]:
        """Convert to Cosmos DB compatible dictionary"""
        data = self.model_dump()
        
        # Ensure datetime is properly serialized
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        
        return data
    
    @classmethod
    def from_cosmos_dict(cls, data: Dict[str, Any]) -> "ConversationLog":
        """Create from Cosmos DB dictionary"""
        # Handle datetime parsing
        if isinstance(data.get("timestamp"), str):
            try:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                data["timestamp"] = datetime.now(timezone.utc)
        
        # Handle nested objects with validation fallbacks
        if "agent_response" in data and isinstance(data["agent_response"], dict):
            agent_data = data["agent_response"]
            # Ensure required fields are present with defaults
            if "agent_type" not in agent_data:
                agent_data["agent_type"] = "unknown"
            if "response" not in agent_data:
                agent_data["response"] = "No response available"
            data["agent_response"] = AgentResponse(**agent_data)
        
        if "formatted_results" in data and isinstance(data["formatted_results"], dict):
            data["formatted_results"] = FormattedResults(**data["formatted_results"])
        
        if "performance" in data and isinstance(data["performance"], dict):
            data["performance"] = ConversationPerformance(**data["performance"])
        
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = ConversationMetadata(**data["metadata"])
        
        return cls(**data)
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        if 'timestamp' in data and isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data


class BusinessAnalytics(BaseModel):
    """Business analytics extracted from conversations"""
    user_id: str
    total_conversations: int = 0
    successful_queries: int = 0
    average_processing_time_ms: float = 0.0
    average_response_time_ms: float = 0.0  # Alias for compatibility
    most_common_query_types: List[str] = Field(default_factory=list)
    average_result_quality: str = "medium"
    cache_efficiency: float = 0.0
    query_complexity_distribution: Dict[str, int] = Field(default_factory=dict)
    conversation_type_distribution: Dict[str, int] = Field(default_factory=dict)
    most_common_insights: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    time_period_days: int = 30
    last_activity: Optional[datetime] = None
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        if 'last_activity' in data and isinstance(data['last_activity'], datetime):
            data['last_activity'] = data['last_activity'].isoformat()
        return data


# ======================================
# Workflow Context Models
# ======================================

class WorkflowContext(BaseModel):
    """Complete workflow context for memory tracking"""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    user_input: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_turn: int = 1
    
    # Workflow stages (will be populated during processing)
    schema_analysis: Optional[Dict[str, Any]] = None
    sql_generation: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    summarization: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    total_processing_time_ms: Optional[int] = None
    cache_hits: List[str] = Field(default_factory=list)
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    
    # Metadata
    workflow_version: str = "v1.0"
    agent_versions: Dict[str, str] = Field(default_factory=dict)
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        if 'timestamp' in data and isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data