"""
Cosmos DB specific models for NL2SQL application.
These models follow the pattern from the C# implementation.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
import uuid


class CacheItem(BaseModel):
    """Cache item model for vector-based semantic caching"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    completion: str
    vectors: List[float]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    similarity_score: Optional[float] = None
    
    @property
    def partition_key(self) -> str:
        """Use id as partition key for cache items"""
        return self.id
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        if 'timestamp' in data and isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class Session(BaseModel):
    """Session model compatible with Cosmos DB hierarchical partitioning"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str = "New Chat Session"
    created_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tokens_used: int = 0
    type: str = "Session"  # Document type discriminator
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure id matches session_id for consistency
        if 'session_id' in data and 'id' not in data:
            self.id = data['session_id']
        elif 'id' in data and 'session_id' not in data:
            self.session_id = data['id']
    
    @property
    def partition_key(self) -> List[str]:
        """Hierarchical partition key [user_id, session_id]"""
        return [self.user_id, self.session_id]
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        for field in ['created_timestamp', 'last_activity_timestamp']:
            if field in data and data[field] is not None:
                if isinstance(data[field], datetime):
                    data[field] = data[field].isoformat()
        return data

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class Message(BaseModel):
    """Message model compatible with Cosmos DB hierarchical partitioning"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sender: str  # "User" or "Assistant"
    text: str
    tokens: int = 0
    type: str = "Message"  # Document type discriminator
    
    @property
    def partition_key(self) -> List[str]:
        """Hierarchical partition key [user_id, session_id]"""
        return [self.user_id, self.session_id]
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        if 'timestamp' in data and isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class ChatContext(BaseModel):
    """Extended chat log entry for comprehensive conversation tracking"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    user_input: str
    assistant_response: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_turn: int = 1
    vector_cache_hit: bool = False
    processing_time_ms: Optional[int] = None
    tokens_used: int = 0
    sql_query: Optional[str] = None
    execution_results: Optional[Dict[str, Any]] = None
    agent_metadata: Dict[str, Any] = Field(default_factory=dict)
    type: str = "ChatContext"  # Document type discriminator
    
    @property
    def partition_key(self) -> List[str]:
        """Hierarchical partition key [user_id, session_id]"""
        return [self.user_id, self.session_id]
    
    def model_dump(self, **kwargs):
        """Custom model_dump method to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        if 'timestamp' in data and isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
