"""
Orchestrator Memory Service for NL2SQL Application.

This service provides intelligent memory management for the NL2SQL orchestrator,
including session management, query history, result caching, and semantic similarity search.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4
import hashlib
import json

from pydantic import BaseModel, Field

from .cosmos_db_service import CosmosDbService, CosmosDbConfig
from Models.agent_response import Session, Message, CacheItem, AgentResponse

logger = logging.getLogger(__name__)


class QueryContext(BaseModel):
    """Context information for a user query."""
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_id: str
    original_query: str
    processed_query: Optional[str] = None
    embedding: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Result of a processed query."""
    query_id: str
    sql_query: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    agent_response: Optional[AgentResponse] = None
    execution_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SimilarQuery(BaseModel):
    """Information about a similar query found in history."""
    query_id: str
    original_query: str
    similarity_score: float
    result: Optional[QueryResult] = None
    timestamp: datetime


class OrchestratorMemoryService:
    """
    Memory service for the NL2SQL orchestrator that provides:
    - Session and conversation management
    - Query history and result caching
    - Semantic similarity search for query reuse
    - Context-aware query processing
    """
    
    def __init__(self, cosmos_service: CosmosDbService):
        """
        Initialize the orchestrator memory service.
        
        Args:
            cosmos_service: Initialized Cosmos DB service instance
        """
        self.cosmos_service = cosmos_service
        self._embedding_cache = {}  # In-memory cache for embeddings
        
    @classmethod
    async def create_from_config(cls, config: CosmosDbConfig = None) -> 'OrchestratorMemoryService':
        """
        Create an orchestrator memory service with automatic configuration.
        
        Args:
            config: Optional Cosmos DB configuration, will load from env if not provided
            
        Returns:
            Initialized orchestrator memory service
        """
        if config is None:
            config = CosmosDbConfig.from_env()
        
        cosmos_service = CosmosDbService(
            endpoint=config.endpoint,
            database_name=config.database_name,
            chat_container_name=config.chat_container_name,
            cache_container_name=config.cache_container_name
        )
        
        await cosmos_service.initialize()
        return cls(cosmos_service)
    
    async def close(self):
        """Close the underlying services."""
        if self.cosmos_service:
            await self.cosmos_service.close()
    
    # Session Management
    async def create_session(self, user_id: str, session_title: str = None) -> Session:
        """
        Create a new chat session for a user.
        
        Args:
            user_id: ID of the user
            session_title: Optional title for the session
            
        Returns:
            Created session object
        """
        session = Session(
            session_id=str(uuid4()),
            user_id=user_id,
            session_name=session_title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            created_at=datetime.now(timezone.utc)
        )
        
        return await self.cosmos_service.insert_session_async(user_id, session)
    
    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for a user."""
        return await self.cosmos_service.get_sessions_async(user_id)
    
    async def get_session_context(self, user_id: str, session_id: str, max_messages: int = 10) -> List[Message]:
        """Get recent conversation context for a session."""
        return await self.cosmos_service.get_session_context_window_async(
            user_id, session_id, max_messages
        )
    
    # Query Processing and History
    async def process_query(
        self, 
        user_id: str, 
        session_id: str, 
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> QueryContext:
        """
        Process a new user query and store it in history.
        
        Args:
            user_id: ID of the user
            session_id: Session identifier
            query: Natural language query
            query_embedding: Optional pre-computed embedding for the query
            
        Returns:
            Query context object
        """
        query_context = QueryContext(
            user_id=user_id,
            session_id=session_id,
            original_query=query,
            embedding=query_embedding,
            metadata={
                "source": "orchestrator",
                "query_length": len(query),
                "word_count": len(query.split())
            }
        )
        
        # Store the user message
        user_message = Message(
            message_id=str(uuid4()),
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=query,
            timestamp=datetime.now(timezone.utc)
        )
        
        await self.cosmos_service.insert_message_async(user_id, user_message)
        
        # Cache the query context for potential similarity search
        if query_embedding:
            await self._cache_query_context(query_context)
        
        logger.info(f"Processed query for user {user_id} in session {session_id}")
        return query_context
    
    async def store_query_result(
        self, 
        query_context: QueryContext, 
        result: QueryResult
    ) -> None:
        """
        Store the result of a processed query.
        
        Args:
            query_context: Original query context
            result: Query execution result
        """
        # Store the assistant response message
        assistant_message = Message(
            message_id=str(uuid4()),
            session_id=query_context.session_id,
            user_id=query_context.user_id,
            role="assistant",
            content=self._format_result_for_storage(result),
            timestamp=datetime.now(timezone.utc),
            metadata={
                "query_id": query_context.query_id,
                "execution_time_ms": result.execution_time_ms,
                "success": result.success
            }
        )
        
        await self.cosmos_service.insert_message_async(
            query_context.user_id, 
            assistant_message
        )
        
        # Cache the complete query-result pair for future similarity search
        await self._cache_query_result(query_context, result)
        
        logger.info(f"Stored result for query {query_context.query_id}")
    
    # Similarity Search and Query Reuse
    async def find_similar_queries(
        self, 
        query_embedding: List[float],
        user_id: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[SimilarQuery]:
        """
        Find similar queries based on embedding similarity.
        
        Args:
            query_embedding: Embedding vector of the query
            user_id: Optional user ID to filter results (None for all users)
            limit: Maximum number of similar queries to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar queries with their results
        """
        try:
            # Search for similar embeddings
            cache_items = await self.cosmos_service.search_similar_embeddings_async(
                query_embedding=query_embedding,
                limit=limit * 2,  # Get more results to filter by user
                similarity_threshold=similarity_threshold
            )
            
            similar_queries = []
            for item in cache_items:
                # Parse the cached query context
                if item.metadata and "query_context" in item.metadata:
                    try:
                        query_data = item.metadata["query_context"]
                        
                        # Filter by user if specified
                        if user_id and query_data.get("user_id") != user_id:
                            continue
                        
                        # Get similarity score
                        similarity_score = item.metadata.get("similarity_score", 0.0)
                        
                        similar_query = SimilarQuery(
                            query_id=query_data["query_id"],
                            original_query=query_data["original_query"],
                            similarity_score=similarity_score,
                            timestamp=datetime.fromisoformat(query_data["timestamp"].replace('Z', '+00:00'))
                        )
                        
                        # Try to get the associated result
                        if "result" in item.metadata:
                            result_data = item.metadata["result"]
                            similar_query.result = QueryResult(**result_data)
                        
                        similar_queries.append(similar_query)
                        
                        if len(similar_queries) >= limit:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error parsing cached query: {str(e)}")
                        continue
            
            logger.info(f"Found {len(similar_queries)} similar queries with threshold {similarity_threshold}")
            return similar_queries
            
        except Exception as e:
            logger.error(f"Error finding similar queries: {str(e)}")
            return []
    
    async def get_cached_result(self, query_hash: str) -> Optional[QueryResult]:
        """
        Get a cached result by query hash.
        
        Args:
            query_hash: Hash of the normalized query
            
        Returns:
            Cached result if found, None otherwise
        """
        cache_key = f"result_{query_hash}"
        cache_item = await self.cosmos_service.get_cache_item_async(cache_key)
        
        if cache_item and cache_item.metadata:
            try:
                result_data = cache_item.metadata.get("result")
                if result_data:
                    return QueryResult(**result_data)
            except Exception as e:
                logger.warning(f"Error parsing cached result: {str(e)}")
        
        return None
    
    # Utility Methods
    async def _cache_query_context(self, query_context: QueryContext) -> None:
        """Cache query context for similarity search."""
        if not query_context.embedding:
            return
        
        cache_key = f"query_{query_context.query_id}"
        metadata = {
            "type": "query_context",
            "query_context": query_context.model_dump(mode='json'),
            "user_id": query_context.user_id,
            "session_id": query_context.session_id
        }
        
        await self.cosmos_service.set_vector_embedding_async(
            key=cache_key,
            embedding=query_context.embedding,
            text=query_context.original_query,
            metadata=metadata
        )
    
    async def _cache_query_result(self, query_context: QueryContext, result: QueryResult) -> None:
        """Cache complete query-result pair."""
        if not query_context.embedding:
            return
        
        cache_key = f"query_result_{query_context.query_id}"
        metadata = {
            "type": "query_result",
            "query_context": query_context.model_dump(mode='json'),
            "result": result.model_dump(mode='json'),
            "user_id": query_context.user_id,
            "session_id": query_context.session_id
        }
        
        await self.cosmos_service.set_vector_embedding_async(
            key=cache_key,
            embedding=query_context.embedding,
            text=f"{query_context.original_query} -> {result.sql_query or 'No SQL'}",
            metadata=metadata
        )
    
    def _format_result_for_storage(self, result: QueryResult) -> str:
        """Format query result for message storage."""
        if not result.success:
            return f"Error: {result.error_message}"
        
        if result.agent_response:
            return result.agent_response.response or "Query executed successfully"
        
        if result.execution_result:
            # Format execution result as a readable string
            return f"Query executed successfully. Results: {json.dumps(result.execution_result, default=str)[:500]}..."
        
        return "Query executed successfully"
    
    @staticmethod
    def create_query_hash(query: str) -> str:
        """Create a hash for query normalization and caching."""
        # Normalize query (lowercase, strip, remove extra spaces)
        normalized = ' '.join(query.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    # Analytics and Insights
    async def get_user_query_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's query patterns."""
        try:
            sessions = await self.get_user_sessions(user_id)
            
            total_sessions = len(sessions)
            total_messages = 0
            
            # Count messages across all sessions
            for session in sessions[:10]:  # Limit to recent sessions for performance
                messages = await self.cosmos_service.get_session_messages_async(
                    user_id, session.session_id
                )
                total_messages += len([m for m in messages if m.role == "user"])
            
            return {
                "user_id": user_id,
                "total_sessions": total_sessions,
                "total_queries": total_messages,
                "average_queries_per_session": total_messages / max(total_sessions, 1),
                "most_recent_session": sessions[0].created_at if sessions else None
            }
            
        except Exception as e:
            logger.error(f"Error getting user stats: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Clean up old data beyond the retention period.
        
        Args:
            days_to_keep: Number of days to keep data
            
        Returns:
            Statistics about cleaned data
        """
        # This would implement cleanup logic based on timestamps
        # For now, return a placeholder
        logger.info(f"Cleanup requested for data older than {days_to_keep} days")
        return {"sessions_cleaned": 0, "messages_cleaned": 0, "cache_items_cleaned": 0}
