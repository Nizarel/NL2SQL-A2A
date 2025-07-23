"""
Azure Cosmos DB service for NL2SQL application.
Provides session and user tracking with chat logs and caching.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from uuid import uuid4

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.identity.aio import DefaultAzureCredential
from pydantic import BaseModel

from Models.agent_response import (
    ChatLogEntry, UserSession, AgentResponse, Session, Message, CacheItem,
    ConversationLog, BusinessAnalytics
)

logger = logging.getLogger(__name__)


class CosmosDbService:
    """Service to access Azure Cosmos DB for NoSQL with Azure Identity authentication."""
    
    def __init__(
        self,
        endpoint: str,
        database_name: str,
        chat_container_name: str = "nl2sql_chatlogs",
        cache_container_name: str = "nl2sql_cache"
    ):
        """
        Initialize the Cosmos DB service using Azure Identity.
        
        Args:
            endpoint: Cosmos DB account endpoint URL
            database_name: Name of the Cosmos DB database
            chat_container_name: Name of the chat logs container
            cache_container_name: Name of the cache container
        """
        self.endpoint = endpoint
        self.database_name = database_name
        self.chat_container_name = chat_container_name
        self.cache_container_name = cache_container_name
        
        self._credential = None
        self._client = None
        self._database = None
        self._chat_container = None
        self._cache_container = None
        
    async def initialize(self):
        """Initialize the Cosmos DB client and containers."""
        try:
            # Create Azure Identity credential
            self._credential = DefaultAzureCredential()
            
            # Create async Cosmos client with Azure Identity
            self._client = AsyncCosmosClient(
                url=self.endpoint,
                credential=self._credential
            )
            
            # Get database reference
            self._database = self._client.get_database_client(self.database_name)
            
            # Get container references
            self._chat_container = self._database.get_container_client(self.chat_container_name)
            self._cache_container = self._database.get_container_client(self.cache_container_name)
            
            logger.info(f"Cosmos DB service initialized successfully for database: {self.database_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB service: {str(e)}")
            raise
    
    async def close(self):
        """Close the Cosmos DB client and credential."""
        try:
            if self._client:
                await self._client.close()
            if self._credential:
                await self._credential.close()
            logger.info("Cosmos DB service closed successfully")
        except Exception as e:
            logger.error(f"Error closing Cosmos DB service: {str(e)}")
    
    @staticmethod
    def _get_partition_key(user_id: str, session_id: str = None) -> List[str]:
        """
        Helper function to generate hierarchical partition key based on parameters.
        
        Args:
            user_id: ID of the user
            session_id: Session ID (optional)
            
        Returns:
            List representing the partition key
        """
        if user_id and session_id:
            return [user_id, session_id]
        elif user_id:
            return [user_id]
        else:
            raise ValueError("user_id is required for partition key")
    
    async def insert_session_async(self, user_id: str, session: Session) -> Session:
        """
        Create a new chat session.
        
        Args:
            user_id: ID of the user
            session: Session object to create
            
        Returns:
            Newly created session object
        """
        try:
            # Ensure session has required fields
            session_dict = session.model_dump()
            session_dict["id"] = session.session_id
            session_dict["user_id"] = user_id
            session_dict["type"] = "session"
            
            # Create the session in Cosmos DB
            created_item = await self._chat_container.create_item(
                body=session_dict
            )
            
            logger.info(f"Session created successfully: {session.session_id}")
            return Session.model_validate(created_item)
            
        except CosmosHttpResponseError as e:
            logger.error(f"Error creating session: {str(e)}")
            raise
    
    async def insert_message_async(self, user_id: str, message: Message) -> Message:
        """
        Create a new chat message.
        
        Args:
            user_id: ID of the user
            message: Message object to create
            
        Returns:
            Newly created message object
        """
        try:
            # Ensure message has required fields
            message_dict = message.model_dump()
            message_dict["id"] = message.message_id
            message_dict["user_id"] = user_id
            message_dict["type"] = "message"
            
            # Create the message in Cosmos DB
            created_item = await self._chat_container.create_item(
                body=message_dict
            )
            
            logger.info(f"Message created successfully: {message.message_id}")
            return Message.model_validate(created_item)
            
        except CosmosHttpResponseError as e:
            logger.error(f"Error creating message: {str(e)}")
            raise
    
    async def get_sessions_async(self, user_id: str) -> List[Session]:
        """
        Get a list of all current chat sessions for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of session objects
        """
        try:
            query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.type = 'session' ORDER BY c.created_at DESC"
            parameters = [{"name": "@user_id", "value": user_id}]
            
            sessions = []
            async for item in self._chat_container.query_items(
                query=query,
                parameters=parameters
            ):
                sessions.append(Session.model_validate(item))
            
            logger.info(f"Retrieved {len(sessions)} sessions for user: {user_id}")
            return sessions
            
        except CosmosHttpResponseError as e:
            logger.error(f"Error retrieving sessions: {str(e)}")
            raise
    
    async def get_session_messages_async(self, user_id: str, session_id: str) -> List[Message]:
        """
        Get all chat messages for a specified session.
        
        Args:
            user_id: ID of the user
            session_id: Session identifier
            
        Returns:
            List of message objects for the session
        """
        try:
            query = """
            SELECT * FROM c 
            WHERE c.user_id = @user_id 
            AND c.session_id = @session_id 
            AND c.type = 'message' 
            ORDER BY c.timestamp ASC
            """
            parameters = [
                {"name": "@user_id", "value": user_id},
                {"name": "@session_id", "value": session_id}
            ]
            
            messages = []
            async for item in self._chat_container.query_items(
                query=query,
                parameters=parameters
            ):
                messages.append(Message.model_validate(item))
            
            logger.info(f"Retrieved {len(messages)} messages for session: {session_id}")
            return messages
            
        except CosmosHttpResponseError as e:
            logger.error(f"Error retrieving session messages: {str(e)}")
            raise
    
    async def get_session_context_window_async(
        self, 
        user_id: str, 
        session_id: str, 
        max_context_window: int = 10
    ) -> List[Message]:
        """
        Get the current context window of chat messages for a session.
        
        Args:
            user_id: ID of the user
            session_id: Session identifier
            max_context_window: Maximum number of messages to return
            
        Returns:
            List of recent message objects for the session
        """
        try:
            query = """
            SELECT TOP @max_context_window * FROM c 
            WHERE c.user_id = @user_id 
            AND c.session_id = @session_id 
            AND c.type = 'message' 
            ORDER BY c.timestamp DESC
            """
            parameters = [
                {"name": "@user_id", "value": user_id},
                {"name": "@session_id", "value": session_id},
                {"name": "@max_context_window", "value": max_context_window}
            ]
            
            messages = []
            async for item in self._chat_container.query_items(
                query=query,
                parameters=parameters
            ):
                messages.append(Message.model_validate(item))
            
            # Reverse to get chronological order
            messages.reverse()
            
            logger.info(f"Retrieved {len(messages)} context messages for session: {session_id}")
            return messages
            
        except CosmosHttpResponseError as e:
            logger.error(f"Error retrieving session context: {str(e)}")
            raise
    
    async def delete_session_async(self, user_id: str, session_id: str) -> bool:
        """
        Delete a chat session and all its messages.
        
        Args:
            user_id: ID of the user
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        try:
            # First, delete all messages in the session
            messages = await self.get_session_messages_async(user_id, session_id)
            for message in messages:
                await self._chat_container.delete_item(
                    item=message.message_id,
                    partition_key=[user_id, session_id]
                )
            
            # Then delete the session
            await self._chat_container.delete_item(
                item=session_id,
                partition_key=[user_id, session_id]
            )
            
            logger.info(f"Session deleted successfully: {session_id}")
            return True
            
        except CosmosResourceNotFoundError:
            logger.warning(f"Session not found for deletion: {session_id}")
            return False
        except CosmosHttpResponseError as e:
            logger.error(f"Error deleting session: {str(e)}")
            raise
    
    # Cache operations
    async def get_cache_item_async(self, cache_key: str) -> Optional[CacheItem]:
        """
        Get a cache item by key.
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            CacheItem if found, None otherwise
        """
        try:
            item = await self._cache_container.read_item(
                item=cache_key,
                partition_key=cache_key
            )
            return CacheItem.model_validate(item)
            
        except CosmosResourceNotFoundError:
            logger.debug(f"Cache item not found: {cache_key}")
            return None
        except CosmosHttpResponseError as e:
            logger.error(f"Error retrieving cache item: {str(e)}")
            raise
    
    async def set_cache_item_async(self, cache_item: CacheItem) -> CacheItem:
        """
        Store a cache item.
        
        Args:
            cache_item: CacheItem to store
            
        Returns:
            Stored CacheItem
        """
        try:
            cache_dict = cache_item.model_dump()
            cache_dict["id"] = cache_item.key
            
            created_item = await self._cache_container.upsert_item(
                body=cache_dict
            )
            
            logger.debug(f"Cache item stored: {cache_item.key}")
            return CacheItem.model_validate(created_item)
            
        except CosmosHttpResponseError as e:
            logger.error(f"Error storing cache item: {str(e)}")
            raise
    
    async def set_vector_embedding_async(
        self, 
        key: str, 
        embedding: List[float], 
        text: str = None,
        metadata: Dict[str, Any] = None
    ) -> CacheItem:
        """
        Store a vector embedding for semantic search.
        
        Args:
            key: Unique key for the embedding
            embedding: Vector embedding array
            text: Original text that was embedded
            metadata: Additional metadata
            
        Returns:
            Stored CacheItem with vector embedding
        """
        try:
            cache_item = CacheItem(
                key=key,
                value=text or "",
                embedding=embedding,
                metadata=metadata or {"type": "vector_embedding"}
            )
            
            cache_dict = cache_item.model_dump()
            cache_dict["id"] = key
            
            created_item = await self._cache_container.upsert_item(
                body=cache_dict
            )
            
            logger.debug(f"Vector embedding stored: {key} (dimension: {len(embedding)})")
            return CacheItem.model_validate(created_item)
            
        except CosmosHttpResponseError as e:
            logger.error(f"Error storing vector embedding: {str(e)}")
            raise
    
    async def search_similar_embeddings_async(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[CacheItem]:
        """
        Search for similar vector embeddings using Cosmos DB Vector Search.
        Falls back to manual similarity calculation if VectorDistance is not available.
        
        Args:
            query_embedding: Query vector to find similar embeddings
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of similar cache items with embeddings ordered by similarity
        """
        try:
            # First, try the advanced vector search with VectorDistance function
            query = """
            SELECT TOP @limit c.key, c["value"], c.embedding, c.metadata, c.created_at,
                   VectorDistance(c.embedding, @queryVector) AS similarity_score
            FROM c 
            WHERE c.embedding != null
            ORDER BY VectorDistance(c.embedding, @queryVector)
            """
            
            parameters = [
                {"name": "@limit", "value": limit},
                {"name": "@queryVector", "value": query_embedding}
            ]
            
            similar_items = []
            try:
                async for item in self._cache_container.query_items(
                    query=query,
                    parameters=parameters
                ):
                    # Check if VectorDistance actually returned a value
                    if "similarity_score" in item and item["similarity_score"] is not None:
                        # Add similarity score to metadata
                        cache_item = CacheItem.model_validate(item)
                        if not cache_item.metadata:
                            cache_item.metadata = {}
                        cache_item.metadata["similarity_score"] = item["similarity_score"]
                        
                        # Filter by similarity threshold (lower distance = higher similarity)
                        similarity_score = item["similarity_score"]
                        if similarity_score <= (1.0 - similarity_threshold):
                            similar_items.append(cache_item)
                    else:
                        # VectorDistance didn't return a value, fall back to manual calculation
                        raise ValueError("VectorDistance function not returning values")
                
                if similar_items:
                    logger.info(f"Vector search found {len(similar_items)} similar embeddings with threshold {similarity_threshold}")
                    return similar_items
                else:
                    logger.info("VectorDistance query worked but no items met similarity threshold")
                    
            except Exception as vector_error:
                logger.warning(f"VectorDistance function not working properly: {str(vector_error)}")
                # Fall through to manual calculation
                
            # Manual similarity calculation fallback
            logger.info("Using manual similarity calculation for vector search")
            return await self._manual_similarity_search(query_embedding, limit, similarity_threshold)
            
        except CosmosHttpResponseError as e:
            logger.warning(f"Vector search failed, falling back to basic search: {str(e)}")
            # Fallback to regular query without vector search
            return await self._fallback_embedding_search(query_embedding, limit)
    
    async def _manual_similarity_search(
        self, 
        query_embedding: List[float], 
        limit: int,
        similarity_threshold: float
    ) -> List[CacheItem]:
        """
        Manual similarity calculation using cosine similarity.
        """
        import math
        
        def cosine_similarity(vec1, vec2):
            """Calculate cosine similarity between two vectors."""
            # Ensure same length
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            
            return dot_product / (magnitude1 * magnitude2)
        
        try:
            # Get all embeddings
            query = "SELECT c.key, c[\"value\"], c.embedding, c.metadata, c.created_at FROM c WHERE c.embedding != null"
            
            similarities = []
            async for item in self._cache_container.query_items(query=query):
                embedding = item['embedding']
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= similarity_threshold:
                    cache_item = CacheItem.model_validate(item)
                    # Add similarity score to metadata
                    if not cache_item.metadata:
                        cache_item.metadata = {}
                    cache_item.metadata["similarity_score"] = similarity
                    similarities.append((similarity, cache_item))
            
            # Sort by similarity (highest first) and limit results
            similarities.sort(key=lambda x: x[0], reverse=True)
            similar_items = [item for _, item in similarities[:limit]]
            
            logger.info(f"Manual similarity search found {len(similar_items)} similar embeddings")
            return similar_items
            
        except CosmosHttpResponseError as e:
            logger.error(f"Manual similarity search failed: {str(e)}")
            return []
    
    async def _fallback_embedding_search(
        self, 
        query_embedding: List[float], 
        limit: int
    ) -> List[CacheItem]:
        """
        Fallback method for embedding search when vector search is not available.
        """
        try:
            query = "SELECT TOP @limit c.key, c[\"value\"], c.embedding, c.metadata, c.created_at FROM c WHERE c.embedding != null"
            parameters = [{"name": "@limit", "value": limit}]
            
            items = []
            async for item in self._cache_container.query_items(
                query=query,
                parameters=parameters
            ):
                items.append(CacheItem.model_validate(item))
            
            logger.debug(f"Fallback search returned {len(items)} items")
            return items
            
        except CosmosHttpResponseError as e:
            logger.error(f"Fallback embedding search failed: {str(e)}")
            return []
    
    async def delete_cache_item_async(self, cache_key: str) -> bool:
        """
        Delete a cache item.
        
        Args:
            cache_key: Cache key to delete
            
        Returns:
            True if successful
        """
        try:
            await self._cache_container.delete_item(
                item=cache_key,
                partition_key=cache_key
            )
            
            logger.debug(f"Cache item deleted: {cache_key}")
            return True
            
        except CosmosResourceNotFoundError:
            logger.debug(f"Cache item not found for deletion: {cache_key}")
            return False
        except CosmosHttpResponseError as e:
            logger.error(f"Error deleting cache item: {str(e)}")
            raise

    # ======================================
    # Conversation Support Methods
    # ======================================

    async def get_user_conversations_async(
        self, 
        user_id: str, 
        session_id: str = None,
        limit: int = 20,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[ConversationLog]:
        """
        Retrieve conversation logs for a user with optional filtering.
        
        Args:
            user_id: User identifier
            session_id: Optional session filter
            limit: Maximum conversations to return
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of conversation logs
        """
        try:
            conversations = []
            
            if session_id:
                # Query specific session with hierarchical partition key
                conditions = ["c.role = 'nl2sql_conversation'"]
                parameters = []
                
                if start_time:
                    conditions.append("c.timestamp >= @start_time")
                    parameters.append({"name": "@start_time", "value": start_time.isoformat()})
                
                if end_time:
                    conditions.append("c.timestamp <= @end_time")
                    parameters.append({"name": "@end_time", "value": end_time.isoformat()})
                
                where_clause = " AND ".join(conditions)
                
                query = f"""
                    SELECT TOP {limit} * FROM c 
                    WHERE {where_clause}
                    ORDER BY c.timestamp DESC
                """
                
                # Use hierarchical partition key [user_id, session_id]
                items = self._chat_container.query_items(
                    query=query,
                    parameters=parameters,
                    partition_key=[user_id, session_id]
                )
                
                async for item in items:
                    try:
                        # Handle both structured data (new format) and JSON strings (legacy format)
                        if isinstance(item["content"], dict):
                            conversation_data = item["content"]  # Already structured data
                        else:
                            import json
                            conversation_data = json.loads(item["content"])  # Legacy JSON string format
                        
                        conversation = ConversationLog.from_cosmos_dict(conversation_data)
                        conversations.append(conversation)
                    except Exception as e:
                        logger.warning(f"Failed to parse conversation log: {e}")
                        continue
            else:
                # For user-wide queries without session, enumerate all user sessions
                logger.debug(f"Retrieving conversations for user {user_id} across all sessions")
                
                # Get all sessions for the user
                sessions = await self.get_sessions_async(user_id)
                logger.debug(f"Found {len(sessions)} sessions for user {user_id}")
                
                if not sessions:
                    logger.warning(f"No sessions found for user {user_id}")
                    return []
                
                # Query each session for conversations
                for session in sessions:
                    logger.debug(f"Querying session {session.session_id} for conversations")
                    # Build query for this specific session
                    conditions = ["c.role = 'nl2sql_conversation'"]
                    parameters = []
                    
                    if start_time:
                        conditions.append("c.timestamp >= @start_time")
                        parameters.append({"name": "@start_time", "value": start_time.isoformat()})
                    
                    if end_time:
                        conditions.append("c.timestamp <= @end_time")
                        parameters.append({"name": "@end_time", "value": end_time.isoformat()})
                    
                    where_clause = " AND ".join(conditions)
                    
                    query = f"""
                        SELECT * FROM c 
                        WHERE {where_clause}
                        ORDER BY c.timestamp DESC
                    """
                    
                    # Use hierarchical partition key [user_id, session_id]
                    try:
                        items = self._chat_container.query_items(
                            query=query,
                            parameters=parameters,
                            partition_key=[user_id, session.session_id]
                        )
                        
                        session_count = 0
                        async for item in items:
                            try:
                                # Handle both structured data (new format) and JSON strings (legacy format)
                                if isinstance(item["content"], dict):
                                    conversation_data = item["content"]  # Already structured data
                                else:
                                    import json
                                    conversation_data = json.loads(item["content"])  # Legacy JSON string format
                                
                                conversation = ConversationLog.from_cosmos_dict(conversation_data)
                                conversations.append(conversation)
                                session_count += 1
                                
                                # Apply limit during collection to avoid memory issues
                                if len(conversations) >= limit:
                                    break
                            except Exception as e:
                                logger.warning(f"Failed to parse conversation log: {e}")
                                continue
                        
                        logger.debug(f"Retrieved {session_count} conversations from session {session.session_id}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to query session {session.session_id}: {e}")
                        continue
                
                # Sort by timestamp and apply final limit
                conversations.sort(key=lambda c: c.timestamp, reverse=True)
                conversations = conversations[:limit]
            
            logger.debug(f"Retrieved {len(conversations)} conversations for user {user_id}")
            return conversations
            
        except Exception as e:
            logger.error(f"Error retrieving user conversations: {str(e)}")
            return []

    async def get_user_conversation_analytics_async(
        self, 
        user_id: str,
        days: int = 30
    ) -> BusinessAnalytics:
        """
        Get comprehensive analytics for a user's conversations.
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
            
        Returns:
            Business analytics summary
        """
        try:
            # Calculate date range properly using timedelta
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            
            # Get conversations in date range
            conversations = await self.get_user_conversations_async(
                user_id=user_id,
                start_time=start_time,
                end_time=end_time,
                limit=1000  # High limit for analytics
            )
            
            if not conversations:
                return BusinessAnalytics(
                    user_id=user_id,
                    total_conversations=0,
                    successful_queries=0,
                    average_processing_time_ms=0.0,
                    average_response_time_ms=0.0,
                    cache_efficiency=0.0,
                    query_complexity_distribution={"simple": 0, "medium": 0, "complex": 0},
                    conversation_type_distribution={},
                    average_result_quality="low",
                    most_common_insights=[],
                    recommended_actions=[],
                    time_period_days=days
                )
            
            # Calculate analytics
            total_conversations = len(conversations)
            successful_queries = sum(1 for c in conversations if c.performance.success)
            
            # Response time analysis
            response_times = [c.performance.processing_time_ms for c in conversations if c.performance.processing_time_ms > 0]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            # Cache efficiency analysis
            cache_efficiencies = [c.performance.cache_efficiency for c in conversations]
            avg_cache_efficiency = sum(cache_efficiencies) / len(cache_efficiencies) if cache_efficiencies else 0.0
            
            # Query complexity distribution
            complexity_counts = {"simple": 0, "medium": 0, "complex": 0}
            for conversation in conversations:
                complexity = conversation.performance.query_complexity
                if complexity in complexity_counts:
                    complexity_counts[complexity] += 1
            
            # Conversation type distribution
            type_counts = {}
            for conversation in conversations:
                conv_type = conversation.metadata.conversation_type
                type_counts[conv_type] = type_counts.get(conv_type, 0) + 1
            
            # Result quality assessment
            quality_scores = {"high": 3, "medium": 2, "low": 1}
            total_quality_score = 0
            quality_count = 0
            
            for conversation in conversations:
                if conversation.metadata.result_quality in quality_scores:
                    total_quality_score += quality_scores[conversation.metadata.result_quality]
                    quality_count += 1
            
            if quality_count > 0:
                avg_quality_score = total_quality_score / quality_count
                if avg_quality_score >= 2.5:
                    avg_result_quality = "high"
                elif avg_quality_score >= 1.5:
                    avg_result_quality = "medium"
                else:
                    avg_result_quality = "low"
            else:
                avg_result_quality = "medium"
            
            # Extract common insights
            all_insights = []
            for conversation in conversations:
                if conversation.agent_response and conversation.agent_response.key_insights:
                    all_insights.extend(conversation.agent_response.key_insights)
            
            # Get most common insights (simplified)
            insight_counts = {}
            for insight in all_insights:
                # Simple keyword extraction
                words = insight.lower().split()
                for word in words:
                    if len(word) > 4:  # Only consider meaningful words
                        insight_counts[word] = insight_counts.get(word, 0) + 1
            
            most_common_insights = sorted(insight_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            most_common_insights = [insight[0] for insight in most_common_insights]
            
            # Generate recommended actions based on patterns
            recommended_actions = []
            if avg_cache_efficiency < 0.5:
                recommended_actions.append("Consider optimizing query patterns to improve cache utilization")
            if avg_response_time > 5000:  # > 5 seconds
                recommended_actions.append("Review query complexity to improve response times")
            if successful_queries / total_conversations < 0.8:
                recommended_actions.append("Investigate query failures to improve success rate")
            
            return BusinessAnalytics(
                user_id=user_id,
                total_conversations=total_conversations,
                successful_queries=successful_queries,
                average_processing_time_ms=avg_response_time,
                average_response_time_ms=avg_response_time,  # Same value for compatibility
                cache_efficiency=avg_cache_efficiency,
                query_complexity_distribution=complexity_counts,
                conversation_type_distribution=type_counts,
                average_result_quality=avg_result_quality,
                most_common_insights=most_common_insights,
                recommended_actions=recommended_actions,
                time_period_days=days
            )
            
        except Exception as e:
            logger.error(f"Error calculating user analytics: {str(e)}")
            # Return empty analytics on error
            return BusinessAnalytics(
                user_id=user_id,
                total_conversations=0,
                successful_queries=0,
                average_processing_time_ms=0.0,
                average_response_time_ms=0.0,
                cache_efficiency=0.0,
                query_complexity_distribution={"simple": 0, "medium": 0, "complex": 0},
                conversation_type_distribution={},
                average_result_quality="low",
                most_common_insights=[],
                recommended_actions=["Error calculating analytics - please try again"],
                time_period_days=days
            )

    async def cleanup_old_conversations_async(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Clean up conversation logs older than specified days.
        
        Args:
            days_to_keep: Number of days to keep conversations
            
        Returns:
            Statistics about cleaned data
        """
        try:
            from datetime import timedelta
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            # Query for old conversations using cross-partition query
            query = """
                SELECT c.id, c.user_id, c.session_id FROM c 
                WHERE c.role = 'nl2sql_conversation' 
                AND c.timestamp < @cutoff_date
            """
            
            items_to_delete = []
            items = self._chat_container.query_items(
                query=query,
                parameters=[{"name": "@cutoff_date", "value": cutoff_date.isoformat()}]
            )
            
            async for item in items:
                items_to_delete.append((item["id"], item["user_id"], item["session_id"]))
            
            # Delete old conversations using hierarchical partition key
            deleted_count = 0
            for item_id, user_id, session_id in items_to_delete:
                try:
                    await self._chat_container.delete_item(
                        item=item_id,
                        partition_key=[user_id, session_id]  # Hierarchical partition key
                    )
                    deleted_count += 1
                except CosmosResourceNotFoundError:
                    # Already deleted
                    pass
                except Exception as e:
                    logger.warning(f"Failed to delete conversation {item_id}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old conversations")
            return {"conversations_deleted": deleted_count}
            
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {str(e)}")
            return {"conversations_deleted": 0, "error": str(e)}


    @staticmethod
    def get_vector_index_policy():
        """
        Get the vector index policy for enabling vector search on the cache container.
        
        This policy should be applied to the nl2sql_cache container to enable
        vector similarity search on the /embedding path.
        
        Returns:
            Dict: Vector index policy for Cosmos DB container
        """
        return {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [
                {
                    "path": "/*"
                }
            ],
            "excludedPaths": [
                {
                    "path": "/\"_etag\"/?"
                }
            ],
            "vectorIndexes": [
                {
                    "path": "/embedding",
                    "type": "diskANN",
                    "quantizationByteSize": 96,
                    "indexingSearchListSize": 100
                }
            ]
        }

class CosmosDbConfig:
    """Configuration class for Cosmos DB service."""
    
    def __init__(
        self,
        endpoint: str,
        database_name: str,
        chat_container_name: str = "nl2sql_chatlogs",
        cache_container_name: str = "nl2sql_cache"
    ):
        self.endpoint = endpoint
        self.database_name = database_name
        self.chat_container_name = chat_container_name
        self.cache_container_name = cache_container_name
    
    @classmethod
    def from_env(cls) -> 'CosmosDbConfig':
        """Create configuration from environment variables."""
        import os
        
        endpoint = os.getenv("COSMOS_ENDPOINT")
        if not endpoint:
            raise ValueError("COSMOS_ENDPOINT environment variable is required")
        
        database_name = os.getenv("COSMOS_DATABASE_NAME", "sales_analytics")
        chat_container_name = os.getenv("COSMOS_CHAT_CONTAINER_NAME", "nl2sql_chatlogs")
        cache_container_name = os.getenv("COSMOS_CACHE_CONTAINER_NAME", "nl2sql_cache")
        
        return cls(
            endpoint=endpoint,
            database_name=database_name,
            chat_container_name=chat_container_name,
            cache_container_name=cache_container_name
        )
