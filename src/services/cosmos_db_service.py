"""
Azure Cosmos DB service for NL2SQL application.
Provides session and user tracking with chat logs and caching.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import uuid4

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.identity.aio import DefaultAzureCredential
from pydantic import BaseModel

from Models.agent_response import ChatLogEntry, UserSession, AgentResponse, Session, Message, CacheItem

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
