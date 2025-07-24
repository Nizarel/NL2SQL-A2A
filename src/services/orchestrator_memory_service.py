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
from Models.agent_response import (
    Session, Message, CacheItem, AgentResponse, FormattedResults,
    ConversationLog, ConversationPerformance, ConversationMetadata, 
    WorkflowContext, BusinessAnalytics
)

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
    Enhanced orchestrator memory service with conversation logging and intelligent caching.
    
    This service provides:
    - Session and conversation management
    - Business-focused conversation logging
    - Intelligent semantic caching
    - User analytics and insights
    - Query complexity assessment
    """
    
    def __init__(self, cosmos_service: CosmosDbService):
        """
        Initialize the orchestrator memory service.
        
        Args:
            cosmos_service: Initialized Cosmos DB service instance
        """
        self.cosmos_service = cosmos_service
        self._embedding_cache = {}  # In-memory cache for embeddings
        self._current_workflows: Dict[str, WorkflowContext] = {}  # Active workflow tracking
        
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
    
    # ======================================
    # Enhanced Conversation Methods
    # ======================================
    
    async def start_workflow_session(self, user_id: str, user_input: str, session_id: str = None) -> WorkflowContext:
        """
        Start a new workflow session and return context for tracking.
        
        Args:
            user_id: User identifier
            user_input: User's natural language query
            session_id: Optional existing session ID, will create new if not provided
            
        Returns:
            WorkflowContext for tracking the workflow
        """
        try:
            # Get or create active session
            if session_id:
                # Check if session exists, create if not
                existing_sessions = await self.get_user_sessions(user_id)
                session_exists = any(s.session_id == session_id for s in existing_sessions)
                
                if not session_exists:
                    # Create session record for provided session_id
                    session = Session(
                        session_id=session_id,
                        user_id=user_id,
                        session_name=f"NL2SQL Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        created_at=datetime.now(timezone.utc)
                    )
                    await self.cosmos_service.insert_session_async(user_id, session)
                    logger.info(f"Created new session record for provided session_id: {session_id}")
                
                active_session_id = session_id
            else:
                # Get or create active session
                sessions = await self.get_user_sessions(user_id)
                active_session = None
                
                for session in sessions:
                    if session.is_active:
                        active_session = session
                        break
                
                if not active_session:
                    # Create new session
                    active_session = await self.create_session(
                        user_id=user_id,
                        session_title=f"NL2SQL Session {datetime.now().strftime('%H:%M')}"
                    )
                
                active_session_id = active_session.session_id
            
            # Create workflow context
            context = WorkflowContext(
                user_id=user_id,
                session_id=active_session_id,
                user_input=user_input,
                conversation_turn=await self._get_next_conversation_turn(user_id, active_session_id)
            )
            
            # Store in memory for active tracking
            self._current_workflows[context.workflow_id] = context
            
            logger.info(f"Started workflow session for user {user_id}, workflow {context.workflow_id}")
            return context
            
        except Exception as e:
            logger.error(f"Error starting workflow session: {str(e)}")
            raise
    
    async def update_workflow_stage(self, workflow_id: str, stage: str, result: Dict[str, Any]) -> None:
        """
        Update a specific workflow stage with results.
        
        Args:
            workflow_id: Workflow identifier
            stage: Stage name (schema_analysis, sql_generation, execution, summarization)
            result: Stage result data
        """
        try:
            if workflow_id not in self._current_workflows:
                logger.warning(f"Workflow context not found: {workflow_id}")
                return
            
            context = self._current_workflows[workflow_id]
            setattr(context, stage, result)
            
            # Track cache hits
            if result.get("metadata", {}).get("cache_hit"):
                cache_type = result.get("metadata", {}).get("cache_type", "unknown")
                context.cache_hits.append(f"{stage}_{cache_type}")
            
            logger.debug(f"Updated workflow {workflow_id} stage {stage}")
            
        except Exception as e:
            logger.error(f"Error updating workflow stage: {str(e)}")
    
    async def complete_workflow_with_conversation_log(self, workflow_id: str) -> Dict[str, Any]:
        """
        Complete workflow and store business-focused conversation log.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Completion summary with performance metrics
        """
        try:
            if workflow_id not in self._current_workflows:
                logger.warning(f"Workflow context not found: {workflow_id}")
                return {"error": "Workflow not found"}
            
            context = self._current_workflows[workflow_id]
            
            # Calculate total processing time
            total_time = (datetime.now(timezone.utc) - context.timestamp).total_seconds() * 1000
            context.total_processing_time_ms = int(total_time)
            
            # Extract business-focused results
            formatted_results = self._extract_formatted_results(context)
            agent_response = self._create_business_agent_response(context)
            
            # Create conversation performance metrics
            performance = ConversationPerformance(
                processing_time_ms=context.total_processing_time_ms,
                cache_hits=context.cache_hits,
                success=context.summarization is not None and context.summarization.get("success", False),
                query_complexity=self._assess_query_complexity(context.user_input),
                cache_efficiency=len(context.cache_hits) / 4.0  # 4 possible stages
            )
            
            # Create conversation metadata
            metadata = ConversationMetadata(
                conversation_type=self._categorize_query(context.user_input),
                result_quality=self._assess_result_quality(context),
                follow_up_suggested=self._should_suggest_followup(agent_response),
                conversation_turn=context.conversation_turn,
                workflow_id=workflow_id
            )
            
            # Create and store conversation log
            conversation_log = ConversationLog(
                user_id=context.user_id,
                session_id=context.session_id,
                user_input=context.user_input,
                formatted_results=formatted_results,
                agent_response=agent_response,
                performance=performance,
                metadata=metadata
            )
            
            # Store conversation log in Cosmos DB
            await self._store_conversation_log(conversation_log)
            
            # Clean up memory
            del self._current_workflows[workflow_id]
            
            completion_summary = {
                "workflow_id": workflow_id,
                "conversation_id": conversation_log.id,
                "total_time_ms": context.total_processing_time_ms,
                "cache_hits": context.cache_hits,
                "cache_efficiency": performance.cache_efficiency,
                "query_complexity": performance.query_complexity,
                "conversation_type": metadata.conversation_type,
                "success": performance.success
            }
            
            logger.info(f"Completed workflow {workflow_id} with conversation log {conversation_log.id}")
            return completion_summary
            
        except Exception as e:
            logger.error(f"Error completing workflow: {str(e)}")
            # Clean up memory even on error
            if workflow_id in self._current_workflows:
                del self._current_workflows[workflow_id]
            return {"error": str(e)}
    
    async def complete_workflow_session(self, workflow_context, formatted_results=None, 
                                      agent_response=None, sql_query=None, processing_time_ms=None):
        """
        Complete a workflow session with conversation logging - simplified interface for orchestrator
        
        Args:
            workflow_context: The workflow context from start_workflow_session
            formatted_results: FormattedResults object
            agent_response: AgentResponse object  
            sql_query: Generated SQL query
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            ConversationLog object if successful, None if failed
        """
        try:
            print(f"🔄 DEBUG: Completing workflow session for {workflow_context.workflow_id}")
            
            # Update workflow context with provided data (storing in expected format)
            if sql_query:
                workflow_context.sql_generation = {"success": True, "data": {"sql_query": sql_query}}
                print(f"✅ DEBUG: Stored SQL query in workflow context")
                
            if formatted_results:
                # Store in the format expected by _extract_formatted_results
                if hasattr(formatted_results, 'model_dump'):
                    formatted_data = formatted_results.model_dump()
                else:
                    formatted_data = formatted_results
                    
                workflow_context.execution = {
                    "success": True,
                    "data": {"formatted_results": formatted_data}
                }
                print(f"✅ DEBUG: Stored formatted_results in workflow context with {formatted_data.get('total_rows', 0)} rows")
                
            if agent_response:
                # Store in the format expected by _create_business_agent_response  
                if hasattr(agent_response, 'model_dump'):
                    agent_data = agent_response.model_dump()
                else:
                    agent_data = agent_response
                    
                workflow_context.summarization = {
                    "success": True,
                    "data": {
                        "executive_summary": agent_data.get("executive_summary", ""),
                        "key_insights": agent_data.get("key_insights", []),
                        "recommendations": agent_data.get("recommendations", []),
                        "confidence_level": agent_data.get("confidence_level", "medium")
                    }
                }
                print(f"✅ DEBUG: Stored agent response in workflow context")
                
            if processing_time_ms:
                workflow_context.total_processing_time_ms = processing_time_ms
                print(f"✅ DEBUG: Stored processing time: {processing_time_ms}ms")
            
            # Create cache entry for the query-result pair
            await self._create_cache_entry_from_workflow(workflow_context, formatted_results, agent_response, sql_query, processing_time_ms)
            
            # Complete the workflow using existing method
            completion_result = await self.complete_workflow_with_conversation_log(workflow_context.workflow_id)
            
            # Return the conversation log if successful
            if "conversation_id" in completion_result and completion_result["conversation_id"]:
                # Create a conversation log object for return
                from Models.agent_response import ConversationLog, ConversationPerformance, ConversationMetadata
                
                conversation_log = ConversationLog(
                    id=completion_result["conversation_id"],
                    user_id=workflow_context.user_id,
                    session_id=workflow_context.session_id,
                    user_input=workflow_context.user_input,
                    formatted_results=formatted_results,
                    agent_response=agent_response,
                    performance=ConversationPerformance(
                        processing_time_ms=processing_time_ms or 0,
                        success=True,
                        query_complexity="medium",
                        cache_efficiency=0.5
                    ),
                    metadata=ConversationMetadata(
                        conversation_type="nl2sql_workflow",
                        result_quality="high",
                        workflow_id=workflow_context.workflow_id
                    )
                )
                return conversation_log
            else:
                logger.warning(f"No conversation ID in completion result: {completion_result}")
                return None
                
        except Exception as e:
            logger.error(f"Error completing workflow session: {str(e)}")
            return None
    
    async def _create_cache_entry_from_workflow(self, workflow_context, formatted_results=None, 
                                              agent_response=None, sql_query=None, processing_time_ms=None):
        """
        Create cache entry from workflow completion data.
        
        Args:
            workflow_context: The workflow context
            formatted_results: FormattedResults object
            agent_response: AgentResponse object
            sql_query: Generated SQL query
            processing_time_ms: Processing time in milliseconds
        """
        try:
            # Generate proper embedding using Azure OpenAI text-embedding-3-small
            query_text = workflow_context.user_input
            embedding = await self._generate_text_embedding(query_text)
            
            if not embedding:
                logger.warning(f"Failed to generate embedding for workflow {workflow_context.workflow_id}")
                return
            
            # Create cache key
            cache_key = f"workflow_result_{workflow_context.workflow_id}"
            
            # Prepare metadata
            metadata = {
                "type": "workflow_result",
                "user_id": workflow_context.user_id,
                "session_id": workflow_context.session_id,
                "workflow_id": workflow_context.workflow_id,
                "sql_query": sql_query,
                "success": formatted_results.success if formatted_results else True,
                "processing_time_ms": processing_time_ms or 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Add agent response info if available
            if agent_response:
                metadata["agent_type"] = agent_response.agent_type
                metadata["confidence_level"] = agent_response.confidence_level
                metadata["key_insights"] = agent_response.key_insights[:3] if agent_response.key_insights else []
            
            # Add results info if available
            if formatted_results:
                metadata["total_rows"] = getattr(formatted_results, 'total_rows', 0)
                metadata["result_success"] = getattr(formatted_results, 'success', False)
            
            # Store in cache
            await self.cosmos_service.set_vector_embedding_async(
                key=cache_key,
                embedding=embedding,
                text=f"Query: {query_text} -> SQL: {sql_query or 'No SQL'}",
                metadata=metadata
            )
            
            logger.info(f"Created cache entry for workflow {workflow_context.workflow_id}")
            
        except Exception as e:
            logger.error(f"Error creating cache entry: {str(e)}")
            # Don't fail the workflow completion due to cache errors

    async def _generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate text embedding using Azure OpenAI text-embedding-3-small.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        try:
            # Try to get the embedding service from a kernel if available
            # This is a simplified approach - in production you might want to inject the service
            import os
            from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
            
            # Load environment from .env file if not already loaded
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                env_path = Path(__file__).parent.parent / ".env"
                load_dotenv(env_path)
            except ImportError:
                # dotenv not available, continue with existing environment
                pass
            
            # Get Azure OpenAI configuration
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") 
            azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            
            if not all([azure_endpoint, azure_api_key, azure_embedding_deployment]):
                logger.warning("Azure OpenAI embedding configuration incomplete, using fallback embedding")
                # Fallback to simple embedding for demo
                return [float(hash(text + str(i)) % 1000) / 1000.0 for i in range(1536)]
            
            # Create embedding service
            embedding_service = AzureTextEmbedding(
                deployment_name=azure_embedding_deployment,
                endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                service_id="cache_embedding_service"
            )
            
            # Generate embedding
            embeddings = await embedding_service.generate_embeddings([text])
            
            if embeddings is not None and len(embeddings) > 0:
                # Extract the embedding vector
                embedding_data = embeddings[0]
                
                # Handle different possible return formats
                if hasattr(embedding_data, 'data') and embedding_data.data:
                    return list(embedding_data.data)
                elif isinstance(embedding_data, (list, tuple)):
                    return list(embedding_data)
                elif hasattr(embedding_data, '__iter__'):  # Handle numpy arrays or similar
                    return list(embedding_data)
                else:
                    logger.warning(f"Unexpected embedding format: {type(embedding_data)}")
                    return None
            else:
                logger.warning("No embeddings returned from service")
                return None
                
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            # Fallback to simple embedding for demo purposes
            return [float(hash(text + str(i)) % 1000) / 1000.0 for i in range(1536)]

    async def get_user_conversation_history(self, user_id: str, session_id: str = None, limit: int = 20) -> List[ConversationLog]:
        """
        Get user's conversation history with business-focused logs.
        
        Args:
            user_id: User identifier
            session_id: Optional session filter
            limit: Maximum conversations to return
            
        Returns:
            List of conversation logs
        """
        try:
            # Get messages from the user's sessions
            if session_id:
                messages = await self.get_session_context(user_id, session_id, limit)
            else:
                # Get recent sessions and their messages
                sessions = await self.get_user_sessions(user_id)
                all_messages = []
                
                for session in sessions[-5:]:  # Last 5 sessions
                    session_messages = await self.get_session_context(user_id, session.session_id, limit // 5)
                    all_messages.extend(session_messages)
                
                # Sort by timestamp and limit
                all_messages.sort(key=lambda x: x.timestamp, reverse=True)
                messages = all_messages[:limit]
            
            # Filter for conversation logs and convert
            conversation_logs = []
            for message in messages:
                if message.role == "nl2sql_conversation":
                    try:
                        # Handle both structured data (new format) and JSON strings (legacy format)
                        if isinstance(message.content, dict):
                            content = message.content  # Already structured data
                        else:
                            content = json.loads(message.content)  # Legacy JSON string format
                        
                        log = ConversationLog(
                            id=message.message_id,
                            user_id=message.user_id,
                            session_id=message.session_id,
                            timestamp=message.timestamp,
                            user_input=content.get("user_input", ""),
                            formatted_results=FormattedResults(**content["formatted_results"]) if content.get("formatted_results") else None,
                            agent_response=AgentResponse(**content["agent_response"]) if content.get("agent_response") else None,
                            performance=ConversationPerformance(**content["performance"]) if content.get("performance") else ConversationPerformance(),
                            metadata=ConversationMetadata(**content["metadata"]) if content.get("metadata") else ConversationMetadata()
                        )
                        conversation_logs.append(log)
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(f"Failed to parse conversation log: {e}")
                        continue
            
            return conversation_logs
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    async def get_user_analytics_enhanced(self, user_id: str, days: int = 7) -> BusinessAnalytics:
        """
        Get enhanced user analytics with business insights.
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
            
        Returns:
            BusinessAnalytics with comprehensive metrics
        """
        try:
            # Use Cosmos service directly to get analytics (more reliable than cross-session queries)
            analytics = await self.cosmos_service.get_user_conversation_analytics_async(
                user_id=user_id,
                days=days
            )
            
            # The Cosmos service returns BusinessAnalytics directly
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting enhanced user analytics: {str(e)}")
            # Return default analytics on error
            return BusinessAnalytics(
                user_id=user_id,
                total_conversations=0,
                successful_queries=0,
                average_processing_time_ms=0.0,
                average_response_time_ms=0.0,
                most_common_query_types=[],
                average_result_quality="low",
                cache_efficiency=0.0,
                query_complexity_distribution={"simple": 0, "medium": 0, "complex": 0},
                conversation_type_distribution={},
                most_common_insights=[],
                recommended_actions=[],
                time_period_days=days
            )
    
    # ======================================
    # Helper Methods for Conversation Processing
    # ======================================
    
    async def _get_next_conversation_turn(self, user_id: str, session_id: str) -> int:
        """Get the next conversation turn number for a session."""
        try:
            messages = await self.get_session_context(user_id, session_id, max_messages=100)
            conversation_messages = [m for m in messages if m.role == "nl2sql_conversation"]
            return len(conversation_messages) + 1
        except Exception:
            return 1
    
    def _extract_formatted_results(self, context: WorkflowContext) -> Optional[FormattedResults]:
        """Extract formatted results from workflow context."""
        try:
            if (context.execution and 
                context.execution.get("success") and 
                "formatted_results" in context.execution.get("data", {})):
                
                exec_data = context.execution["data"]["formatted_results"]
                return FormattedResults(
                    success=True,
                    headers=exec_data.get("headers", []),
                    rows=exec_data.get("rows", []),
                    total_rows=exec_data.get("total_rows", 0),
                    chart_recommendations=exec_data.get("chart_recommendations", [])
                )
            return None
        except Exception as e:
            logger.warning(f"Failed to extract formatted results: {e}")
            return None
    
    def _create_business_agent_response(self, context: WorkflowContext) -> Optional[AgentResponse]:
        """Create business-focused agent response from workflow context."""
        try:
            if context.summarization and context.summarization.get("success"):
                summary_data = context.summarization.get("data", {})
                
                return AgentResponse(
                    agent_type="orchestrator",
                    response=summary_data.get("summary", "Analysis completed successfully."),
                    success=True,
                    processing_time_ms=context.total_processing_time_ms,
                    executive_summary=summary_data.get("executive_summary", ""),
                    key_insights=summary_data.get("key_insights", []),
                    recommendations=summary_data.get("recommendations", []),
                    confidence_level=summary_data.get("confidence_level", "medium")
                )
            
            # Fallback response if no summarization
            return AgentResponse(
                agent_type="orchestrator",
                response="Query processed successfully.",
                success=context.execution is not None and context.execution.get("success", False),
                processing_time_ms=context.total_processing_time_ms,
                confidence_level="medium"
            )
            
        except Exception as e:
            logger.warning(f"Failed to create agent response: {e}")
            return None
    
    def _assess_query_complexity(self, user_input: str) -> str:
        """Assess query complexity for analytics."""
        complexity_indicators = {
            "simple": ["show", "list", "what is", "how many", "get", "find"],
            "medium": ["compare", "trend", "analyze", "group by", "top", "average", "sum"],
            "complex": ["correlation", "forecast", "predict", "advanced", "complex", "join", "subquery"]
        }
        
        user_lower = user_input.lower()
        
        for level in ["complex", "medium", "simple"]:  # Check complex first
            if any(indicator in user_lower for indicator in complexity_indicators[level]):
                return level
        
        return "simple"
    
    def _categorize_query(self, user_input: str) -> str:
        """Categorize the type of business query."""
        categories = {
            "sales_analytics": ["sales", "revenue", "income", "profit", "sold", "selling"],
            "customer_analytics": ["customer", "client", "user", "buyer", "consumer"],
            "product_analytics": ["product", "item", "inventory", "stock", "catalog"],
            "financial_analytics": ["financial", "cost", "expense", "budget", "finance"],
            "operational_analytics": ["operational", "process", "efficiency", "performance", "operations"]
        }
        
        user_lower = user_input.lower()
        
        for category, keywords in categories.items():
            if any(keyword in user_lower for keyword in keywords):
                return category
        
        return "general_analytics"
    
    def _assess_result_quality(self, context: WorkflowContext) -> str:
        """Assess the quality of results based on workflow success."""
        if not context.execution or not context.execution.get("success"):
            return "low"
        
        if context.summarization and context.summarization.get("success"):
            summary_data = context.summarization.get("data", {})
            if summary_data.get("key_insights") and summary_data.get("recommendations"):
                return "high"
            return "medium"
        
        return "medium"
    
    def _should_suggest_followup(self, agent_response: Optional[AgentResponse]) -> bool:
        """Determine if follow-up suggestions should be made."""
        if not agent_response:
            return False
        
        return (len(agent_response.recommendations) > 0 or 
                len(agent_response.key_insights) > 1 or
                agent_response.confidence_level == "high")
    
    async def _store_conversation_log(self, conversation_log: ConversationLog) -> None:
        """Store conversation log in Cosmos DB with structured data format."""
        try:
            # Convert to structured data format for storage (similar to API endpoint)
            conversation_data = conversation_log.to_cosmos_dict()
            
            # Apply the same make_json_serializable logic as the API endpoint
            def make_json_serializable(obj):
                """Convert complex objects to JSON-serializable format while preserving structure."""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    return {key: make_json_serializable(value) for key, value in obj.__dict__.items()}
                elif hasattr(obj, '_asdict'):  # namedtuple
                    return make_json_serializable(obj._asdict())
                elif isinstance(obj, dict):
                    return {key: make_json_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            # Convert to properly structured data (not JSON string)
            structured_data = make_json_serializable(conversation_data)
            
            message = Message(
                session_id=conversation_log.session_id,
                user_id=conversation_log.user_id,
                role="nl2sql_conversation",
                content=structured_data,  # Store as structured data, not JSON string
                timestamp=conversation_log.timestamp,
                metadata={}
            )
            
            await self.cosmos_service.insert_message_async(conversation_log.user_id, message)
            
        except Exception as e:
            logger.error(f"Failed to store conversation log: {e}")
            raise
    
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
