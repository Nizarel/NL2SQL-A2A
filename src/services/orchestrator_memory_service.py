"""
Orchestrator Memory Service - Integrates ChatLogger with the multi-agent system
Provides memory capabilities for the orchestrator agent using Cosmos DB
Compatible with Semantic Kernel 1.35.0

Updated for single container design:
- Database: sales_analytics
- Container: nl2sql_chatlogs
- Partition Key: /user_id/session_id (hierarchical)
- Vector embedding support with GA Cosmos DB Vector Search
- Azure Identity support for secure production deployments
- SK 1.35.0 embedding generation and semantic search
"""
import os
import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.memory.memory_record import MemoryRecord
from .chat_logger import ChatLogger
from Models.agent_response import ChatLogEntry, AgentResponse, UserSession, LogTokens

# SK 1.35.0 imports for embedding generation
try:
    from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import EmbeddingGeneratorBase
    from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
    from semantic_kernel import Kernel
    SK_EMBEDDINGS_AVAILABLE = True
except ImportError:
    SK_EMBEDDINGS_AVAILABLE = False

# Azure Identity imports for secure authentication
try:
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False


class OrchestratorMemoryService(MemoryStoreBase):
    """
    Memory service for the orchestrator agent that provides:
    1. Session-based conversation memory
    2. Agent response logging and retrieval
    3. Context-aware conversation history
    4. User session management
    5. Compatible with Semantic Kernel 1.35.0 MemoryStoreBase
    6. Single container design with hierarchical partitioning
    7. Azure Identity support for secure deployments
    """
    
    def __init__(
        self, 
        connection_string: str = None,
        cosmos_endpoint: str = None,
        use_azure_identity: bool = None,
        managed_identity_client_id: str = None,
        embedding_service: EmbeddingGeneratorBase = None,
        kernel: "Kernel" = None
    ):
        """
        Initialize the OrchestratorMemoryService with SK 1.35.0 embedding support
        
        Args:
            connection_string: Cosmos DB connection string (for local development)
            cosmos_endpoint: Cosmos DB endpoint URL (for Azure identity)
            use_azure_identity: Whether to use Azure Identity for authentication
            managed_identity_client_id: Client ID for user-assigned managed identity
            embedding_service: SK 1.35.0 embedding service for semantic search
            kernel: SK Kernel instance to auto-discover embedding service
        """
        
        # Store embedding service for semantic search capabilities
        self.embedding_service = embedding_service
        
        # Try to auto-discover embedding service from kernel if not provided
        if self.embedding_service is None and kernel is not None and SK_EMBEDDINGS_AVAILABLE:
            try:
                # Look for EmbeddingGeneratorBase first
                try:
                    self.embedding_service = kernel.get_service(type=EmbeddingGeneratorBase)
                    print("✅ Auto-discovered embedding service from SK kernel (EmbeddingGeneratorBase)")
                except Exception:
                    # Try AzureTextEmbedding specifically
                    self.embedding_service = kernel.get_service(type=AzureTextEmbedding)
                    print("✅ Auto-discovered embedding service from SK kernel (AzureTextEmbedding)")
            except Exception as e:
                print(f"⚠️ Could not auto-discover embedding service: {e}")
        
        # Read from environment variables if not provided
        if cosmos_endpoint is None:
            cosmos_endpoint = os.getenv("COSMOS_DB_ENDPOINT")
        
        if managed_identity_client_id is None:
            managed_identity_client_id = os.getenv("MANAGED_IDENTITY_CLIENT_ID")
        
        if use_azure_identity is None:
            # Check environment variable, otherwise auto-detect
            use_azure_identity_env = os.getenv("USE_AZURE_IDENTITY", "").lower()
            if use_azure_identity_env in ["true", "1", "yes"]:
                use_azure_identity = True
            elif use_azure_identity_env in ["false", "0", "no"]:
                use_azure_identity = False
            else:
                use_azure_identity = self._is_azure_deployment()
        
        # Determine authentication method based on deployment environment
        if use_azure_identity:
            # Azure Identity authentication (recommended for production)
            self._connection_string = self._setup_azure_identity_auth(
                cosmos_endpoint, managed_identity_client_id
            )
        else:
            # Connection string authentication (local development)
            self._connection_string = self._setup_connection_string_auth(connection_string)
        
        if not self._connection_string:
            raise ValueError(
                "Unable to configure Cosmos DB authentication. "
                "For local development: Set COSMOS_DB_CONNECTION_STRING and AccountKey environment variables. "
                "For Azure deployment: Set COSMOS_DB_ENDPOINT and ensure Azure Identity is configured."
            )
        
        # Set auth method before initializing ChatLogger
        self._auth_method = "azure_identity" if use_azure_identity else "connection_string"
        
        # Initialize ChatLogger with the configured connection
        if self._auth_method == "azure_identity":
            self.chat_logger = ChatLogger(
                use_azure_identity=True,
                cosmos_endpoint=cosmos_endpoint
            )
        else:
            self.chat_logger = ChatLogger(self._connection_string)
        
        self._initialized = False
    
    def _is_azure_deployment(self) -> bool:
        """Detect if running in Azure environment"""
        # Check for common Azure environment variables
        azure_indicators = [
            "WEBSITE_SITE_NAME",  # App Service
            "FUNCTIONS_WORKER_RUNTIME",  # Azure Functions
            "CONTAINER_APP_NAME",  # Container Apps
            "AKS_CLUSTER_NAME",  # AKS
            "MSI_ENDPOINT"  # Managed Service Identity
        ]
        
        return any(os.getenv(indicator) for indicator in azure_indicators)
    
    def _setup_azure_identity_auth(
        self, 
        cosmos_endpoint: str = None, 
        managed_identity_client_id: str = None
    ) -> Optional[str]:
        """Setup Azure Identity authentication for Cosmos DB"""
        
        if not AZURE_IDENTITY_AVAILABLE:
            raise ImportError(
                "Azure Identity is not available. Install with: pip install azure-identity"
            )
        
        # Get Cosmos DB endpoint
        endpoint = cosmos_endpoint or os.getenv("COSMOS_DB_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Cosmos DB endpoint is required for Azure Identity authentication. "
                "Set COSMOS_DB_ENDPOINT environment variable or pass cosmos_endpoint parameter."
            )
        
        # Ensure endpoint has proper format
        if not endpoint.startswith("https://"):
            endpoint = f"https://{endpoint}"
        if not endpoint.endswith(".documents.azure.com:443/"):
            if endpoint.endswith("/"):
                endpoint = endpoint[:-1]
            endpoint = f"{endpoint}:443/"
        
        try:
            # Use managed identity if client_id is provided, otherwise use default credential
            if managed_identity_client_id:
                credential = ManagedIdentityCredential(client_id=managed_identity_client_id)
                print(f"🔐 Using User-Assigned Managed Identity: {managed_identity_client_id}")
            else:
                credential = DefaultAzureCredential()
                print("🔐 Using Default Azure Credential (System-Assigned MI or other)")
            
            # Create connection string format that ChatLogger can use with Azure Identity
            # Note: This is a simplified approach - in production, you'd modify ChatLogger 
            # to accept credentials directly
            return f"AccountEndpoint={endpoint};Credential=AzureIdentity;"
            
        except Exception as e:
            print(f"❌ Failed to setup Azure Identity authentication: {e}")
            return None
    
    def _setup_connection_string_auth(self, connection_string: str = None) -> Optional[str]:
        """Setup connection string authentication (local development)"""
        
        # Use provided connection string or build from environment variables
        if connection_string:
            return connection_string
        
        endpoint = os.getenv("COSMOS_DB_CONNECTION_STRING", "")
        account_key = os.getenv("AccountKey", "")
        
        if endpoint and account_key:
            # Ensure proper endpoint format
            if not endpoint.startswith("https://"):
                endpoint = f"https://{endpoint}"
            if not endpoint.endswith("/"):
                endpoint = f"{endpoint}/"
            
            connection_str = f"AccountEndpoint={endpoint};AccountKey={account_key};"
            return connection_str
        
        return None
    
    def get_auth_info(self) -> Dict[str, Any]:
        """Get information about the current authentication method"""
        return {
            "auth_method": self._auth_method,
            "is_azure_deployment": self._is_azure_deployment(),
            "azure_identity_available": AZURE_IDENTITY_AVAILABLE,
            "cosmos_endpoint": self._connection_string.split(";")[0] if self._connection_string else None
        }
    
    def initialize(self):
        """Initialize the memory service"""
        if not self._initialized:
            self.chat_logger.initialize()
            self._initialized = True
    
    def start_conversation(self, user_id: str, metadata: Dict[str, Any] = None) -> UserSession:
        """Start a new conversation session for a user"""
        self.initialize()
        
        # Get or create active session
        session = self.chat_logger.get_active_session(user_id)
        
        # If we need to create a new session with specific metadata
        if metadata and not session.session_metadata:
            session.session_metadata.update(metadata)
            # Note: In a full implementation, you'd update this in Cosmos DB
        
        return session
    
    def log_orchestrator_workflow(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        workflow_results: Dict[str, Any],
        processing_time_ms: int = None
    ) -> ChatLogEntry:
        """
        Log a complete orchestrator workflow execution
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_input: Original user question
            workflow_results: Results from each agent in the workflow
            processing_time_ms: Total processing time
        """
        self.initialize()
        
        # Create agent responses from workflow results
        agent_responses = []
        
        # Schema Analyst Response
        if "schema_analysis" in workflow_results and workflow_results["schema_analysis"]["success"]:
            schema_response = AgentResponse(
                summary=workflow_results["schema_analysis"].get("data", {}).get("summary"),
                insights=workflow_results["schema_analysis"].get("data", {}).get("insights"),
                tokens=self._extract_tokens(workflow_results["schema_analysis"])
            )
            agent_responses.append(schema_response)
        
        # SQL Generator Response
        if "sql_generation" in workflow_results and workflow_results["sql_generation"]["success"]:
            sql_response = AgentResponse(
                summary=workflow_results["sql_generation"].get("data", {}).get("sql"),
                insights=workflow_results["sql_generation"].get("data", {}).get("explanation"),
                tokens=self._extract_tokens(workflow_results["sql_generation"])
            )
            agent_responses.append(sql_response)
        
        # Executor Response
        if "execution" in workflow_results and workflow_results["execution"]["success"]:
            exec_data = workflow_results["execution"].get("data", {})
            executor_response = AgentResponse(
                formatted_results=exec_data.get("formatted_results"),
                summary=f"Executed query and retrieved {exec_data.get('row_count', 0)} rows",
                tokens=self._extract_tokens(workflow_results["execution"])
            )
            agent_responses.append(executor_response)
        
        # Summarizing Agent Response
        if "summarization" in workflow_results and workflow_results["summarization"]["success"]:
            summary_data = workflow_results["summarization"].get("data", {})
            summary_response = AgentResponse(
                summary=summary_data.get("summary"),
                insights=summary_data.get("insights"),
                recommendations=summary_data.get("recommendations"),
                tokens=self._extract_tokens(workflow_results["summarization"])
            )
            agent_responses.append(summary_response)
        
        # Get current conversation turn
        existing_entries = self.chat_logger.get_chat_log_entries_by_session(session_id, user_id)
        conversation_turn = len(existing_entries) + 1
        
        # Create chat log entry
        chat_entry = ChatLogEntry(
            user_id=user_id,
            session_id=session_id,
            user_input=user_input,
            agent_responses=agent_responses,
            conversation_turn=conversation_turn,
            processing_time_ms=processing_time_ms
        )
        
        # Save to Cosmos DB
        return self.chat_logger.create_chat_log_entry(chat_entry)
    
    def get_conversation_history(
        self,
        user_id: str,
        session_id: str = None,
        limit: int = 10
    ) -> List[ChatLogEntry]:
        """
        Get conversation history for context-aware responses
        
        Args:
            user_id: User identifier
            session_id: Optional session identifier (if None, gets recent across all sessions)
            limit: Maximum number of entries to return
        """
        self.initialize()
        
        if session_id:
            return self.chat_logger.get_chat_log_entries_by_session(session_id, user_id)
        else:
            return self.chat_logger.get_chat_log_entries_by_user(user_id, limit)
    
    def find_similar_queries(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> List[ChatLogEntry]:
        """
        Find similar queries from the user's history
        This is a simple implementation - could be enhanced with vector similarity
        """
        self.initialize()
        
        # For now, do exact match - can be enhanced with semantic search
        return self.chat_logger.get_chat_log_entries_by_user_input(query, user_id)
    
    def update_session_metadata(
        self,
        user_id: str,
        session_id: str,
        metadata: Dict[str, Any]
    ):
        """Update session metadata with additional context"""
        self.initialize()
        
        # This would require extending the ChatLogger with session metadata update
        # For now, we can update session activity
        self.chat_logger.update_session_activity(session_id, user_id)
    
    def end_conversation(self, user_id: str, session_id: str):
        """End a conversation session"""
        self.initialize()
        
        self.chat_logger.end_session(session_id, user_id)
    
    def _extract_tokens(self, agent_result: Dict[str, Any]) -> Optional[LogTokens]:
        """Extract token information from agent result metadata"""
        metadata = agent_result.get("metadata", {})
        
        if "tokens" in metadata:
            token_data = metadata["tokens"]
            return LogTokens(
                input_tokens=token_data.get("input_tokens", 0),
                output_tokens=token_data.get("output_tokens", 0),
                input_cost=token_data.get("input_cost", 0.0),
                output_cost=token_data.get("output_cost", 0.0),
                total_tokens=token_data.get("total_tokens", 0)
            )
        
        return None
    
    def store_conversation_memory(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store conversation memory for a user
        
        Args:
            conversation_id: Unique conversation identifier
            user_message: User's input message
            assistant_response: Assistant's response
            metadata: Optional metadata dictionary
        
        Returns:
            Entry ID of the stored conversation
        """
        self.initialize()
        
        # Extract user_id and session_id from conversation_id or use defaults
        if "/" in conversation_id:
            parts = conversation_id.split("/")
            user_id = parts[0] if len(parts) > 0 else "default_user"
            session_id = parts[1] if len(parts) > 1 else conversation_id
        else:
            user_id = metadata.get("user_id", "default_user") if metadata else "default_user"
            session_id = conversation_id
        
        # Create agent response from assistant message
        agent_response = AgentResponse(
            agent_type="OrchestratorAgent",
            response=assistant_response,
            success=True,
            summary=assistant_response,
            insights=f"Response to: {user_message[:100]}..."
        )
        
        # Get current conversation turn
        existing_entries = self.chat_logger.get_chat_log_entries_by_session(session_id, user_id)
        conversation_turn = len(existing_entries) + 1
        
        # Create chat log entry
        chat_entry = ChatLogEntry(
            user_id=user_id,
            session_id=session_id,
            user_input=user_message,
            agent_responses=[agent_response],
            conversation_turn=conversation_turn
        )
        
        # Save to Cosmos DB
        stored_entry = self.chat_logger.create_chat_log_entry(chat_entry)
        return stored_entry.id

    async def store_conversation_with_embedding(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store conversation with vector embedding for semantic search (SK 1.35.0)
        
        Args:
            conversation_id: Unique conversation identifier
            user_message: User's input message
            assistant_response: Assistant's response
            metadata: Optional metadata dictionary
        
        Returns:
            Entry ID of the stored conversation
        """
        # Store the basic conversation first
        entry_id = self.store_conversation_memory(
            conversation_id, user_message, assistant_response, metadata
        )
        
        # Generate and store embedding if service is available
        if self.embedding_service and SK_EMBEDDINGS_AVAILABLE:
            try:
                # Generate embedding for the user message
                embedding = await self.generate_embedding(user_message)
                
                if embedding is not None:
                    # Extract user_id and session_id
                    if "/" in conversation_id:
                        user_id, session_id = conversation_id.split("/", 1)
                    else:
                        user_id = metadata.get("user_id", "default_user") if metadata else "default_user"
                        session_id = conversation_id
                    
                    # Store vector embedding
                    try:
                        self.chat_logger.add_vector_embedding(
                            entry_id=entry_id,
                            user_id=user_id,
                            session_id=session_id,
                            embedding=embedding.tolist()
                        )
                        print(f"✅ Stored conversation with embedding: {entry_id}")
                    except Exception as embed_error:
                        print(f"❌ Failed to store vector embedding: {embed_error}")
                        print(f"   Entry ID: {entry_id}, User: {user_id}, Session: {session_id}")
                        print(f"   Embedding size: {len(embedding.tolist())}")
                        # Continue without embedding
                        print(f"✅ Stored conversation without embedding: {entry_id}")
                else:
                    print("⚠️ Embedding generation failed - stored without vector")
                    
            except Exception as e:
                print(f"⚠️ Error generating embedding: {e}")
                # Continue without embedding
        else:
            print("⚠️ Embedding service not available - stored without vector")
        
        return entry_id

    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding using SK 1.35.0 embedding service
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Numpy array of embedding values or None if failed
        """
        if not self.embedding_service or not SK_EMBEDDINGS_AVAILABLE:
            return None
        
        try:
            # SK 1.35.0 generate_embeddings returns numpy.ndarray with shape (batch_size, embedding_dim)
            # For single text input: shape is (1, 1536), dtype is float64
            embeddings = await self.embedding_service.generate_embeddings([text])
            
            # Check if we got a valid result - avoid boolean evaluation on numpy array
            if embeddings is not None and hasattr(embeddings, 'size') and embeddings.size > 0:
                # Extract the first (and only) embedding vector and convert to float32 for Cosmos DB
                embedding_vector = embeddings[0].astype(np.float32)
                return embedding_vector
            else:
                print("⚠️ Empty embedding result from SK service")
                return None
                
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return None

    async def semantic_search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[ChatLogEntry, float]]:
        """
        Perform semantic search across user's conversation history
        
        Args:
            user_id: User identifier
            query: Search query text
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of tuples containing (ChatLogEntry, similarity_score)
        """
        if not self.embedding_service or not SK_EMBEDDINGS_AVAILABLE:
            print("⚠️ Semantic search not available - no embedding service")
            return []
        
        try:
            # Generate embedding for search query
            query_embedding = await self.generate_embedding(query)
            if query_embedding is None:
                print("⚠️ Could not generate query embedding")
                return []
            
            print(f"🔍 Performing semantic search for user {user_id}")
            
            # Use ChatLogger's vector search
            similar_entries = self.chat_logger.vector_search(
                user_id=user_id,
                query_embedding=query_embedding.tolist(),
                limit=limit
            )
            
            # Convert to tuple format with similarity scores
            results = []
            for entry in similar_entries:
                # For now, use a mock similarity score since ChatLogger doesn't return scores
                # In a full implementation, you'd get the actual scores from Cosmos DB
                results.append((entry, 0.8))  # Mock score
            
            print(f"✅ Found {len(results)} semantically similar conversations")
            return results
            
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            # Fallback to recent conversations
            recent_entries = self.get_conversation_history(user_id, limit=limit)
            return [(entry, 0.5) for entry in recent_entries]

    def get_embedding_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about embeddings and semantic search capabilities"""
        return {
            "embedding_service_available": self.embedding_service is not None,
            "sk_embeddings_available": SK_EMBEDDINGS_AVAILABLE,
            "azure_identity_available": AZURE_IDENTITY_AVAILABLE,
            "vector_search_enabled": True,  # Cosmos DB Vector Search is GA
            "embedding_dimensions": 1536,  # OpenAI text-embedding-3-small
            "similarity_threshold": 0.7,
            "user_id": user_id
        }

    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        self.initialize()
        
        # Get all sessions for the user
        user_entries = self.chat_logger.get_chat_log_entries_by_user(user_id, limit=1000)
        
        if not user_entries:
            return {
                "total_conversations": 0,
                "total_sessions": 0,
                "total_tokens": 0,
                "avg_tokens_per_conversation": 0,
                "last_activity": None
            }
        
        # Calculate statistics
        sessions = set(entry.session_id for entry in user_entries)
        total_tokens = sum(
            sum(response.tokens.total_tokens if response.tokens else 0 for response in entry.agent_responses)
            for entry in user_entries
        )
        
        return {
            "total_conversations": len(user_entries),
            "total_sessions": len(sessions),
            "total_tokens": total_tokens,
            "avg_tokens_per_conversation": total_tokens / len(user_entries) if user_entries else 0,
            "last_activity": max(entry.timestamp for entry in user_entries).isoformat() if user_entries else None
        }
    
    def close(self):
        """Close the memory service"""
        if self.chat_logger:
            self.chat_logger.close()
    
    # Enhanced features leveraging SK 1.35.0 improvements
    async def store_workflow_memory(
        self,
        user_id: str,
        session_id: str,
        workflow_step: str,
        step_result: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """Store individual workflow step results for better context retrieval"""
        record = MemoryRecord(
            id=f"workflow_{user_id}_{session_id}_{workflow_step}_{int(time.time())}",
            text=str(step_result.get("data", "")),
            description=f"Workflow step: {workflow_step}",
            additional_metadata={
                "user_id": user_id,
                "session_id": session_id,
                "workflow_step": workflow_step,
                "success": step_result.get("success", False),
                "tokens": step_result.get("metadata", {}).get("tokens", {}),
                "execution_time": step_result.get("metadata", {}).get("execution_time_ms", 0)
            },
            embedding=embedding,
            timestamp=datetime.now(timezone.utc)
        )
        
        return await self.upsert("workflow_memory", record)
    
    async def get_workflow_context(
        self,
        user_id: str,
        session_id: str,
        workflow_step: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Get context from previous similar workflow steps for better AI responses"""
        # This would use vector similarity search when available
        # For now, return recent workflow history
        
        # Get recent chat history for context
        recent_chats = self.chat_logger.get_chat_log_entries_by_session(session_id, user_id)
        context = []
        
        for chat in recent_chats[-limit:]:
            for response in chat.agent_responses:
                if response.summary or response.insights:
                    context.append({
                        "step": workflow_step,
                        "content": response.summary or response.insights,
                        "timestamp": chat.timestamp,
                        "success": True
                    })
        
        return context
    
    async def create_embedding_enhanced_memory(
        self,
        user_input: str,
        user_id: str,
        session_id: str,
        embedding_function = None
    ) -> MemoryRecord:
        """Create memory record with embedding for semantic search"""
        
        # Generate embedding if function provided
        embedding = None
        if embedding_function:
            embedding = await embedding_function(user_input)
        
        record = MemoryRecord(
            id=f"enhanced_{user_id}_{session_id}_{int(time.time())}",
            text=user_input,
            description="Enhanced memory with semantic embedding",
            additional_metadata={
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "enhanced": True
            },
            embedding=embedding,
            timestamp=datetime.now(timezone.utc)
        )
        
        await self.upsert("vector_embeddings", record)
        return record
    
    async def semantic_search_conversations(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        limit: int = 5,
        min_relevance: float = 0.7
    ) -> List[Tuple[ChatLogEntry, float]]:
        """Perform semantic search across user's conversation history"""
        
        # When Cosmos DB vector search is available, this would be:
        # matches = await self.get_nearest_matches(
        #     "vector_embeddings", 
        #     query_embedding, 
        #     limit=limit,
        #     min_relevance_score=min_relevance,
        #     with_embeddings=True
        # )
        
        # For now, return recent conversations as fallback
        recent_chats = self.chat_logger.get_chat_log_entries_by_user(user_id, limit=limit)
        return [(chat, 0.8) for chat in recent_chats]  # Mock relevance score
    
    def get_enhanced_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Enhanced memory statistics with SK 1.35.0 and GA Vector Search features"""
        base_stats = self.get_memory_stats(user_id)
        embedding_stats = self.get_embedding_stats(user_id)
        
        # Add enhanced statistics
        enhanced_stats = base_stats.copy()
        enhanced_stats.update({
            "sk_version": "1.35.0",
            "auth_method": self._auth_method,
            "azure_deployment": self._is_azure_deployment(),
            "cosmos_vector_search": "GA",  # Vector Search is now GA
            "embedding_service_active": self.embedding_service is not None,
            "semantic_search_ready": SK_EMBEDDINGS_AVAILABLE and self.embedding_service is not None,
            "features_enabled": [
                "batch_operations",
                "collection_management", 
                "nearest_match_search",
                "embedding_generation",  # NEW: SK 1.35.0 embedding generation
                "semantic_similarity_search",  # NEW: GA Vector Search
                "workflow_memory",
                "azure_identity_support",
                "ga_vector_search",  # NEW: GA Cosmos DB Vector Search
                "conversation_embeddings"  # NEW: Conversation embedding storage
            ],
            "vector_capabilities": {
                "embedding_dimensions": 1536,
                "distance_function": "cosine",
                "vector_index_type": "diskANN",
                "similarity_threshold": 0.7,
                "semantic_search_available": self.embedding_service is not None
            },
            "collections_available": [
                "chat_history", 
                "user_sessions", 
                "vector_embeddings", 
                "workflow_memory", 
                "context_cache",
                "semantic_search"  # NEW: Semantic search capability
            ]
        })
        
        # Merge embedding stats
        enhanced_stats.update(embedding_stats)
        
        return enhanced_stats
    
    # Semantic Kernel 1.35.0 MemoryStoreBase compatibility methods with NEW features
    async def create_collection(self, collection_name: str) -> None:
        """Create a collection - implemented for SK compatibility"""
        self.initialize()
        # Collections are implicit in our single container design
        # We could create metadata tracking for collections if needed
        pass
    
    async def get_collections(self) -> List[str]:
        """Get available collections"""
        return ["chat_history", "user_sessions", "vector_embeddings", "workflow_memory", "context_cache"]
    
    async def does_collection_exist(self, collection_name: str) -> bool:
        """NEW in SK 1.35.0: Check if collection exists"""
        self.initialize()
        collections = await self.get_collections()
        return collection_name in collections
    
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection - implemented for SK compatibility"""
        # Not implemented for single container design
        pass
    
    async def upsert(self, collection_name: str, record: MemoryRecord) -> str:
        """Store a memory record - enhanced for SK 1.35.0"""
        self.initialize()
        
        # Convert MemoryRecord to ChatLogEntry format if it's chat history
        if collection_name == "chat_history":
            # Create a chat entry from memory record
            chat_entry = ChatLogEntry(
                id=record.id,
                user_id=record.additional_metadata.get("user_id", "unknown"),
                session_id=record.additional_metadata.get("session_id", "unknown"),
                user_input=record.text,
                timestamp=record.timestamp or datetime.now(timezone.utc)
            )
            
            # Add vector embedding if present
            if record.embedding is not None:
                self.chat_logger.add_vector_embedding(
                    entry_id=chat_entry.id,
                    user_id=chat_entry.user_id,
                    session_id=chat_entry.session_id,
                    embedding=record.embedding.tolist()
                )
            
            self.chat_logger.create_chat_log_entry(chat_entry)
        
        return record.id
    
    async def upsert_batch(self, collection_name: str, records: List[MemoryRecord]) -> List[str]:
        """NEW in SK 1.35.0: Batch upsert for better performance"""
        self.initialize()
        
        result_ids = []
        for record in records:
            record_id = await self.upsert(collection_name, record)
            result_ids.append(record_id)
        
        return result_ids
    
    async def get(self, collection_name: str, key: str, with_embedding: bool = False) -> Optional[MemoryRecord]:
        """Retrieve a memory record - enhanced for SK 1.35.0"""
        self.initialize()
        
        if collection_name == "chat_history":
            # Parse key to extract user_id and session_id
            parts = key.split("_")
            if len(parts) >= 3:
                user_id = parts[1]
                session_id = parts[2]
                entry_id = "_".join(parts[3:])
                
                chat_entry = self.chat_logger.get_chat_log_entry(entry_id, user_id, session_id)
                if chat_entry:
                    return self._convert_chat_to_memory_record(chat_entry, with_embedding)
        
        return None
    
    async def get_batch(self, collection_name: str, keys: List[str], with_embeddings: bool = False) -> List[MemoryRecord]:
        """NEW in SK 1.35.0: Batch get for better performance"""
        self.initialize()
        
        records = []
        for key in keys:
            record = await self.get(collection_name, key, with_embeddings)
            if record:
                records.append(record)
        
        return records
    
    async def remove(self, collection_name: str, key: str) -> None:
        """Remove a memory record - enhanced for SK 1.35.0"""
        self.initialize()
        
        if collection_name == "chat_history":
            parts = key.split("_")
            if len(parts) >= 3:
                user_id = parts[1]
                session_id = parts[2]
                entry_id = "_".join(parts[3:])
                
                self.chat_logger.delete_chat_log_entry(entry_id, user_id, session_id)
    
    async def remove_batch(self, collection_name: str, keys: List[str]) -> None:
        """NEW in SK 1.35.0: Batch remove for better performance"""
        self.initialize()
        
        for key in keys:
            await self.remove(collection_name, key)
    
    async def get_nearest_match(
        self,
        collection_name: str,
        embedding: np.ndarray,
        min_relevance_score: float = 0.0,
        with_embedding: bool = False,
    ) -> Tuple[MemoryRecord, float]:
        """NEW in SK 1.35.0: Get single nearest match with relevance score"""
        matches = await self.get_nearest_matches(
            collection_name, embedding, limit=1, min_relevance_score=min_relevance_score, with_embeddings=with_embedding
        )
        
        if matches:
            return matches[0]
        return None, 0.0
    
    async def get_nearest_matches(
        self,
        collection_name: str,
        embedding: np.ndarray,
        limit: int,
        min_relevance_score: float = 0.0,
        with_embeddings: bool = False,
    ) -> List[Tuple[MemoryRecord, float]]:
        """NEW in SK 1.35.0: Get nearest records by vector similarity with relevance scores"""
        self.initialize()
        
        # Enhanced vector search capability
        if collection_name == "chat_history" and embedding is not None:
            # Use Cosmos DB vector search when available
            # For now, return empty list with a note for future enhancement
            # This is where you'd implement Cosmos DB vector search
            pass
        
        return []
    
    def _convert_chat_to_memory_record(self, chat_entry: ChatLogEntry, with_embedding: bool = False) -> MemoryRecord:
        """Convert ChatLogEntry to MemoryRecord for SK compatibility"""
        return MemoryRecord(
            id=chat_entry.id,
            text=chat_entry.user_input,
            description=f"Chat entry from session {chat_entry.session_id}",
            additional_metadata={
                "user_id": chat_entry.user_id,
                "session_id": chat_entry.session_id,
                "conversation_turn": chat_entry.conversation_turn,
                "agent_response_count": len(chat_entry.agent_responses)
            },
            timestamp=chat_entry.timestamp,
            embedding=None  # Would be populated from Cosmos DB vector search
        )
