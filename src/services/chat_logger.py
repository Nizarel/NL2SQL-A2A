import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# Azure Identity imports for secure authentication
try:
    from azure.identity import DefaultAzureCredential
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

from Models.agent_response import ChatLogEntry, AgentResponse, UserSession

class ChatLogger:
    """
    Chat logging service for single Cosmos DB container with hierarchical partitioning
    Database: sales_analytics
    Container: nl2sql_chatlogs  
    Partition Key: /user_id/session_id (hierarchical)
    Supports: Chat logs, User sessions, Vector embedding
    """
    
    def __init__(
        self, 
        connection_string: str = None,
        use_azure_identity: bool = False,
        cosmos_endpoint: str = None,
        managed_identity_client_id: str = None
    ):
        """
        Initialize ChatLogger with flexible authentication options
        
        Args:
            connection_string: Traditional connection string (for local dev)
            use_azure_identity: Use Azure Identity for authentication
            cosmos_endpoint: Cosmos DB endpoint URL (for Azure Identity)
            managed_identity_client_id: Client ID for user-assigned managed identity
        """
        # Connection details
        self.connection_string = connection_string
        self.use_azure_identity = use_azure_identity
        self.cosmos_endpoint = cosmos_endpoint
        self.managed_identity_client_id = managed_identity_client_id
        
        # Cosmos DB settings
        self.database_id = "sales_analytics"
        self.container_id = "nl2sql_chatlogs"
        
        # Initialize clients
        self.client = None
        self.database = None
        self.container = None
        self._initialized = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize Cosmos DB connection with enhanced authentication support"""
        if self._initialized:
            return
            
        try:
            if self.use_azure_identity and AZURE_IDENTITY_AVAILABLE:
                # Azure Identity authentication
                if not self.cosmos_endpoint:
                    raise ValueError("cosmos_endpoint is required for Azure Identity authentication")
                
                # Create appropriate credential
                if self.managed_identity_client_id:
                    from azure.identity import ManagedIdentityCredential
                    credential = ManagedIdentityCredential(client_id=self.managed_identity_client_id)
                    self.logger.info(f"🔐 Using User-Assigned Managed Identity: {self.managed_identity_client_id}")
                else:
                    credential = DefaultAzureCredential()
                    self.logger.info("🔐 Using Default Azure Credential (System-Assigned MI or other)")
                
                self.client = CosmosClient(self.cosmos_endpoint, credential=credential)
                
            else:
                # Traditional connection string authentication
                if not self.connection_string:
                    # Try to get from environment
                    endpoint = os.getenv("COSMOS_DB_CONNECTION_STRING", "")
                    account_key = os.getenv("AccountKey", "")
                    
                    if endpoint and account_key:
                        if not endpoint.startswith("https://"):
                            endpoint = f"https://{endpoint}"
                        if not endpoint.endswith("/"):
                            endpoint = f"{endpoint}/"
                        self.connection_string = f"AccountEndpoint={endpoint};AccountKey={account_key};"
                    
                if not self.connection_string:
                    raise ValueError(
                        "Connection string is required. Set COSMOS_DB_CONNECTION_STRING and AccountKey "
                        "environment variables or pass connection_string parameter."
                    )
                
                self.client = CosmosClient.from_connection_string(self.connection_string)
                self.logger.info("🔐 Using connection string authentication")
            
            # Create database if it doesn't exist
            self.database = self.client.create_database_if_not_exists(id=self.database_id)
            
            # Create single container with hierarchical partition key and vector embedding support
            # Cosmos DB Vector Search is now GA - enabling vector capabilities
            
            # Try to get existing container first (to handle RBAC restrictions)
            try:
                self.container = self.database.get_container_client(self.container_id)
                self.logger.info(f"Using existing container: {self.container_id}")
            except exceptions.CosmosResourceNotFoundError:
                # Container doesn't exist, try to create it with GA Vector Search support
                try:
                    self.container = self.database.create_container(
                        id=self.container_id,
                        partition_key=PartitionKey(path="/partition_key"),
                        vector_embedding_policy={
                            "vectorEmbeddings": [
                                {
                                    "path": "/embedding",
                                    "dataType": "float32",
                                    "distanceFunction": "cosine",
                                    "dimensions": 1536  # OpenAI text-embedding-3-small
                                }
                            ]
                        },
                        indexing_policy={
                            "includedPaths": [
                                {"path": "/*"}
                            ],
                            "excludedPaths": [
                                {"path": "/embedding/*"}
                            ],
                            "vectorIndexes": [
                                {
                                    "path": "/embedding",
                                    "type": "diskANN"
                                }
                            ]
                        },
                        offer_throughput=1000
                    )
                    self.logger.info(f"Created new container with GA Vector Search: {self.container_id}")
                except exceptions.CosmosHttpResponseError as e:
                    if "Forbidden" in str(e) or "RBAC" in str(e):
                        self.logger.warning(f"RBAC permission issue - container may already exist: {e}")
                        # Try to get the container again
                        try:
                            self.container = self.database.get_container_client(self.container_id)
                            self.logger.info(f"Successfully connected to existing container: {self.container_id}")
                        except Exception as inner_e:
                            self.logger.error(f"Cannot access container {self.container_id}: {inner_e}")
                            raise inner_e
                    else:
                        raise e
            
            self._initialized = True
            self.logger.info(f"Cosmos DB initialized with vector embedding: {self.database_id}/{self.container_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Cosmos DB: {e}")
            raise

    # === SINGLE CONTAINER DOCUMENT TYPES ===
    def _create_chat_document(self, chat_log_entry: ChatLogEntry) -> Dict[str, Any]:
        """Create document for chat log entry with discriminator"""
        doc = chat_log_entry.model_dump()
        doc["document_type"] = "chat_log"
        doc["partition_key"] = f"{chat_log_entry.user_id}/{chat_log_entry.session_id}"
        return doc
    
    def _create_session_document(self, session: UserSession) -> Dict[str, Any]:
        """Create document for user session with discriminator"""
        doc = session.model_dump()
        doc["id"] = session.session_id  # Ensure id matches session_id for Cosmos DB
        doc["document_type"] = "user_session"
        doc["partition_key"] = session.user_id  # Use just user_id for sessions
        return doc

    # === USER SESSION MANAGEMENT ===
    def get_active_session(self, user_id: str) -> UserSession:
        """Get or create an active session for the user"""
        self.initialize()
        
        query = """
            SELECT * FROM c 
            WHERE c.document_type = 'user_session' 
            AND c.user_id = @user_id 
            AND c.is_active = true 
            ORDER BY c.start_time DESC
        """
        parameters = [{"name": "@user_id", "value": user_id}]
        
        try:
            items = list(self.container.query_items(
                query=query, 
                parameters=parameters,
                max_item_count=1,
                enable_cross_partition_query=True
            ))
            
            if items:
                item = items[0]
                return UserSession(**{k: v for k, v in item.items() if k not in ["document_type", "partition_key"]})
                
        except Exception as e:
            self.logger.warning(f"No active session found for user {user_id}: {e}")
        
        # Create new session if none found
        session = UserSession(
            user_id=user_id,
            is_active=True,
            start_time=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc)
        )
        
        # Save to Cosmos DB
        doc = self._create_session_document(session)
        try:
            self.container.create_item(doc)
            self.logger.info(f"Created new session: {session.session_id} for user: {user_id} with partition key: {session.user_id}")
        except Exception as create_error:
            self.logger.error(f"Failed to create session: {create_error}")
            # Still return the session object even if save failed
        
        return session
    
    def update_session_activity(self, session_id: str, user_id: str):
        """Update the last activity time for a session"""
        self.initialize()
        
        # Try both partition key strategies for backward compatibility
        partition_keys_to_try = [
            user_id,  # New strategy: just user_id for sessions
            f"{user_id}/{session_id}"  # Old strategy: user_id/session_id
        ]
        
        # Try to find and update the session document
        for partition_key in partition_keys_to_try:
            try:
                doc = self.container.read_item(item=session_id, partition_key=partition_key)
                doc["last_activity"] = datetime.now(timezone.utc).isoformat()
                
                # If using old partition key, migrate to new strategy
                if partition_key != user_id:
                    self.logger.info(f"Migrating session {session_id} to new partition key strategy")
                    # Delete old document and create with new partition key
                    self.container.delete_item(item=session_id, partition_key=partition_key)
                    doc["partition_key"] = user_id  # Update to new strategy
                    self.container.create_item(doc)
                else:
                    # Update existing document with new partition key
                    self.container.replace_item(item=doc, body=doc)
                
                self.logger.debug(f"Updated session activity: {session_id} with partition key: {partition_key}")
                return
            except Exception as e:
                self.logger.debug(f"Failed to update session with partition key {partition_key}: {e}")
                continue
        
        # If not found, try to create a fresh session record
        self.logger.info(f"Session {session_id} not found, attempting to create/update session record for user {user_id}")
        try:
            # Try to read the session first (might exist with wrong partition key)
            query = """
                SELECT * FROM c 
                WHERE c.id = @session_id 
                AND c.document_type = 'user_session'
            """
            parameters = [{"name": "@session_id", "value": session_id}]
            
            items = list(self.container.query_items(
                query=query, 
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if items:
                # Session exists, just update it regardless of partition key
                doc = items[0]
                doc["last_activity"] = datetime.now(timezone.utc).isoformat()
                doc["partition_key"] = user_id  # Ensure correct partition key
                self.container.replace_item(item=doc, body=doc)
                self.logger.info(f"Updated existing session: {session_id} for user: {user_id}")
                return
            
            # Create a new session document for the missing session
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                is_active=True,
                start_time=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc)
            )
            doc = self._create_session_document(session)
            self.container.create_item(doc)
            self.logger.info(f"Created missing session: {session_id} for user: {user_id}")
            return
            
        except Exception as create_error:
            if "Entity with the specified id already exists" in str(create_error):
                self.logger.info(f"Session {session_id} already exists (concurrent creation), will skip update")
                return
            else:
                self.logger.error(f"Failed to create/update session {session_id}: {create_error}")
        
        self.logger.warning(f"Failed to update session activity for {session_id} - all attempts failed")
    
    def end_session(self, session_id: str, user_id: str):
        """Mark a session as inactive"""
        self.initialize()
        
        # Try both partition key strategies for backward compatibility
        partition_keys_to_try = [
            user_id,  # New strategy: just user_id for sessions
            f"{user_id}/{session_id}"  # Old strategy: user_id/session_id
        ]
        
        for partition_key in partition_keys_to_try:
            try:
                doc = self.container.read_item(item=session_id, partition_key=partition_key)
                doc["is_active"] = False
                doc["end_time"] = datetime.now(timezone.utc).isoformat()
                
                # If using old partition key, migrate to new strategy
                if partition_key != user_id:
                    self.logger.info(f"Migrating session {session_id} to new partition key strategy during end")
                    # Delete old document and create with new partition key
                    self.container.delete_item(item=session_id, partition_key=partition_key)
                    doc["partition_key"] = user_id  # Update to new strategy
                    self.container.create_item(doc)
                else:
                    # Update existing document
                    self.container.replace_item(item=doc, body=doc)
                
                self.logger.info(f"Ended session: {session_id}")
                return
            except Exception as e:
                self.logger.debug(f"Failed to end session with partition key {partition_key}: {e}")
                continue
        
        self.logger.warning(f"Failed to end session {session_id} - session not found")

    # === CHAT LOG MANAGEMENT ===
    def create_chat_log_entry(self, chat_log_entry: ChatLogEntry) -> ChatLogEntry:
        """Create a new chat log entry in Cosmos DB"""
        self.initialize()
        
        # Create document with discriminator and partition key
        doc = self._create_chat_document(chat_log_entry)
        
        # Insert into Cosmos DB
        result = self.container.create_item(doc)
        
        # Update session activity
        self.update_session_activity(chat_log_entry.session_id, chat_log_entry.user_id)
        
        self.logger.info(f"Created chat log entry: {chat_log_entry.id}")
        return chat_log_entry

    def get_chat_log_entry(self, entry_id: str, user_id: str, session_id: str) -> Optional[ChatLogEntry]:
        """Get a specific chat log entry by ID"""
        self.initialize()
        
        partition_key = f"{user_id}/{session_id}"
        
        try:
            item = self.container.read_item(item=entry_id, partition_key=partition_key)
            return ChatLogEntry(**{k: v for k, v in item.items() if k not in ["document_type", "partition_key"]})
        except exceptions.CosmosResourceNotFoundError:
            self.logger.debug(f"Chat log entry not found: {entry_id}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get chat log entry: {e}")
            return None

    def get_chat_log_entries_by_session(self, session_id: str, user_id: str) -> List[ChatLogEntry]:
        """Get all chat log entries for a specific session"""
        self.initialize()
        
        query = """
            SELECT * FROM c 
            WHERE c.document_type = 'chat_log' 
            AND c.session_id = @session_id 
            AND c.user_id = @user_id 
            ORDER BY c.conversation_turn
        """
        parameters = [
            {"name": "@session_id", "value": session_id},
            {"name": "@user_id", "value": user_id}
        ]
        
        try:
            items = list(self.container.query_items(
                query=query, 
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return [ChatLogEntry(**{k: v for k, v in item.items() if k not in ["document_type", "partition_key"]}) 
                   for item in items]
        except Exception as e:
            self.logger.error(f"Failed to get chat log entries by session: {e}")
            return []

    def get_chat_log_entries_by_user(self, user_id: str, limit: int = 50) -> List[ChatLogEntry]:
        """Get recent chat log entries for a user across all sessions"""
        self.initialize()
        
        query = """
            SELECT * FROM c 
            WHERE c.document_type = 'chat_log' 
            AND c.user_id = @user_id 
            ORDER BY c.timestamp DESC 
            OFFSET 0 LIMIT @limit
        """
        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@limit", "value": limit}
        ]
        
        try:
            items = list(self.container.query_items(
                query=query, 
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return [ChatLogEntry(**{k: v for k, v in item.items() if k not in ["document_type", "partition_key"]}) 
                   for item in items]
        except Exception as e:
            self.logger.error(f"Failed to get chat log entries by user: {e}")
            return []

    def get_chat_log_entries_by_user_input(self, user_input: str, user_id: str) -> List[ChatLogEntry]:
        """Query chat log entries by user input within a user's context"""
        self.initialize()
        
        query = """
            SELECT * FROM c 
            WHERE c.document_type = 'chat_log' 
            AND c.user_id = @user_id 
            AND CONTAINS(c.user_input, @user_input, true)
            ORDER BY c.timestamp DESC
        """
        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@user_input", "value": user_input}
        ]
        
        try:
            items = list(self.container.query_items(
                query=query, 
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return [ChatLogEntry(**{k: v for k, v in item.items() if k not in ["document_type", "partition_key"]}) 
                   for item in items]
        except Exception as e:
            self.logger.error(f"Failed to query chat log entries by user input: {e}")
            return []

    def delete_chat_log_entry(self, entry_id: str, user_id: str, session_id: str):
        """Delete a specific chat log entry"""
        self.initialize()
        
        partition_key = f"{user_id}/{session_id}"
        
        try:
            self.container.delete_item(item=entry_id, partition_key=partition_key)
            self.logger.info(f"Deleted chat log entry: {entry_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete chat log entry: {e}")
    
    # === VECTOR EMBEDDING SUPPORT FOR GA COSMOS DB VECTOR SEARCH ===
    
    def add_vector_embedding(self, entry_id: str, user_id: str, session_id: str, embedding: List[float]):
        """Add vector embedding to an existing chat log entry for GA Vector Search"""
        self.initialize()
        
        partition_key = f"{user_id}/{session_id}"
        
        try:
            # Get the existing document
            doc = self.container.read_item(item=entry_id, partition_key=partition_key)
            
            # Add the embedding
            doc["embedding"] = embedding
            
            # Update the document
            self.container.replace_item(item=doc, body=doc)
            self.logger.info(f"Added vector embedding to entry: {entry_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add vector embedding: {e}")
    
    def vector_search(self, user_id: str, query_embedding: List[float], limit: int = 5, similarity_threshold: float = 0.7) -> List[ChatLogEntry]:
        """
        Perform vector similarity search using GA Cosmos DB Vector Search
        
        Args:
            user_id: User to search within
            query_embedding: Vector representation of the query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of ChatLogEntry objects ordered by similarity (most similar first)
        """
        self.initialize()
        
        try:
            # GA Vector Search query - VectorDistance automatically orders by similarity
            query = f"""
                SELECT TOP @limit
                    c.id,
                    c.user_id,
                    c.session_id,
                    c.user_input,
                    c.timestamp,
                    c.conversation_turn,
                    VectorDistance(c.embedding, @queryVector) AS similarityScore
                FROM c 
                WHERE c.document_type = 'chat_log' 
                AND c.user_id = @user_id
                AND c.embedding != null
                AND VectorDistance(c.embedding, @queryVector) > @threshold
            """
            
            parameters = [
                {"name": "@user_id", "value": user_id},
                {"name": "@queryVector", "value": query_embedding},
                {"name": "@threshold", "value": similarity_threshold},
                {"name": "@limit", "value": limit}
            ]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            # Convert to ChatLogEntry objects
            results = []
            for item in items:
                # Create ChatLogEntry from the selected fields
                chat_entry = ChatLogEntry(
                    id=item["id"],
                    user_id=item["user_id"],
                    session_id=item["session_id"],
                    user_input=item["user_input"],
                    timestamp=datetime.fromisoformat(item["timestamp"].replace('Z', '+00:00')),
                    conversation_turn=item["conversation_turn"],
                    agent_responses=[]  # We didn't select this for performance
                )
                results.append(chat_entry)
            
            self.logger.info(f"Vector search found {len(results)} similar entries")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector similarity search failed: {e}")
            # Return empty list on failure
            return []

    def vector_similarity_search(self, user_id: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Alternative vector search method with more detailed results
        
        Returns:
            List of dictionaries with entry details and similarity scores
        """
        self.initialize()
        
        try:
            query = f"""
                SELECT TOP @topK
                    c.id,
                    c.user_input,
                    c.timestamp,
                    c.conversation_turn,
                    VectorDistance(c.embedding, @queryVector) AS similarity
                FROM c 
                WHERE c.document_type = 'chat_log'
                AND c.user_id = @userId
                AND c.embedding != null
            """
            
            parameters = [
                {"name": "@userId", "value": user_id},
                {"name": "@queryVector", "value": query_vector},
                {"name": "@topK", "value": top_k}
            ]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            return items
            
        except Exception as e:
            self.logger.error(f"Vector similarity search failed: {e}")
            return []
    
    def close(self):
        """Close the connection and clean up resources"""
        if self.client:
            # CosmosClient doesn't have an explicit close method in the latest SDK
            # The connection is automatically managed
            pass
        
        self.logger.info("ChatLogger closed")
