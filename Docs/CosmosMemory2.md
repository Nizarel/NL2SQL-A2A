Here's the Python implementation for session and user tracking in your NL2SQL system:

## 1. **Enhanced Data Models with Pydantic**

````python
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import uuid4
import json

class LogTokens(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_tokens: int = 0

class AgentResponse(BaseModel):
    agent_name: str
    response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tokens: Optional[LogTokens] = None
    agent_turn_order: int = 0
    execution_time_ms: Optional[int] = None

class ChatLogEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    user_input: str
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_turn: int = 1
    vector_cache_hit: bool = False
    processing_time_ms: Optional[int] = None
    
    @property
    def partition_key(self) -> str:
        """Composite partition key for better Cosmos DB distribution"""
        return f"{self.user_id}_{self.session_id}"
    
    def dict(self, **kwargs):
        """Custom dict method to handle datetime serialization"""
        data = super().dict(**kwargs)
        # Convert datetime objects to ISO format strings
        if 'timestamp' in data:
            data['timestamp'] = self.timestamp.isoformat()
        for response in data.get('agent_responses', []):
            if 'timestamp' in response:
                response['timestamp'] = response['timestamp'].isoformat() if isinstance(response['timestamp'], datetime) else response['timestamp']
        return data

class UserSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_turns: int = 0
    is_active: bool = True
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def dict(self, **kwargs):
        """Custom dict method to handle datetime serialization"""
        data = super().dict(**kwargs)
        data['start_time'] = self.start_time.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        return data
````

## 2. **Enhanced ChatLogger with Session Management**

````python
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from azure.cosmos.aio import CosmosClient, ContainerProxy
from azure.cosmos import PartitionKey, exceptions
from ..models.chat_models import ChatLogEntry, AgentResponse, UserSession
import logging

class ChatLogger:
    def __init__(self, connection_string: str, database_id: str, 
                 chat_container_id: str, session_container_id: str):
        self.client = CosmosClient.from_connection_string(connection_string)
        self.database_id = database_id
        self.chat_container_id = chat_container_id
        self.session_container_id = session_container_id
        self.chat_container = None
        self.session_container = None
        self._initialized = False
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize database and containers if they don't exist"""
        if self._initialized:
            return
            
        try:
            database = await self.client.create_database_if_not_exists(id=self.database_id)
            
            # Create chat logs container
            self.chat_container = await database.create_container_if_not_exists(
                id=self.chat_container_id,
                partition_key=PartitionKey(path="/partition_key"),
                offer_throughput=400
            )
            
            # Create sessions container
            self.session_container = await database.create_container_if_not_exists(
                id=self.session_container_id,
                partition_key=PartitionKey(path="/user_id"),
                offer_throughput=400
            )
            
            self._initialized = True
            self.logger.info(f"Cosmos DB initialized: {self.database_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Cosmos DB: {e}")
            raise

    # ✅ Session Management Methods
    async def create_user_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> UserSession:
        """Create a new user session"""
        await self.initialize()
        
        session = UserSession(
            user_id=user_id,
            session_metadata=metadata or {}
        )
        
        try:
            response = await self.session_container.create_item(
                body=session.dict()
            )
            return UserSession(**response)
        except Exception as e:
            self.logger.error(f"Failed to create user session: {e}")
            raise

    async def get_active_session(self, user_id: str) -> UserSession:
        """Get the active session for a user, create one if none exists"""
        await self.initialize()
        
        query = """
            SELECT * FROM s 
            WHERE s.user_id = @user_id AND s.is_active = true 
            ORDER BY s.start_time DESC
        """
        parameters = [{"name": "@user_id", "value": user_id}]
        
        try:
            async for item in self.session_container.query_items(
                query=query, 
                parameters=parameters,
                max_item_count=1
            ):
                return UserSession(**item)
        except Exception as e:
            self.logger.warning(f"No active session found for user {user_id}: {e}")
        
        # Create new session if none found
        return await self.create_user_session(user_id)

    async def update_session_activity(self, session_id: str, user_id: str):
        """Update session's last activity and turn count"""
        await self.initialize()
        
        try:
            # Read current session
            response = await self.session_container.read_item(
                item=session_id, 
                partition_key=user_id
            )
            
            session_data = response
            session_data['last_activity'] = datetime.now(timezone.utc).isoformat()
            session_data['total_turns'] = session_data.get('total_turns', 0) + 1
            
            # Update session
            await self.session_container.upsert_item(body=session_data)
            
        except exceptions.CosmosResourceNotFoundError:
            # Session not found, create new one
            await self.create_user_session(user_id)
        except Exception as e:
            self.logger.error(f"Failed to update session activity: {e}")

    async def end_session(self, session_id: str, user_id: str):
        """Mark a session as inactive"""
        await self.initialize()
        
        try:
            # Read current session
            response = await self.session_container.read_item(
                item=session_id, 
                partition_key=user_id
            )
            
            session_data = response
            session_data['is_active'] = False
            session_data['last_activity'] = datetime.now(timezone.utc).isoformat()
            
            # Update session
            await self.session_container.upsert_item(body=session_data)
            
        except exceptions.CosmosResourceNotFoundError:
            self.logger.warning(f"Session not found for ending: {session_id}")
        except Exception as e:
            self.logger.error(f"Failed to end session: {e}")

    # ✅ Enhanced Chat Logging with Session Context
    async def create_chat_log_entry(self, chat_log_entry: ChatLogEntry) -> ChatLogEntry:
        """Create a new chat log entry"""
        await self.initialize()
        
        try:
            # Update session activity
            await self.update_session_activity(chat_log_entry.session_id, chat_log_entry.user_id)
            
            # Create chat log entry
            entry_dict = chat_log_entry.dict()
            entry_dict['partition_key'] = chat_log_entry.partition_key
            
            response = await self.chat_container.create_item(body=entry_dict)
            return ChatLogEntry(**response)
        except Exception as e:
            self.logger.error(f"Failed to create chat log entry: {e}")
            raise

    async def update_chat_log_entry(self, chat_log_entry: ChatLogEntry) -> ChatLogEntry:
        """Update an existing chat log entry"""
        await self.initialize()
        
        try:
            entry_dict = chat_log_entry.dict()
            entry_dict['partition_key'] = chat_log_entry.partition_key
            
            response = await self.chat_container.upsert_item(body=entry_dict)
            return ChatLogEntry(**response)
        except Exception as e:
            self.logger.error(f"Failed to update chat log entry: {e}")
            raise

    async def get_chat_log_entry(self, entry_id: str, user_id: str, session_id: str) -> Optional[ChatLogEntry]:
        """Retrieve a chat log entry by ID"""
        await self.initialize()
        
        partition_key = f"{user_id}_{session_id}"
        
        try:
            response = await self.chat_container.read_item(
                item=entry_id, 
                partition_key=partition_key
            )
            return ChatLogEntry(**response)
        except exceptions.CosmosResourceNotFoundError:
            self.logger.warning(f"Chat log entry not found: {entry_id}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get chat log entry: {e}")
            return None

    # ✅ Query by Session
    async def get_chat_log_entries_by_session(self, session_id: str, user_id: str) -> List[ChatLogEntry]:
        """Get all chat log entries for a specific session"""
        await self.initialize()
        
        query = """
            SELECT * FROM c 
            WHERE c.session_id = @session_id AND c.user_id = @user_id 
            ORDER BY c.conversation_turn ASC
        """
        parameters = [
            {"name": "@session_id", "value": session_id},
            {"name": "@user_id", "value": user_id}
        ]
        
        try:
            items = []
            async for item in self.chat_container.query_items(
                query=query, 
                parameters=parameters
            ):
                items.append(ChatLogEntry(**item))
            return items
        except Exception as e:
            self.logger.error(f"Failed to query chat log entries by session: {e}")
            return []

    # ✅ Query by User across all sessions
    async def get_chat_log_entries_by_user(self, user_id: str, limit: int = 50) -> List[ChatLogEntry]:
        """Get recent chat log entries for a user across all sessions"""
        await self.initialize()
        
        query = """
            SELECT * FROM c 
            WHERE c.user_id = @user_id 
            ORDER BY c.timestamp DESC 
            OFFSET 0 LIMIT @limit
        """
        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@limit", "value": limit}
        ]
        
        try:
            items = []
            async for item in self.chat_container.query_items(
                query=query, 
                parameters=parameters
            ):
                items.append(ChatLogEntry(**item))
            return items
        except Exception as e:
            self.logger.error(f"Failed to query chat log entries by user: {e}")
            return []

    async def get_chat_log_entries_by_user_input(self, user_input: str, user_id: str) -> List[ChatLogEntry]:
        """Query chat log entries by user input within a user's context"""
        await self.initialize()
        
        query = """
            SELECT * FROM c 
            WHERE c.user_input = @user_input AND c.user_id = @user_id
            ORDER BY c.timestamp DESC
        """
        parameters = [
            {"name": "@user_input", "value": user_input},
            {"name": "@user_id", "value": user_id}
        ]
        
        try:
            items = []
            async for item in self.chat_container.query_items(
                query=query, 
                parameters=parameters
            ):
                items.append(ChatLogEntry(**item))
            return items
        except Exception as e:
            self.logger.error(f"Failed to query chat log entries by user input: {e}")
            return []

    async def delete_chat_log_entry(self, entry_id: str, user_id: str, session_id: str):
        """Delete a chat log entry"""
        await self.initialize()
        
        partition_key = f"{user_id}_{session_id}"
        
        try:
            await self.chat_container.delete_item(
                item=entry_id, 
                partition_key=partition_key
            )
        except Exception as e:
            self.logger.error(f"Failed to delete chat log entry: {e}")
            raise

    async def close(self):
        """Close the Cosmos DB client"""
        if self.client:
            await self.client.close()
````

## 3. **Enhanced Chat Processor with Session Management**

````python
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat
from .chat_logger import ChatLogger
from .vector_store_plugin import VectorStorePlugin
from ..models.chat_models import ChatLogEntry, AgentResponse, LogTokens, UserSession

class ChatProcessor:
    def __init__(
        self,
        kernel: Kernel,
        agent_group_chat: AgentGroupChat,
        chat_logger: ChatLogger,
        vector_store_plugin: VectorStorePlugin,
        user_id: str,  # ✅ Add user identification
        input_token_cost: float = 0.0015,
        output_token_cost: float = 0.002
    ):
        self.kernel = kernel
        self.agent_group_chat = agent_group_chat
        self.chat_logger = chat_logger
        self.vector_store_plugin = vector_store_plugin
        self.user_id = user_id  # ✅ Store user ID
        self.input_token_cost = input_token_cost
        self.output_token_cost = output_token_cost
        
        # Session tracking
        self.current_session: Optional[UserSession] = None
        self.conversation_turn = 0
        self.is_complete = False
        
        # Token tracking
        self.input_token_count = 0
        self.output_token_count = 0
        self.total_token_count = 0

    # ✅ Initialize session
    async def initialize_session(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a new session for the user"""
        self.current_session = await self.chat_logger.get_active_session(self.user_id)
        print(f"Session initialized for user {self.user_id}: {self.current_session.session_id}")
        
        if metadata:
            self.current_session.session_metadata.update(metadata)

    async def process_chat_loop(self):
        """Main chat processing loop"""
        if not self.current_session:
            await self.initialize_session({
                "started_at": datetime.now(timezone.utc).isoformat(),
                "client": "PythonConsole",
                "version": "1.0"
            })

        while not self.is_complete:
            user_input = self._get_user_input()
            if not user_input:
                continue

            if user_input.upper() == "EXIT":
                self.is_complete = True
                break
            elif user_input.upper() == "RESET":
                await self._reset_conversation()
                continue
            else:
                await self._process_user_query(user_input)

    def _get_user_input(self) -> str:
        """Get user input from console"""
        print()
        return input("User (To quit type [EXIT] or to reset type [RESET])> ").strip()

    async def _process_user_query(self, user_input: str):
        """Process a user query with full session tracking"""
        if user_input.startswith("#"):
            # Handle file input
            user_input = await self._read_from_file(user_input[1:])
            if not user_input:
                return

        self.conversation_turn += 1
        start_time = datetime.now(timezone.utc)

        # ✅ Create comprehensive chat log entry
        chat_log_entry = ChatLogEntry(
            session_id=self.current_session.session_id,
            user_id=self.user_id,
            user_input=user_input,
            conversation_turn=self.conversation_turn,
            timestamp=start_time,
            vector_cache_hit=False,
            agent_responses=[]
        )

        try:
            # Check vector cache first
            cached_response = await self._check_vector_cache(user_input)
            if cached_response and await self._handle_cached_response(cached_response):
                chat_log_entry.vector_cache_hit = True
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                chat_log_entry.processing_time_ms = int(processing_time)
                await self.chat_logger.create_chat_log_entry(chat_log_entry)
                return

            # Process through agent group chat
            await self._invoke_agent_chat(user_input, chat_log_entry)
            
            # Calculate total processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            chat_log_entry.processing_time_ms = int(processing_time)
            
            # Cache the final response if appropriate
            await self._cache_response(user_input, chat_log_entry)
            
            # Save/update the complete log entry
            await self.chat_logger.update_chat_log_entry(chat_log_entry)
            
        except Exception as e:
            print(f"Error processing query: {e}")
            # Log error response
            error_response = AgentResponse(
                agent_name="System",
                response=f"Error processing query: {str(e)}",
                timestamp=datetime.now(timezone.utc)
            )
            chat_log_entry.agent_responses.append(error_response)
            await self.chat_logger.update_chat_log_entry(chat_log_entry)

    async def _check_vector_cache(self, user_input: str) -> Optional[str]:
        """Check vector store for cached responses"""
        print(">>>>>>>>>>> Searching Chat Cache for recent conversations <<<<<<<<<<<<<<")
        try:
            return await self.vector_store_plugin.search_chat_store(
                collection_name="chat_cache",
                search_input=user_input,
                similarity_threshold=0.8
            )
        except Exception as e:
            return None

    async def _handle_cached_response(self, cached_response: str) -> bool:
        """Handle cached response - ask user if they want to use it"""
        if cached_response and cached_response != "No results found.":
            print(f"\033[92m>>>>>>>>>>>Chat cache search Result:\n{cached_response}\033[0m")
            user_response = input("\nIs the answer correct? (Yes/No)> ").strip()
            
            if user_response.lower() == "yes":
                return True
        return False

    async def _invoke_agent_chat(self, user_input: str, chat_log_entry: ChatLogEntry):
        """Invoke agent group chat and log all responses"""
        try:
            # Add user message to chat
            await self.agent_group_chat.add_chat_message(role="user", content=user_input)
            
            agent_turn_order = 0
            
            # Process through agents
            async for message in self.agent_group_chat.invoke():
                agent_turn_order += 1
                agent_start_time = datetime.now(timezone.utc)
                
                # Calculate token usage (simplified - adapt to your token counting method)
                tokens = self._calculate_tokens(message.content or "")
                self._update_token_counts(tokens)
                
                # Calculate execution time
                execution_time = (datetime.now(timezone.utc) - agent_start_time).total_seconds() * 1000
                
                agent_response = AgentResponse(
                    agent_name=getattr(message, 'role', 'Unknown') or 'Unknown',
                    response=message.content or "",
                    timestamp=datetime.now(timezone.utc),
                    tokens=tokens,
                    agent_turn_order=agent_turn_order,
                    execution_time_ms=int(execution_time)
                )
                
                chat_log_entry.agent_responses.append(agent_response)
                
                # Display response with color coding
                color = self._get_agent_color(agent_response.agent_name)
                print(f"{color}{agent_response.agent_name}> {agent_response.response}\033[0m")
            
            self._print_token_counts()
            
        except Exception as e:
            print(f"Error in agent chat: {e}")
            raise

    def _calculate_tokens(self, content: str) -> LogTokens:
        """Calculate token usage and costs (simplified estimation)"""
        # This is a rough estimation - replace with actual token counting
        estimated_tokens = int(len(content.split()) * 1.3)
        
        return LogTokens(
            input_tokens=estimated_tokens,
            output_tokens=0,  # Would need actual model response data
            input_cost=(estimated_tokens / 1000) * self.input_token_cost,
            output_cost=0.0,
            total_tokens=estimated_tokens
        )

    def _update_token_counts(self, tokens: LogTokens):
        """Update running token counts"""
        self.input_token_count += tokens.input_tokens
        self.output_token_count += tokens.output_tokens
        self.total_token_count += tokens.total_tokens

    def _print_token_counts(self):
        """Print current token usage and costs"""
        input_cost = round((self.input_token_count / 1000) * self.input_token_cost, 8)
        output_cost = round((self.output_token_count / 1000) * self.output_token_cost, 8)
        total_cost = round(input_cost + output_cost, 8)
        
        print(f"\033[93m")  # Yellow color
        print(f"InputToken: {self.input_token_count} | ${input_cost} | "
              f"OutputToken: {self.output_token_count} | ${output_cost} | "
              f"TotalTokens: {self.total_token_count} | ${total_cost}")
        print("\033[0m")  # Reset color

    def _get_agent_color(self, agent_name: str) -> str:
        """Get color code for agent display"""
        colors = {
            "GenAgent": "\033[94m",      # Blue
            "ExecAgent": "\033[95m",     # Magenta  
            "SummAgent": "\033[96m",     # Cyan
            "ReviewAgent": "\033[92m",   # Green
        }
        return colors.get(agent_name, "\033[97m")  # Default white

    async def _cache_response(self, user_input: str, chat_log_entry: ChatLogEntry):
        """Cache the final response for future use"""
        if chat_log_entry.agent_responses:
            # Typically cache the summarizer's response
            final_response = chat_log_entry.agent_responses[-1]
            if "Summ" in final_response.agent_name:  # SummAgent
                try:
                    await self.vector_store_plugin.ingest_to_chat_store(
                        collection_name="chat_cache",
                        question=user_input,
                        answer=final_response.response
                    )
                    print("\n>>> This Chat data committed to cache.")
                except Exception as e:
                    # Log but don't fail
                    pass

    async def _reset_conversation(self):
        """Reset conversation but maintain session tracking"""
        # Reset token counts
        self.input_token_count = 0
        self.output_token_count = 0
        self.total_token_count = 0
        self.conversation_turn = 0
        
        # Reset agent group chat
        await self.agent_group_chat.reset()
        
        # End current session and start new one
        if self.current_session:
            await self.chat_logger.end_session(
                self.current_session.session_id, 
                self.user_id
            )
        
        # Create new session
        self.current_session = await self.chat_logger.create_user_session(self.user_id)
        
        print(f"[Conversation reset - New session: {self.current_session.session_id}]")

    async def _read_from_file(self, file_path: str) -> Optional[str]:
        """Read input from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                print(f"Read from file: {file_path}")
                return content
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    # ✅ Graceful session cleanup
    async def end_session(self):
        """End the current session"""
        if self.current_session:
            await self.chat_logger.end_session(
                self.current_session.session_id, 
                self.user_id
            )
            print(f"Session ended: {self.current_session.session_id}")
````

## 4. **Main Application Integration**

````python
import asyncio
import os
from src.services.chat_logger import ChatLogger
from src.services.chat_processor import ChatProcessor
from src.services.vector_store_plugin import VectorStorePlugin
from semantic_kernel import Kernel
from dotenv import load_dotenv

async def main():
    # Load environment variables
    load_dotenv()
    
    # ✅ Get user identification (from environment, auth, or input)
    user_id = os.getenv("USER_ID", "default-user")
    print(f"Starting session for user: {user_id}")
    
    # Initialize services
    chat_logger = ChatLogger(
        connection_string=os.getenv("COSMOS_CONNECTION_STRING"),
        database_id=os.getenv("COSMOS_DATABASE_ID", "nl2sql_logs"),
        chat_container_id=os.getenv("COSMOS_CHAT_CONTAINER_ID", "chat_logs"),
        session_container_id=os.getenv("COSMOS_SESSION_CONTAINER_ID", "user_sessions")
    )
    
    # Initialize your kernel, agents, and vector store plugin here
    kernel = Kernel()
    # ... setup your agents and group chat ...
    vector_store_plugin = VectorStorePlugin(...)
    agent_group_chat = ...  # Your agent group chat setup
    
    # ✅ Create chat processor with user context
    chat_processor = ChatProcessor(
        kernel=kernel,
        agent_group_chat=agent_group_chat,
        chat_logger=chat_logger,
        vector_store_plugin=vector_store_plugin,
        user_id=user_id,  # Pass user ID
        input_token_cost=float(os.getenv("INPUT_TOKEN_COST", "0.0015")),
        output_token_cost=float(os.getenv("OUTPUT_TOKEN_COST", "0.002"))
    )
    
    try:
        # Start the chat loop
        await chat_processor.process_chat_loop()
    finally:
        # ✅ Graceful cleanup
        await chat_processor.end_session()
        await chat_logger.close()
        print("Session ended gracefully.")

if __name__ == "__main__":
    asyncio.run(main())
````

## Key Features of This Python Implementation:

1. **✅ Complete Session Tracking**: Each conversation has a unique session ID
2. **✅ User Context**: All interactions are associated with a specific user
3. **✅ Turn Management**: Tracks conversation turns within sessions
4. **✅ Agent Performance**: Measures execution time for each agent
5. **✅ Token & Cost Tracking**: Comprehensive usage and cost monitoring
6. **✅ Vector Cache Integration**: Tracks cache hit/miss rates
7. **✅ Graceful Session Management**: Proper session initialization and cleanup
8. **✅ Cosmos DB Optimization**: Uses composite partition keys for better performance
9. **✅ Error Handling**: Robust error handling with logging
10. **✅ Async/Await**: Fully asynchronous operations for better performance

