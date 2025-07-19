Here are best practices for implementing Cosmos DB memory logging for NL2SQL using Semantic Kernel 1.35 in Python:

## 1. **Data Models with Pydantic**

````python
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import uuid4

class LogTokens(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_tokens: int = 0

class AgentResponse(BaseModel):
    agent_name: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tokens: Optional[LogTokens] = None
    execution_time_ms: Optional[int] = None

class ChatLogEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_input: str
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    conversation_turn: int = 1
    vector_cache_hit: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
````

## 2. **Async Cosmos DB Logger**

````python
import asyncio
from typing import List, Optional
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey
from .models.chat_models import ChatLogEntry, AgentResponse
import logging

class ChatLogger:
    def __init__(self, connection_string: str, database_id: str, container_id: str):
        self.client = CosmosClient.from_connection_string(connection_string)
        self.database_id = database_id
        self.container_id = container_id
        self.container = None
        self._initialized = False
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize database and container if they don't exist"""
        if self._initialized:
            return
            
        try:
            database = await self.client.create_database_if_not_exists(id=self.database_id)
            self.container = await database.create_container_if_not_exists(
                id=self.container_id,
                partition_key=PartitionKey(path="/id"),
                offer_throughput=400  # Adjust based on your needs
            )
            self._initialized = True
            self.logger.info(f"Cosmos DB initialized: {self.database_id}/{self.container_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Cosmos DB: {e}")
            raise

    async def create_chat_log_entry(self, chat_log_entry: ChatLogEntry) -> ChatLogEntry:
        """Create a new chat log entry"""
        await self.initialize()
        
        try:
            response = await self.container.create_item(body=chat_log_entry.dict())
            return ChatLogEntry(**response)
        except Exception as e:
            self.logger.error(f"Failed to create chat log entry: {e}")
            raise

    async def update_chat_log_entry(self, chat_log_entry: ChatLogEntry) -> ChatLogEntry:
        """Update an existing chat log entry"""
        await self.initialize()
        
        try:
            response = await self.container.upsert_item(body=chat_log_entry.dict())
            return ChatLogEntry(**response)
        except Exception as e:
            self.logger.error(f"Failed to update chat log entry: {e}")
            raise

    async def get_chat_log_entry(self, entry_id: str) -> Optional[ChatLogEntry]:
        """Retrieve a chat log entry by ID"""
        await self.initialize()
        
        try:
            response = await self.container.read_item(item=entry_id, partition_key=entry_id)
            return ChatLogEntry(**response)
        except Exception as e:
            self.logger.warning(f"Chat log entry not found: {entry_id}")
            return None

    async def query_by_user_input(self, user_input: str, limit: int = 10) -> List[ChatLogEntry]:
        """Query chat log entries by user input"""
        await self.initialize()
        
        query = "SELECT * FROM c WHERE c.user_input = @user_input ORDER BY c.timestamp DESC"
        parameters = [{"name": "@user_input", "value": user_input}]
        
        try:
            items = []
            async for item in self.container.query_items(
                query=query, 
                parameters=parameters,
                max_item_count=limit
            ):
                items.append(ChatLogEntry(**item))
            return items
        except Exception as e:
            self.logger.error(f"Failed to query chat log entries: {e}")
            return []

    async def close(self):
        """Close the Cosmos DB client"""
        if self.client:
            await self.client.close()
````

## 3. **Enhanced Chat Processor with Memory**

````python
import asyncio
from datetime import datetime
from typing import Optional, List
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
from .chat_logger import ChatLogger
from .vector_store_plugin import VectorStorePlugin
from .models.chat_models import ChatLogEntry, AgentResponse, LogTokens

class ChatProcessor:
    def __init__(
        self,
        kernel: Kernel,
        agent_group_chat: AgentGroupChat,
        chat_logger: ChatLogger,
        vector_store_plugin: VectorStorePlugin,
        input_token_cost: float = 0.0015,  # Per 1K tokens
        output_token_cost: float = 0.002   # Per 1K tokens
    ):
        self.kernel = kernel
        self.agent_group_chat = agent_group_chat
        self.chat_logger = chat_logger
        self.vector_store_plugin = vector_store_plugin
        self.input_token_cost = input_token_cost
        self.output_token_cost = output_token_cost
        self.session_id = str(uuid4())
        self.conversation_turn = 0

    async def process_user_query(self, user_input: str) -> str:
        """Process user query with memory logging"""
        self.conversation_turn += 1
        
        # Create initial chat log entry
        chat_log_entry = ChatLogEntry(
            user_input=user_input,
            session_id=self.session_id,
            conversation_turn=self.conversation_turn
        )

        try:
            # Check vector cache first
            cached_response = await self._check_vector_cache(user_input)
            if cached_response:
                chat_log_entry.vector_cache_hit = True
                await self.chat_logger.create_chat_log_entry(chat_log_entry)
                return cached_response

            # Process through agent group chat
            response = await self._invoke_agent_chat(user_input, chat_log_entry)
            
            # Cache the response
            await self._cache_response(user_input, response)
            
            # Save final log entry
            await self.chat_logger.update_chat_log_entry(chat_log_entry)
            
            return response

        except Exception as e:
            # Log error
            error_response = AgentResponse(
                agent_name="System",
                response=f"Error processing query: {str(e)}",
                timestamp=datetime.utcnow()
            )
            chat_log_entry.agent_responses.append(error_response)
            await self.chat_logger.update_chat_log_entry(chat_log_entry)
            raise

    async def _check_vector_cache(self, user_input: str) -> Optional[str]:
        """Check vector store for cached responses"""
        try:
            return await self.vector_store_plugin.search_chat_store(
                collection_name="chat_cache",
                search_input=user_input,
                similarity_threshold=0.8
            )
        except Exception as e:
            # Log but don't fail
            return None

    async def _invoke_agent_chat(self, user_input: str, chat_log_entry: ChatLogEntry) -> str:
        """Invoke agent group chat and log responses"""
        start_time = datetime.utcnow()
        
        # Add user message to chat
        await self.agent_group_chat.add_chat_message(
            role="user", 
            content=user_input
        )
        
        # Process through agents
        async for message in self.agent_group_chat.invoke():
            end_time = datetime.utcnow()
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            # Calculate token usage (if available)
            tokens = self._calculate_tokens(message)
            
            agent_response = AgentResponse(
                agent_name=message.role or "Unknown",
                response=message.content or "",
                timestamp=end_time,
                tokens=tokens,
                execution_time_ms=execution_time
            )
            
            chat_log_entry.agent_responses.append(agent_response)
            start_time = end_time

        # Return the final response
        if chat_log_entry.agent_responses:
            return chat_log_entry.agent_responses[-1].response
        return "No response generated"

    def _calculate_tokens(self, message) -> LogTokens:
        """Calculate token usage and costs"""
        # This is a simplified calculation - adapt based on your token counting method
        input_tokens = len(message.content.split()) * 1.3  # Rough estimate
        output_tokens = 0  # Would need to track from the model response
        
        return LogTokens(
            input_tokens=int(input_tokens),
            output_tokens=output_tokens,
            input_cost=(input_tokens / 1000) * self.input_token_cost,
            output_cost=(output_tokens / 1000) * self.output_token_cost,
            total_tokens=int(input_tokens + output_tokens)
        )

    async def _cache_response(self, user_input: str, response: str):
        """Cache the response for future use"""
        try:
            await self.vector_store_plugin.ingest_to_chat_store(
                collection_name="chat_cache",
                question=user_input,
                answer=response
            )
        except Exception as e:
            # Log but don't fail
            pass

    async def reset_conversation(self):
        """Reset the conversation"""
        self.session_id = str(uuid4())
        self.conversation_turn = 0
        await self.agent_group_chat.reset()
````

## 4. **Vector Store Plugin for Python**

````python
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from typing import Optional, List
import asyncio

class VectorStorePlugin:
    def __init__(self, memory_store: AzureCognitiveSearchMemoryStore):
        self.memory = SemanticTextMemory(storage=memory_store, embeddings_generator=None)

    async def search_chat_store(
        self, 
        collection_name: str, 
        search_input: str,
        similarity_threshold: float = 0.8,
        limit: int = 1
    ) -> Optional[str]:
        """Search cached responses"""
        try:
            results = await self.memory.search_async(
                collection=collection_name,
                query=search_input,
                limit=limit,
                min_relevance_score=similarity_threshold
            )
            
            if results and len(results) > 0:
                return results[0].text
            return None
            
        except Exception as e:
            return None

    async def ingest_to_chat_store(
        self, 
        collection_name: str, 
        question: str, 
        answer: str
    ):
        """Cache a Q&A pair"""
        try:
            await self.memory.save_information_async(
                collection=collection_name,
                text=answer,
                id=str(hash(question)),
                description=question
            )
        except Exception as e:
            # Log error but don't fail
            pass
````

## 5. **Configuration and Best Practices**

````python
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Cosmos DB
    COSMOS_CONNECTION_STRING: str
    COSMOS_DATABASE_ID: str = "nl2sql_logs"
    COSMOS_CONTAINER_ID: str = "chat_logs"
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    
    # Token Costs
    INPUT_TOKEN_COST: float = 0.0015  # per 1K tokens
    OUTPUT_TOKEN_COST: float = 0.002  # per 1K tokens
    
    # Performance
    VECTOR_SIMILARITY_THRESHOLD: float = 0.8
    CACHE_COLLECTION_NAME: str = "chat_cache"
    FAQ_COLLECTION_NAME: str = "faq"
    
    class Config:
        env_file = ".env"

settings = Settings()
````

## 6. **Main Application Integration**

````python
import asyncio
from src.config import settings
from src.services.chat_logger import ChatLogger
from src.services.chat_processor import ChatProcessor
from src.services.vector_store_plugin import VectorStorePlugin
from semantic_kernel import Kernel

async def main():
    # Initialize services
    chat_logger = ChatLogger(
        connection_string=settings.COSMOS_CONNECTION_STRING,
        database_id=settings.COSMOS_DATABASE_ID,
        container_id=settings.COSMOS_CONTAINER_ID
    )
    
    # Initialize kernel and agents (your existing setup)
    kernel = Kernel()
    # ... setup agents and group chat
    
    # Create chat processor
    chat_processor = ChatProcessor(
        kernel=kernel,
        agent_group_chat=agent_group_chat,
        chat_logger=chat_logger,
        vector_store_plugin=vector_store_plugin,
        input_token_cost=settings.INPUT_TOKEN_COST,
        output_token_cost=settings.OUTPUT_TOKEN_COST
    )
    
    # Main chat loop
    try:
        while True:
            user_input = input("User> ")
            if user_input.lower() in ['exit', 'quit']:
                break
            elif user_input.lower() == 'reset':
                await chat_processor.reset_conversation()
                continue
                
            response = await chat_processor.process_user_query(user_input)
            print(f"Assistant> {response}")
            
    finally:
        await chat_logger.close()

if __name__ == "__main__":
    asyncio.run(main())
````

## Key Best Practices:

1. **Async/Await Pattern**: Use async operations for all I/O operations
2. **Error Handling**: Graceful degradation when logging fails
3. **Connection Pooling**: Reuse Cosmos DB connections
4. **Data Validation**: Use Pydantic for type safety and validation
5. **Configuration Management**: Centralized settings with environment variables
6. **Token Tracking**: Monitor and log API usage costs
7. **Vector Caching**: Implement semantic similarity caching
8. **Session Management**: Track conversation sessions and turns
9. **Performance Monitoring**: Log execution times and token usage
10. **Clean Architecture**: Separate concerns between logging, processing, and storage

