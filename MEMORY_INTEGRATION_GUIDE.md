# NL2SQL Orchestrator Memory Integration Guide

## 🎯 Overview

Your NL2SQL application now has a sophisticated memory system that provides:

- **Session Management**: Track user conversations and query history
- **Semantic Similarity Search**: Find related past queries to provide context
- **Query Analytics**: Monitor usage patterns and performance metrics
- **Context Preservation**: Maintain conversation history across sessions
- **User Insights**: Analyze query patterns and provide suggestions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NL2SQL Application                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Your Agents   │  │  Orchestrator   │  │   API Layer  │ │
│  │                 │  │    Memory       │  │              │ │
│  │ • Schema Agent  │◄─┤    Service      ├─►│ • Sessions   │ │
│  │ • SQL Agent     │  │                 │  │ • Queries    │ │
│  │ • Executor      │  │ • Query Context │  │ • Analytics  │ │
│  └─────────────────┘  │ • Similarity    │  └──────────────┘ │
│                       │ • Analytics     │                   │
│                       └─────────────────┘                   │
├─────────────────────────────────────────────────────────────┤
│                    Cosmos DB Service                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  nl2sql_chatlogs│  │  nl2sql_cache   │                   │
│  │                 │  │                 │                   │
│  │ • Sessions      │  │ • Schema Cache  │                   │
│  │ • Messages      │  │ • Query Cache   │                   │
│  │ • Embeddings    │  │ • Results       │                   │
│  │ • Vector Search │  │                 │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Integration

### 1. Import the Memory Service

```python
from src.services.orchestrator_memory_service import OrchestratorMemoryService
from src.Models.agent_response import Session, Message, QueryResult, AgentResponse
```

### 2. Initialize in Your Orchestrator

```python
class YourNL2SQLOrchestrator:
    def __init__(self):
        self.memory_service = None
        
    async def initialize(self):
        # Initialize memory service with Azure Identity
        self.memory_service = await OrchestratorMemoryService.create_from_config()
        
    async def process_query(self, user_id: str, session_id: str, query: str):
        # Generate embedding for the query (use your embedding model)
        embedding = await self.generate_embedding(query)
        
        # Store query context
        context = await self.memory_service.process_query(
            user_id=user_id,
            session_id=session_id, 
            query=query,
            query_embedding=embedding
        )
        
        # Find similar past queries
        similar = await self.memory_service.find_similar_queries(
            query_embedding=embedding,
            user_id=user_id,
            limit=3
        )
        
        # Process with your existing agents
        sql_result = await self.your_existing_processing(query, similar)
        
        # Store the result
        result = QueryResult(
            query_id=context.query_id,
            sql_query=sql_result.sql,
            execution_result=sql_result.data,
            agent_response=AgentResponse(
                agent_type="orchestrator",
                response=sql_result.response,
                success=sql_result.success
            )
        )
        
        await self.memory_service.store_query_result(context, result)
        
        return result
```

## 📊 Key Features

### Session Management
- **Create Sessions**: `await memory_service.create_session(user_id, session_title)`
- **Get Context**: `await memory_service.get_session_context(user_id, session_id)`
- **List Sessions**: `await memory_service.get_user_sessions(user_id)`

### Query Processing
- **Process Queries**: `await memory_service.process_query(user_id, session_id, query, embedding)`
- **Store Results**: `await memory_service.store_query_result(context, result)`
- **Find Similar**: `await memory_service.find_similar_queries(embedding, user_id)`

### Analytics & Insights
- **User Stats**: `await memory_service.get_user_query_stats(user_id)`
- **Query History**: Filter and analyze past queries
- **Performance Metrics**: Track processing times and success rates

## 🔧 Configuration

### Environment Variables
```bash
# Required for Azure Identity
AZURE_TENANT_ID=433ec967-f454-49f2-b132-d07f81545e02
AZURE_SUBSCRIPTION_ID=79f24240-60f9-497c-8ce8-43af104aec8c

# Cosmos DB Configuration  
COSMOS_ACCOUNT_NAME=cosmos-acrasalesanalytics2
COSMOS_DATABASE_NAME=sales_analytics
COSMOS_CHATLOGS_CONTAINER=nl2sql_chatlogs
COSMOS_CACHE_CONTAINER=nl2sql_cache
```

### Azure Cosmos DB Setup
The memory service automatically handles:
- ✅ Database and container creation
- ✅ Hierarchical partitioning (`/user_id/session_id`)
- ✅ Vector indexing for similarity search
- ✅ TTL for cache management

## 🧪 Testing Your Integration

```python
# Test basic functionality
async def test_integration():
    service = await OrchestratorMemoryService.create_from_config()
    
    # Create a session
    session = await service.create_session("test_user", "Test Session")
    
    # Process a query
    context = await service.process_query(
        user_id="test_user",
        session_id=session.session_id,
        query="Show me sales data",
        query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    # Verify it works
    assert context.query_id is not None
    print("✅ Integration test passed!")
```

## 🚀 Production Deployment

### 1. Update your existing orchestrator
```python
# In your main orchestrator file
from src.services.orchestrator_memory_service import OrchestratorMemoryService

class YourOrchestrator:
    async def __init__(self):
        # Add memory service to your existing initialization
        self.memory_service = await OrchestratorMemoryService.create_from_config()
        
        # Your existing initialization code
        self.schema_agent = SchemaAnalystAgent()
        self.sql_agent = SQLGeneratorAgent()
        # ... etc
```

### 2. Enhance query processing
```python
async def process_nl_query(self, user_id: str, session_id: str, query: str):
    # Generate embedding (integrate with your embedding model)
    embedding = await self.generate_query_embedding(query)
    
    # Check for similar queries to provide context
    similar_queries = await self.memory_service.find_similar_queries(
        query_embedding=embedding,
        user_id=user_id,
        limit=5
    )
    
    # Use similar queries to enhance context for your agents
    enhanced_context = self.build_context_from_history(similar_queries)
    
    # Process with your existing pipeline
    result = await self.your_existing_pipeline(query, enhanced_context)
    
    # Store everything in memory
    query_context = await self.memory_service.process_query(
        user_id=user_id,
        session_id=session_id,
        query=query,
        query_embedding=embedding
    )
    
    await self.memory_service.store_query_result(query_context, result)
    
    return result
```

### 3. Add API endpoints for memory features
```python
# Add to your API server
@app.post("/api/sessions")
async def create_session(user_id: str, session_name: str):
    session = await orchestrator.memory_service.create_session(user_id, session_name)
    return session

@app.get("/api/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    sessions = await orchestrator.memory_service.get_user_sessions(user_id)
    return sessions

@app.get("/api/analytics/{user_id}")
async def get_user_analytics(user_id: str):
    stats = await orchestrator.memory_service.get_user_query_stats(user_id)
    return stats
```

## 🎯 Next Steps

1. **Run the Integration Demo**: `python nl2sql_orchestrator_integration_demo.py`
2. **Update Your Main Orchestrator**: Add memory service initialization
3. **Enhance Query Processing**: Integrate similarity search with your agents
4. **Add Analytics**: Expose memory insights through your API
5. **Test in Production**: Start with a subset of users to validate performance

## 📈 Benefits You'll See

- **Faster Query Resolution**: Reuse similar past results
- **Better User Experience**: Context-aware responses
- **Analytics Insights**: Understand user behavior patterns
- **Improved Accuracy**: Learn from successful past queries
- **Operational Intelligence**: Monitor system performance

Your NL2SQL application now has enterprise-grade memory capabilities! 🚀
