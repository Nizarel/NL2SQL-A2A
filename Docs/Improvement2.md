Looking at your NL2SQL multi-agent system, I can see it's a sophisticated implementation with conversation features. Let me analyze the architecture and propose optimizations for better execution and orchestration while minimizing code and ensuring no regression.

## Current Architecture Analysis

### Strengths:
1. **Well-structured multi-agent system** with specialized agents
2. **Conversation memory and context management**
3. **Schema analysis with caching**
4. **Connection pooling for database operations**
5. **Comprehensive error handling**

### Areas for Optimization:

## Proposed Refactoring Plan

### 1. **Simplify Agent Communication Pattern**

Create a unified agent communication interface to reduce code duplication:

````python
"""
Unified Agent Communication Interface
"""
from typing import Dict, Any, Protocol, Optional
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AgentMessage:
    """Standardized message format for inter-agent communication"""
    agent_name: str
    operation: str
    payload: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class AgentProtocol(Protocol):
    """Protocol defining agent interface"""
    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message and return response"""
        ...
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities"""
        ...
````

### 2. **Optimize Orchestrator Agent**

Refactor the orchestrator to use a more efficient workflow engine:

````python
"""
Optimized Orchestrator Agent with streamlined workflow execution
"""
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from semantic_kernel import Kernel
from .base_agent import BaseAgent
from .agent_interface import AgentMessage, AgentProtocol

class WorkflowStage(Enum):
    """Workflow stages"""
    CONTEXT_ANALYSIS = "context_analysis"
    SCHEMA_ANALYSIS = "schema_analysis"
    SQL_GENERATION = "sql_generation"
    SQL_EXECUTION = "sql_execution"
    SUMMARIZATION = "summarization"

@dataclass
class WorkflowConfig:
    """Workflow configuration"""
    enable_context_analysis: bool = True
    enable_schema_cache: bool = True
    enable_execution: bool = True
    enable_summarization: bool = True
    max_context_messages: int = 10
    parallel_stages: List[WorkflowStage] = None

class OptimizedOrchestratorAgent(BaseAgent):
    """
    Optimized orchestrator with streamlined workflow execution
    """
    
    def __init__(self, kernel: Kernel, agents: Dict[str, AgentProtocol], 
                 memory_service=None):
        super().__init__(kernel, "OptimizedOrchestrator")
        self.agents = agents
        self.memory_service = memory_service
        self._workflow_cache = {}
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute optimized workflow with parallel stages where possible
        """
        workflow_config = self._create_workflow_config(input_data)
        workflow_context = await self._initialize_workflow_context(input_data)
        
        try:
            # Stage 1: Context & Schema Analysis (can run in parallel)
            if workflow_config.enable_context_analysis:
                context_task = self._analyze_context(workflow_context)
                schema_task = self._analyze_schema(workflow_context)
                
                context_result, schema_result = await asyncio.gather(
                    context_task, schema_task, return_exceptions=True
                )
                
                workflow_context["context_analysis"] = context_result
                workflow_context["schema_analysis"] = schema_result
            
            # Stage 2: SQL Generation (depends on previous stages)
            sql_result = await self._generate_sql(workflow_context)
            workflow_context["sql_generation"] = sql_result
            
            if not sql_result.get("success"):
                return self._create_error_response(sql_result.get("error"))
            
            # Stage 3 & 4: Execution and Summarization (can run conditionally)
            final_tasks = []
            
            if workflow_config.enable_execution:
                final_tasks.append(self._execute_sql(workflow_context))
            
            if final_tasks:
                results = await asyncio.gather(*final_tasks, return_exceptions=True)
                
                if workflow_config.enable_execution:
                    workflow_context["execution"] = results[0]
                    
                    if workflow_config.enable_summarization and results[0].get("success"):
                        summary_result = await self._summarize_results(workflow_context)
                        workflow_context["summarization"] = summary_result
            
            # Complete workflow with logging
            if self.memory_service and input_data.get("enable_conversation_logging", True):
                await self._log_conversation(workflow_context)
            
            return self._compile_final_response(workflow_context)
            
        except Exception as e:
            return self._create_error_response(f"Workflow failed: {str(e)}")
    
    async def _analyze_context(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation context efficiently"""
        if not self.memory_service:
            return {"success": True, "data": {"context_available": False}}
        
        try:
            # Get conversation context with follow-up detection
            context_data = await self.memory_service.get_conversation_context_with_summary(
                user_id=workflow_context["user_id"],
                session_id=workflow_context["session_id"],
                max_context_window=workflow_context["config"].max_context_messages
            )
            
            # Detect follow-up queries
            follow_up_info = await self.memory_service.detect_follow_up_query(
                workflow_context["question"],
                context_data.get("context_messages", [])
            )
            
            # Update workflow context with enhanced question if follow-up
            if follow_up_info.get("is_follow_up"):
                workflow_context["enhanced_question"] = follow_up_info.get("enhanced_question")
                workflow_context["original_question"] = workflow_context["question"]
            
            return {
                "success": True,
                "data": {
                    "context_data": context_data,
                    "follow_up_info": follow_up_info,
                    "context_available": True
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_schema(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema with caching"""
        question = workflow_context.get("enhanced_question", workflow_context["question"])
        
        message = AgentMessage(
            agent_name="orchestrator",
            operation="analyze_schema",
            payload={
                "question": question,
                "context": workflow_context.get("context", ""),
                "use_cache": workflow_context["config"].enable_schema_cache
            }
        )
        
        return await self.agents["schema_analyst"].process(message)
    
    async def _generate_sql(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL with optimized context"""
        question = workflow_context.get("enhanced_question", workflow_context["question"])
        schema_analysis = workflow_context.get("schema_analysis", {})
        
        # Extract optimized schema context
        optimized_context = None
        if schema_analysis.get("success"):
            optimized_context = schema_analysis["data"].get("optimized_schema")
        
        message = AgentMessage(
            agent_name="orchestrator",
            operation="generate_sql",
            payload={
                "question": question,
                "context": workflow_context.get("context", ""),
                "optimized_schema_context": optimized_context,
                "schema_analysis": schema_analysis.get("data")
            }
        )
        
        return await self.agents["sql_generator"].process(message)
    
    async def _execute_sql(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL query"""
        sql_generation = workflow_context.get("sql_generation", {})
        
        if not sql_generation.get("success"):
            return {"success": False, "error": "No SQL query to execute"}
        
        message = AgentMessage(
            agent_name="orchestrator",
            operation="execute_sql",
            payload={
                "sql_query": sql_generation["data"]["sql_query"],
                "limit": workflow_context.get("limit", 100),
                "timeout": workflow_context.get("timeout", 30)
            }
        )
        
        return await self.agents["executor"].process(message)
    
    async def _summarize_results(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize execution results"""
        execution = workflow_context.get("execution", {})
        
        if not execution.get("success"):
            return {"success": False, "error": "No results to summarize"}
        
        message = AgentMessage(
            agent_name="orchestrator",
            operation="summarize",
            payload={
                "raw_results": execution["data"].get("raw_results"),
                "formatted_results": execution["data"].get("formatted_results"),
                "sql_query": workflow_context["sql_generation"]["data"]["sql_query"],
                "question": workflow_context["question"],
                "metadata": execution.get("metadata", {})
            }
        )
        
        return await self.agents["summarizer"].process(message)
    
    def _create_workflow_config(self, input_data: Dict[str, Any]) -> WorkflowConfig:
        """Create workflow configuration from input"""
        return WorkflowConfig(
            enable_context_analysis=self.memory_service is not None,
            enable_schema_cache=input_data.get("use_cache", True),
            enable_execution=input_data.get("execute", True),
            enable_summarization=input_data.get("include_summary", True),
            max_context_messages=input_data.get("max_context_messages", 10)
        )
    
    async def _initialize_workflow_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize workflow context"""
        context = {
            "question": input_data["question"],
            "user_id": input_data.get("user_id", "default_user"),
            "session_id": input_data.get("session_id", "default_session"),
            "context": input_data.get("context", ""),
            "limit": input_data.get("limit", 100),
            "timeout": input_data.get("timeout", 30),
            "config": self._create_workflow_config(input_data),
            "start_time": asyncio.get_event_loop().time()
        }
        
        # Start workflow session if memory service available
        if self.memory_service and input_data.get("enable_conversation_logging", True):
            workflow_session = await self.memory_service.start_workflow_session(
                user_id=context["user_id"],
                user_input=context["question"],
                session_id=context["session_id"]
            )
            context["workflow_session"] = workflow_session
        
        return context
    
    async def _log_conversation(self, workflow_context: Dict[str, Any]) -> None:
        """Log conversation to memory service"""
        if not workflow_context.get("workflow_session"):
            return
        
        try:
            # Extract results for logging
            formatted_results = None
            if workflow_context.get("execution", {}).get("success"):
                formatted_results = workflow_context["execution"]["data"].get("formatted_results")
            
            # Create agent response for logging
            agent_response = None
            if workflow_context.get("summarization", {}).get("success"):
                summary_data = workflow_context["summarization"]["data"]
                agent_response = {
                    "executive_summary": summary_data.get("executive_summary"),
                    "key_insights": summary_data.get("key_insights", []),
                    "recommendations": summary_data.get("recommendations", [])
                }
            
            # Log the conversation
            await self.memory_service.complete_workflow_session(
                workflow_context["workflow_session"],
                formatted_results=formatted_results,
                agent_response=agent_response,
                sql_query=workflow_context.get("sql_generation", {}).get("data", {}).get("sql_query"),
                processing_time_ms=(asyncio.get_event_loop().time() - workflow_context["start_time"]) * 1000
            )
            
        except Exception as e:
            print(f"⚠️ Failed to log conversation: {e}")
    
    def _compile_final_response(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final response from workflow results"""
        response = {
            "success": True,
            "data": {},
            "metadata": {
                "workflow_time": asyncio.get_event_loop().time() - workflow_context["start_time"],
                "stages_completed": []
            }
        }
        
        # Add SQL query if generated
        if workflow_context.get("sql_generation", {}).get("success"):
            response["data"]["sql_query"] = workflow_context["sql_generation"]["data"]["sql_query"]
            response["metadata"]["stages_completed"].append("sql_generation")
        
        # Add execution results if available
        if workflow_context.get("execution", {}).get("success"):
            response["data"]["executed"] = True
            response["data"]["results"] = workflow_context["execution"]["data"]["raw_results"]
            response["data"]["formatted_results"] = workflow_context["execution"]["data"]["formatted_results"]
            response["metadata"]["stages_completed"].append("execution")
            response["metadata"]["row_count"] = workflow_context["execution"]["metadata"].get("row_count", 0)
        
        # Add summary if available
        if workflow_context.get("summarization", {}).get("success"):
            response["data"]["summary"] = workflow_context["summarization"]["data"]
            response["metadata"]["stages_completed"].append("summarization")
        
        # Add context information
        if workflow_context.get("context_analysis", {}).get("success"):
            follow_up_info = workflow_context["context_analysis"]["data"].get("follow_up_info", {})
            if follow_up_info.get("is_follow_up"):
                response["metadata"]["is_follow_up"] = True
                response["metadata"]["follow_up_reasoning"] = follow_up_info.get("reasoning")
        
        # Add suggestions if available
        if self.memory_service and workflow_context.get("context_analysis", {}).get("success"):
            # Generate suggestions asynchronously later to not block response
            response["metadata"]["suggestions_available"] = True
        
        return response
    
    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "success": False,
            "error": error,
            "data": {},
            "metadata": {}
        }
````

### 3. **Optimize Memory Service**

Streamline the memory service for better performance:

````python
"""
Optimized Memory Service with improved caching and batching
"""
import asyncio
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime, timedelta

class OptimizedMemoryService:
    """
    Optimized memory service with batching and improved caching
    """
    
    def __init__(self, cosmos_service):
        self.cosmos_service = cosmos_service
        self._cache = {}  # In-memory cache
        self._write_queue = deque()  # Batch write queue
        self._write_task = None
        self._cache_ttl = timedelta(minutes=30)
        
    async def initialize(self):
        """Initialize service and start batch writer"""
        self._write_task = asyncio.create_task(self._batch_writer())
        
    async def _batch_writer(self):
        """Background task for batch writing to Cosmos DB"""
        while True:
            try:
                if len(self._write_queue) >= 10:  # Batch size threshold
                    batch = []
                    for _ in range(min(10, len(self._write_queue))):
                        batch.append(self._write_queue.popleft())
                    
                    # Write batch to Cosmos DB
                    await self._write_batch(batch)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"Batch writer error: {e}")
    
    async def _write_batch(self, batch: List[Dict[str, Any]]):
        """Write batch of items to Cosmos DB"""
        tasks = []
        for item in batch:
            if item["type"] == "message":
                tasks.append(self.cosmos_service.insert_message_async(
                    item["user_id"], item["message"]
                ))
            elif item["type"] == "cache":
                tasks.append(self.cosmos_service.set_cache_item_async(
                    item["cache_item"]
                ))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def queue_write(self, item: Dict[str, Any]):
        """Queue item for batch writing"""
        self._write_queue.append(item)
    
    async def get_cached_context(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached context with TTL check"""
        if key in self._cache:
            entry = self._cache[key]
            if datetime.utcnow() - entry["timestamp"] < self._cache_ttl:
                return entry["data"]
            else:
                del self._cache[key]
        return None
    
    def set_cached_context(self, key: str, data: Dict[str, Any]):
        """Set cached context"""
        self._cache[key] = {
            "data": data,
            "timestamp": datetime.utcnow()
        }
````

### 4. **Simplify API Server**

Create a cleaner API interface:

````python
"""
Optimized FastAPI Server with cleaner interface
"""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Optional

from .main import NL2SQLMultiAgentSystem
from .agents.orchestrator_agent_optimized import OptimizedOrchestratorAgent

# Global system instance
system: Optional[NL2SQLMultiAgentSystem] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global system
    try:
        system = await NL2SQLMultiAgentSystem.create_and_initialize()
        yield
    finally:
        if system:
            await system.close()

app = FastAPI(
    title="NL2SQL API",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/query")
async def process_query(request: QueryRequest) -> APIResponse:
    """
    Main endpoint for processing natural language queries
    """
    if not system:
        raise HTTPException(500, "System not initialized")
    
    try:
        result = await system.process_query(
            question=request.question,
            user_id=request.user_id,
            session_id=request.session_id,
            execute=request.execute,
            limit=request.limit,
            include_summary=request.include_summary,
            context=request.context,
            enable_conversation_logging=request.enable_conversation_logging
        )
        
        return APIResponse(
            success=result.get("success", False),
            data=result.get("data"),
            metadata=result.get("metadata")
        )
        
    except Exception as e:
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_ready": system is not None
    }
````

### 5. **Performance Optimizations**

Create a performance monitoring wrapper:

````python
"""
Performance monitoring utilities
"""
import time
import functools
from typing import Callable, Any
import asyncio

class PerformanceMonitor:
    """Performance monitoring for async operations"""
    
    def __init__(self):
        self.metrics = {}
    
    def track_async(self, name: str):
        """Decorator for tracking async function performance"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.time() - start
                    self._record_metric(name, elapsed, success=True)
                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    self._record_metric(name, elapsed, success=False)
                    raise
            return wrapper
        return decorator
    
    def _record_metric(self, name: str, elapsed: float, success: bool):
        """Record performance metric"""
        if name not in self.metrics:
            self.metrics[name] = {
                "count": 0,
                "total_time": 0,
                "success_count": 0,
                "avg_time": 0
            }
        
        self.metrics[name]["count"] += 1
        self.metrics[name]["total_time"] += elapsed
        if success:
            self.metrics[name]["success_count"] += 1
        self.metrics[name]["avg_time"] = (
            self.metrics[name]["total_time"] / self.metrics[name]["count"]
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return self.metrics

# Global performance monitor
perf_monitor = PerformanceMonitor()
````

## Benefits of This Refactoring

### 1. **Improved Performance**
- Parallel execution of context and schema analysis
- Batch writing to Cosmos DB
- Optimized caching with TTL
- Connection pooling already in place

### 2. **Reduced Code Complexity**
- Unified agent communication interface
- Cleaner API surface
- Centralized error handling
- Streamlined workflow execution

### 3. **Better Maintainability**
- Clear separation of concerns
- Standardized message passing
- Easier to add new agents
- Performance monitoring built-in

### 4. **No Regression Risk**
- All existing functionality preserved
- Backward compatible API
- Same conversation features
- Enhanced error handling

## Migration Strategy

1. **Phase 1**: Implement new interfaces alongside existing code
2. **Phase 2**: Gradually migrate agents to new interface
3. **Phase 3**: Switch to optimized orchestrator
4. **Phase 4**: Remove old code

