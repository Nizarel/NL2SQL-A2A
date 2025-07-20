"""
FastAPI Server for NL2SQL Multi-Agent System
Provides REST API endpoints for orchestrator and all specialized agents
"""

import os
import sys
import asyncio
import time
import re
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Load environment variables first
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
print(f"🔍 API Server loading .env from: {dotenv_path}")
print(f"🔍 API Server .env file exists: {os.path.exists(dotenv_path)}")
result = load_dotenv(dotenv_path, override=True)
print(f"🔍 API Server load_dotenv result: {result}")

# Test environment variables
mcp_url = os.getenv('MCP_SERVER_URL')
print(f"🔍 API Server MCP_SERVER_URL: {mcp_url[:50] if mcp_url else 'None'}...")

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import NL2SQLMultiAgentSystem


def make_json_serializable(obj):
    """
    Convert objects to JSON serializable format
    """
    if hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dictionary
        return {key: make_json_serializable(value) for key, value in obj.__dict__.items()}
    elif hasattr(obj, '_asdict'):
        # Handle namedtuples
        return make_json_serializable(obj._asdict())
    elif isinstance(obj, dict):
        # Recursively handle dictionaries
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively handle lists and tuples
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'data') and hasattr(obj, 'content'):
        # Handle CallToolResult objects
        return str(obj)
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        # Convert other non-serializable objects to string
        return str(obj)
    else:
        # Return serializable objects as-is
        return obj


# Pydantic Models for Request/Response
class QueryRequest(BaseModel):
    """Request model for natural language queries"""
    question: str = Field(..., description="Natural language question about the data")
    execute: bool = Field(True, description="Whether to execute the generated SQL query")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of rows to return")
    include_summary: bool = Field(True, description="Whether to generate AI summary and insights")
    context: str = Field("", description="Optional additional context for the question")


class SQLGenerationRequest(BaseModel):
    """Request model for SQL generation"""
    question: str = Field(..., description="Natural language question")
    context: str = Field("", description="Optional additional context")


class SQLExecutionRequest(BaseModel):
    """Request model for SQL execution"""
    sql_query: str = Field(..., description="SQL query to execute")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of rows to return")
    timeout: int = Field(30, ge=5, le=300, description="Query timeout in seconds")


class SummarizationRequest(BaseModel):
    """Request model for data summarization"""
    raw_results: str = Field(..., description="Raw query results")
    formatted_results: Dict[str, Any] = Field(..., description="Formatted query results")
    sql_query: str = Field(..., description="Original SQL query")
    question: str = Field(..., description="Original user question")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query execution metadata")


class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC"))


# Global system instance
nl2sql_system: Optional[NL2SQLMultiAgentSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the NL2SQL system"""
    global nl2sql_system
    
    print("🚀 Starting NL2SQL Multi-Agent API Server...")
    
    try:
        # Initialize the system
        nl2sql_system = await NL2SQLMultiAgentSystem.create_and_initialize()
        print("✅ NL2SQL Multi-Agent System initialized successfully!")
        yield
    except Exception as e:
        print(f"❌ Failed to initialize NL2SQL system: {str(e)}")
        raise
    finally:
        # Cleanup
        if nl2sql_system:
            await nl2sql_system.close()
            print("🔐 NL2SQL Multi-Agent System closed successfully")


# Create FastAPI app
app = FastAPI(
    title="NL2SQL Multi-Agent API",
    description="REST API for Natural Language to SQL Multi-Agent System with Semantic Kernel orchestration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_system() -> NL2SQLMultiAgentSystem:
    """Dependency to get the initialized system"""
    if nl2sql_system is None:
        raise HTTPException(status_code=503, detail="NL2SQL system not initialized")
    return nl2sql_system


# Health and Status Endpoints
@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API information"""
    return APIResponse(
        success=True,
        data={
            "message": "NL2SQL Multi-Agent API Server",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    )


@app.get("/health", response_model=APIResponse)
async def health_check(system: NL2SQLMultiAgentSystem = Depends(get_system)):
    """Health check endpoint"""
    try:
        # Check system status
        status = await system.get_workflow_status()
        
        # Check database connectivity
        db_info = await system.get_database_info()
        
        return APIResponse(
            success=True,
            data={
                "status": "healthy",
                "system_status": status,
                "database_connected": "Query Results" in str(db_info) or "Available tables" in str(db_info)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.get("/status", response_model=APIResponse)
async def get_system_status(system: NL2SQLMultiAgentSystem = Depends(get_system)):
    """Get detailed system status"""
    try:
        status = await system.get_workflow_status()
        return APIResponse(success=True, data=status["data"], metadata=status.get("metadata"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


# Orchestrator Endpoints
@app.post("/orchestrator/query", response_model=APIResponse)
async def orchestrator_query(
    request: QueryRequest,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """
    Process natural language query using the full orchestrator workflow
    This is the main endpoint for end-to-end NL2SQL processing
    """
    try:
        print(f"🎯 Orchestrator query: {request.question}")
        
        result = await system.ask_question(
            question=request.question,
            execute=request.execute,
            limit=request.limit,
            include_summary=request.include_summary,
            context=request.context
        )
        
        if result.get("success"):
            # Debug: Check data structure for serialization issues
            try:
                data = result.get("data")
                metadata = result.get("metadata")
                
                # Convert any non-serializable objects to strings
                if data:
                    data = make_json_serializable(data)
                if metadata:
                    metadata = make_json_serializable(metadata)
                
                return APIResponse(
                    success=True,
                    data=data,
                    metadata=metadata
                )
            except Exception as serialization_error:
                print(f"❌ Serialization error: {serialization_error}")
                # Fallback: return string representation
                return APIResponse(
                    success=True,
                    data={"result": str(result.get("data"))},
                    metadata={"original_metadata": str(result.get("metadata"))}
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Query processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestrator query failed: {str(e)}")


@app.post("/orchestrator/formatedresults", response_model=APIResponse)
async def orchestrator_formatted_results(
    request: QueryRequest,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """
    Process natural language query and return only formatted results and summary
    Excludes SQL query and raw results from the response
    """
    try:
        print(f"📊 Orchestrator formatted results: {request.question}")
        
        result = await system.ask_question(
            question=request.question,
            execute=request.execute,
            limit=request.limit,
            include_summary=request.include_summary,
            context=request.context
        )
        
        if result.get("success"):
            # Extract only formatted results and summary from the data
            original_data = result.get("data", {})
            filtered_data = {}
            
            # Include only formatted results and summary, exclude sql_query and results
            if "formatted_results" in original_data:
                filtered_data["formatted_results"] = make_json_serializable(original_data["formatted_results"])
            if "summary" in original_data:
                filtered_data["summary"] = make_json_serializable(original_data["summary"])
            if "insights" in original_data:
                filtered_data["insights"] = make_json_serializable(original_data["insights"])
            if "recommendations" in original_data:
                filtered_data["recommendations"] = make_json_serializable(original_data["recommendations"])
            
            return APIResponse(
                success=True,
                data=filtered_data,
                metadata=make_json_serializable(result.get("metadata"))
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Query processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestrator formatted results failed: {str(e)}")


@app.post("/orchestrator/query-async", response_model=APIResponse)
async def orchestrator_query_async(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """
    Process natural language query asynchronously
    Returns immediately with a task ID for tracking
    """
    try:
        task_id = f"task_{int(time.time() * 1000)}"
        
        # Store task status (in production, use a proper task queue like Celery)
        background_tasks.add_task(
            process_query_background,
            system, request, task_id
        )
        
        return APIResponse(
            success=True,
            data={
                "task_id": task_id,
                "status": "processing",
                "message": "Query is being processed in the background"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start async query: {str(e)}")


async def process_query_background(
    system: NL2SQLMultiAgentSystem,
    request: QueryRequest,
    task_id: str
):
    """Background task processor (placeholder for production task queue)"""
    try:
        print(f"🔄 Processing background task {task_id}: {request.question}")
        
        result = await system.ask_question(
            question=request.question,
            execute=request.execute,
            limit=request.limit,
            include_summary=request.include_summary,
            context=request.context
        )
        
        # In production, store result in cache/database for retrieval
        print(f"✅ Background task {task_id} completed successfully")
        
    except Exception as e:
        print(f"❌ Background task {task_id} failed: {str(e)}")


# SQL Generator Agent Endpoints
@app.post("/agents/sql-generator/generate", response_model=APIResponse)
async def generate_sql(
    request: SQLGenerationRequest,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """Generate SQL query from natural language question"""
    try:
        print(f"🧠 SQL Generation: {request.question}")
        
        result = await system.sql_generator_agent.process({
            "question": request.question,
            "context": request.context
        })
        
        if result.get("success"):
            return APIResponse(
                success=True,
                data=result.get("data"),
                metadata=result.get("metadata")
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"SQL generation failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")


@app.get("/agents/sql-generator/schema", response_model=APIResponse)
async def get_database_schema(system: NL2SQLMultiAgentSystem = Depends(get_system)):
    """Get the database schema context used by SQL generator"""
    try:
        schema = await system.get_schema_context()
        return APIResponse(
            success=True,
            data={"schema": schema}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")


# Executor Agent Endpoints
@app.post("/agents/executor/execute", response_model=APIResponse)
async def execute_sql(
    request: SQLExecutionRequest,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """Execute SQL query against the database"""
    try:
        print(f"⚡ SQL Execution: {request.sql_query[:100]}...")
        
        result = await system.executor_agent.process({
            "sql_query": request.sql_query,
            "limit": request.limit,
            "timeout": request.timeout
        })
        
        if result.get("success"):
            return APIResponse(
                success=True,
                data=result.get("data"),
                metadata=result.get("metadata")
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"SQL execution failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL execution failed: {str(e)}")


@app.get("/agents/executor/database-info", response_model=APIResponse)
async def get_database_info(system: NL2SQLMultiAgentSystem = Depends(get_system)):
    """Get database connection information and available tables"""
    try:
        db_info = await system.get_database_info()
        return APIResponse(
            success=True,
            data={"database_info": str(db_info)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database info: {str(e)}")


# Summarizing Agent Endpoints
@app.post("/agents/summarizer/analyze", response_model=APIResponse)
async def analyze_data(
    request: SummarizationRequest,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """Generate summary and insights from query results"""
    try:
        print(f"📊 Data Analysis: {request.question}")
        
        result = await system.summarizing_agent.process({
            "raw_results": request.raw_results,
            "formatted_results": request.formatted_results,
            "sql_query": request.sql_query,
            "question": request.question,
            "metadata": request.metadata
        })
        
        if result.get("success"):
            return APIResponse(
                success=True,
                data=result.get("data"),
                metadata=result.get("metadata")
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Data analysis failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data analysis failed: {str(e)}")


# Combined Workflow Endpoints
@app.post("/workflows/sql-only", response_model=APIResponse)
async def sql_generation_only(
    request: SQLGenerationRequest,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """Generate SQL without execution (SQL generation only)"""
    try:
        result = await system.ask_question(
            question=request.question,
            context=request.context,
            execute=False,
            include_summary=False
        )
        
        if result.get("success"):
            return APIResponse(
                success=True,
                data=result.get("data"),
                metadata=result.get("metadata")
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"SQL generation workflow failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL generation workflow failed: {str(e)}")


@app.post("/workflows/sql-and-execute", response_model=APIResponse)
async def sql_generation_and_execution(
    request: QueryRequest,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """Generate and execute SQL without summarization"""
    try:
        result = await system.ask_question(
            question=request.question,
            context=request.context,
            execute=True,
            limit=request.limit,
            include_summary=False
        )
        
        if result.get("success"):
            return APIResponse(
                success=True,
                data=result.get("data"),
                metadata=result.get("metadata")
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"SQL execution workflow failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL execution workflow failed: {str(e)}")


# Utility Endpoints
@app.get("/database/tables", response_model=APIResponse)
async def list_database_tables(system: NL2SQLMultiAgentSystem = Depends(get_system)):
    """List all available database tables"""
    try:
        # Use the MCP plugin to list tables
        tables_result = await system.mcp_plugin.list_tables()
        
        return APIResponse(
            success=True,
            data={"tables": str(tables_result)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")


@app.get("/database/table/{table_name}/schema", response_model=APIResponse)
async def get_table_schema(
    table_name: str,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """Get schema information for a specific table"""
    try:
        # Use the MCP plugin to describe table
        schema_result = await system.mcp_plugin.describe_table(table_name)
        
        return APIResponse(
            success=True,
            data={"table_schema": str(schema_result)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get table schema: {str(e)}")


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse(
            success=False,
            error=exc.detail,
            metadata={"status_code": exc.status_code}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            error=f"Internal server error: {str(exc)}",
            metadata={"error_type": type(exc).__name__}
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"🚀 Starting NL2SQL Multi-Agent API Server on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )
