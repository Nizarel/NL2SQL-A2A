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
from Models.api_models import QueryRequest, SQLGenerationRequest, SQLExecutionRequest, SummarizationRequest, ConversationAnalysisRequest, APIResponse
from Models.schema_analysis_result import SchemaAnalysisResult


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
        print(f"👤 User: {request.user_id}, Session: {request.session_id}")
        
        result = await system.orchestrator_agent.process({
            "question": request.question,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "execute": request.execute,
            "limit": request.limit,
            "include_summary": request.include_summary,
            "context": request.context,
            "enable_conversation_logging": request.enable_conversation_logging
        })
        
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
        print(f"👤 User: {request.user_id}, Session: {request.session_id}")
        
        result = await system.orchestrator_agent.process({
            "question": request.question,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "execute": request.execute,
            "limit": request.limit,
            "include_summary": request.include_summary,
            "context": request.context,
            "enable_conversation_logging": request.enable_conversation_logging
        })
        
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


@app.post("/conversation/analyze", response_model=APIResponse)
async def analyze_conversation_from_cosmos(
    request: ConversationAnalysisRequest,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """Analyze conversation data retrieved from Cosmos DB"""
    try:
        print(f"🗄️ Analyzing conversation from Cosmos DB")
        print(f"👤 User: {request.user_id}, Session: {request.session_id}")
        
        if not system.memory_service:
            raise HTTPException(
                status_code=503,
                detail="Memory service (Cosmos DB) not available"
            )
        
        # Get conversation history from Cosmos DB
        print("📖 Retrieving conversation history from Cosmos DB...")
        conversation_history = await system.memory_service.get_user_conversation_history(
            user_id=request.user_id,
            session_id=request.session_id,
            limit=request.limit
        )
        
        if not conversation_history:
            return APIResponse(
                success=True,
                data={
                    "message": "No conversation history found for the specified user and session",
                    "user_id": request.user_id,
                    "session_id": request.session_id
                },
                metadata={"conversation_count": 0}
            )
        
        # Get session context (recent messages)
        print("💬 Retrieving session messages...")
        session_messages = await system.memory_service.get_session_context(
            user_id=request.user_id,
            session_id=request.session_id,
            max_messages=request.limit * 2
        )
        
        # Compile conversation analysis
        conversation_analysis = {
            "session_summary": {
                "user_id": request.user_id,
                "session_id": request.session_id,
                "total_conversations": len(conversation_history),
                "total_messages": len(session_messages),
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC")
            },
            "conversation_highlights": [],
            "key_insights": [],
            "conversation_patterns": {},
            "business_summary": {
                "executive_summary": "",
                "main_topics": [],
                "recommendations": []
            }
        }
        
        # Process each conversation log
        for idx, conv in enumerate(conversation_history):
            highlight = {
                "conversation_number": idx + 1,
                "user_input": conv.user_input,
                "timestamp": conv.timestamp.isoformat() if conv.timestamp else "",
                "conversation_type": conv.metadata.conversation_type if conv.metadata else "unknown",
                "result_quality": conv.metadata.result_quality if conv.metadata else "unknown"
            }
            
            # Add agent response summary if available
            if conv.agent_response:
                highlight["executive_summary"] = conv.agent_response.executive_summary
                highlight["confidence_level"] = conv.agent_response.confidence_level
                highlight["key_insights_count"] = len(conv.agent_response.key_insights)
                highlight["recommendations_count"] = len(conv.agent_response.recommendations)
                
                # Collect key insights
                for insight in conv.agent_response.key_insights:
                    conversation_analysis["key_insights"].append({
                        "conversation": idx + 1,
                        "insight": insight,  # insight is a string
                        "timestamp": conv.timestamp.isoformat() if conv.timestamp else ""
                    })
            
            conversation_analysis["conversation_highlights"].append(highlight)
        
        # Analyze conversation patterns
        conversation_types = [conv.metadata.conversation_type if conv.metadata else "unknown" for conv in conversation_history]
        result_qualities = [conv.metadata.result_quality if conv.metadata else "unknown" for conv in conversation_history]
        
        conversation_analysis["conversation_patterns"] = {
            "conversation_types": {ctype: conversation_types.count(ctype) for ctype in set(conversation_types)},
            "quality_distribution": {quality: result_qualities.count(quality) for quality in set(result_qualities)},
            "most_common_conversation_type": max(set(conversation_types), key=conversation_types.count) if conversation_types else "unknown",
            "average_insights_per_conversation": sum(len(conv.agent_response.key_insights) if conv.agent_response else 0 for conv in conversation_history) / len(conversation_history) if conversation_history else 0
        }
        
        # Generate business summary
        all_topics = []
        all_recommendations = []
        all_summaries = []
        
        for conv in conversation_history:
            if conv.agent_response:
                if hasattr(conv.agent_response, 'executive_summary'):
                    all_summaries.append(conv.agent_response.executive_summary)
                
                # Extract topics from user inputs and insights
                if conv.user_input:
                    # Simple topic extraction based on keywords
                    user_input_lower = conv.user_input.lower()
                    if any(word in user_input_lower for word in ["revenue", "sales", "income", "profit"]):
                        all_topics.append("Revenue Analysis")
                    if any(word in user_input_lower for word in ["customer", "client", "user"]):
                        all_topics.append("Customer Analytics")
                    if any(word in user_input_lower for word in ["product", "inventory", "item"]):
                        all_topics.append("Product Performance")
                    if any(word in user_input_lower for word in ["trend", "time", "month", "year"]):
                        all_topics.append("Time Series Analysis")
                
                # Collect recommendations
                for rec in conv.agent_response.recommendations:
                    all_recommendations.append(rec)  # rec is already a string
        
        conversation_analysis["business_summary"] = {
            "executive_summary": f"Analyzed {len(conversation_history)} conversations from session {request.session_id}. " + 
                               (f"Key focus areas include {', '.join(list(set(all_topics))[:3])}." if all_topics else "Mixed business analytics topics discussed."),
            "main_topics": list(set(all_topics)),
            "recommendations": list(set(all_recommendations))[:5],  # Top 5 unique recommendations
            "conversation_summaries": all_summaries[:3]  # Recent summaries
        }
        
        return APIResponse(
            success=True,
            data=conversation_analysis,
            metadata={
                "source": "cosmos_db",
                "conversation_count": len(conversation_history),
                "message_count": len(session_messages),
                "analysis_type": "conversation_retrospective"
            }
        )
        
    except Exception as e:
        print(f"❌ Conversation analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation analysis failed: {str(e)}")


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
        ).model_dump()
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
        ).model_dump()
    )


# Debug and Monitoring Endpoints
@app.get("/debug/pool-metrics", response_model=APIResponse)
async def get_pool_metrics(system: NL2SQLMultiAgentSystem = Depends(get_system)):
    """Get MCP connection pool metrics for debugging and monitoring"""
    try:
        metrics = system.get_connection_pool_metrics()
        if metrics:
            # Add connection history and operation log if available
            if hasattr(system, 'mcp_plugin') and system.mcp_plugin and system.mcp_plugin.connection_pool:
                pool = system.mcp_plugin.connection_pool
                metrics["connection_history"] = pool.get_connection_history(50)
                metrics["operation_log"] = pool.get_operation_log(50)
                
                # Add validation results
                validation = pool.validate_connection_reuse()
                metrics["reuse_validation"] = validation
        
        return APIResponse(
            success=True,
            data=metrics or {"message": "Connection pooling not enabled"}
        )
    except Exception as e:
        return APIResponse(
            success=False,
            error=f"Failed to get pool metrics: {str(e)}"
        )


@app.post("/debug/test-database", response_model=APIResponse)
async def test_database_operation(
    operation_data: dict,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """Test database operation for connection pool validation"""
    try:
        operation = operation_data.get("operation", "list_tables")
        
        if operation == "list_tables":
            # Use the database plugin directly to test connection pool
            if hasattr(system, 'mcp_plugin') and system.mcp_plugin:
                result = await system.mcp_plugin.list_tables()
                return APIResponse(
                    success=True,
                    data={"operation": operation, "result": result}
                )
            else:
                return APIResponse(
                    success=False,
                    error="MCP plugin not available"
                )
        else:
            return APIResponse(
                success=False,
                error=f"Unsupported test operation: {operation}"
            )
            
    except Exception as e:
        return APIResponse(
            success=False,
            error=f"Test operation failed: {str(e)}"
        )


@app.get("/debug/connection-history", response_model=APIResponse)
async def get_connection_history(
    limit: int = 100,
    system: NL2SQLMultiAgentSystem = Depends(get_system)
):
    """Get connection pool history for debugging"""
    try:
        if hasattr(system, 'mcp_plugin') and system.mcp_plugin and system.mcp_plugin.connection_pool:
            history = system.mcp_plugin.connection_pool.get_connection_history(limit)
            return APIResponse(
                success=True,
                data={"connection_history": history, "count": len(history)}
            )
        else:
            return APIResponse(
                success=False,
                error="Connection pool not available"
            )
    except Exception as e:
        return APIResponse(
            success=False,
            error=f"Failed to get connection history: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8005"))
    
    print(f"🚀 Starting NL2SQL Multi-Agent API Server on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print(f"🔧 Debug Endpoints: http://{host}:{port}/debug/pool-metrics")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )
