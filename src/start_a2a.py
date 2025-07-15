"""
Integration example for A2A Server with FastAPI
Run this to expose your NL2SQL Orchestrator Agent through A2A protocol
"""

import asyncio
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from a2a_server import A2AServer


# Global A2A server instance
a2a_server_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup A2A server"""
    global a2a_server_instance
    
    print("🚀 Starting A2A Server for NL2SQL Orchestrator...")
    
    try:
        # Create HTTP client for A2A communications
        httpx_client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize A2A server
        a2a_server_instance = A2AServer(
            httpx_client=httpx_client,
            host="localhost", 
            port=8001  # Different port from main API
        )
        
        print("✅ A2A Server initialized successfully!")
        print(f"🔗 A2A Endpoint: http://localhost:8001/")
        print(f"📋 Agent Card: http://localhost:8001/agent-card")
        
        yield
        
    except Exception as e:
        print(f"❌ Failed to initialize A2A server: {str(e)}")
        raise
    finally:
        # Cleanup
        if httpx_client:
            await httpx_client.aclose()
        print("🔄 A2A Server shutdown complete")


# Create FastAPI app with A2A integration
app = FastAPI(
    title="NL2SQL A2A Server",
    description="Agent-to-Agent server for NL2SQL Multi-Agent Orchestrator",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with A2A server information"""
    if not a2a_server_instance:
        raise HTTPException(status_code=503, detail="A2A server not initialized")
    
    return {
        "message": "NL2SQL A2A Server",
        "status": "running",
        "agent": "NL2SQL Orchestrator Agent",
        "capabilities": ["streaming", "nl2sql", "multi-agent"],
        "endpoints": {
            "agent_card": "/agent-card",
            "health": "/health",
            "a2a_protocol": "/a2a/*"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not a2a_server_instance:
        raise HTTPException(status_code=503, detail="A2A server not initialized")
    
    health_status = a2a_server_instance.health_check()
    
    if health_status.get("a2a_server") != "healthy":
        raise HTTPException(status_code=503, detail="A2A server unhealthy")
    
    return health_status


@app.get("/agent-card")
async def get_agent_card():
    """Get the agent card for external agents"""
    if not a2a_server_instance:
        raise HTTPException(status_code=503, detail="A2A server not initialized")
    
    return a2a_server_instance._get_agent_card().dict()


# Mount the A2A Starlette application
if a2a_server_instance:
    app.mount("/a2a", a2a_server_instance.get_starlette_app())


if __name__ == "__main__":
    print("🎯 Starting NL2SQL A2A Server...")
    print("📖 A2A Protocol Documentation: https://a2aproject.github.io/A2A/")
    
    uvicorn.run(
        "start_a2a:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
