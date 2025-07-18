#!/usr/bin/env python3
"""
Simple startup script for the NL2SQL API Server
"""

import os
import sys
import subprocess

def main():
    """Start the NL2SQL API server"""
    
    # Set working directory to src
    src_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_dir)
    
    # Configuration from environment or defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = os.getenv("API_PORT", "8000")
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    print("🚀 Starting NL2SQL Multi-Agent API Server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    
    # Detect environment and provide appropriate URLs
    if os.getenv('CODESPACES'):
        codespace_name = os.getenv('CODESPACE_NAME', 'unknown')
        github_user = os.getenv('GITHUB_USER', 'user')
        base_url = f"https://{codespace_name}-{port}.app.github.dev"
        print(f"🌐 GitHub Codespaces Environment Detected")
        print(f"📚 Swagger Documentation: {base_url}/docs")
        print(f"📋 ReDoc Documentation: {base_url}/redoc")
        print(f"🔍 Health Check: {base_url}/health")
        print(f"🎯 Main Endpoint: {base_url}/orchestrator/query")
    elif os.getenv('VSCODE_REMOTE'):
        print(f"🔗 VS Code Remote Environment")
        print(f"📚 Documentation: http://localhost:{port}/docs (check VS Code PORTS tab)")
        print(f"🔍 Health Check: http://localhost:{port}/health")
        print(f"🎯 Main Endpoint: http://localhost:{port}/orchestrator/query")
    else:
        print(f"💻 Local Development Environment")
        print(f"📚 Documentation: http://{host}:{port}/docs")
        print(f"🔍 Health Check: http://{host}:{port}/health")
        print(f"🎯 Main Endpoint: http://{host}:{port}/orchestrator/query")
    print("")
    
    # Start the server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api_server:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "info"
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
