#!/usr/bin/env python3
"""
Test script to diagnose import and execution issues
"""

import sys
import os

print("🔍 Starting diagnostic test...")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
print("✓ Python path configured")

try:
    from dotenv import load_dotenv
    print("✓ dotenv imported")
    
    load_dotenv()
    print("✓ Environment loaded")
    
    # Test environment variables
    mcp_url = os.getenv('MCP_SERVER_URL')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    print(f"✓ MCP URL: {mcp_url[:50]}..." if mcp_url else "✗ MCP URL not found")
    print(f"✓ Azure endpoint: {azure_endpoint[:50]}..." if azure_endpoint else "✗ Azure endpoint not found")
    
except Exception as e:
    print(f"✗ Environment setup failed: {e}")
    sys.exit(1)

try:
    from semantic_kernel import Kernel
    print("✓ Semantic Kernel imported")
except Exception as e:
    print(f"✗ Semantic Kernel import failed: {e}")
    sys.exit(1)

try:
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
    print("✓ OpenAI connectors imported")
except Exception as e:
    print(f"✗ OpenAI connectors import failed: {e}")
    sys.exit(1)

try:
    from plugins.mcp_database_plugin import MCPDatabasePlugin
    print("✓ MCP Database Plugin imported")
except Exception as e:
    print(f"✗ MCP Database Plugin import failed: {e}")
    sys.exit(1)

try:
    from services.schema_service import SchemaService
    print("✓ Schema Service imported")
except Exception as e:
    print(f"✗ Schema Service import failed: {e}")
    sys.exit(1)

try:
    from services.nl2sql_service import NL2SQLService
    print("✓ NL2SQL Service imported")
except Exception as e:
    print(f"✗ NL2SQL Service import failed: {e}")
    sys.exit(1)

print("\n🎉 All imports successful! The NL2SQL system should work.")
print("🚀 Testing basic initialization...")

try:
    # Test basic kernel creation
    kernel = Kernel()
    print("✓ Kernel created")
    
    # Test MCP plugin initialization
    mcp_url = os.getenv('MCP_SERVER_URL')
    if mcp_url:
        mcp_plugin = MCPDatabasePlugin(mcp_url)
        print("✓ MCP Plugin initialized")
    else:
        print("✗ Cannot test MCP Plugin - no URL in environment")
    
    print("\n✅ Basic initialization test passed!")
    
except Exception as e:
    print(f"✗ Initialization test failed: {e}")
    import traceback
    traceback.print_exc()
