#!/usr/bin/env python3
"""
Debug script to test schema service methods
"""
import sys
import os
import asyncio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plugins.mcp_database_plugin import MCPDatabasePlugin
from services.schema_service import SchemaService

async def test_schema_service():
    """Test the schema service independently"""
    print("🔍 Testing Schema Service...")
    
    # Initialize MCP Plugin
    mcp_server_url = "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/"
    mcp_plugin = MCPDatabasePlugin(mcp_server_url)
    
    # Initialize Schema Service
    schema_service = SchemaService(mcp_plugin)
    
    # Check available methods
    print(f"📋 Schema Service methods: {[method for method in dir(schema_service) if not method.startswith('_')]}")
    
    # Check if get_full_schema_summary exists
    if hasattr(schema_service, 'get_full_schema_summary'):
        print("✅ get_full_schema_summary method exists")
    else:
        print("❌ get_full_schema_summary method missing")
    
    if hasattr(schema_service, 'get_schema_context'):
        print("✅ get_schema_context method exists")
    else:
        print("❌ get_schema_context method missing")
    
    # Try to initialize schema context
    try:
        context = await schema_service.initialize_schema_context()
        print(f"✅ Schema context initialized: {len(context)} keys")
        
        # Try to get full schema summary
        summary = schema_service.get_full_schema_summary()
        print(f"✅ Schema summary generated: {len(summary)} characters")
        
    except Exception as e:
        print(f"❌ Error testing schema service: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_schema_service())
