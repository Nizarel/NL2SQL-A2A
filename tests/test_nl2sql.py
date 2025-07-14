"""
Simple test script for the NL2SQL Agent
Tests basic functionality without requiring API keys
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plugins.mcp_database_plugin import MCPDatabasePlugin
from services.schema_service import SchemaService


async def test_mcp_plugin():
    """Test the MCP Database Plugin functionality"""
    print("🧪 Testing MCP Database Plugin...")
    
    try:
        # Initialize MCP plugin
        mcp_url = "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/"
        plugin = MCPDatabasePlugin(mcp_url)
        
        # Test listing tables
        print("\n1. Testing list_tables()...")
        tables = await plugin.list_tables()
        print(f"✅ Tables: {tables}")
        
        # Test describing a table
        print("\n2. Testing describe_table('cliente')...")
        schema = await plugin.describe_table("cliente")
        print(f"✅ Schema preview: {schema[:200]}...")
        
        # Test database info
        print("\n3. Testing database_info()...")
        db_info = await plugin.database_info()
        print(f"✅ Database info: {db_info}")
        
        # Test a simple query
        print("\n4. Testing a simple query...")
        simple_query = "SELECT TOP 5 customer_id, Nombre_cliente FROM dev.cliente WHERE Nombre_cliente IS NOT NULL"
        result = await plugin.read_data(simple_query, 5)
        print(f"✅ Query result preview: {result[:300]}...")
        
        print("\n🎉 MCP Plugin tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ MCP Plugin test failed: {str(e)}")
        return False


async def test_schema_service():
    """Test the Schema Service functionality"""
    print("\n🧪 Testing Schema Service...")
    
    try:
        # Initialize components
        mcp_url = "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/"
        plugin = MCPDatabasePlugin(mcp_url)
        schema_service = SchemaService(plugin)
        
        # Test schema initialization
        print("\n1. Testing schema context initialization...")
        context = await schema_service.initialize_schema_context()
        print(f"✅ Schema context keys: {list(context.keys())}")
        
        # Test schema summary
        print("\n2. Testing schema summary...")
        summary = schema_service.get_full_schema_summary()
        print(f"✅ Schema summary preview: {summary[:300]}...")
        
        print("\n🎉 Schema Service tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Schema Service test failed: {str(e)}")
        return False


async def test_fallback_sql_generation():
    """Test SQL generation without AI service (fallback mode)"""
    print("\n🧪 Testing Fallback SQL Generation...")
    
    try:
        # Import the NL2SQL service
        from services.nl2sql_service import NL2SQLService
        
        # Create mock components
        from semantic_kernel import Kernel
        kernel = Kernel()  # Empty kernel for testing
        
        mcp_url = "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/"
        plugin = MCPDatabasePlugin(mcp_url)
        schema_service = SchemaService(plugin)
        await schema_service.initialize_schema_context()
        
        nl2sql_service = NL2SQLService(kernel, schema_service, plugin)
        
        # Test fallback generation
        print("\n1. Testing fallback SQL generation...")
        questions = [
            "Show me top 10 customers",
            "Show me sales data",
            "Show me product information"
        ]
        
        for question in questions:
            sql = await nl2sql_service._fallback_sql_generation(question)
            print(f"✅ Question: '{question}'")
            print(f"   SQL: {sql.strip().replace(chr(10), ' ')[:100]}...")
        
        print("\n🎉 Fallback SQL generation tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Fallback SQL generation test failed: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting NL2SQL Agent Tests\n")
    
    # Test results
    results = []
    
    # Test MCP Plugin
    results.append(await test_mcp_plugin())
    
    # Test Schema Service
    results.append(await test_schema_service())
    
    # Test Fallback SQL Generation
    results.append(await test_fallback_sql_generation())
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    test_names = ["MCP Plugin", "Schema Service", "Fallback SQL Generation"]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'🎉 ALL TESTS PASSED' if all_passed else '⚠️  SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎯 Your NL2SQL Agent is ready to use!")
        print("💡 To use with AI services, set environment variables:")
        print("   - OPENAI_API_KEY (for OpenAI)")
        print("   - Or AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME (for Azure OpenAI)")


if __name__ == "__main__":
    asyncio.run(main())
