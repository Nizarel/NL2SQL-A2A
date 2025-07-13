"""
Test script for NL2SQL Agent
Simple validation of MCP connection and basic functionality
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plugins.mcp_database_plugin import MCPDatabasePlugin


async def test_mcp_connection():
    """Test MCP database connection and basic functionality"""
    
    print("🔧 Testing MCP Database Connection...")
    
    # Initialize MCP plugin
    mcp_server_url = "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/"
    plugin = MCPDatabasePlugin(mcp_server_url=mcp_server_url)
    
    try:
        # Test database info
        print("\n📊 Getting database info...")
        db_info = await plugin.database_info()
        print(f"Database Info: {db_info[:200]}...")
        
        # Test list tables
        print("\n📋 Listing tables...")
        tables = await plugin.list_tables()
        print(f"Tables: {tables}")
        
        # Test describe table
        print("\n🔍 Describing 'cliente' table...")
        schema = await plugin.describe_table("cliente")
        print(f"Schema: {schema[:300]}...")
        
        # Test simple query
        print("\n🔄 Testing simple query...")
        query_result = await plugin.read_data("SELECT TOP 3 customer_id, Nombre_cliente FROM dev.cliente WHERE Nombre_cliente IS NOT NULL", 3)
        print(f"Query Results: {query_result[:300]}...")
        
        print("\n✅ All MCP tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ MCP test failed: {str(e)}")
        return False
    
    finally:
        await plugin.close()
    
    return True


async def main():
    """Main test function"""
    print("🚀 Starting NL2SQL Agent Tests...")
    print("="*50)
    
    # Test MCP connection
    mcp_success = await test_mcp_connection()
    
    print("\n" + "="*50)
    if mcp_success:
        print("🎉 All tests completed successfully!")
        print("\n💡 Next steps:")
        print("1. Configure your AI service (OpenAI/Azure OpenAI) in .env file")
        print("2. Run the main application: python src/main.py")
        print("3. Start asking natural language questions!")
    else:
        print("❌ Some tests failed. Please check the configuration.")


if __name__ == "__main__":
    asyncio.run(main())
