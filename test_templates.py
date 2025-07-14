"""
Test the Jinja2 template integration with SQL Generator Agent
"""
import asyncio
import os
import sys

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

async def test_template_loading():
    """Test that templates are loaded correctly"""
    print("🧪 Testing Jinja2 Template Integration...")
    
    try:
        # Import required modules
        from semantic_kernel import Kernel
        from agents.sql_generator_agent import SQLGeneratorAgent
        from services.schema_service import SchemaService
        from plugins.mcp_database_plugin import MCPDatabasePlugin
        
        print("✅ Imports successful")
        
        # Create a mock MCP plugin for testing
        mcp_url = "https://test-url.com"  # This won't be used in template test
        mock_plugin = MCPDatabasePlugin(mcp_url)
        
        # Create schema service
        schema_service = SchemaService(mock_plugin)
        
        # Create kernel (minimal setup for template testing)
        kernel = Kernel()
        
        # Initialize SQL Generator Agent (this will test template loading)
        agent = SQLGeneratorAgent(kernel, schema_service)
        
        print("✅ SQL Generator Agent initialized successfully")
        print("✅ Jinja2 templates loaded successfully")
        
        # Verify template functions exist
        assert hasattr(agent, 'intent_analysis_function'), "Intent analysis function not created"
        assert hasattr(agent, 'sql_generation_function'), "SQL generation function not created"
        
        print("✅ Template functions created successfully")
        print("🎉 All template integration tests passed!")
        
    except Exception as e:
        print(f"❌ Template test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_template_loading())
