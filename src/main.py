"""
NL2SQL Agent - Natural Language to SQL Converter
Main application using Semantic Kernel with MCP Database Integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion

from plugins.mcp_database_plugin import MCPDatabasePlugin
from services.schema_service import SchemaService
from services.nl2sql_service import NL2SQLService


class NL2SQLAgent:
    """
    Main NL2SQL Agent class that orchestrates the conversion process
    """
    
    def __init__(self):
        # Load environment variables from same directory (override existing)
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        print(f"🔍 Loading .env from: {dotenv_path}")
        print(f"🔍 .env file exists: {os.path.exists(dotenv_path)}")
        result = load_dotenv(dotenv_path, override=True)
        print(f"🔍 load_dotenv result: {result}")
        
        # Test reading environment variables
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        print(f"🔍 AZURE_OPENAI_ENDPOINT: {endpoint[:50] if endpoint else 'None'}...")
        print(f"🔍 AZURE_OPENAI_DEPLOYMENT_NAME: {deployment}")
        
        # Initialize components
        self.kernel = None
        self.mcp_plugin = None
        self.schema_service = None
        self.nl2sql_service = None
        
    async def initialize(self):
        """
        Initialize all components of the NL2SQL Agent
        """
        try:
            # Initialize Semantic Kernel
            self.kernel = Kernel()
            
            # Setup AI service (OpenAI or Azure OpenAI)
            await self._setup_ai_service()
            
            # Initialize MCP Database Plugin
            mcp_server_url = os.getenv("MCP_SERVER_URL", "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/")
            self.mcp_plugin = MCPDatabasePlugin(mcp_server_url=mcp_server_url)
            
            # Add MCP plugin to kernel
            self.kernel.add_plugin(self.mcp_plugin, plugin_name="database")
            
            # Initialize Schema Service
            self.schema_service = SchemaService(self.mcp_plugin)
            
            # Initialize schema context
            print("Initializing database schema context...")
            schema_context = await self.schema_service.initialize_schema_context()
            print("✅ Schema context initialized successfully!")
            
            # Initialize NL2SQL Service
            self.nl2sql_service = NL2SQLService(self.kernel, self.schema_service, self.mcp_plugin)
            
            print("🚀 NL2SQL Agent initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing NL2SQL Agent: {str(e)}")
            raise
    
    async def _setup_ai_service(self):
        """
        Setup AI service (OpenAI or Azure OpenAI) for Semantic Kernel
        """
        # Try Azure OpenAI first
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if azure_endpoint and azure_api_key and azure_deployment:
            print("🔧 Setting up Azure OpenAI service...")
            print(f"   Endpoint: {azure_endpoint}")
            print(f"   Deployment: {azure_deployment}")
            print(f"   API Version: {azure_api_version}")
            
            ai_service = AzureChatCompletion(
                deployment_name=azure_deployment,
                endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                service_id="azure_openai"
            )
            self.kernel.add_service(ai_service)
            print("✅ Azure OpenAI service configured")
            return
        
        # Fallback to OpenAI only if no Azure configuration
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        if openai_api_key:
            print("🔧 Setting up OpenAI service...")
            print(f"   Model: {openai_model}")
            
            ai_service = OpenAIChatCompletion(
                ai_model_id=openai_model,
                api_key=openai_api_key,
                service_id="openai"
            )
            self.kernel.add_service(ai_service)
            print("✅ OpenAI service configured")
            return
        
        raise ValueError("No AI service configuration found. Please set up either Azure OpenAI or OpenAI credentials in .env file")
    
    async def ask_question(self, question: str, execute: bool = True, limit: int = 100) -> dict:
        """
        Process a natural language question
        
        Args:
            question: Natural language question about the data
            execute: Whether to execute the generated SQL query
            limit: Maximum number of rows to return
            
        Returns:
            Dictionary containing the results
        """
        if not self.nl2sql_service:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        print(f"\n🤔 Question: {question}")
        print("🔄 Converting to SQL...")
        
        # Convert question to SQL and optionally execute
        result = await self.nl2sql_service.convert_question_to_sql(question, execute, limit)
        
        return result
    
    async def get_database_info(self) -> str:
        """
        Get database information and connection status
        """
        if not self.mcp_plugin:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        return await self.mcp_plugin.database_info()
    
    async def get_schema_context(self) -> str:
        """
        Get the full database schema context
        """
        if not self.schema_service:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        return self.schema_service.get_full_schema_summary()
    
    async def close(self):
        """
        Close all connections and cleanup resources
        """
        if self.mcp_plugin:
            await self.mcp_plugin.close()
        print("🔐 NL2SQL Agent closed successfully")


async def main():
    """
    Main function to demonstrate the NL2SQL Agent
    """
    # Initialize the agent
    agent = NL2SQLAgent()
    await agent.initialize()
    
    try:
        # Example questions
        test_questions = [
            "Show me the top 10 customers by total revenue",
            "What are the best selling products by category?",
            "Show me sales data for the last month",
            "Which territories have the highest sales?",
            "List all customers in the Universidad segment"
        ]
        
        print("\n" + "="*60)
        print("🎯 NL2SQL AGENT DEMO")
        print("="*60)
        
        # Test with a sample question
        sample_question = "Show me the top 5 customers with their names and total revenue"
        
        result = await agent.ask_question(sample_question, execute=True, limit=10)
        
        if result["error"] is None:
            print(f"\n✅ SUCCESS!")
            print(f"📝 Generated SQL: {result['sql_query']}")
            if result["executed"] and result["results"]:
                print(f"\n📊 Results:\n{result['results']}")
                print(f"⏱️  Execution time: {result['execution_time']}s")
                print(f"� Rows returned: {result['row_count']}")
        else:
            print(f"\n❌ ERROR: {result['error']}")
        
        print("\n" + "="*60)
        print("🎮 INTERACTIVE MODE")
        print("Type your questions or 'quit' to exit")
        print("="*60)
        
        # Interactive mode
        while True:
            try:
                user_question = input("\n🤔 Your question: ").strip()
                
                if user_question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_question:
                    continue
                
                result = await agent.ask_question(user_question, execute=True, limit=20)
                
                if result["error"] is None:
                    print(f"\n📝 SQL Query:\n{result['sql_query']}")
                    print(f"\n📊 Results:\n{result['results']}")
                else:
                    print(f"\n❌ Error: {result['error']}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
        
    finally:
        await agent.close()


if __name__ == "__main__":
    print("🚀 Starting NL2SQL Agent...")
    asyncio.run(main())
