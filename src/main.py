"""
NL2SQL Multi-Agent System - Natural Language to SQL Converter
Main application using Semantic Kernel 1.34.0 with specialized agents
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
from agents import SQLGeneratorAgent, ExecutorAgent, SummarizingAgent, OrchestratorAgent


class NL2SQLMultiAgentSystem:
    """
    Multi-Agent NL2SQL System that orchestrates specialized agents
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
        
        # Initialize specialized agents
        self.sql_generator_agent = None
        self.executor_agent = None
        self.summarizing_agent = None
        self.orchestrator_agent = None
        
    async def initialize(self):
        """
        Initialize all components of the Multi-Agent NL2SQL System
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
            
            # Initialize specialized agents
            print("🤖 Initializing specialized agents...")
            
            # SQL Generator Agent
            self.sql_generator_agent = SQLGeneratorAgent(self.kernel, self.schema_service)
            print("✅ SQL Generator Agent initialized")
            
            # Executor Agent  
            self.executor_agent = ExecutorAgent(self.kernel, self.mcp_plugin)
            print("✅ Executor Agent initialized")
            
            # Summarizing Agent
            self.summarizing_agent = SummarizingAgent(self.kernel)
            print("✅ Summarizing Agent initialized")
            
            # Orchestrator Agent
            self.orchestrator_agent = OrchestratorAgent(
                self.kernel, 
                self.sql_generator_agent,
                self.executor_agent, 
                self.summarizing_agent
            )
            print("✅ Orchestrator Agent initialized")
            
            print("🚀 Multi-Agent NL2SQL System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing Multi-Agent NL2SQL System: {str(e)}")
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
    
    async def ask_question(self, question: str, execute: bool = True, limit: int = 100, 
                          include_summary: bool = True, context: str = "") -> dict:
        """
        Process a natural language question using the multi-agent system
        
        Args:
            question: Natural language question about the data
            execute: Whether to execute the generated SQL query
            limit: Maximum number of rows to return
            include_summary: Whether to generate AI summary and insights
            context: Optional additional context for the question
            
        Returns:
            Dictionary containing the complete workflow results
        """
        if not self.orchestrator_agent:
            raise ValueError("Multi-Agent System not initialized. Call initialize() first.")
        
        print(f"\n🤔 Question: {question}")
        print("🔄 Starting multi-agent workflow...")
        
        # Use orchestrator to coordinate the workflow
        result = await self.orchestrator_agent.process({
            "question": question,
            "context": context,
            "execute": execute,
            "limit": limit,
            "include_summary": include_summary
        })
        
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
    
    async def get_workflow_status(self) -> dict:
        """
        Get the status of all agents in the system
        """
        if not self.orchestrator_agent:
            raise ValueError("Multi-Agent System not initialized. Call initialize() first.")
        
        return await self.orchestrator_agent.get_workflow_status()
    
    async def execute_single_step(self, step: str, input_data: dict) -> dict:
        """
        Execute a single workflow step for testing or debugging
        
        Args:
            step: Step name ('sql_generation', 'execution', 'summarization')
            input_data: Input data for the specific step
        """
        if not self.orchestrator_agent:
            raise ValueError("Multi-Agent System not initialized. Call initialize() first.")
        
        return await self.orchestrator_agent.execute_single_step(step, input_data)
    
    async def close(self):
        """
        Close all connections and cleanup resources
        """
        if self.mcp_plugin:
            await self.mcp_plugin.close()
        print("🔐 Multi-Agent NL2SQL System closed successfully")


async def main():
    """
    Main function to demonstrate the Multi-Agent NL2SQL System
    """
    # Initialize the multi-agent system
    agent = NL2SQLMultiAgentSystem()
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
        print("🤖 MULTI-AGENT NL2SQL SYSTEM DEMO")
        print("="*60)
        
        # Test with a sample question
        sample_question = "Show me the top 5 customers with their names and total revenue"
        
        result = await agent.ask_question(
            question=sample_question, 
            execute=True, 
            limit=10,
            include_summary=True
        )
        
        # Display results based on multi-agent workflow
        if result["success"]:
            print(f"\n✅ WORKFLOW COMPLETED SUCCESSFULLY!")
            
            data = result["data"]
            metadata = result["metadata"]
            
            print(f"📝 Generated SQL: {data['sql_query']}")
            
            if data.get("executed"):
                print(f"\n📊 Query Results:\n{data['results']}")
                print(f"⏱️  Execution time: {metadata.get('execution_time', 'N/A')}s")
                print(f"📈 Rows returned: {metadata.get('row_count', 'N/A')}")
                
                # Display AI-generated summary if available
                if data.get("summary"):
                    summary = data["summary"]
                    print(f"\n🧠 AI INSIGHTS & SUMMARY:")
                    print(f"Executive Summary: {summary['executive_summary']}")
                    print(f"\nKey Insights:")
                    for i, insight in enumerate(summary.get('key_insights', []), 1):
                        print(f"  {i}. {insight.get('finding', 'N/A')}")
                    
                    print(f"\nRecommendations:")
                    for i, rec in enumerate(summary.get('recommendations', []), 1):
                        print(f"  {i}. {rec.get('action', 'N/A')} (Priority: {rec.get('priority', 'N/A')})")
            else:
                if data.get("execution_error"):
                    print(f"\n❌ Execution Error: {data['execution_error']}")
        else:
            print(f"\n❌ WORKFLOW FAILED: {result['error']}")
        
        print("\n" + "="*60)
        print("🎮 INTERACTIVE MODE")
        print("Type your questions, 'status' for system status, or 'quit' to exit")
        print("="*60)
        
        # Interactive mode
        while True:
            try:
                user_input = input("\n🤔 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'status':
                    status_result = await agent.get_workflow_status()
                    if status_result["success"]:
                        status = status_result["data"]
                        print(f"\n📊 System Status:")
                        print(f"  Orchestrator: {status['orchestrator']}")
                        print(f"  Agents: {status['agents']}")
                        print(f"  Capabilities: {status['workflow_capabilities']}")
                    continue
                
                if not user_input:
                    continue
                
                result = await agent.ask_question(
                    question=user_input, 
                    execute=True, 
                    limit=20,
                    include_summary=True
                )
                
                if result["success"]:
                    data = result["data"]
                    print(f"\n📝 SQL Query:\n{data['sql_query']}")
                    
                    if data.get("executed"):
                        print(f"\n📊 Results:\n{data['results']}")
                        
                        # Show summary if available
                        if data.get("summary"):
                            summary = data["summary"]
                            print(f"\n🧠 Executive Summary: {summary['executive_summary']}")
                    else:
                        if data.get("execution_error"):
                            print(f"\n❌ Execution Error: {data['execution_error']}")
                else:
                    print(f"\n❌ Error: {result['error']}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
        
    finally:
        await agent.close()


if __name__ == "__main__":
    print("🚀 Starting Multi-Agent NL2SQL System...")
    asyncio.run(main())
