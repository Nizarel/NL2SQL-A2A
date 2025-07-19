"""
NL2SQL Multi-Agent System - Natural Language to SQL Converter
Main application using Semantic Kernel 1.34.0 with specialized agents
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion, AzureTextEmbedding

from plugins.mcp_database_plugin import MCPDatabasePlugin
from services.schema_service import SchemaService
from agents import SQLGeneratorAgent, ExecutorAgent, SummarizingAgent, OrchestratorAgent
from agents.schema_analyst_agent import SchemaAnalystAgent


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
        self.schema_analyst_agent = None
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
            
            # Initialize Enhanced MCP Database Plugin with Connection Pooling
            mcp_server_url = os.getenv("MCP_SERVER_URL")
            if not mcp_server_url:
                raise ValueError("MCP_SERVER_URL environment variable is required")
            
            # Create enhanced plugin with connection pooling
            self.mcp_plugin = MCPDatabasePlugin(
                mcp_server_url=mcp_server_url, 
                enable_pooling=True
            )
            
            # Initialize connection pool
            print("🔗 Initializing MCP connection pool...")
            await self.mcp_plugin.initialize()
            print("✅ MCP connection pool initialized successfully!")
            
            # Add MCP plugin to kernel
            self.kernel.add_plugin(self.mcp_plugin, plugin_name="database")
            
            # Initialize Schema Service
            self.schema_service = SchemaService(self.mcp_plugin)
            
            # Initialize schema context
            print("Initializing database schema context...")
            schema_context = await self.schema_service.initialize_schema_context()
            print("✅ Schema context initialized successfully!")
            
            # Initialize specialized agents in parallel for better performance
            print("🤖 Initializing specialized agents in parallel...")
            
            # Define agent initialization functions
            def init_sql_generator():
                """Initialize SQL Generator Agent"""
                agent = SQLGeneratorAgent(self.kernel)
                print("✅ SQL Generator Agent initialized")
                return agent
                
            def init_executor():
                """Initialize Executor Agent"""
                agent = ExecutorAgent(self.kernel, self.mcp_plugin)
                print("✅ Executor Agent initialized")
                return agent
                
            def init_summarizer():
                """Initialize Summarizing Agent"""
                agent = SummarizingAgent(self.kernel)
                print("✅ Summarizing Agent initialized")
                return agent
                
            def init_schema_analyst():
                """Initialize Schema Analyst Agent"""
                agent = SchemaAnalystAgent(self.kernel, self.schema_service)
                print("✅ Schema Analyst Agent initialized")
                return agent
            
            # Run agent initializations in parallel using ThreadPoolExecutor
            # Since all agent constructors are synchronous, we use ThreadPoolExecutor
            max_workers = int(os.getenv("AGENT_INIT_MAX_WORKERS", "4"))
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AgentInit") as executor:
                loop = asyncio.get_event_loop()
                
                # Submit all agent initializations concurrently
                print("🚀 Starting parallel agent initialization...")
                futures = [
                    loop.run_in_executor(executor, init_sql_generator),
                    loop.run_in_executor(executor, init_executor),
                    loop.run_in_executor(executor, init_summarizer),
                    loop.run_in_executor(executor, init_schema_analyst)
                ]
                
                # Wait for all agents to initialize concurrently
                try:
                    results = await asyncio.gather(*futures)
                    
                    # Assign results in the order they were submitted
                    self.sql_generator_agent = results[0]
                    self.executor_agent = results[1]
                    self.summarizing_agent = results[2]
                    self.schema_analyst_agent = results[3]
                    
                    print("✅ All specialized agents initialized successfully in parallel!")
                    
                except Exception as init_error:
                    print(f"❌ Parallel agent initialization failed: {str(init_error)}")
                    # Fallback to sequential initialization
                    print("⚠️ Falling back to sequential agent initialization...")
                    
                    self.sql_generator_agent = SQLGeneratorAgent(self.kernel)
                    print("✅ SQL Generator Agent initialized (fallback)")
                    
                    self.executor_agent = ExecutorAgent(self.kernel, self.mcp_plugin)
                    print("✅ Executor Agent initialized (fallback)")
                    
                    self.summarizing_agent = SummarizingAgent(self.kernel)
                    print("✅ Summarizing Agent initialized (fallback)")
                    
                    self.schema_analyst_agent = SchemaAnalystAgent(self.kernel, self.schema_service)
                    print("✅ Schema Analyst Agent initialized (fallback)")
            
            # Initialize Orchestrator Agent (must be last as it depends on all others)
            print("🎯 Initializing Orchestrator Agent...")
            self.orchestrator_agent = OrchestratorAgent(
                self.kernel,
                self.schema_analyst_agent,  # NEW: Pass Schema Analyst first
                self.sql_generator_agent,
                self.executor_agent, 
                self.summarizing_agent
            )
            print("✅ Orchestrator Agent initialized with Schema Analyst integration")
            
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
            
            # Add chat completion service
            ai_service = AzureChatCompletion(
                deployment_name=azure_deployment,
                endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                service_id="azure_openai"
            )
            self.kernel.add_service(ai_service)
            
            # Add embedding service if configured
            azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            if azure_embedding_deployment:
                print(f"🔧 Setting up Azure OpenAI embedding service...")
                print(f"   Embedding Deployment: {azure_embedding_deployment}")
                
                embedding_service = AzureTextEmbedding(
                    deployment_name=azure_embedding_deployment,
                    endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                    service_id="azure_openai_embedding"
                )
                self.kernel.add_service(embedding_service)
                print("✅ Azure OpenAI embedding service configured")
            
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
    
    async def ask_question(self, question: str, execute: bool = True, 
                          limit: int = None, include_summary: bool = True, context: str = "") -> dict:
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
        
        # Set default limit from environment variable if not provided
        if limit is None:
            limit = int(os.getenv("DEFAULT_QUERY_LIMIT", "100"))
        
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

    @classmethod
    async def create_and_initialize(cls):
        """
        Factory method to create and initialize the system in one call
        Production-ready convenience method
        """
        system = cls()
        await system.initialize()
        return system

    async def process_query(self, question: str, **kwargs) -> dict:
        """
        Simplified interface for processing queries in production
        
        Args:
            question: Natural language question
            **kwargs: Optional parameters (execute, limit, include_summary, context)
        
        Returns:
            Processed result dictionary
        """
        return await self.ask_question(question, **kwargs)
    
    async def cleanup(self):
        """Cleanup resources and close connections"""
        try:
            if hasattr(self, 'mcp_plugin') and self.mcp_plugin:
                print("🔒 Cleaning up MCP connection pool...")
                await self.mcp_plugin.close()
            print("✅ Cleanup completed successfully")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
    
    def get_connection_pool_metrics(self):
        """Get connection pool performance metrics"""
        if hasattr(self, 'mcp_plugin') and self.mcp_plugin:
            return self.mcp_plugin.get_pool_metrics()
        return None
    
    def print_connection_pool_metrics(self):
        """Print connection pool performance metrics"""
        if hasattr(self, 'mcp_plugin') and self.mcp_plugin:
            self.mcp_plugin.print_pool_metrics()
        else:
            print("📊 Connection pool not available")


async def main():
    """
    Production-ready main function for the Multi-Agent NL2SQL System with enhanced connection pooling
    """
    print("🚀 Initializing Enhanced Multi-Agent NL2SQL System...")
    
    # Initialize the multi-agent system
    agent = NL2SQLMultiAgentSystem()
    
    try:
        await agent.initialize()
        print("✅ Enhanced system ready for processing queries")
        
        # Print initial connection pool metrics
        print("\n📊 Initial Connection Pool Status:")
        agent.print_connection_pool_metrics()
        
        # Example usage (can be removed in production)
        # result = await agent.ask_question("Your question here")
        # print(result)
        
        return agent
        
    except Exception as e:
        print(f"❌ Failed to initialize enhanced system: {str(e)}")
        # Ensure cleanup on failure
        if agent:
            await agent.cleanup()
        raise


if __name__ == "__main__":
    """
    Entry point for production deployment with enhanced connection pooling
    """
    import signal
    import sys
    
    agent_system = None
    
    async def shutdown_handler():
        """Graceful shutdown handler"""
        if agent_system:
            print("\n🔒 Gracefully shutting down enhanced system...")
            await agent_system.cleanup()
        print("👋 Enhanced system shutdown complete")
        sys.exit(0)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n⚠️ Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(shutdown_handler())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        agent_system = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        if agent_system:
            asyncio.run(agent_system.cleanup())
