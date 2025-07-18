"""
NL2SQL Multi-Agent System - Natural Language to SQL Converter
Main application using Semantic Kernel 1.34.0 with specialized agents
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion

from config import get_settings
from plugins.mcp_database_plugin import MCPDatabasePlugin
from services.schema_service import SchemaService
from services.query_cache import QueryCache
from agents import SQLGeneratorAgent, ExecutorAgent, SummarizingAgent, OrchestratorAgent


class NL2SQLMultiAgentSystem:
    """
    Multi-Agent NL2SQL System that orchestrates specialized agents
    """
    
    def __init__(self):
        # Use centralized settings
        self.settings = get_settings()
        
        # Initialize components
        self.kernel = None
        self.mcp_plugin = None
        self.schema_service = None
        self.query_cache = None
        
        # Initialize specialized agents
        self.sql_generator_agent = None
        self.executor_agent = None
        self.summarizing_agent = None
        self.orchestrator_agent = None
        
    async def initialize(self):
        """
        Initialize all components of the Multi-Agent NL2SQL System
        Optimized with parallel initialization where possible
        """
        try:
            # Initialize Semantic Kernel
            self.kernel = Kernel()
            
            # Setup AI service (OpenAI or Azure OpenAI)
            await self._setup_ai_service()
            
            # Initialize MCP Database Plugin
            self.mcp_plugin = MCPDatabasePlugin(mcp_server_url=self.settings.mcp_server_url)
            
            # Add MCP plugin to kernel
            self.kernel.add_plugin(self.mcp_plugin, plugin_name="database")
            
            # Initialize services in parallel with agent creation
            print("🔄 Initializing services and agents in parallel...")
            
            # Initialize Schema Service and Query Cache
            self.schema_service = SchemaService(self.mcp_plugin)
            self.query_cache = QueryCache()
            
            # Start schema context initialization (non-blocking)
            schema_task = self.schema_service.initialize_schema_context()
            
            # Initialize agents (they don't need schema to be fully loaded)
            print("🤖 Initializing specialized agents...")
            
            self.sql_generator_agent = SQLGeneratorAgent(self.kernel, self.schema_service)
            print("✅ SQL Generator Agent initialized")
            
            self.executor_agent = ExecutorAgent(self.kernel, self.mcp_plugin)
            print("✅ Executor Agent initialized")
            
            self.summarizing_agent = SummarizingAgent(self.kernel)
            print("✅ Summarizing Agent initialized")
            
            # Wait for schema initialization to complete
            print("⏳ Waiting for schema context initialization...")
            await schema_task
            print("✅ Schema context initialized successfully!")
            
            # Initialize orchestrator with query cache
            self.orchestrator_agent = OrchestratorAgent(
                self.kernel, 
                self.sql_generator_agent,
                self.executor_agent, 
                self.summarizing_agent,
                self.query_cache
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
        settings = self.settings
        
        # Try Azure OpenAI first
        if settings.azure_openai_endpoint and settings.azure_openai_api_key and settings.azure_openai_deployment_name:
            print("🔧 Setting up Azure OpenAI service...")
            print(f"   Endpoint: {settings.azure_openai_endpoint}")
            print(f"   Deployment: {settings.azure_openai_deployment_name}")
            print(f"   API Version: {settings.azure_openai_api_version}")
            
            ai_service = AzureChatCompletion(
                deployment_name=settings.azure_openai_deployment_name,
                endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                service_id="azure_openai"
            )
            self.kernel.add_service(ai_service)
            print("✅ Azure OpenAI service configured")
            return
        
        # Fallback to OpenAI
        if settings.openai_api_key:
            print("🔧 Setting up OpenAI service...")
            print(f"   Model: {settings.openai_model}")
            
            ai_service = OpenAIChatCompletion(
                ai_model_id=settings.openai_model,
                api_key=settings.openai_api_key,
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


async def main():
    """
    Production-ready main function for the Multi-Agent NL2SQL System
    """
    print("🚀 Initializing Multi-Agent NL2SQL System...")
    
    # Initialize the multi-agent system
    agent = NL2SQLMultiAgentSystem()
    
    try:
        await agent.initialize()
        print("✅ System ready for processing queries")
        
        # Example usage (can be removed in production)
        # result = await agent.ask_question("Your question here")
        # print(result)
        
        return agent
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Entry point for production deployment
    """
    asyncio.run(main())
