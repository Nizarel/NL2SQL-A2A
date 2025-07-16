"""
Pure A2A Server for NL2SQL Orchestrator Agent
Following the reference pattern from a2a-sdk samples
"""

import logging

import click
import httpx

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import BasePushNotificationSender, InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# Import with fallback for different execution contexts
try:
    from a2a_executors.orchestrator_executor import OrchestratorAgentExecutor
except ImportError:
    from .a2a_executors.orchestrator_executor import OrchestratorAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', default='localhost', help='Host to bind the server to')
@click.option('--port', default=8002, help='Port to bind the server to')
def main(host, port):
    """Starts the NL2SQL Orchestrator Agent server using A2A protocol."""
    
    print(f"🚀 Starting NL2SQL A2A Server on {host}:{port}")
    print("📖 A2A Protocol Documentation: https://a2aproject.github.io/A2A/")
    print("🔄 Initializing NL2SQL Multi-Agent System...")
    
    # Initialize the orchestrator agent properly
    import asyncio
    
    async def create_initialized_executor():
        """Create and initialize the orchestrator executor with the full NL2SQL system"""
        try:
            # Import the main system
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            print("🔄 Importing NL2SQL system...")
            from main import NL2SQLMultiAgentSystem
            
            # Create and initialize the NL2SQL system
            print("🔄 Creating NL2SQL Multi-Agent System...")
            nl2sql_system = await NL2SQLMultiAgentSystem.create_and_initialize()
            print("✅ NL2SQL Multi-Agent System initialized!")
            
            # Verify orchestrator agent is available
            if not nl2sql_system.orchestrator_agent:
                raise Exception("Orchestrator agent is None after system initialization")
            
            # Create executor with the initialized orchestrator
            executor = OrchestratorAgentExecutor(nl2sql_system.orchestrator_agent)
            print("✅ OrchestratorAgentExecutor initialized with active orchestrator!")
            
            return executor
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to initialize NL2SQL system: {str(e)}")
            print(f"❌ Detailed error: {traceback.format_exc()}")
            print("⚠️ Creating executor without orchestrator (placeholder mode)")
            return OrchestratorAgentExecutor()
    
    # Get the initialized executor
    executor = asyncio.run(create_initialized_executor())
    
    # Create HTTP client for A2A communications
    httpx_client = httpx.AsyncClient()
    
    # Setup push notification components for a2a-sdk 0.2.12
    config_store = InMemoryPushNotificationConfigStore()
    push_sender = BasePushNotificationSender(httpx_client, config_store)
    
    # Create request handler following a2a-sdk 0.2.12 pattern
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        queue_manager=None,  # Defaults to InMemoryQueueManager
        push_config_store=config_store,
        push_sender=push_sender,
    )

    # Create agent card
    agent_card = get_agent_card(host, port)
    logger.info(f"✅ Agent card created: {agent_card.name}")

    # Create A2A Starlette application
    server = A2AStarletteApplication(
        agent_card=agent_card, 
        http_handler=request_handler
    )
    
    # Start the server
    import uvicorn
    print(f"✅ NL2SQL A2A Server ready!")
    print(f"🔗 A2A Endpoint: http://{host}:{port}/")
    print(f"📋 Agent Card: http://{host}:{port}/agent-card")
    print(f"🔍 Agent Card JSON: {agent_card.model_dump_json(indent=2)}")
    
    uvicorn.run(server.build(), host=host, port=port)


def get_agent_card(host: str, port: int) -> AgentCard:
    """Returns the Agent Card for the NL2SQL Orchestrator Agent."""
    
    # Build agent capabilities
    capabilities = AgentCapabilities(
        streaming=True,
        pushNotifications=True
    )
    
    # Define NL2SQL orchestration skill
    skill_nl2sql = AgentSkill(
        id='nl2sql_orchestration',
        name='NL2SQL Multi-Agent Orchestration',
        description=(
            'Orchestrates a sequential multi-agent workflow to process natural language queries: '
            '1. SQLGeneratorAgent converts natural language to SQL queries, '
            '2. SQLExecutorAgent executes the generated SQL against the database, '
            '3. SummarizingAgent analyzes results and generates business insights. '
            'Supports streaming responses and comprehensive data analysis.'
        ),
        tags=['nl2sql', 'multi-agent', 'orchestration', 'sql', 'semantic-kernel', 'streaming'],
        examples=[
            'Analyze revenue by region and show which region performs best in 2025?',
            'Show the top performing distribution centers (CEDIs) by total sales in 2025',
            "Generate a query to find customers who haven't made purchases in the last 6 months?",
            'Which products have declining sales trends and in which regions in May 2025?',
            'What are the top 5 products by sales in the last quarter?',
            'Compare performance metrics between different business units',
        ],
    )

    # Create agent card
    agent_card = AgentCard(
        name='NL2SQL Orchestrator Agent',
        description=(
            'Advanced NL2SQL Orchestrator Agent powered by Semantic Kernel that coordinates '
            'multiple specialized agents to transform natural language questions into SQL queries, '
            'execute them safely, and generate comprehensive business insights with streaming support.'
        ),
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=capabilities,
        skills=[skill_nl2sql],
    )

    logger.info(f"✅ Agent card created: {agent_card.name}")
    return agent_card


if __name__ == '__main__':
    main()
