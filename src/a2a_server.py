import logging
import httpx
from typing import Optional

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import BasePushNotificationSender, InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from a2a_executors import OrchestratorAgentExecutor

logger = logging.getLogger(__name__)


class A2AServer:
    """A2A Server wrapper for Natural Language to SQL Orchestrator Agent"""
    
    def __init__(self, httpx_client: httpx.AsyncClient, host: str = "localhost", port: int = 8000):
        self.httpx_client = httpx_client
        self.host = host
        self.port = port
        self._setup_server()
    
    def _setup_server(self):
        """Setup the A2A server with the Orchestrator Agent"""
        try:
            # Setup A2A components for a2a-sdk 0.2.12
            task_store = InMemoryTaskStore()
            config_store = InMemoryPushNotificationConfigStore()
            push_sender = BasePushNotificationSender(self.httpx_client, config_store)
            
            # Create agent executor with validation
            try:
                agent_executor = OrchestratorAgentExecutor()
            except Exception as e:
                logger.error(f"Failed to create OrchestratorAgentExecutor: {str(e)}")
                raise
            
            request_handler = DefaultRequestHandler(
                agent_executor=agent_executor,
                task_store=task_store,
                queue_manager=None,  # Defaults to InMemoryQueueManager
                push_config_store=config_store,
                push_sender=push_sender,
            )

            # Create A2A Starlette application
            self.a2a_app = A2AStarletteApplication(
                agent_card=self._get_agent_card(),
                http_handler=request_handler
            )
            
            logger.info(f"✅ A2A server configured successfully for {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup A2A server: {str(e)}")
            raise
    
    def _get_agent_card(self) -> AgentCard:
        """Returns the Agent Card for the Orchestrator Agent."""
        try:
            # Enhanced capabilities for NL2SQL workflow
            capabilities = AgentCapabilities(
                streaming=True,
                pushNotifications=True
            )

            skill_orchestration = AgentSkill(
                id='nl2sql_orchestration',
                name='NL2SQL Multi-Agent Orchestration',
                description=(
                    'Agent responsible for orchestrating the sequential multi-agent workflow: '
                    '1. SQLGeneratorAgent: Converts natural language to SQL, '
                    '2. SQLExecutorAgent: Executes the generated SQL query, '
                    '3. SummarizingAgent: Analyzes results and generates insights'
                ),
                tags=['nl2sql', 'multi-agent', 'orchestration', 'sql', 'semantic-kernel'],
                examples=[
                    'Analyze revenue by region and show which region performs best in 2025?',
                    'Show the top performing distribution centers (CEDIs) by total sales in 2025',
                    "Generate a query to find customers who haven't made purchases in the last 6 months?",
                    'Which products have declining sales trends and in which regions in May 2025?',
                    'What are the top 5 products by sales in the last quarter?',
                ],
            )

            agent_card = AgentCard(
                name='NL2SQL Orchestrator Agent',
                description=(
                    'The OrchestratorAgent coordinates the multi-agent NL2SQL workflow by sequentially invoking the SQL generation, '
                    'execution, and summarization agents to process natural language queries into SQL, execute them, and generate business insights. '
                    'It supports streaming responses and provides comprehensive data analysis capabilities.'
                ),
                url=f'http://{self.host}:{self.port}/',
                version='1.0.0',
                defaultInputModes=['text'],
                defaultOutputModes=['text'],
                capabilities=capabilities,
                skills=[skill_orchestration],
            )

            logger.info(f"✅ Agent card created successfully: {agent_card.name}")
            return agent_card
            
        except Exception as e:
            logger.error(f"❌ Failed to create agent card: {str(e)}")
            raise
    
    def get_starlette_app(self):
        """Get the Starlette app for mounting in FastAPI"""
        try:
            if not self.a2a_app:
                raise RuntimeError("A2A application not initialized")
            return self.a2a_app.build()
        except Exception as e:
            logger.error(f"❌ Failed to build Starlette app: {str(e)}")
            raise
    
    def health_check(self) -> dict:
        """Perform health check on A2A server components"""
        try:
            health_status = {
                "a2a_server": "healthy",
                "starlette_app": "ready" if self.a2a_app else "not_initialized",
                "host": self.host,
                "port": self.port,
                "capabilities": {
                    "streaming": True,
                    "push_notifications": True,
                    "task_cancellation": False
                }
            }
            logger.info("✅ A2A health check passed")
            return health_status
        except Exception as e:
            logger.error(f"❌ A2A health check failed: {str(e)}")
            return {"a2a_server": "unhealthy", "error": str(e)}