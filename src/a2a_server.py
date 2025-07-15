"""
FastA2A Agent-to-Agent Server for NL2SQL Multi-Agent System
Provides A2A protocol endpoints for orchestrator and all specialized agents
"""

import os
import sys
import asyncio
import time
import json
import uuid
import anyio
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import logging
from datetime import datetime

import uvicorn
from fasta2a import FastA2A, Skill, Storage, Broker, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import (
    AgentProvider, AgentCapabilities, Message, Task, 
    Artifact, TaskSendParams, TaskIdParams
)


class FixedInMemoryBroker(Broker):
    """Fixed version of InMemoryBroker with proper stream implementation"""
    
    def __init__(self):
        from contextlib import AsyncExitStack
        self.aexit_stack = None
        self._write_stream = None
        self._read_stream = None
        self._cancelled_tasks = set()
        print(f"🔧 FixedInMemoryBroker initialized")
    
    async def __aenter__(self):
        """Async context manager entry - creates the memory streams"""
        from contextlib import AsyncExitStack
        self.aexit_stack = AsyncExitStack()
        await self.aexit_stack.__aenter__()

        # Create memory object streams like the original FastA2A InMemoryBroker
        self._write_stream, self._read_stream = anyio.create_memory_object_stream()
        await self.aexit_stack.enter_async_context(self._read_stream)
        await self.aexit_stack.enter_async_context(self._write_stream)
        
        print(f"🔧 FixedInMemoryBroker: Memory streams created")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleans up streams"""
        if self.aexit_stack:
            await self.aexit_stack.__aexit__(exc_type, exc_val, exc_tb)
        print(f"🔧 FixedInMemoryBroker: Streams cleaned up")
        
    async def receive_task_operations(self):
        """Return an async iterator of task operations from the read stream"""
        print(f"🔧 FixedInMemoryBroker: Starting to receive task operations")
        async for task_operation in self._read_stream:
            print(f"🔧 FixedInMemoryBroker: Received task operation: {task_operation['operation']}")
            yield task_operation
                
    async def run_task(self, task_send_params):
        """Send a run task operation to the write stream"""
        from opentelemetry.trace import get_current_span
        operation = {
            'operation': 'run',
            'params': task_send_params,
            '_current_span': get_current_span()
        }
        print(f"🔧 FixedInMemoryBroker: Sending run task operation: {task_send_params.get('id', 'unknown')}")
        await self._write_stream.send(operation)
        
    async def cancel_task(self, task_id_params):
        """Send a cancel task operation to the write stream"""
        from opentelemetry.trace import get_current_span
        operation = {
            'operation': 'cancel',
            'params': task_id_params,
            '_current_span': get_current_span()
        }
        print(f"🔧 FixedInMemoryBroker: Sending cancel task operation: {task_id_params.get('id', 'unknown')}")
        await self._write_stream.send(operation)
        if 'id' in task_id_params:
            self._cancelled_tasks.add(task_id_params['id'])

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import NL2SQLMultiAgentSystem


# Environment configuration
A2A_HOST = os.getenv("A2A_HOST", "0.0.0.0")
A2A_BASE_PORT = int(os.getenv("A2A_BASE_PORT", "8100"))
A2A_DEBUG = os.getenv("A2A_DEBUG", "true").lower() == "true"
A2A_LOG_LEVEL = os.getenv("A2A_LOG_LEVEL", "info").lower()

# Configure logging
logging.basicConfig(
    level=getattr(logging, A2A_LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NL2SQLAgentStorage(InMemoryStorage[Dict[str, Any]]):
    """Custom storage for NL2SQL agents with conversation context"""
    
    def __init__(self):
        super().__init__()
        self.agent_conversations: Dict[str, List[Dict[str, Any]]] = {}


class NL2SQLWorker(Worker[Dict[str, Any]]):
    """Worker implementation for NL2SQL agents"""
    
    def __init__(self, broker: Broker, storage: Storage[Dict[str, Any]], 
                 nl2sql_system: NL2SQLMultiAgentSystem, agent_type: str):
        super().__init__(broker, storage)
        self.nl2sql_system = nl2sql_system
        self.agent_type = agent_type
    
    def build_artifacts(self, result: Any) -> List[Artifact]:
        """Build artifacts from agent result"""
        artifacts = []
        
        if isinstance(result, dict):
            # Add result as text artifact
            if result.get('success'):
                artifacts.append(Artifact(
                    id=str(uuid.uuid4()),
                    type='text',
                    name=f'{self.agent_type}_result',
                    content=json.dumps(result, indent=2)
                ))
            
            # Add specific artifacts based on agent type
            if self.agent_type == 'sql_generator' and 'data' in result:
                data = result['data']
                if 'sql_query' in data:
                    artifacts.append(Artifact(
                        id=str(uuid.uuid4()),
                        type='text',
                        name='generated_sql',
                        content=data['sql_query']
                    ))
            
            elif self.agent_type == 'executor' and 'data' in result:
                data = result['data']
                if 'raw_results' in data:
                    artifacts.append(Artifact(
                        id=str(uuid.uuid4()),
                        type='text',
                        name='query_results',
                        content=str(data['raw_results'])
                    ))
            
            elif self.agent_type == 'summarizer' and 'data' in result:
                data = result['data']
                if 'summary' in data:
                    artifacts.append(Artifact(
                        id=str(uuid.uuid4()),
                        type='text',
                        name='summary',
                        content=data['summary']
                    ))
        
        return artifacts
    
    def build_message_history(self, history: List[Message]) -> List[Any]:
        """Build message history for agent context"""
        return [
            {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", time.time())
            }
            for msg in history
        ]
    
    async def process_task(self, task_id: str, context_id: str, message: Message) -> Dict[str, Any]:
        """Process task using appropriate NL2SQL agent"""
        print(f"🔍 [{self.agent_type}] Processing task {task_id}, message: {message}")
        
        # Extract content from A2A message parts with improved error handling
        content = ""
        if 'parts' in message and message['parts']:
            for part in message['parts']:
                if part.get('kind') == 'text':
                    content += part.get('text', '')
        
        # Fallback to direct content if no parts found
        if not content and 'content' in message:
            content = str(message['content'])
        
        print(f"📝 [{self.agent_type}] Extracted content: '{content[:100]}...'")
        
        if not content:
            error_result = {
                "success": False, 
                "error": f"No content found in message for {self.agent_type}",
                "agent_type": self.agent_type
            }
            print(f"❌ [{self.agent_type}] No content found")
            return error_result
        
        try:
            print(f"🎯 [{self.agent_type}] Starting processing...")
            
            if self.agent_type == 'orchestrator':
                # Use content as question for orchestrator
                print(f"🤖 [{self.agent_type}] Calling orchestrator with question")
                result = await self.nl2sql_system.ask_question(content)
                
            elif self.agent_type == 'sql_generator':
                # Parse input for SQL generation
                try:
                    content_dict = json.loads(content)
                except json.JSONDecodeError:
                    content_dict = {"question": content}
                
                print(f"🤖 [{self.agent_type}] Calling SQL generator")
                result = await self.nl2sql_system.sql_generator_agent.process(content_dict)
                
            elif self.agent_type == 'executor':
                # Parse input for SQL execution
                try:
                    content_dict = json.loads(content)
                except json.JSONDecodeError:
                    content_dict = {"sql_query": content}
                
                print(f"🤖 [{self.agent_type}] Calling executor")
                result = await self.nl2sql_system.executor_agent.process(content_dict)
                
            elif self.agent_type == 'summarizer':
                # Parse input for summarization
                try:
                    content_dict = json.loads(content)
                except json.JSONDecodeError:
                    error_result = {
                        "success": False, 
                        "error": f"Summarizer requires structured JSON input, received: {content[:100]}...",
                        "agent_type": self.agent_type
                    }
                    print(f"❌ [{self.agent_type}] Invalid JSON input")
                    return error_result
                
                print(f"🤖 [{self.agent_type}] Calling summarizer")
                result = await self.nl2sql_system.summarizing_agent.process(content_dict)
                
            else:
                error_result = {"success": False, "error": f"Unknown agent type: {self.agent_type}"}
                print(f"❌ [{self.agent_type}] Unknown agent type")
                return error_result
            
            print(f"✅ [{self.agent_type}] Processing completed, success: {result.get('success', False)}")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Error processing {self.agent_type} task: {str(e)}",
                "agent_type": self.agent_type
            }
            print(f"❌ [{self.agent_type}] Exception during processing: {e}")
            import traceback
            traceback.print_exc()
            return error_result
    
    async def run_task(self, params: TaskSendParams) -> None:
        """Execute a task and update storage"""
        print(f"🚀 [{self.agent_type}] run_task called with params: {params}")
        
        task_id = params['id']
        context_id = params['context_id']
        message = params['message']
        
        print(f"🔧 [{self.agent_type}] Running task {task_id} with context {context_id}")
        print(f"📝 [{self.agent_type}] Message content: {message}")

        try:
            # Update task to working state
            print(f"🔄 [{self.agent_type}] Setting task {task_id} to working state")
            await self.storage.update_task(task_id, 'working')
            
            # Process the task
            print(f"⚙️ [{self.agent_type}] Processing task {task_id}")
            result = await self.process_task(task_id, context_id, message)
            print(f"✅ [{self.agent_type}] Task {task_id} processed, result: {result.get('success', False)}")
            
            # Build artifacts from result
            artifacts = self.build_artifacts(result)
            print(f"📄 [{self.agent_type}] Built {len(artifacts)} artifacts")
            
            # Create response message with correct schema
            response_message = Message(
                role='agent',
                kind='message',
                message_id=f"response-{int(time.time() * 1000)}",
                parts=[
                    {
                        'kind': 'text',
                        'text': json.dumps(result)
                    }
                ]
            )
            print(f"📨 [{self.agent_type}] Created response message")
            
            # Update task with completion
            if result.get('success', False):
                print(f"✅ [{self.agent_type}] Marking task {task_id} as completed")
                await self.storage.update_task(
                    task_id, 'completed', 
                    new_artifacts=artifacts,
                    new_messages=[response_message]
                )
            else:
                print(f"❌ [{self.agent_type}] Marking task {task_id} as failed")
                await self.storage.update_task(
                    task_id, 'failed',
                    new_artifacts=artifacts,
                    new_messages=[response_message]
                )
            
            print(f"📝 [{self.agent_type}] Task {task_id} completed and stored")
                
        except Exception as e:
            print(f"❌ [{self.agent_type}] Exception in task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Handle errors
            error_message = Message(
                role='agent',
                kind='message',
                message_id=f"error-{int(time.time() * 1000)}",
                parts=[
                    {
                        'kind': 'text',
                        'text': json.dumps({"success": False, "error": str(e)})
                    }
                ]
            )
            
            await self.storage.update_task(
                task_id, 'failed',
                new_messages=[error_message]
            )
    
    async def cancel_task(self, params: TaskIdParams) -> None:
        """Cancel a task"""
        task_id = params['id']
        await self.storage.update_task(task_id, 'cancelled')


class NL2SQLAgentServer:
    """A2A Server for NL2SQL agents"""
    
    def __init__(self):
        self.nl2sql_system: Optional[NL2SQLMultiAgentSystem] = None
        self.servers: Dict[str, FastA2A] = {}
        self.workers: Dict[str, NL2SQLWorker] = {}
    
    async def initialize(self):
        """Initialize the NL2SQL system and create A2A servers"""
        print("🚀 Initializing NL2SQL Multi-Agent A2A Server...")
        
        # Initialize NL2SQL system
        self.nl2sql_system = await NL2SQLMultiAgentSystem.create_and_initialize()
        print("✅ NL2SQL Multi-Agent System initialized!")
        
        # Create A2A servers for each agent type
        await self._create_agent_servers()
        print("🤖 A2A Agent servers created!")
    
    async def _create_agent_servers(self):
        """Create FastA2A servers for each agent"""
        agent_configs = {
            'orchestrator': {
                'name': 'NL2SQL Orchestrator Agent',
                'description': 'Coordinates the complete NL2SQL workflow with SQL generation, execution, and summarization',
                'port': A2A_BASE_PORT,
                'skills': [
                    {
                        'id': 'process_query',
                        'description': 'Process natural language question through complete NL2SQL pipeline',
                        'input_schema': {
                            'type': 'object',
                            'properties': {
                                'question': {'type': 'string', 'description': 'Natural language question'}
                            },
                            'required': ['question']
                        },
                        'output_schema': {
                            'type': 'object',
                            'properties': {
                                'success': {'type': 'boolean'},
                                'data': {'type': 'object'},
                                'error': {'type': 'string'}
                            }
                        },
                        'examples': [
                            {
                                'input': {'question': 'Show me the top 10 customers by revenue'},
                                'output': {'success': True, 'data': {'sql_query': 'SELECT ...', 'results': []}}
                            }
                        ]
                    }
                ]
            },
            'sql_generator': {
                'name': 'NL2SQL Generator Agent',
                'description': 'Converts natural language questions to SQL queries using database schema context',
                'port': A2A_BASE_PORT + 1,
                'skills': [
                    {
                        'id': 'generate_sql',
                        'description': 'Generate SQL query from natural language question',
                        'input_schema': {
                            'type': 'object',
                            'properties': {
                                'question': {'type': 'string', 'description': 'Natural language question'},
                                'context': {'type': 'string', 'description': 'Optional additional context'}
                            },
                            'required': ['question']
                        },
                        'output_schema': {
                            'type': 'object',
                            'properties': {
                                'sql_query': {'type': 'string'},
                                'intent_analysis': {'type': 'object'},
                                'confidence': {'type': 'number'}
                            }
                        },
                        'examples': [
                            {
                                'input': {'question': 'What are the top selling products?'},
                                'output': {'sql_query': 'SELECT producto, SUM(quantity) FROM sales...'}
                            }
                        ]
                    }
                ]
            },
            'executor': {
                'name': 'NL2SQL Executor Agent',
                'description': 'Executes SQL queries against the database and formats results',
                'port': A2A_BASE_PORT + 2,
                'skills': [
                    {
                        'id': 'execute_sql',
                        'description': 'Execute SQL query and return formatted results',
                        'input_schema': {
                            'type': 'object',
                            'properties': {
                                'sql_query': {'type': 'string', 'description': 'SQL query to execute'},
                                'limit': {'type': 'integer', 'description': 'Maximum rows to return', 'default': 100}
                            },
                            'required': ['sql_query']
                        },
                        'output_schema': {
                            'type': 'object',
                            'properties': {
                                'raw_results': {'type': 'string'},
                                'formatted_results': {'type': 'object'},
                                'row_count': {'type': 'integer'}
                            }
                        },
                        'examples': [
                            {
                                'input': {'sql_query': 'SELECT * FROM customers LIMIT 10'},
                                'output': {'raw_results': '...', 'row_count': 10}
                            }
                        ]
                    }
                ]
            },
            'summarizer': {
                'name': 'NL2SQL Summarizer Agent',
                'description': 'Analyzes query results and generates business insights and summaries',
                'port': A2A_BASE_PORT + 3,
                'skills': [
                    {
                        'id': 'summarize_results',
                        'description': 'Generate summary and insights from query results',
                        'input_schema': {
                            'type': 'object',
                            'properties': {
                                'raw_results': {'type': 'string'},
                                'formatted_results': {'type': 'object'},
                                'sql_query': {'type': 'string'},
                                'question': {'type': 'string'}
                            },
                            'required': ['formatted_results', 'question']
                        },
                        'output_schema': {
                            'type': 'object',
                            'properties': {
                                'summary': {'type': 'string'},
                                'insights': {'type': 'array'},
                                'recommendations': {'type': 'array'}
                            }
                        },
                        'examples': [
                            {
                                'input': {'question': 'Top customers', 'formatted_results': {}},
                                'output': {'summary': 'Analysis shows...', 'insights': []}
                            }
                        ]
                    }
                ]
            }
        }
        
        for agent_type, config in agent_configs.items():
            await self._create_single_agent_server(agent_type, config)
    
    async def _create_single_agent_server(self, agent_type: str, config: Dict[str, Any]):
        """Create a single A2A server for an agent"""
        # Create storage and broker
        storage = NL2SQLAgentStorage()
        broker = FixedInMemoryBroker()  # Use our fixed broker
        print(f"🔧 Using FixedInMemoryBroker for {agent_type}")
        
        # Create worker with the same broker
        worker = NL2SQLWorker(broker, storage, self.nl2sql_system, agent_type)
        self.workers[agent_type] = worker
        
        # Create FastA2A server with CUSTOM task handler
        server = FastA2A(
            storage=storage,
            broker=broker,
            name=config['name'],
            url=f"http://localhost:{config['port']}",
            version="1.0.0",
            description=config['description'],
            provider=AgentProvider(
                organization="NL2SQL Multi-Agent System",
                url="https://github.com/your-org/nl2sql"
            ),
            skills=config['skills'],
            debug=A2A_DEBUG
        )
        
        # Override the task handler to use our NL2SQL worker
        original_task_handler = server._task_handler
        
        async def custom_task_handler(context_id: str, task_id: str, messages: list):
            """Custom task handler that uses our NL2SQL worker"""
            print(f"🎯 Custom task handler called for {agent_type}")
            print(f"   Context ID: {context_id}")
            print(f"   Task ID: {task_id}")
            print(f"   Messages: {len(messages)} messages")
            
            try:
                # Use our worker to process the task
                if messages:
                    # Get the last message (most recent)
                    message = messages[-1] if messages else {}
                    print(f"🔍 Processing message: {message}")
                    
                    # Process with our NL2SQL worker
                    result = await worker.process_task(task_id, context_id, message)
                    print(f"✅ Task {task_id} completed successfully")
                    return result
                else:
                    print(f"⚠️ No messages provided for task {task_id}")
                    return {"success": False, "error": "No messages provided"}
                    
            except Exception as e:
                print(f"❌ Task {task_id} failed: {e}")
                import traceback
                traceback.print_exc()
                return {"success": False, "error": str(e)}
        
        # Replace the task handler
        server._task_handler = custom_task_handler
        
        self.servers[agent_type] = server
        print(f"📡 Created A2A server for {agent_type} on port {config['port']} with custom task handler")
    
    async def start_servers(self):
        """Start all A2A servers with integrated task handling"""
        print("🌐 Starting A2A servers...")
        
        # Start HTTP servers only - task handling is integrated
        server_tasks = []
        ports = {
            'orchestrator': A2A_BASE_PORT,
            'sql_generator': A2A_BASE_PORT + 1,
            'executor': A2A_BASE_PORT + 2,
            'summarizer': A2A_BASE_PORT + 3
        }
        
        for agent_type, server in self.servers.items():
            port = ports[agent_type]
            task = asyncio.create_task(self._run_server(server, port, agent_type))
            server_tasks.append(task)
        
        print("✅ All A2A servers started with integrated task handling!")
        print("\n📋 Available A2A Agent Endpoints:")
        print("=" * 50)
        for agent_type, port in ports.items():
            print(f"🤖 {agent_type.title()} Agent:")
            print(f"   • Agent Card: http://localhost:{port}/.well-known/agent.json")
            print(f"   • A2A Endpoint: http://localhost:{port}/")
            print(f"   • Documentation: http://localhost:{port}/docs")
            print()
        
        # Wait for all server tasks
        await asyncio.gather(*server_tasks)
    
    async def _run_worker_properly(self, worker: NL2SQLWorker, agent_type: str):
        """Run a worker using the proper FastA2A Worker.run() method"""
        try:
            print(f"🔧 Starting proper worker for {agent_type}...")
            # First, enter the broker's async context to initialize the streams
            async with worker.broker:
                print(f"🔧 Broker for {agent_type} initialized")
                # Now run the worker with the properly initialized broker
                async with worker.run():
                    print(f"✅ Worker for {agent_type} is now running and listening for tasks")
                    # The context manager will handle the task processing loop internally
                    try:
                        # This will run until the task group is cancelled
                        await anyio.sleep_forever()
                    except anyio.get_cancelled_exc_class():
                        print(f"🛑 Worker for {agent_type} cancelled")
        except Exception as e:
            print(f"❌ Worker error for {agent_type}: {e}")
            import traceback
            traceback.print_exc()
    
    async def _run_worker(self, worker: NL2SQLWorker, agent_type: str):
        """Run a worker in the background with enhanced error handling"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                async with worker.broker:
                    print(f"🔧 Worker for {agent_type} started and listening for tasks (attempt {retry_count + 1})")
                    async for task_operation in worker.broker.receive_task_operations():
                        try:
                            print(f"📨 [{agent_type}] Received task operation: {task_operation.get('operation', 'unknown')}")
                            
                            if task_operation['operation'] == 'run':
                                print(f"🚀 [{agent_type}] Starting task execution")
                                await worker.run_task(task_operation['params'])
                                print(f"✅ [{agent_type}] Task execution completed")
                            elif task_operation['operation'] == 'cancel':
                                print(f"🛑 [{agent_type}] Cancelling task")
                                await worker.cancel_task(task_operation['params'])
                                print(f"✅ [{agent_type}] Task cancelled")
                            else:
                                print(f"❓ [{agent_type}] Unknown operation: {task_operation['operation']}")
                                
                        except Exception as task_error:
                            print(f"⚠️ Task error in {agent_type} worker: {task_error}")
                            import traceback
                            traceback.print_exc()
                            # Continue processing other tasks
                            continue
                            
                # If we get here, worker completed normally
                break
                
            except Exception as e:
                retry_count += 1
                print(f"❌ Worker for {agent_type} failed (attempt {retry_count}/{max_retries}): {e}")
                import traceback
                traceback.print_exc()
                
                if retry_count >= max_retries:
                    print(f"🚫 Worker for {agent_type} permanently failed after {max_retries} attempts")
                    break
                else:
                    # Wait before retry
                    print(f"⏳ [{agent_type}] Waiting {2 ** retry_count} seconds before retry")
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
    
    async def _run_server(self, server: FastA2A, port: int, agent_type: str):
        """Run an A2A server with enhanced error handling"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                config = uvicorn.Config(
                    server,
                    host=A2A_HOST,
                    port=port,
                    log_level=A2A_LOG_LEVEL,
                    access_log=False  # Reduce log noise
                )
                server_instance = uvicorn.Server(config)
                print(f"🚀 A2A server for {agent_type} starting on port {port}")
                await server_instance.serve()
                
                # If we get here, server completed normally
                break
                
            except Exception as e:
                retry_count += 1
                print(f"❌ A2A server for {agent_type} failed (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    print(f"🚫 A2A server for {agent_type} permanently failed after {max_retries} attempts")
                    break
                else:
                    # Wait before retry
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
    
    async def close(self):
        """Close all resources"""
        if self.nl2sql_system:
            await self.nl2sql_system.close()
        print("🔐 NL2SQL A2A servers closed")
    
    async def health_check(self) -> dict:
        """
        Perform health check on all agents and services
        """
        try:
            if not self.nl2sql_system:
                return {"status": "unhealthy", "error": "System not initialized"}
            
            # Check database connectivity
            db_info = await self.nl2sql_system.get_database_info()
            
            # Check workflow status
            workflow_status = await self.nl2sql_system.get_workflow_status()
            
            return {
                "status": "healthy",
                "agents": {
                    "orchestrator": "running" if "orchestrator" in self.servers else "stopped",
                    "sql_generator": "running" if "sql_generator" in self.servers else "stopped", 
                    "executor": "running" if "executor" in self.servers else "stopped",
                    "summarizer": "running" if "summarizer" in self.servers else "stopped"
                },
                "database_connected": "Query Results" in str(db_info) or "Available tables" in str(db_info),
                "workflow_status": workflow_status
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e)
            }
    
    async def get_system_metrics(self) -> dict:
        """
        Get system performance metrics
        """
        try:
            metrics = {
                "active_servers": len(self.servers),
                "active_workers": len(self.workers),
                "server_ports": {
                    agent_type: {
                        "orchestrator": 8100,
                        "sql_generator": 8101,
                        "executor": 8102,
                        "summarizer": 8103
                    }.get(agent_type, "unknown")
                    for agent_type in self.servers.keys()
                },
                "system_status": "operational" if self.nl2sql_system else "not_initialized"
            }
            
            if self.nl2sql_system:
                # Add database metrics
                db_info = await self.nl2sql_system.get_database_info()
                schema_context = await self.nl2sql_system.get_schema_context()
                
                metrics.update({
                    "database_available": True,
                    "schema_loaded": len(schema_context) > 0 if schema_context else False
                })
            
            return metrics
        except Exception as e:
            return {"error": str(e), "system_status": "error"}
        
    async def list_database_tables(self) -> dict:
        """
        List all available database tables
        """
        try:
            if not self.nl2sql_system:
                return {"success": False, "error": "System not initialized"}
            
            tables_result = await self.nl2sql_system.mcp_plugin.list_tables()
            return {
                "success": True,
                "data": {"tables": str(tables_result)}
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to list tables: {str(e)}"}
    
    async def get_table_schema(self, table_name: str) -> dict:
        """
        Get schema information for a specific table
        """
        try:
            if not self.nl2sql_system:
                return {"success": False, "error": "System not initialized"}
            
            schema_result = await self.nl2sql_system.mcp_plugin.describe_table(table_name)
            return {
                "success": True,
                "data": {"table_schema": str(schema_result)}
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get table schema: {str(e)}"}
    
    async def execute_workflow_step(self, step: str, input_data: dict) -> dict:
        """
        Execute a single workflow step for testing/debugging
        """
        try:
            if not self.nl2sql_system:
                return {"success": False, "error": "System not initialized"}
            
            result = await self.nl2sql_system.execute_single_step(step, input_data)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": f"Failed to execute step: {str(e)}"}
    
    async def get_agent_capabilities(self, agent_type: str) -> dict:
        """
        Get capabilities of a specific agent
        """
        capabilities = {
            'orchestrator': {
                'description': 'Coordinates the complete NL2SQL workflow',
                'capabilities': ['question_processing', 'workflow_coordination', 'result_aggregation'],
                'input_formats': ['natural_language_text'],
                'output_formats': ['structured_json_with_sql_and_results']
            },
            'sql_generator': {
                'description': 'Converts natural language to SQL queries',
                'capabilities': ['intent_analysis', 'sql_generation', 'schema_awareness'],
                'input_formats': ['natural_language_question', 'json_with_question_and_context'],
                'output_formats': ['sql_query', 'intent_analysis', 'confidence_score']
            },
            'executor': {
                'description': 'Executes SQL queries safely and formats results',
                'capabilities': ['sql_execution', 'result_formatting', 'error_handling'],
                'input_formats': ['sql_query_string', 'json_with_sql_and_params'],
                'output_formats': ['query_results', 'execution_metadata', 'error_details']
            },
            'summarizer': {
                'description': 'Analyzes results and generates business insights',
                'capabilities': ['data_analysis', 'insight_generation', 'recommendations'],
                'input_formats': ['query_results_with_context'],
                'output_formats': ['summary', 'insights_list', 'recommendations']
            }
        }
        
        return {
            "success": True,
            "data": capabilities.get(agent_type, {"error": f"Unknown agent type: {agent_type}"})
        }
        

async def main():
    """Main entry point"""
    server = NL2SQLAgentServer()
    
    try:
        await server.initialize()
        await server.start_servers()
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down A2A servers...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(main())
