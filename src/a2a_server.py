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
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import uvicorn
from fasta2a import FastA2A, Skill, Storage, Broker, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import (
    AgentProvider, AgentCapabilities, Message, Task, 
    Artifact, TaskSendParams, TaskIdParams
)

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import NL2SQLMultiAgentSystem


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
        # Extract content from A2A message parts
        content = ""
        if 'parts' in message and message['parts']:
            for part in message['parts']:
                if part.get('kind') == 'text':
                    content += part.get('text', '')
        
        try:
            if self.agent_type == 'orchestrator':
                # Use content as question for orchestrator
                result = await self.nl2sql_system.process_query(content)
                
            elif self.agent_type == 'sql_generator':
                # Parse input for SQL generation
                try:
                    content_dict = json.loads(content)
                except json.JSONDecodeError:
                    content_dict = {"question": content}
                
                result = await self.nl2sql_system.sql_generator.process(content_dict)
                
            elif self.agent_type == 'executor':
                # Parse input for SQL execution
                try:
                    content_dict = json.loads(content)
                except json.JSONDecodeError:
                    content_dict = {"sql_query": content}
                
                result = await self.nl2sql_system.executor.process(content_dict)
                
            elif self.agent_type == 'summarizer':
                # Parse input for summarization
                try:
                    content_dict = json.loads(content)
                except json.JSONDecodeError:
                    return {"success": False, "error": "Invalid input format for summarizer"}
                
                result = await self.nl2sql_system.summarizer.process(content_dict)
                
            else:
                return {"success": False, "error": f"Unknown agent type: {self.agent_type}"}
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing {self.agent_type} task: {str(e)}",
                "agent_type": self.agent_type
            }
    
    async def run_task(self, params: TaskSendParams) -> None:
        """Execute a task and update storage"""
        task_id = params['id']
        context_id = params['context_id']
        message = params['message']
        
        try:
            # Update task to working state
            await self.storage.update_task(task_id, 'working')
            
            # Process the task
            result = await self.process_task(task_id, context_id, message)
            
            # Build artifacts from result
            artifacts = self.build_artifacts(result)
            
            # Create response message
            response_message = Message(
                role='agent',
                kind='message',
                messageId=f"response-{int(time.time() * 1000)}",
                parts=[
                    {
                        'kind': 'text',
                        'text': json.dumps(result)
                    }
                ],
                timestamp=time.time()
            )
            
            # Update task with completion
            if result.get('success', False):
                await self.storage.update_task(
                    task_id, 'completed', 
                    new_artifacts=artifacts,
                    new_messages=[response_message]
                )
            else:
                await self.storage.update_task(
                    task_id, 'failed',
                    new_artifacts=artifacts,
                    new_messages=[response_message]
                )
                
        except Exception as e:
            # Handle errors
            error_message = Message(
                role='agent',
                kind='message',
                messageId=f"error-{int(time.time() * 1000)}",
                parts=[
                    {
                        'kind': 'text',
                        'text': json.dumps({"success": False, "error": str(e)})
                    }
                ],
                timestamp=time.time()
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
                'port': 8100,
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
                'port': 8101,
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
                'port': 8102,
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
                'port': 8103,
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
        broker = InMemoryBroker()
        
        # Create worker
        worker = NL2SQLWorker(broker, storage, self.nl2sql_system, agent_type)
        self.workers[agent_type] = worker
        
        # Create FastA2A server
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
            debug=True
        )
        
        self.servers[agent_type] = server
        print(f"📡 Created A2A server for {agent_type} on port {config['port']}")
    
    async def start_servers(self):
        """Start all A2A servers"""
        print("🌐 Starting A2A servers...")
        
        # Start worker tasks
        worker_tasks = []
        for agent_type, worker in self.workers.items():
            task = asyncio.create_task(self._run_worker(worker, agent_type))
            worker_tasks.append(task)
        
        # Start HTTP servers
        server_tasks = []
        ports = {
            'orchestrator': 8100,
            'sql_generator': 8101,
            'executor': 8102,
            'summarizer': 8103
        }
        
        for agent_type, server in self.servers.items():
            port = ports[agent_type]
            task = asyncio.create_task(self._run_server(server, port, agent_type))
            server_tasks.append(task)
        
        print("✅ All A2A servers started!")
        print("\n📋 Available A2A Agent Endpoints:")
        print("=" * 50)
        for agent_type, port in ports.items():
            print(f"🤖 {agent_type.title()} Agent:")
            print(f"   • Agent Card: http://localhost:{port}/.well-known/agent.json")
            print(f"   • A2A Endpoint: http://localhost:{port}/")
            print(f"   • Documentation: http://localhost:{port}/docs")
            print()
        
        # Wait for all tasks
        await asyncio.gather(*worker_tasks, *server_tasks)
    
    async def _run_worker(self, worker: NL2SQLWorker, agent_type: str):
        """Run a worker in the background"""
        try:
            async with worker.broker:
                print(f"🔧 Worker for {agent_type} started")
                async for task_operation in worker.broker.receive_task_operations():
                    if task_operation['operation'] == 'run':
                        await worker.run_task(task_operation['params'])
                    elif task_operation['operation'] == 'cancel':
                        await worker.cancel_task(task_operation['params'])
        except Exception as e:
            print(f"❌ Worker for {agent_type} failed: {e}")
    
    async def _run_server(self, server: FastA2A, port: int, agent_type: str):
        """Run an A2A server"""
        try:
            config = uvicorn.Config(
                server,
                host="0.0.0.0",
                port=port,
                log_level="info"
            )
            server_instance = uvicorn.Server(config)
            print(f"🚀 A2A server for {agent_type} starting on port {port}")
            await server_instance.serve()
        except Exception as e:
            print(f"❌ A2A server for {agent_type} failed: {e}")
    
    async def close(self):
        """Close all resources"""
        if self.nl2sql_system:
            await self.nl2sql_system.close()
        print("🔐 NL2SQL A2A servers closed")


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
