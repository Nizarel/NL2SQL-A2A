"""
NL2SQL FastA2A Client
Comprehensive client for communicating with NL2SQL Multi-Agent System using FastA2A protocol
"""

import asyncio
import json
import time
import uuid
import httpx
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

from fasta2a.client import A2AClient


class AgentType(Enum):
    """Available agent types in the NL2SQL system"""
    ORCHESTRATOR = "orchestrator"
    SQL_GENERATOR = "sql_generator"
    EXECUTOR = "executor"
    SUMMARIZER = "summarizer"


@dataclass
class AgentEndpoint:
    """Agent endpoint configuration"""
    agent_type: AgentType
    base_url: str
    port: int
    name: str
    description: str


class NL2SQLException(Exception):
    """Custom exception for NL2SQL client errors"""
    pass


class NL2SQLClient:
    """
    FastA2A Client for NL2SQL Multi-Agent System
    
    Provides high-level methods to interact with all NL2SQL agents:
    - Orchestrator: Complete NL2SQL workflow
    - SQL Generator: Natural language to SQL conversion
    - Executor: SQL query execution
    - Summarizer: Result analysis and insights
    """
    
    def __init__(self, base_host: str = "localhost", base_port: int = 8100):
        """
        Initialize NL2SQL FastA2A Client
        
        Args:
            base_host: Host where A2A servers are running
            base_port: Base port number (orchestrator port)
        """
        self.base_host = base_host
        self.base_port = base_port
        
        # Define agent endpoints
        self.agents = {
            AgentType.ORCHESTRATOR: AgentEndpoint(
                agent_type=AgentType.ORCHESTRATOR,
                base_url=f"http://{base_host}:{base_port}",
                port=base_port,
                name="NL2SQL Orchestrator Agent",
                description="Coordinates the complete NL2SQL workflow"
            ),
            AgentType.SQL_GENERATOR: AgentEndpoint(
                agent_type=AgentType.SQL_GENERATOR,
                base_url=f"http://{base_host}:{base_port + 1}",
                port=base_port + 1,
                name="NL2SQL Generator Agent",
                description="Converts natural language to SQL queries"
            ),
            AgentType.EXECUTOR: AgentEndpoint(
                agent_type=AgentType.EXECUTOR,
                base_url=f"http://{base_host}:{base_port + 2}",
                port=base_port + 2,
                name="NL2SQL Executor Agent",
                description="Executes SQL queries against the database"
            ),
            AgentType.SUMMARIZER: AgentEndpoint(
                agent_type=AgentType.SUMMARIZER,
                base_url=f"http://{base_host}:{base_port + 3}",
                port=base_port + 3,
                name="NL2SQL Summarizer Agent",
                description="Analyzes results and generates insights"
            )
        }
        
        # Initialize A2A clients for each agent
        self.a2a_clients: Dict[AgentType, A2AClient] = {}
        for agent_type, endpoint in self.agents.items():
            # Create httpx client with appropriate timeout for each agent type
            timeout = 120.0 if agent_type == AgentType.ORCHESTRATOR else 60.0
            http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
            
            self.a2a_clients[agent_type] = A2AClient(
                base_url=endpoint.base_url, 
                http_client=http_client
            )
    
    def _create_message(self, content: Union[str, Dict[str, Any]], role: str = "user") -> Dict[str, Any]:
        """
        Create a FastA2A message format
        
        Args:
            content: Message content (string or dict)
            role: Message role (user, agent, etc.)
            
        Returns:
            Formatted message for FastA2A protocol
        """
        # Convert content to string if it's a dict
        if isinstance(content, dict):
            text_content = json.dumps(content)
        else:
            text_content = str(content)
        
        return {
            "role": role,
            "kind": "message",
            "message_id": f"msg-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
            "parts": [
                {
                    "kind": "text",
                    "text": text_content
                }
            ]
        }
    
    def _extract_result(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract result from FastA2A JSON-RPC response
        
        Args:
            response: JSON-RPC response from A2A client
            
        Returns:
            Extracted result data
            
        Raises:
            NL2SQLException: If response contains error
        """
        if "error" in response and response["error"]:
            error_msg = response["error"]
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
            raise NL2SQLException(f"A2A Error: {error_msg}")
        
        if "result" not in response:
            raise NL2SQLException("No result found in A2A response")
        
        return response["result"]
    
    async def ask_question(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ask a natural language question using the Orchestrator agent (complete workflow)
        
        Args:
            question: Natural language question about the data
            metadata: Optional metadata for the request
            
        Returns:
            Complete NL2SQL workflow result including SQL, execution results, and summary
        """
        try:
            client = self.a2a_clients[AgentType.ORCHESTRATOR]
            message = self._create_message(question)
            
            # Use a longer timeout for the complex orchestrator workflow
            import httpx
            if hasattr(client, 'http_client') and isinstance(client.http_client, httpx.AsyncClient):
                # Update the timeout for this specific request
                original_timeout = client.http_client.timeout
                client.http_client.timeout = httpx.Timeout(120.0)  # 2 minutes for complex queries
            
            response = await client.send_message(
                message=message,
                metadata=metadata or {}
            )
            
            # Restore original timeout
            if hasattr(client, 'http_client') and isinstance(client.http_client, httpx.AsyncClient):
                client.http_client.timeout = original_timeout
            
            result = self._extract_result(response)
            
            # Parse the result if it's a JSON string
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            
            return result
            
        except Exception as e:
            raise NL2SQLException(f"Failed to process question with orchestrator: {str(e)}")
    
    async def generate_sql(self, question: str, context: Optional[str] = None, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate SQL query from natural language using SQL Generator agent
        
        Args:
            question: Natural language question
            context: Optional additional context
            metadata: Optional metadata for the request
            
        Returns:
            SQL generation result with query, intent analysis, and confidence
        """
        try:
            client = self.a2a_clients[AgentType.SQL_GENERATOR]
            
            content = {
                "question": question
            }
            if context:
                content["context"] = context
            
            message = self._create_message(content)
            
            response = await client.send_message(
                message=message,
                metadata=metadata or {}
            )
            
            result = self._extract_result(response)
            
            # Parse the result if it's a JSON string
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            
            return result
            
        except Exception as e:
            raise NL2SQLException(f"Failed to generate SQL: {str(e)}")
    
    async def execute_sql(self, sql_query: str, limit: int = 100, 
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute SQL query using Executor agent
        
        Args:
            sql_query: SQL query to execute
            limit: Maximum number of rows to return
            metadata: Optional metadata for the request
            
        Returns:
            Execution result with raw results, formatted data, and metadata
        """
        try:
            client = self.a2a_clients[AgentType.EXECUTOR]
            
            content = {
                "sql_query": sql_query,
                "limit": limit
            }
            
            message = self._create_message(content)
            
            response = await client.send_message(
                message=message,
                metadata=metadata or {}
            )
            
            result = self._extract_result(response)
            
            # Parse the result if it's a JSON string
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            
            return result
            
        except Exception as e:
            raise NL2SQLException(f"Failed to execute SQL: {str(e)}")
    
    async def summarize_results(self, raw_results: str, formatted_results: Dict[str, Any],
                               sql_query: str, question: str, 
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate summary and insights from query results using Summarizer agent
        
        Args:
            raw_results: Raw query results as string
            formatted_results: Formatted query results
            sql_query: The SQL query that was executed
            question: Original natural language question
            metadata: Optional metadata for the request
            
        Returns:
            Summary with insights, recommendations, and analysis
        """
        try:
            client = self.a2a_clients[AgentType.SUMMARIZER]
            
            content = {
                "raw_results": raw_results,
                "formatted_results": formatted_results,
                "sql_query": sql_query,
                "question": question
            }
            
            message = self._create_message(content)
            
            response = await client.send_message(
                message=message,
                metadata=metadata or {}
            )
            
            result = self._extract_result(response)
            
            # Parse the result if it's a JSON string
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            
            return result
            
        except Exception as e:
            raise NL2SQLException(f"Failed to summarize results: {str(e)}")
    
    async def get_task_status(self, agent_type: AgentType, task_id: str) -> Dict[str, Any]:
        """
        Get status of a specific task
        
        Args:
            agent_type: Type of agent to query
            task_id: Task ID to check
            
        Returns:
            Task status information
        """
        try:
            client = self.a2a_clients[agent_type]
            response = await client.get_task(task_id)
            return self._extract_result(response)
            
        except Exception as e:
            raise NL2SQLException(f"Failed to get task status: {str(e)}")
    
    async def execute_step_by_step(self, question: str) -> Dict[str, Any]:
        """
        Execute the NL2SQL workflow step-by-step using individual agents
        
        This method demonstrates how to use each agent individually and
        provides more control over the workflow process.
        
        Args:
            question: Natural language question
            
        Returns:
            Complete workflow result with step-by-step details
        """
        workflow_result = {
            "question": question,
            "steps": {},
            "final_result": None,
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Generate SQL
            print(f"🔍 Step 1: Generating SQL for question: '{question}'")
            sql_result = await self.generate_sql(question)
            workflow_result["steps"]["sql_generation"] = sql_result
            
            if not sql_result.get("success", False):
                raise NL2SQLException(f"SQL generation failed: {sql_result.get('error', 'Unknown error')}")
            
            sql_query = sql_result["data"]["sql_query"]
            print(f"✅ Generated SQL: {sql_query}")
            
            # Step 2: Execute SQL
            print(f"🚀 Step 2: Executing SQL query")
            exec_result = await self.execute_sql(sql_query)
            workflow_result["steps"]["sql_execution"] = exec_result
            
            if not exec_result.get("success", False):
                raise NL2SQLException(f"SQL execution failed: {exec_result.get('error', 'Unknown error')}")
            
            print(f"✅ Query executed successfully")
            
            # Step 3: Summarize results
            print(f"📊 Step 3: Generating summary and insights")
            summary_result = await self.summarize_results(
                raw_results=exec_result["data"]["raw_results"],
                formatted_results=exec_result["data"]["formatted_results"],
                sql_query=sql_query,
                question=question
            )
            workflow_result["steps"]["summarization"] = summary_result
            
            if not summary_result.get("success", False):
                raise NL2SQLException(f"Summarization failed: {summary_result.get('error', 'Unknown error')}")
            
            print(f"✅ Summary generated successfully")
            
            # Compile final result
            workflow_result["final_result"] = {
                "question": question,
                "sql_query": sql_query,
                "results": exec_result["data"]["formatted_results"],
                "summary": summary_result["data"]["summary"],
                "insights": summary_result["data"].get("insights", []),
                "recommendations": summary_result["data"].get("recommendations", [])
            }
            
            workflow_result["execution_time"] = time.time() - start_time
            workflow_result["success"] = True
            
            return workflow_result
            
        except Exception as e:
            workflow_result["execution_time"] = time.time() - start_time
            workflow_result["success"] = False
            workflow_result["error"] = str(e)
            raise NL2SQLException(f"Step-by-step execution failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all agents
        
        Returns:
            Health status of all agents
        """
        health_status = {
            "overall_status": "healthy",
            "agents": {},
            "timestamp": time.time()
        }
        
        for agent_type, endpoint in self.agents.items():
            try:
                # Simple HTTP check instead of sending A2A messages
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    # Check if the agent card endpoint is accessible
                    agent_card_url = f"{endpoint.base_url}/.well-known/agent.json"
                    async with session.get(agent_card_url, timeout=5) as response:
                        if response.status == 200:
                            health_status["agents"][agent_type.value] = {
                                "status": "healthy",
                                "url": endpoint.base_url,
                                "port": endpoint.port
                            }
                        else:
                            raise Exception(f"HTTP {response.status}")
                
            except Exception as e:
                health_status["agents"][agent_type.value] = {
                    "status": "unhealthy",
                    "url": endpoint.base_url,
                    "port": endpoint.port,
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        return health_status
    
    def get_agent_info(self, agent_type: AgentType) -> Dict[str, Any]:
        """
        Get information about a specific agent
        
        Args:
            agent_type: Type of agent to get info for
            
        Returns:
            Agent information including capabilities and endpoints
        """
        if agent_type not in self.agents:
            raise NL2SQLException(f"Unknown agent type: {agent_type}")
        
        endpoint = self.agents[agent_type]
        
        capabilities = {
            AgentType.ORCHESTRATOR: {
                "primary_function": "Complete NL2SQL workflow coordination",
                "capabilities": ["question_processing", "workflow_coordination", "result_aggregation"],
                "input_formats": ["natural_language_text"],
                "output_formats": ["complete_workflow_result"]
            },
            AgentType.SQL_GENERATOR: {
                "primary_function": "Natural language to SQL conversion",
                "capabilities": ["intent_analysis", "sql_generation", "schema_awareness"],
                "input_formats": ["natural_language_question", "question_with_context"],
                "output_formats": ["sql_query", "intent_analysis", "confidence_score"]
            },
            AgentType.EXECUTOR: {
                "primary_function": "SQL query execution and result formatting",
                "capabilities": ["sql_execution", "result_formatting", "error_handling"],
                "input_formats": ["sql_query", "sql_with_parameters"],
                "output_formats": ["query_results", "execution_metadata"]
            },
            AgentType.SUMMARIZER: {
                "primary_function": "Result analysis and insight generation",
                "capabilities": ["data_analysis", "insight_generation", "recommendations"],
                "input_formats": ["query_results_with_context"],
                "output_formats": ["summary", "insights", "recommendations"]
            }
        }
        
        return {
            "agent_type": agent_type.value,
            "name": endpoint.name,
            "description": endpoint.description,
            "base_url": endpoint.base_url,
            "port": endpoint.port,
            "capabilities": capabilities[agent_type]
        }
    
    def list_all_agents(self) -> List[Dict[str, Any]]:
        """
        List information about all available agents
        
        Returns:
            List of agent information
        """
        return [self.get_agent_info(agent_type) for agent_type in AgentType]


# Convenience functions for common operations
async def quick_question(question: str, host: str = "localhost", port: int = 8100) -> Dict[str, Any]:
    """
    Quick way to ask a question using the orchestrator (complete workflow)
    
    Args:
        question: Natural language question
        host: Host where A2A servers are running
        port: Base port for orchestrator
        
    Returns:
        Complete NL2SQL result
    """
    client = NL2SQLClient(base_host=host, base_port=port)
    return await client.ask_question(question)


async def quick_sql_generation(question: str, host: str = "localhost", port: int = 8100) -> str:
    """
    Quick way to generate SQL from natural language
    
    Args:
        question: Natural language question
        host: Host where A2A servers are running
        port: Base port for orchestrator
        
    Returns:
        Generated SQL query
    """
    client = NL2SQLClient(base_host=host, base_port=port)
    result = await client.generate_sql(question)
    
    if result.get("success") and "data" in result:
        return result["data"].get("sql_query", "")
    else:
        raise NL2SQLException(f"SQL generation failed: {result.get('error', 'Unknown error')}")


# Example usage and testing
async def main():
    """Example usage of the NL2SQL FastA2A Client"""
    client = NL2SQLClient()
    
    print("🚀 NL2SQL FastA2A Client Example\n")
    
    # Health check
    print("🏥 Performing health check...")
    health = await client.health_check()
    print(f"Overall status: {health['overall_status']}")
    for agent, status in health['agents'].items():
        print(f"  {agent}: {status['status']} (Port {status['port']})")
    print()
    
    # List agents
    print("🤖 Available agents:")
    agents = client.list_all_agents()
    for agent in agents:
        print(f"  • {agent['name']} ({agent['agent_type']})")
        print(f"    URL: {agent['base_url']}")
        print(f"    Primary Function: {agent['capabilities']['primary_function']}")
    print()
    
    # Example question
    question = "What are the top 5 products by total sales revenue?"
    
    print(f"❓ Question: {question}\n")
    
    # Method 1: Use orchestrator (complete workflow)
    print("🔄 Method 1: Using Orchestrator (Complete Workflow)")
    try:
        result = await client.ask_question(question)
        print(f"✅ Success: {result.get('success', False)}")
        if result.get('success'):
            data = result.get('data', {})
            print(f"SQL Query: {data.get('sql_query', 'N/A')}")
            print(f"Results: {len(data.get('results', []))} rows")
            print(f"Summary: {data.get('summary', 'N/A')[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()
    
    # Method 2: Step-by-step execution
    print("🔄 Method 2: Step-by-Step Execution")
    try:
        result = await client.execute_step_by_step(question)
        print(f"✅ Success: {result.get('success', False)}")
        print(f"Execution time: {result.get('execution_time', 0):.2f} seconds")
        if result.get('success'):
            final = result.get('final_result', {})
            print(f"SQL Query: {final.get('sql_query', 'N/A')}")
            print(f"Summary: {final.get('summary', 'N/A')[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
