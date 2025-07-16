"""
NL2SQL Client Agent using Semantic Kernel
Connects to the A2A NL2SQL Orchestrator Server to process natural language queries
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional, AsyncIterable
from datetime import datetime

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

import httpx
from a2a.client import A2AClient
from a2a.client.helpers import create_text_message_object
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendMessageResponse,
    MessageSendParams,
    Message,
    TextPart,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    Task,
    Role
)
from dotenv import load_dotenv

# Handle import for direct execution vs module import
try:
    from .remote_agent_connection import RemoteAgentConnections
except ImportError:
    # Fallback for direct execution
    from remote_agent_connection import RemoteAgentConnections

# Load environment variables from Host_Agent/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NL2SQLClientAgent:
    """
    Semantic Kernel Agent that connects to A2A NL2SQL Orchestrator Server
    """
    
    def __init__(self, kernel: Kernel, a2a_server_url: str = "http://localhost:8002"):
        self.kernel = kernel
        self.a2a_server_url = a2a_server_url
        self.remote_connection: Optional[RemoteAgentConnections] = None
        self._session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create Semantic Kernel Chat Agent
        self.sk_agent = ChatCompletionAgent(
            service=kernel.get_service("azure_openai_chat"),
            name="NL2SQLClientAgent",
            instructions=(
                "You are an intelligent data analysis assistant that helps users query databases using natural language. "
                "You connect to a remote NL2SQL orchestrator service that can: "
                "1. Convert natural language to SQL queries "
                "2. Execute SQL queries safely against databases "
                "3. Analyze results and provide business insights "
                "4. Generate comprehensive data summaries "
                "When users ask data questions, you forward them to the NL2SQL service and present the results in a user-friendly way."
            )
        )
        
        # Add NL2SQL function to the kernel
        self.kernel.add_plugin(self, plugin_name="NL2SQLPlugin")
        
        # Debug: Check what plugins and functions are available
        logger.info(f"🔍 Available plugins: {list(self.kernel.plugins.keys())}")
        for plugin_name, plugin in self.kernel.plugins.items():
            logger.info(f"🔍 Plugin '{plugin_name}' functions: {list(plugin.functions.keys())}")
        
    async def initialize_connection(self):
        """Initialize connection to the A2A NL2SQL server"""
        try:
            logger.info(f"🔗 Connecting to A2A NL2SQL server at {self.a2a_server_url}")
            
            # First, try to get the agent card from the A2A server
            try:
                async with httpx.AsyncClient() as client:
                    # Try different possible endpoints for agent card
                    for endpoint in ["/agent-card", "/agent_card", "/"]:
                        try:
                            response = await client.get(f"{self.a2a_server_url}{endpoint}", timeout=5.0)
                            if response.status_code == 200:
                                agent_card_data = response.json()
                                agent_card = AgentCard.model_validate(agent_card_data)
                                logger.info(f"✅ Retrieved agent card from {endpoint}: {agent_card.name}")
                                break
                        except Exception:
                            continue
                    else:
                        # If no agent card endpoint works, create a default one
                        logger.warning("⚠️ Could not retrieve agent card, creating default connection")
                        agent_card = None
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to get agent card: {e}, using direct connection")
                agent_card = None
            
            # Create connection with or without agent card
            self.remote_connection = RemoteAgentConnections(agent_card, self.a2a_server_url)
            logger.info("✅ A2A connection initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to A2A server: {str(e)}")
            raise

    @kernel_function(
        description="Process natural language queries about data using the remote NL2SQL orchestrator service. Use this function for ANY question about data, customers, revenue, sales, or database queries.",
        name="query_database"
    )
    async def query_database(self, query: str) -> str:
        """
        Send a natural language query to the remote NL2SQL orchestrator using the same logic as the API server
        
        Args:
            query: Natural language question about data
            
        Returns:
            Formatted response with SQL results and insights
        """
        logger.info(f"🎯 QUERY_DATABASE FUNCTION CALLED with: {query}")
        
        if not self.remote_connection:
            await self.initialize_connection()
        
        try:
            logger.info(f"📤 Sending orchestrator query: {query}")
            
            # Create a comprehensive query request similar to API server's QueryRequest
            query_request = {
                "question": query,
                "execute": True,  # Execute the generated SQL query
                "limit": 100,     # Maximum number of rows to return
                "include_summary": True,  # Generate AI summary and insights
                "context": ""     # Optional additional context
            }
            
            # Format as A2A message with the full orchestrator request
            orchestrator_prompt = f"""Process this natural language query using the full NL2SQL orchestrator workflow:

Question: {query_request['question']}
Parameters:
- Execute Query: {query_request['execute']}
- Row Limit: {query_request['limit']}
- Include Summary: {query_request['include_summary']}
- Context: {query_request['context'] or 'None'}

Please follow the complete orchestrator workflow:
1. Generate SQL query from the natural language question
2. Execute the SQL query against the database  
3. Analyze results and provide business insights
4. Return formatted results with summary and recommendations

Return the complete workflow results including SQL query, execution results, and AI-generated insights."""
            
            # Create message request using the correct JSON-RPC structure
            message = create_text_message_object(
                role=Role.user,
                content=orchestrator_prompt
            )
            
            request = SendMessageRequest(
                id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                params=MessageSendParams(
                    message=message
                )
            )
            
            # Send message and get response
            response = await self.remote_connection.send_message(request)
            
            # Debug: Log the actual response
            logger.info(f"🔍 DEBUG: Received response type: {type(response)}")
            logger.info(f"🔍 DEBUG: Response has root: {hasattr(response, 'root')}")
            if hasattr(response, 'root'):
                logger.info(f"🔍 DEBUG: Root type: {type(response.root)}")
                logger.info(f"🔍 DEBUG: Root has result: {hasattr(response.root, 'result')}")
                if hasattr(response.root, 'result'):
                    logger.info(f"🔍 DEBUG: Result type: {type(response.root.result)}")
                    logger.info(f"🔍 DEBUG: Result content: {response.root.result}")
            
            # Process response - handle JSON-RPC response structure
            if hasattr(response, 'root') and response.root:
                if hasattr(response.root, 'result'):
                    # Success response
                    result_data = response.root.result
                    if hasattr(result_data, 'status') and hasattr(result_data.status, 'state'):
                        # It's a Task
                        if result_data.status.state == "completed":
                            result = self._format_orchestrator_response(result_data, query)
                            logger.info("✅ Orchestrator query processed successfully")
                            return result
                        else:
                            error_msg = f"Orchestrator query failed with status: {result_data.status.state}"
                            logger.error(f"❌ {error_msg}")
                            return f"❌ Error: {error_msg}"
                    else:
                        # It's a Message - format it directly
                        result = self._format_orchestrator_message_response(result_data, query)
                        logger.info("✅ Orchestrator query processed successfully")
                        return result
                elif hasattr(response.root, 'error'):
                    # Error response
                    error_msg = f"A2A Orchestrator Error: {response.root.error}"
                    logger.error(f"❌ {error_msg}")
                    return f"❌ Error: {error_msg}"
            
            # Fallback
            logger.warning("⚠️ Unexpected response format")
            return "❌ Error: Unexpected response format from A2A server"
                
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return f"❌ Error: {error_msg}"

    async def query_database_streaming(self, query: str) -> AsyncIterable[str]:
        """
        Send a natural language query with streaming responses
        
        Args:
            query: Natural language question about data
            
        Yields:
            Streaming response chunks
        """
        if not self.remote_connection:
            await self.initialize_connection()
        
        try:
            logger.info(f"📤 Sending streaming query: {query}")
            
            # For streaming, we would need the A2A client to support streaming
            # For now, we'll simulate streaming by yielding the regular response in chunks
            result = await self.query_database(query)
            
            # Simulate streaming by yielding chunks
            words = result.split()
            for i in range(0, len(words), 10):  # Yield 10 words at a time
                chunk = " ".join(words[i:i+10])
                yield chunk + " "
                await asyncio.sleep(0.1)  # Small delay to simulate streaming
                
        except Exception as e:
            yield f"❌ Streaming error: {str(e)}"

    def _format_response(self, response: SendMessageResponse, original_query: str) -> str:
        """Format the A2A response into a user-friendly format"""
        try:
            if not response.artifacts:
                return "❌ No results returned from the NL2SQL service"
            
            # Get the main artifact (should be the NL2SQL result)
            artifact = response.artifacts[0]
            if not artifact.parts:
                return "❌ Empty response from NL2SQL service"
            
            # Extract the content
            content = ""
            for part in artifact.parts:
                if hasattr(part, 'text'):
                    content += part.text
                elif hasattr(part, 'content'):
                    content += str(part.content)
            
            # Format the response nicely
            formatted_response = f"""🎯 **Query Processing Complete**

📝 **Your Question:** {original_query}

📊 **Results:**
{content}

---
*Processed by NL2SQL Orchestrator Agent via A2A Protocol*
"""
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return f"❌ Error formatting response: {str(e)}"

    def _format_message_response(self, message, original_query: str) -> str:
        """Format a direct Message response into a user-friendly format"""
        try:
            # Extract content from message parts
            content = ""
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'text'):
                        content += part.text
                    elif hasattr(part, 'content'):
                        content += str(part.content)
            elif hasattr(message, 'content'):
                content = str(message.content)
            else:
                content = str(message)
            
            # Format the response nicely
            formatted_response = f"""🎯 **Query Processing Complete**

📝 **Your Question:** {original_query}

📊 **Results:**
{content}

---
*Processed by NL2SQL Orchestrator Agent via A2A Protocol*
"""
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting message response: {str(e)}")
            return f"❌ Error formatting message response: {str(e)}"

    def _format_orchestrator_response(self, response, original_query: str) -> str:
        """Format the A2A orchestrator response into a user-friendly format"""
        try:
            if not response.artifacts:
                return "❌ No results returned from the NL2SQL orchestrator"
            
            # Get the main artifact (should be the orchestrator result)
            artifact = response.artifacts[0]
            if not artifact.parts:
                return "❌ Empty response from NL2SQL orchestrator"
            
            # Extract the content
            content = ""
            for part in artifact.parts:
                if hasattr(part, 'text'):
                    content += part.text
                elif hasattr(part, 'content'):
                    content += str(part.content)
            
            # Format the response nicely with orchestrator context
            formatted_response = f"""🎯 **NL2SQL Orchestrator Results**

📝 **Your Question:** {original_query}

🔄 **Orchestrator Workflow Completed:**
{content}

---
*Processed by Multi-Agent NL2SQL Orchestrator via A2A Protocol*
*Workflow: SQL Generation → Execution → Analysis*
"""
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting orchestrator response: {str(e)}")
            return f"❌ Error formatting orchestrator response: {str(e)}"

    def _format_orchestrator_message_response(self, message, original_query: str) -> str:
        """Format a direct orchestrator Message response into a user-friendly format"""
        try:
            # Extract content from message parts
            content = ""
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'text'):
                        content += part.text
                    elif hasattr(part, 'content'):
                        content += str(part.content)
            elif hasattr(message, 'content'):
                content = str(message.content)
            else:
                content = str(message)
            
            # Format the response nicely with orchestrator context
            formatted_response = f"""🎯 **NL2SQL Orchestrator Results**

📝 **Your Question:** {original_query}

🔄 **Multi-Agent Workflow Results:**
{content}

---
*Processed by Multi-Agent NL2SQL Orchestrator via A2A Protocol*
*Workflow: SQL Generation → Execution → Analysis & Insights*
"""
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting orchestrator message response: {str(e)}")
            return f"❌ Error formatting orchestrator message response: {str(e)}"

    async def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of the remote NL2SQL agent"""
        if not self.remote_connection:
            await self.initialize_connection()
        
        agent_card = self.remote_connection.get_agent()
        return {
            "name": agent_card.name,
            "description": agent_card.description,
            "version": agent_card.version,
            "capabilities": {
                "streaming": agent_card.capabilities.streaming if agent_card.capabilities else False,
                "pushNotifications": agent_card.capabilities.pushNotifications if agent_card.capabilities else False
            },
            "skills": [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "tags": skill.tags,
                    "examples": skill.examples
                }
                for skill in (agent_card.skills or [])
            ]
        }

    async def ask_question(self, question: str) -> str:
        """
        High-level method to ask a data question
        
        Args:
            question: Natural language question about data
            
        Returns:
            Formatted answer
        """
        return await self.query_database(question)

    async def ask_question_with_context(self, question: str, context: str = "") -> str:
        """
        Ask a question with additional context
        
        Args:
            question: Natural language question
            context: Additional context for the question
            
        Returns:
            Formatted answer
        """
        if context:
            full_query = f"Context: {context}\n\nQuestion: {question}"
        else:
            full_query = question
            
        return await self.query_database(full_query)

    async def get_session_id(self) -> str:
        """Get the current session ID"""
        return self._session_id

    async def reset_session(self):
        """Reset the session ID"""
        self._session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🔄 Session reset: {self._session_id}")

    async def close(self):
        """Close connections and cleanup"""
        if self.remote_connection and hasattr(self.remote_connection, '_httpx_client'):
            await self.remote_connection._httpx_client.aclose()
        logger.info("🔐 NL2SQL Client Agent closed")


class SemanticKernelNL2SQLAgent:
    """
    Higher-level Semantic Kernel Agent that integrates with NL2SQL Client
    """
    
    def __init__(self, a2a_server_url: str = "http://localhost:8002"):
        self.kernel = Kernel()
        
        # Setup Azure OpenAI service
        self._setup_ai_service()
        
        # Create NL2SQL client
        self.nl2sql_client = NL2SQLClientAgent(self.kernel, a2a_server_url)
        
        # Create main chat agent with function calling enabled and access to the NL2SQL functions
        self.chat_agent = ChatCompletionAgent(
            service=self.kernel.get_service("azure_openai_chat"),
            kernel=self.nl2sql_client.kernel,  # Use the kernel that has the NL2SQL functions
            name="DataAnalysisAssistant",
            instructions=(
                "You are a helpful data analysis assistant. You can help users query databases using natural language. "
                "When users ask questions about data, you MUST use the query_database function to process their questions. "
                "Always call the query_database function for any question that involves data analysis, SQL queries, or database information. "
                "Present the results in a clear, user-friendly format and provide insights when appropriate. "
                "You can handle follow-up questions and help users understand their data better."
            ),
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )

    def _setup_ai_service(self):
        """Setup Azure OpenAI service for Semantic Kernel using Host_Agent/.env variables"""
        try:
            # Get Azure OpenAI configuration from environment variables
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
            azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            
            # Validate required environment variables
            if not azure_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
            if not azure_api_key:
                raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
            
            logger.info(f"🔧 Configuring Azure OpenAI service:")
            logger.info(f"   Endpoint: {azure_endpoint}")
            logger.info(f"   Deployment: {azure_deployment}")
            logger.info(f"   API Version: {azure_api_version}")
            
            # Add Azure OpenAI chat completion service with explicit configuration
            azure_chat_service = AzureChatCompletion(
                service_id="azure_openai_chat",
                endpoint=azure_endpoint,
                api_key=azure_api_key,
                deployment_name=azure_deployment,
                api_version=azure_api_version
            )
            self.kernel.add_service(azure_chat_service)
            logger.info("✅ Azure OpenAI service configured successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup AI service: {str(e)}")
            logger.error("💡 Make sure your Host_Agent/.env file contains:")
            logger.error("   AZURE_OPENAI_ENDPOINT=your_endpoint")
            logger.error("   AZURE_OPENAI_API_KEY=your_api_key")
            logger.error("   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment")
            logger.error("   AZURE_OPENAI_API_VERSION=your_api_version")
            raise

    async def initialize(self):
        """Initialize the agent and connections"""
        await self.nl2sql_client.initialize_connection()
        logger.info("✅ Semantic Kernel NL2SQL Agent initialized")

    async def chat(self, user_input: str) -> str:
        """
        Chat with the agent about data questions
        
        Args:
            user_input: User's question or message
            
        Returns:
            Agent's response
        """
        try:
            # Use Semantic Kernel 1.35.0 get_response method which returns a single response
            response = await self.chat_agent.get_response(user_input)
            
            # Extract content from the AgentResponseItem
            if response and hasattr(response, 'content') and response.content:
                return str(response.content)
            elif response and hasattr(response, 'message') and response.message and response.message.content:
                return str(response.message.content)
            else:
                return "No response received"
            
        except Exception as e:
            logger.error(f"❌ Chat error: {str(e)}")
            return f"❌ Error processing your question: {str(e)}"

    async def get_capabilities(self) -> Dict[str, Any]:
        """Get information about the connected NL2SQL service"""
        return await self.nl2sql_client.get_agent_capabilities()

    async def close(self):
        """Cleanup resources"""
        await self.nl2sql_client.close()
        logger.info("🔐 Semantic Kernel NL2SQL Agent closed")


# Example usage and testing functions
async def example_usage():
    """Example of how to use the NL2SQL Client Agent"""
    
    # Validate environment variables before starting
    required_env_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("💡 Please check your Host_Agent/.env file")
        return
    
    print("🔧 Environment variables loaded:")
    print(f"   Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"   Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    print(f"   API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    
    # Create and initialize the agent
    agent = SemanticKernelNL2SQLAgent("http://localhost:8002")
    
    try:
        await agent.initialize()
        
        print("\n🎉 A2A Client Agent initialized successfully!")
        print("🔗 Connection to A2A server established")
        
        # Test the specific NL2SQL question
        test_question = "Show top 3 customers by revenue with their details in March 2025."
        print(f"\n🤔 Testing NL2SQL question: {test_question}")
        answer = await agent.chat(test_question)
        print(f"💡 Answer: {answer}")
        
        # Test a follow-up question
        follow_up = "Can you also show their contact information?"
        print(f"\n🤔 Follow-up question: {follow_up}")
        answer2 = await agent.chat(follow_up)
        print(f"💡 Follow-up Answer: {answer2}")
        
    except Exception as e:
        print(f"❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
