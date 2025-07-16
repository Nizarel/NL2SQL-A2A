"""
End-to-End Test for NL2SQL A2A Orchestrator Agent
Tests the complete A2A protocol integration with real queries
"""

import asyncio
import httpx
import logging
import time
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# A2A SDK imports
from a2a.client import A2AClient
from a2a.client.helpers import create_text_message_object
from a2a.types import SendMessageRequest, SendStreamingMessageRequest, Role

# Local imports
from a2a_server import A2AServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class A2AEndToEndTester:
    """Comprehensive end-to-end tester for A2A NL2SQL Orchestrator"""
    
    def __init__(self):
        self.a2a_server = None
        self.a2a_client = None
        self.httpx_client = None
        self.server_host = "localhost"
        self.server_port = 8002  # Use different port for testing
        self.test_queries = [
            'Analyze revenue by region and show which region performs best in 2025?',
            'Show the top performing distribution centers (CEDIs) by total sales in 2025',
            "Generate a query to find customers who haven't made purchases in the last 6 months?",
            'Which products have declining sales trends and in which regions in May 2025?',
            'What are the top 5 products by sales in the last quarter?',
        ]
        
    async def setup_test_environment(self):
        """Setup A2A server and client for testing"""
        try:
            logger.info("🚀 Setting up A2A test environment...")
            
            # Create HTTP client
            self.httpx_client = httpx.AsyncClient(timeout=60.0)
            
            # Initialize A2A server
            self.a2a_server = A2AServer(
                httpx_client=self.httpx_client,
                host=self.server_host,
                port=self.server_port
            )
            
            # Get agent card for client initialization
            agent_card = self.a2a_server._get_agent_card()
            
            # Initialize A2A client
            self.a2a_client = A2AClient(
                httpx_client=self.httpx_client,
                agent_card=agent_card
            )
            
            logger.info(f"✅ A2A test environment setup complete")
            logger.info(f"🔗 Server URL: {agent_card.url}")
            logger.info(f"🤖 Agent: {agent_card.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {str(e)}")
            return False
    
    async def test_agent_card_retrieval(self) -> bool:
        """Test agent card retrieval and validation"""
        try:
            logger.info("🧪 Testing agent card retrieval...")
            
            agent_card = self.a2a_server._get_agent_card()
            
            # Validate agent card structure
            assert agent_card.name == "NL2SQL Orchestrator Agent"
            assert len(agent_card.skills) == 1
            assert agent_card.skills[0].id == "nl2sql_orchestration"
            assert len(agent_card.skills[0].examples) >= 5
            assert agent_card.capabilities.streaming is True
            
            logger.info("✅ Agent card validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Agent card test failed: {str(e)}")
            return False
    
    async def test_health_check(self) -> bool:
        """Test A2A server health check"""
        try:
            logger.info("🧪 Testing A2A server health check...")
            
            health_status = self.a2a_server.health_check()
            
            # Validate health status
            assert health_status["a2a_server"] == "healthy"
            assert health_status["starlette_app"] == "ready"
            assert health_status["capabilities"]["streaming"] is True
            
            logger.info("✅ Health check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return False
    
    async def test_single_query(self, query: str) -> Dict[str, Any]:
        """Test a single NL2SQL query through A2A protocol"""
        try:
            logger.info(f"🧪 Testing query: {query[:50]}...")
            
            start_time = time.time()
            
            # Create message object
            message = create_text_message_object(
                role=Role.user,
                content=query
            )
            
            # Create send message request  
            request = SendMessageRequest(
                message=message
            )
            
            # Send message through A2A client
            response = await self.a2a_client.send_message(request)
            
            execution_time = time.time() - start_time
            
            # Validate response
            if hasattr(response, 'task') and response.task:
                logger.info(f"✅ Query processed successfully in {execution_time:.2f}s")
                logger.info(f"📋 Task ID: {response.task.id}")
                logger.info(f"📊 Task State: {response.task.status.state if response.task.status else 'Unknown'}")
                
                return {
                    "success": True,
                    "query": query,
                    "task_id": response.task.id,
                    "execution_time": execution_time,
                    "response": response
                }
            else:
                logger.warning(f"⚠️ Unexpected response format for query: {query[:30]}...")
                return {
                    "success": False,
                    "query": query,
                    "error": "Unexpected response format",
                    "response": response
                }
                
        except Exception as e:
            logger.error(f"❌ Query test failed: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    async def test_streaming_query(self, query: str) -> Dict[str, Any]:
        """Test a streaming NL2SQL query through A2A protocol"""
        try:
            logger.info(f"🧪 Testing streaming query: {query[:50]}...")
            
            start_time = time.time()
            streaming_events = []
            
            # Create message object
            message = create_text_message_object(
                role=Role.user,
                content=query
            )
            
            # Create streaming send message request
            request = SendStreamingMessageRequest(
                message=message
            )
            
            # Send streaming message through A2A client
            async for response in self.a2a_client.send_message_streaming(request):
                streaming_events.append({
                    "timestamp": time.time(),
                    "event_type": type(response).__name__,
                    "content": str(response)[:200]  # Truncate for logging
                })
                logger.info(f"📡 Streaming event: {type(response).__name__}")
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Streaming query completed in {execution_time:.2f}s")
            logger.info(f"📊 Total streaming events: {len(streaming_events)}")
            
            return {
                "success": True,
                "query": query,
                "execution_time": execution_time,
                "streaming_events": len(streaming_events),
                "events": streaming_events
            }
                
        except Exception as e:
            logger.error(f"❌ Streaming query test failed: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite for A2A NL2SQL integration"""
        logger.info("🎯 Starting comprehensive A2A end-to-end test suite...")
        
        test_results = {
            "test_start_time": time.time(),
            "setup_success": False,
            "agent_card_test": False,
            "health_check_test": False,
            "query_tests": [],
            "streaming_tests": [],
            "summary": {}
        }
        
        try:
            # 1. Setup test environment
            test_results["setup_success"] = await self.setup_test_environment()
            if not test_results["setup_success"]:
                logger.error("❌ Test suite aborted due to setup failure")
                return test_results
            
            # 2. Test agent card retrieval
            test_results["agent_card_test"] = await self.test_agent_card_retrieval()
            
            # 3. Test health check
            test_results["health_check_test"] = await self.test_health_check()
            
            # 4. Test individual queries (non-streaming)
            logger.info("🧪 Testing non-streaming queries...")
            for i, query in enumerate(self.test_queries, 1):
                logger.info(f"📝 Testing query {i}/{len(self.test_queries)}")
                result = await self.test_single_query(query)
                test_results["query_tests"].append(result)
                
                # Add delay between queries to avoid overwhelming the system
                await asyncio.sleep(1)
            
            # 5. Test streaming queries (subset for performance)
            logger.info("🧪 Testing streaming queries...")
            streaming_test_queries = self.test_queries[:2]  # Test first 2 queries for streaming
            for i, query in enumerate(streaming_test_queries, 1):
                logger.info(f"📡 Testing streaming query {i}/{len(streaming_test_queries)}")
                result = await self.test_streaming_query(query)
                test_results["streaming_tests"].append(result)
                
                # Add delay between streaming tests
                await asyncio.sleep(2)
            
            # 6. Generate test summary
            test_results["summary"] = self._generate_test_summary(test_results)
            
            logger.info("🎉 Comprehensive test suite completed!")
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {str(e)}")
            test_results["summary"] = {"error": str(e)}
        
        finally:
            await self.cleanup_test_environment()
        
        return test_results
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        total_execution_time = time.time() - test_results["test_start_time"]
        
        query_tests = test_results["query_tests"]
        streaming_tests = test_results["streaming_tests"]
        
        successful_queries = sum(1 for test in query_tests if test.get("success"))
        successful_streaming = sum(1 for test in streaming_tests if test.get("success"))
        
        avg_query_time = (
            sum(test.get("execution_time", 0) for test in query_tests if test.get("success")) / 
            max(successful_queries, 1)
        )
        
        avg_streaming_time = (
            sum(test.get("execution_time", 0) for test in streaming_tests if test.get("success")) / 
            max(successful_streaming, 1)
        )
        
        summary = {
            "total_execution_time": round(total_execution_time, 2),
            "setup_success": test_results["setup_success"],
            "agent_card_test": test_results["agent_card_test"],
            "health_check_test": test_results["health_check_test"],
            "query_tests": {
                "total": len(query_tests),
                "successful": successful_queries,
                "failed": len(query_tests) - successful_queries,
                "success_rate": round((successful_queries / max(len(query_tests), 1)) * 100, 1),
                "avg_execution_time": round(avg_query_time, 2)
            },
            "streaming_tests": {
                "total": len(streaming_tests),
                "successful": successful_streaming,
                "failed": len(streaming_tests) - successful_streaming,
                "success_rate": round((successful_streaming / max(len(streaming_tests), 1)) * 100, 1),
                "avg_execution_time": round(avg_streaming_time, 2)
            },
            "overall_success": (
                test_results["setup_success"] and
                test_results["agent_card_test"] and
                test_results["health_check_test"] and
                successful_queries == len(query_tests) and
                successful_streaming == len(streaming_tests)
            )
        }
        
        return summary
    
    async def cleanup_test_environment(self):
        """Cleanup test environment"""
        try:
            if self.httpx_client:
                await self.httpx_client.aclose()
            logger.info("✅ Test environment cleanup complete")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {str(e)}")
    
    def print_test_results(self, test_results: Dict[str, Any]):
        """Print formatted test results"""
        print("\n" + "="*80)
        print("🎯 A2A NL2SQL END-TO-END TEST RESULTS")
        print("="*80)
        
        summary = test_results.get("summary", {})
        
        print(f"⏱️  Total Execution Time: {summary.get('total_execution_time', 0)}s")
        print(f"🔧 Setup Success: {'✅' if summary.get('setup_success') else '❌'}")
        print(f"📋 Agent Card Test: {'✅' if summary.get('agent_card_test') else '❌'}")
        print(f"💚 Health Check: {'✅' if summary.get('health_check_test') else '❌'}")
        
        # Query tests summary
        query_summary = summary.get("query_tests", {})
        print(f"\n📝 NON-STREAMING QUERY TESTS:")
        print(f"   Total: {query_summary.get('total', 0)}")
        print(f"   Successful: {query_summary.get('successful', 0)}")
        print(f"   Failed: {query_summary.get('failed', 0)}")
        print(f"   Success Rate: {query_summary.get('success_rate', 0)}%")
        print(f"   Avg Execution Time: {query_summary.get('avg_execution_time', 0)}s")
        
        # Streaming tests summary
        streaming_summary = summary.get("streaming_tests", {})
        print(f"\n📡 STREAMING QUERY TESTS:")
        print(f"   Total: {streaming_summary.get('total', 0)}")
        print(f"   Successful: {streaming_summary.get('successful', 0)}")
        print(f"   Failed: {streaming_summary.get('failed', 0)}")
        print(f"   Success Rate: {streaming_summary.get('success_rate', 0)}%")
        print(f"   Avg Execution Time: {streaming_summary.get('avg_execution_time', 0)}s")
        
        # Overall result
        overall_success = summary.get("overall_success", False)
        print(f"\n🎉 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        # Detailed query results
        print(f"\n📊 DETAILED QUERY RESULTS:")
        for i, test in enumerate(test_results.get("query_tests", []), 1):
            status = "✅" if test.get("success") else "❌"
            query = test.get("query", "Unknown")[:50]
            exec_time = test.get("execution_time", 0)
            print(f"   {i}. {status} {query}... ({exec_time:.2f}s)")
        
        print("="*80)


async def main():
    """Main test runner"""
    print("🚀 Starting A2A NL2SQL End-to-End Test Suite")
    print("This will test the complete A2A protocol integration with real queries")
    
    tester = A2AEndToEndTester()
    
    try:
        # Run comprehensive test suite
        test_results = await tester.run_comprehensive_test_suite()
        
        # Print results
        tester.print_test_results(test_results)
        
        # Return exit code based on overall success
        overall_success = test_results.get("summary", {}).get("overall_success", False)
        return 0 if overall_success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test suite failed with unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
