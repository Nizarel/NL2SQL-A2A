"""
Simplified A2A End-to-End Test Runner
Starts the A2A server and tests it with real queries
"""

import asyncio
import httpx
import logging
import time
import uvicorn
import threading
from multiprocessing import Process
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test queries from the agent card examples
TEST_QUERIES = [
    'Analyze revenue by region and show which region performs best in 2025?',
    'Show the top performing distribution centers (CEDIs) by total sales in 2025',
    "Generate a query to find customers who haven't made purchases in the last 6 months?",
    'Which products have declining sales trends and in which regions in May 2025?',
    'What are the top 5 products by sales in the last quarter?',
]


class SimpleA2ATester:
    """Simplified A2A tester that works with the actual server"""
    
    def __init__(self, server_url: str = "http://localhost:8004"):
        self.server_url = server_url
        self.httpx_client = None
        
    async def setup(self):
        """Setup HTTP client"""
        self.httpx_client = httpx.AsyncClient(timeout=60.0)
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.httpx_client:
            await self.httpx_client.aclose()
    
    async def test_server_health(self) -> bool:
        """Test if the A2A server is healthy and responding"""
        try:
            response = await self.httpx_client.get(f"{self.server_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Server health check passed: {health_data}")
                return True
            else:
                logger.error(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to server: {str(e)}")
            return False
    
    async def test_agent_card(self) -> bool:
        """Test agent card retrieval"""
        try:
            response = await self.httpx_client.get(f"{self.server_url}/agent-card")
            if response.status_code == 200:
                agent_card = response.json()
                logger.info(f"✅ Agent card retrieved: {agent_card.get('name', 'Unknown')}")
                logger.info(f"📋 Skills: {len(agent_card.get('skills', []))}")
                logger.info(f"💡 Examples: {len(agent_card.get('skills', [{}])[0].get('examples', []))}")
                return True
            else:
                logger.error(f"❌ Agent card retrieval failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Agent card test failed: {str(e)}")
            return False
    
    async def test_a2a_message(self, query: str) -> Dict[str, Any]:
        """Test sending a message through A2A protocol"""
        try:
            logger.info(f"🧪 Testing A2A message: {query[:50]}...")
            
            start_time = time.time()
            
            # A2A message format (simplified JSON-RPC)
            message_data = {
                "jsonrpc": "2.0",
                "method": "send_message",
                "params": {
                    "message": {
                        "messageId": f"test-{int(time.time() * 1000)}",
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": query
                            }
                        ]
                    }
                },
                "id": f"test-{int(time.time() * 1000)}"
            }
            
            # Send A2A message
            response = await self.httpx_client.post(
                f"{self.server_url}/a2a/rpc",
                json=message_data,
                headers={"Content-Type": "application/json"}
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ A2A message processed successfully in {execution_time:.2f}s")
                logger.info(f"📋 Response ID: {result.get('id', 'Unknown')}")
                
                # Check if we got a task response
                if "result" in result and "task" in result["result"]:
                    task = result["result"]["task"]
                    logger.info(f"📊 Task created: {task.get('id', 'Unknown')}")
                    logger.info(f"🎯 Task status: {task.get('status', {}).get('state', 'Unknown')}")
                    
                return {
                    "success": True,
                    "query": query,
                    "execution_time": execution_time,
                    "response": result
                }
            else:
                logger.error(f"❌ A2A message failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {
                    "success": False,
                    "query": query,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "execution_time": execution_time
                }
                
        except Exception as e:
            logger.error(f"❌ A2A message test failed: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🎯 Starting A2A End-to-End Test Suite")
        
        test_results = {
            "start_time": time.time(),
            "server_health": False,
            "agent_card": False,
            "message_tests": [],
            "summary": {}
        }
        
        try:
            await self.setup()
            
            # Test 1: Server Health
            logger.info("🧪 Testing server health...")
            test_results["server_health"] = await self.test_server_health()
            
            if not test_results["server_health"]:
                logger.error("❌ Server health check failed, aborting tests")
                return test_results
            
            # Test 2: Agent Card
            logger.info("🧪 Testing agent card retrieval...")
            test_results["agent_card"] = await self.test_agent_card()
            
            # Test 3: A2A Messages
            logger.info("🧪 Testing A2A messages...")
            for i, query in enumerate(TEST_QUERIES, 1):
                logger.info(f"📝 Testing query {i}/{len(TEST_QUERIES)}")
                result = await self.test_a2a_message(query)
                test_results["message_tests"].append(result)
                
                # Small delay between tests
                await asyncio.sleep(1)
            
            # Generate summary
            test_results["summary"] = self._generate_summary(test_results)
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {str(e)}")
            test_results["summary"] = {"error": str(e)}
        
        finally:
            await self.cleanup()
        
        return test_results
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary"""
        total_time = time.time() - test_results["start_time"]
        message_tests = test_results["message_tests"]
        
        successful_messages = sum(1 for test in message_tests if test.get("success"))
        
        avg_execution_time = (
            sum(test.get("execution_time", 0) for test in message_tests if test.get("success")) /
            max(successful_messages, 1)
        )
        
        return {
            "total_execution_time": round(total_time, 2),
            "server_health": test_results["server_health"],
            "agent_card": test_results["agent_card"],
            "message_tests": {
                "total": len(message_tests),
                "successful": successful_messages,
                "failed": len(message_tests) - successful_messages,
                "success_rate": round((successful_messages / max(len(message_tests), 1)) * 100, 1),
                "avg_execution_time": round(avg_execution_time, 2)
            },
            "overall_success": (
                test_results["server_health"] and
                test_results["agent_card"] and
                successful_messages == len(message_tests)
            )
        }
    
    def print_results(self, test_results: Dict[str, Any]):
        """Print formatted test results"""
        print("\n" + "="*80)
        print("🎯 A2A NL2SQL END-TO-END TEST RESULTS")
        print("="*80)
        
        summary = test_results.get("summary", {})
        
        print(f"⏱️  Total Execution Time: {summary.get('total_execution_time', 0)}s")
        print(f"💚 Server Health: {'✅' if summary.get('server_health') else '❌'}")
        print(f"📋 Agent Card: {'✅' if summary.get('agent_card') else '❌'}")
        
        # Message tests summary
        message_summary = summary.get("message_tests", {})
        print(f"\n📝 A2A MESSAGE TESTS:")
        print(f"   Total: {message_summary.get('total', 0)}")
        print(f"   Successful: {message_summary.get('successful', 0)}")
        print(f"   Failed: {message_summary.get('failed', 0)}")
        print(f"   Success Rate: {message_summary.get('success_rate', 0)}%")
        print(f"   Avg Execution Time: {message_summary.get('avg_execution_time', 0)}s")
        
        # Overall result
        overall_success = summary.get("overall_success", False)
        print(f"\n🎉 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        # Detailed results
        print(f"\n📊 DETAILED TEST RESULTS:")
        for i, test in enumerate(test_results.get("message_tests", []), 1):
            status = "✅" if test.get("success") else "❌"
            query = test.get("query", "Unknown")[:50]
            exec_time = test.get("execution_time", 0)
            print(f"   {i}. {status} {query}... ({exec_time:.2f}s)")
            if not test.get("success") and test.get("error"):
                print(f"      Error: {test.get('error')}")
        
        print("="*80)


async def wait_for_server(url: str, max_wait: int = 30) -> bool:
    """Wait for server to be ready"""
    async with httpx.AsyncClient() as client:
        for _ in range(max_wait):
            try:
                response = await client.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    return True
            except:
                pass
            await asyncio.sleep(1)
    return False


async def main():
    """Main test runner"""
    print("🚀 A2A NL2SQL End-to-End Test")
    print("="*50)
    
    server_url = "http://localhost:8004"
    
    # Check if server is already running
    print("🔍 Checking if A2A server is running...")
    tester = SimpleA2ATester(server_url)
    await tester.setup()
    
    server_running = await tester.test_server_health()
    await tester.cleanup()
    
    if not server_running:
        print("❌ A2A server is not running!")
        print("💡 Please start the A2A server first:")
        print("   cd src && python start_a2a.py")
        print("   Then run this test again.")
        return 1
    
    print("✅ A2A server is running, proceeding with tests...")
    
    # Run test suite
    tester = SimpleA2ATester(server_url)
    test_results = await tester.run_full_test_suite()
    
    # Print results
    tester.print_results(test_results)
    
    # Return appropriate exit code
    overall_success = test_results.get("summary", {}).get("overall_success", False)
    return 0 if overall_success else 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
