"""
A2A Client for testing NL2SQL Agent-to-Agent communication
Demonstrates how to interact with A2A agents using JSON-RPC protocol
"""

import asyncio
import json
import aiohttp
import time
from typing import Dict, Any, Optional


class A2AClient:
    """Client for communicating with A2A agents using JSON-RPC protocol"""
    
    def __init__(self, base_urls: Optional[Dict[str, str]] = None):
        if base_urls is None:
            # Default localhost URLs
            self.base_urls = {
                'orchestrator': 'http://localhost:8100',
                'sql_generator': 'http://localhost:8101',
                'executor': 'http://localhost:8102',
                'summarizer': 'http://localhost:8103'
            }
        else:
            self.base_urls = base_urls
    
    async def get_agent_card(self, agent_type: str) -> Dict[str, Any]:
        """Get agent card information"""
        url = f"{self.base_urls[agent_type]}/.well-known/agent.json"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get agent card: {response.status}")
    
    async def send_message(self, agent_type: str, content: Any, context_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a message to an agent using A2A protocol"""
        url = self.base_urls[agent_type]
        
        # Create JSON-RPC request with proper A2A message format
        request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": str(int(time.time() * 1000)),  # Use timestamp as ID
            "params": {
                "message": {
                    "role": "user",
                    "kind": "message",
                    "messageId": f"msg-{int(time.time() * 1000)}",
                    "parts": [
                        {
                            "kind": "text",
                            "text": content if isinstance(content, str) else json.dumps(content)
                        }
                    ],
                    "timestamp": time.time()
                }
            }
        }
        
        if context_id:
            request["params"]["message"]["contextId"] = context_id
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=request,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    raise Exception(f"Request failed: {response.status} - {text}")
    
    async def get_task(self, agent_type: str, task_id: str) -> Dict[str, Any]:
        """Get task status and results"""
        url = self.base_urls[agent_type]
        
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "id": str(int(time.time() * 1000)),
            "params": {
                "id": task_id
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=request,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    raise Exception(f"Request failed: {response.status} - {text}")
    
    async def wait_for_completion(self, agent_type: str, task_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Wait for task completion with polling"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = await self.get_task(agent_type, task_id)
            
            if 'result' in result:
                task = result['result']
                status = task.get('status', {})
                state = status.get('state', 'unknown')
                
                if state in ['completed', 'failed', 'cancelled']:
                    return result
            
            await asyncio.sleep(1)  # Poll every second
        
        raise Exception(f"Task {task_id} did not complete within {timeout} seconds")


async def test_orchestrator_workflow():
    """Test complete workflow through orchestrator"""
    print("🎯 Testing Orchestrator Workflow")
    print("=" * 40)
    
    client = A2AClient()
    
    try:
        # Test agent card
        print("📋 Getting agent card...")
        card = await client.get_agent_card('orchestrator')
        print(f"✅ Agent: {card['name']}")
        print(f"   Skills: {len(card.get('skills', []))} available")
        print()
        
        # Send question
        question = "Show me the top 5 customers by revenue"
        print(f"💬 Sending question: {question}")
        
        response = await client.send_message('orchestrator', question)
        
        if 'result' in response:
            task = response['result']
            task_id = task['id']
            print(f"✅ Task created: {task_id}")
            print(f"   Status: {task['status']['state']}")
            
            # Wait for completion
            print("⏳ Waiting for completion...")
            final_result = await client.wait_for_completion('orchestrator', task_id)
            
            if 'result' in final_result:
                final_task = final_result['result']
                print(f"✅ Task completed: {final_task['status']['state']}")
                
                # Show artifacts
                artifacts = final_task.get('artifacts', [])
                if artifacts:
                    print(f"📎 Generated {len(artifacts)} artifacts:")
                    for artifact in artifacts:
                        print(f"   • {artifact['name']} ({artifact['type']})")
                
                # Show final message
                history = final_task.get('history', [])
                if history:
                    last_message = history[-1]
                    if last_message.get('role') == 'assistant':
                        try:
                            content = json.loads(last_message['content'])
                            if content.get('success'):
                                print("✅ Orchestrator workflow completed successfully!")
                            else:
                                print(f"❌ Workflow failed: {content.get('error')}")
                        except json.JSONDecodeError:
                            print(f"📝 Response: {last_message['content'][:100]}...")
                
            else:
                print(f"❌ Error: {final_result.get('error', 'Unknown error')}")
        
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")


async def test_individual_agents():
    """Test individual agents separately"""
    print("\n🤖 Testing Individual Agents")
    print("=" * 40)
    
    client = A2AClient()
    
    # Test SQL Generator
    try:
        print("🧠 Testing SQL Generator...")
        response = await client.send_message('sql_generator', {
            "question": "What are the top selling products?",
            "context": "Focus on revenue analysis"
        })
        
        if 'result' in response:
            task_id = response['result']['id']
            result = await client.wait_for_completion('sql_generator', task_id)
            print("✅ SQL Generator completed")
        else:
            print(f"❌ SQL Generator error: {response.get('error')}")
    
    except Exception as e:
        print(f"❌ SQL Generator test failed: {e}")
    
    # Test Executor
    try:
        print("🔧 Testing Executor...")
        response = await client.send_message('executor', {
            "sql_query": "SELECT COUNT(*) as total_records FROM dev.cliente",
            "limit": 10
        })
        
        if 'result' in response:
            task_id = response['result']['id']
            result = await client.wait_for_completion('executor', task_id)
            print("✅ Executor completed")
        else:
            print(f"❌ Executor error: {response.get('error')}")
    
    except Exception as e:
        print(f"❌ Executor test failed: {e}")


async def test_agent_cards():
    """Test all agent cards"""
    print("\n📋 Testing Agent Cards")
    print("=" * 40)
    
    client = A2AClient()
    
    for agent_type in ['orchestrator', 'sql_generator', 'executor', 'summarizer']:
        try:
            card = await client.get_agent_card(agent_type)
            print(f"✅ {agent_type.title()}: {card['name']}")
            print(f"   Description: {card['description'][:80]}...")
            print(f"   Skills: {len(card.get('skills', []))}")
            print()
        except Exception as e:
            print(f"❌ {agent_type} card failed: {e}")


async def main():
    """Main test function"""
    print("🧪 NL2SQL A2A Client Test Suite")
    print("=" * 50)
    print()
    
    # Wait a moment for servers to be ready
    print("⏳ Waiting for servers to be ready...")
    await asyncio.sleep(2)
    
    # Test agent cards first
    await test_agent_cards()
    
    # Test individual agents
    await test_individual_agents()
    
    # Test complete workflow
    await test_orchestrator_workflow()
    
    print("\n🎉 Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main())
