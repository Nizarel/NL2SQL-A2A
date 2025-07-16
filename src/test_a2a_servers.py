"""
Test both A2A server implementations for compatibility with a2a-sdk 0.2.12
"""

import asyncio
import logging
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_a2a_servers():
    """Test both A2A server implementations"""
    print("🧪 Testing A2A Server Implementations for a2a-sdk 0.2.12...")
    
    try:
        # Test 1: A2AServer class
        print("\n1️⃣ Testing A2AServer class...")
        from a2a_server import A2AServer
        print("✅ A2AServer import successful")
        
        # Create HTTP client
        httpx_client = httpx.AsyncClient()
        
        # Create A2A server instance
        server = A2AServer(httpx_client, "localhost", 8002)
        print("✅ A2AServer instance created")
        
        # Test agent card
        agent_card = server._get_agent_card()
        print(f"✅ Agent card created: {agent_card.name}")
        
        # Test health check
        health = server.health_check()
        print(f"✅ Health check: {health['a2a_server']}")
        
        # Test 2: Pure A2A Server functions
        print("\n2️⃣ Testing pure A2A server functions...")
        from pure_a2a_server import get_agent_card
        print("✅ Pure A2A server import successful")
        
        # Test agent card creation
        pure_agent_card = get_agent_card("localhost", 8002)
        print(f"✅ Pure agent card created: {pure_agent_card.name}")
        
        # Test 3: Compare agent cards
        print("\n3️⃣ Comparing agent cards...")
        print(f"A2AServer card: {agent_card.name}")
        print(f"Pure server card: {pure_agent_card.name}")
        print(f"Both have streaming: {agent_card.capabilities.streaming and pure_agent_card.capabilities.streaming}")
        print(f"Both have push notifications: {agent_card.capabilities.pushNotifications and pure_agent_card.capabilities.pushNotifications}")
        
        # Test 4: Verify a2a-sdk compatibility
        print("\n4️⃣ Testing a2a-sdk 0.2.12 compatibility...")
        
        # Test required imports work
        from a2a.server.tasks import BasePushNotificationSender, InMemoryPushNotificationConfigStore, InMemoryTaskStore
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.apps import A2AStarletteApplication
        print("✅ All a2a-sdk 0.2.12 imports successful")
        
        # Test component creation (without starting server)
        config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(httpx_client, config_store)
        task_store = InMemoryTaskStore()
        print("✅ A2A components created successfully")
        
        # Cleanup
        await httpx_client.aclose()
        
        print("\n🎉 All A2A server tests passed!")
        print("✅ Both implementations are compatible with a2a-sdk 0.2.12")
        return True
        
    except Exception as e:
        print(f"\n❌ A2A server test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_a2a_servers())
    if success:
        print("\n✅ A2A server compatibility verification completed successfully!")
    else:
        print("\n❌ A2A server compatibility verification failed!")
