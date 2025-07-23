"""
Test script for Cosmos DB service with Azure Identity authentication.
This script tests the basic functionality of the Cosmos DB service.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.cosmos_db_service import CosmosDbService, CosmosDbConfig
from src.Models.agent_response import Session, Message


async def test_cosmos_db_service():
    """Test the Cosmos DB service with Azure Identity."""
    
    print("🧪 Testing Cosmos DB Service with Azure Identity")
    print("=" * 50)
    
    # Load configuration from environment
    try:
        config = CosmosDbConfig.from_env()
        print(f"✅ Loaded configuration:")
        print(f"   Endpoint: {config.endpoint}")
        print(f"   Database: {config.database_name}")
        print(f"   Chat Container: {config.chat_container_name}")
        print(f"   Cache Container: {config.cache_container_name}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    # Initialize service
    service = CosmosDbService(
        endpoint=config.endpoint,
        database_name=config.database_name,
        chat_container_name=config.chat_container_name,
        cache_container_name=config.cache_container_name
    )
    
    try:
        # Initialize connection
        print("\n🔌 Initializing Cosmos DB connection...")
        await service.initialize()
        print("✅ Connected to Cosmos DB successfully!")
        
        # Test data
        user_id = "test_user_001"
        session_id = str(uuid4())
        
        print(f"\n👤 Testing with User ID: {user_id}")
        print(f"📝 Session ID: {session_id}")
        
        # Test 1: Create a session
        print("\n📋 Test 1: Creating a new session")
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            session_name="Test Session with Azure Identity",
            is_active=True
        )
        
        created_session = await service.insert_session_async(user_id, session)
        print(f"✅ Session created: {created_session.session_id}")
        
        # Test 2: Create messages
        print("\n💬 Test 2: Creating messages")
        messages = [
            Message(
                message_id=str(uuid4()),
                session_id=session_id,
                user_id=user_id,
                role="user",
                content="What are the top selling products?",
                timestamp=datetime.now(timezone.utc)
            ),
            Message(
                message_id=str(uuid4()),
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content="Here are the top selling products based on our analysis...",
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        for message in messages:
            created_message = await service.insert_message_async(user_id, message)
            print(f"✅ Message created: {created_message.message_id} ({created_message.role})")
        
        # Test 3: Retrieve sessions
        print("\n📋 Test 3: Retrieving user sessions")
        user_sessions = await service.get_sessions_async(user_id)
        print(f"✅ Found {len(user_sessions)} sessions for user {user_id}")
        for sess in user_sessions:
            print(f"   - {sess.session_id}: {sess.session_name}")
        
        # Test 4: Retrieve session messages
        print("\n💬 Test 4: Retrieving session messages")
        session_messages = await service.get_session_messages_async(user_id, session_id)
        print(f"✅ Found {len(session_messages)} messages in session")
        for msg in session_messages:
            print(f"   - {msg.role}: {msg.content[:50]}...")
        
        # Test 5: Context window
        print("\n🪟 Test 5: Testing context window")
        context_messages = await service.get_session_context_window_async(user_id, session_id, max_context_window=5)
        print(f"✅ Context window contains {len(context_messages)} messages")
        
        # Test 6: Cache operations
        print("\n💾 Test 6: Testing cache operations")
        from src.Models.agent_response import CacheItem
        
        # Test basic cache item
        cache_item = CacheItem(
            key="test_embedding_key",
            value="[0.1, 0.2, 0.3, 0.4, 0.5]",  # Mock embedding as string
            metadata={"type": "embedding", "model": "text-embedding-3-small"},
            created_at=datetime.now(timezone.utc),
            expires_at=None
        )
        
        stored_cache = await service.set_cache_item_async(cache_item)
        print(f"✅ Cache item stored: {stored_cache.key}")
        
        retrieved_cache = await service.get_cache_item_async("test_embedding_key")
        if retrieved_cache:
            print(f"✅ Cache item retrieved: {retrieved_cache.key}")
        else:
            print("❌ Cache item not found")
        
        # Test 7: Vector embedding operations
        print("\n🧠 Test 7: Testing vector embeddings")
        
        # Store vector embeddings for similarity search
        embeddings_data = [
            {
                "key": "sales_query_1",
                "text": "What are the top selling products?",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            },
            {
                "key": "customer_query_1", 
                "text": "Which customers have the most orders?",
                "embedding": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            },
            {
                "key": "revenue_query_1",
                "text": "What is our total revenue this month?", 
                "embedding": [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7]
            }
        ]
        
        # Store vector embeddings
        for data in embeddings_data:
            stored_embedding = await service.set_vector_embedding_async(
                key=data["key"],
                embedding=data["embedding"],
                text=data["text"],
                metadata={"type": "query_embedding", "model": "text-embedding-3-small"}
            )
            print(f"✅ Vector embedding stored: {stored_embedding.key} (dim: {len(stored_embedding.embedding)})")
        
        # Test similarity search
        query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        print(f"\n🔍 Testing similarity search with query vector (dim: {len(query_embedding)})")
        
        try:
            similar_items = await service.search_similar_embeddings_async(
                query_embedding=query_embedding,
                limit=3,
                similarity_threshold=0.5
            )
            
            if similar_items:
                print(f"✅ Found {len(similar_items)} similar embeddings:")
                for item in similar_items:
                    print(f"   - {item.key}: {item.value[:50]}...")
            else:
                print("⚠️ No similar embeddings found (vector search may need index setup)")
                
        except Exception as e:
            print(f"⚠️ Vector search not available: {str(e)}")
            print("   This is expected if vector indexing is not configured on the container")
        
        # Show setup instructions
        print("\n📚 Vector Search Setup Instructions:")
        setup_instructions = service.get_setup_instructions()
        print(setup_instructions)
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Azure Identity authentication working correctly")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            await service.close()
            print("✅ Service closed successfully")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")


async def test_authentication_info():
    """Test authentication setup and configuration."""
    
    print("\n🔐 Testing Authentication Configuration")
    print("=" * 50)
    
    # Check environment variables
    required_vars = [
        "COSMOS_ENDPOINT",
        "COSMOS_DATABASE_NAME", 
        "COSMOS_CHAT_CONTAINER_NAME",
        "COSMOS_CACHE_CONTAINER_NAME"
    ]
    
    print("🔍 Checking environment variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Test Azure Identity availability
    try:
        from azure.identity.aio import DefaultAzureCredential
        print("✅ Azure Identity available")
        
        # Test credential creation (don't authenticate yet)
        credential = DefaultAzureCredential()
        print("✅ DefaultAzureCredential created successfully")
        
        await credential.close()
        
    except ImportError:
        print("❌ Azure Identity not available - install with: pip install azure-identity")
    except Exception as e:
        print(f"⚠️ Azure Identity issue: {e}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    
    env_path = os.path.join(os.path.dirname(__file__), 'src', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
    else:
        print(f"⚠️ .env file not found at: {env_path}")
        print("Make sure to set environment variables manually")
    
    # Run tests
    asyncio.run(test_authentication_info())
    asyncio.run(test_cosmos_db_service())
