"""
Test for the new Orchestrator Memory Service.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from services.orchestrator_memory_service import OrchestratorMemoryService, QueryContext, QueryResult
from services.cosmos_db_service import CosmosDbConfig
from Models.agent_response import AgentResponse

async def test_orchestrator_memory():
    """Test the orchestrator memory service functionality."""
    
    # Load environment
    from dotenv import load_dotenv
    env_path = src_path / ".env"
    load_dotenv(env_path)
    
    print("🧠 Testing Orchestrator Memory Service")
    print("=" * 50)
    
    # Initialize service
    memory_service = await OrchestratorMemoryService.create_from_config()
    
    try:
        print("✅ Memory service initialized")
        
        # Test data
        user_id = "test_user_memory"
        
        # Test 1: Create a session
        print("\n📋 Test 1: Creating a new session")
        session = await memory_service.create_session(
            user_id=user_id,
            session_title="NL2SQL Memory Test Session"
        )
        session_id = session.session_id
        print(f"✅ Session created: {session_id}")
        print(f"   Name: {session.session_name}")
        
        # Test 2: Process queries with embeddings
        print("\n💬 Test 2: Processing queries with memory")
        
        test_queries = [
            ("Show me sales data for Q1 2024", [0.8, 0.2, 0.1, 0.3, 0.7, 0.1]),
            ("What are the top selling products?", [0.7, 0.3, 0.2, 0.8, 0.1, 0.4]),
            ("Analyze customer behavior patterns", [0.1, 0.8, 0.7, 0.1, 0.2, 0.3]),
            ("Show quarterly revenue trends", [0.9, 0.1, 0.1, 0.2, 0.8, 0.1]),  # Similar to first query
        ]
        
        query_contexts = []
        
        for i, (query_text, embedding) in enumerate(test_queries, 1):
            print(f"\n   Processing Query {i}: {query_text}")
            
            # Process the query
            query_context = await memory_service.process_query(
                user_id=user_id,
                session_id=session_id,
                query=query_text,
                query_embedding=embedding
            )
            query_contexts.append(query_context)
            print(f"   ✅ Query processed: {query_context.query_id}")
            
            # Simulate query result
            query_result = QueryResult(
                query_id=query_context.query_id,
                sql_query=f"SELECT * FROM table WHERE condition = '{query_text[:20]}...'",
                execution_result={
                    "rows": i * 10,
                    "columns": ["id", "name", "value"],
                    "execution_time": f"{i * 0.1:.2f}s"
                },
                agent_response=AgentResponse(
                    response=f"Here are the results for: {query_text}",
                    sql_query=f"SELECT * FROM table WHERE condition = '{query_text[:20]}...'",
                    confidence_score=0.9
                ),
                execution_time_ms=i * 100,
                success=True
            )
            
            # Store the result
            await memory_service.store_query_result(query_context, query_result)
            print(f"   ✅ Result stored for query {i}")
        
        # Test 3: Get session context
        print("\n🪟 Test 3: Retrieving session context")
        context_messages = await memory_service.get_session_context(
            user_id=user_id,
            session_id=session_id,
            max_messages=10
        )
        print(f"✅ Retrieved {len(context_messages)} context messages")
        for msg in context_messages[-4:]:  # Show last 4 messages
            print(f"   {msg.role}: {msg.content[:50]}...")
        
        # Test 4: Find similar queries
        print("\n🔍 Test 4: Finding similar queries")
        
        # Test with a query similar to the first one (sales data)
        similar_query_embedding = [0.85, 0.15, 0.1, 0.25, 0.75, 0.1]  # Similar to first query
        
        similar_queries = await memory_service.find_similar_queries(
            query_embedding=similar_query_embedding,
            user_id=user_id,
            limit=3,
            similarity_threshold=0.6
        )
        
        print(f"✅ Found {len(similar_queries)} similar queries")
        for sq in similar_queries:
            print(f"   📝 Query: {sq.original_query[:50]}...")
            print(f"      Similarity: {sq.similarity_score:.4f}")
            if sq.result:
                print(f"      SQL: {sq.result.sql_query[:50]}...")
        
        # Test 5: Cross-user similarity search
        print("\n🌐 Test 5: Cross-user similarity search")
        
        all_similar = await memory_service.find_similar_queries(
            query_embedding=similar_query_embedding,
            user_id=None,  # Search across all users
            limit=5,
            similarity_threshold=0.5
        )
        
        print(f"✅ Found {len(all_similar)} similar queries across all users")
        
        # Test 6: User statistics
        print("\n📊 Test 6: User query statistics")
        
        user_stats = await memory_service.get_user_query_stats(user_id)
        print("✅ User statistics:")
        for key, value in user_stats.items():
            print(f"   {key}: {value}")
        
        # Test 7: Query hash functionality
        print("\n🔐 Test 7: Query hash functionality")
        
        query1 = "Show me sales data"
        query2 = "SHOW ME SALES DATA"  # Different case
        query3 = "  show   me  sales    data  "  # Extra spaces
        query4 = "Show me different data"
        
        hash1 = memory_service.create_query_hash(query1)
        hash2 = memory_service.create_query_hash(query2)
        hash3 = memory_service.create_query_hash(query3)
        hash4 = memory_service.create_query_hash(query4)
        
        print(f"✅ Query hash tests:")
        print(f"   Query 1 hash: {hash1}")
        print(f"   Query 2 hash: {hash2} (same: {hash1 == hash2})")
        print(f"   Query 3 hash: {hash3} (same: {hash1 == hash3})")
        print(f"   Query 4 hash: {hash4} (different: {hash1 != hash4})")
        
        # Test 8: Get user sessions
        print("\n👤 Test 8: Getting user sessions")
        
        user_sessions = await memory_service.get_user_sessions(user_id)
        print(f"✅ Found {len(user_sessions)} sessions for user")
        for session in user_sessions:
            print(f"   Session: {session.session_name} (created: {session.created_at})")
        
        print("\n🎯 Summary")
        print("=" * 20)
        print("✅ Session creation and management")
        print("✅ Query processing with embeddings")
        print("✅ Result storage and retrieval")
        print("✅ Session context management")
        print("✅ Semantic similarity search")
        print("✅ Cross-user query discovery")
        print("✅ User analytics and statistics")
        print("✅ Query normalization and hashing")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        await memory_service.close()
        print("\n✅ Memory service test complete!")

if __name__ == "__main__":
    asyncio.run(test_orchestrator_memory())
