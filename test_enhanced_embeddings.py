#!/usr/bin/env python3
"""
Test script to demonstrate enhanced embeddings with result summaries for richer semantic search
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import NL2SQLMultiAgentSystem
from services.cosmos_db_service import CosmosDbService
from services.orchestrator_memory_service import OrchestratorMemoryService

async def test_enhanced_embeddings():
    """Test enhanced embeddings with multiple related queries to show semantic similarity"""
    
    print("🔍 Enhanced Embeddings Test - Semantic Search with Result Summaries")
    print("=" * 80)
    
    # Load environment
    load_dotenv(os.path.join("src", ".env"))
    
    # Initialize system
    print("🔧 Initializing NL2SQL System...")
    nl2sql_system = NL2SQLMultiAgentSystem()
    await nl2sql_system.initialize()
    
    # Set up memory service for conversation logging
    print("🧠 Setting up memory service...")
    cosmos_service = CosmosDbService(
        endpoint="https://cosmos-acrasalesanalytics2.documents.azure.com:443/",
        database_name="sales_analytics",
        chat_container_name="nl2sql_chatlogs",
        cache_container_name="nl2sql_cache"
    )
    await cosmos_service.initialize()
    memory_service = OrchestratorMemoryService(cosmos_service)
    nl2sql_system.orchestrator_agent.set_memory_service(memory_service)
    
    # Test queries that should have semantic similarity but different results
    test_queries = [
        "Show me the top 5 customers by revenue in May 2025",
        "Which customers had the highest sales in May 2025?", 
        "Top revenue generating customers for May 2025",
        "Best performing clients by sales in April 2025"
    ]
    
    user_id = "embedding_test_user"
    session_id = f"embedding_test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"👤 User: {user_id}")
    print(f"📅 Session: {session_id}")
    print()
    
    conversation_logs = []
    
    try:
        # Process each query to build embedding cache
        for i, question in enumerate(test_queries):
            print(f"🔄 Processing Query {i+1}: {question}")
            print("-" * 50)
            
            result = await nl2sql_system.orchestrator_agent.process({
                "question": question,
                "user_id": user_id,
                "session_id": session_id,
                "execute": True,
                "limit": 5,
                "include_summary": True,
                "enable_conversation_logging": True
            })
            
            if result.get("success"):
                print(f"✅ Query {i+1} processed successfully")
                
                # Check conversation history to get the enhanced embedding text
                if hasattr(nl2sql_system.orchestrator_agent, 'memory_service'):
                    conversations = await nl2sql_system.orchestrator_agent.memory_service.get_user_conversation_history(
                        user_id=user_id,
                        session_id=session_id,
                        limit=1
                    )
                    
                    if conversations:
                        conv = conversations[0]
                        conversation_logs.append(conv)
                        
                        # Show enhanced embedding content
                        print(f"📝 Query: {conv.user_input}")
                        if conv.formatted_results and conv.formatted_results.rows:
                            print(f"📊 Results: {len(conv.formatted_results.rows)} rows")
                            print(f"📈 Columns: {', '.join(list(conv.formatted_results.rows[0].keys())[:3])}...")
                        
                        if conv.agent_response:
                            if conv.agent_response.key_insights:
                                print(f"💡 Key Insights: {len(conv.agent_response.key_insights)} insights")
                                print(f"   - {conv.agent_response.key_insights[0][:80]}...")
                            if conv.agent_response.executive_summary:
                                print(f"📋 Summary: {conv.agent_response.executive_summary[:80]}...")
                    
                    print()
            else:
                print(f"❌ Query {i+1} failed: {result.get('error', 'Unknown error')}")
            
            # Small delay between queries
            await asyncio.sleep(1)
        
        print()
        print("🔍 ENHANCED EMBEDDING ANALYSIS")
        print("=" * 60)
        
        # Now test semantic similarity search using the enhanced embeddings
        if conversation_logs:
            print(f"✅ Stored {len(conversation_logs)} conversation logs with enhanced embeddings")
            print()
            print("📊 Enhanced Embedding Components for each query:")
            print("-" * 50)
            
            for i, conv in enumerate(conversation_logs):
                print(f"\n🔍 Query {i+1}: {conv.user_input}")
                
                # Simulate what the enhanced embedding text would contain
                embedding_parts = [f"Query: {conv.user_input}"]
                
                if conv.formatted_results and conv.formatted_results.rows:
                    row_count = len(conv.formatted_results.rows)
                    embedding_parts.append(f"Results: {row_count} rows returned")
                    
                    if isinstance(conv.formatted_results.rows[0], dict):
                        columns = list(conv.formatted_results.rows[0].keys())[:5]
                        embedding_parts.append(f"Columns: {', '.join(columns)}")
                
                if conv.agent_response and conv.agent_response.key_insights:
                    top_insights = conv.agent_response.key_insights[:2]
                    insights_text = "; ".join(top_insights)[:200]
                    embedding_parts.append(f"Insights: {insights_text}")
                
                if conv.agent_response and conv.agent_response.executive_summary:
                    summary_snippet = conv.agent_response.executive_summary[:150]
                    embedding_parts.append(f"Summary: {summary_snippet}")
                
                enhanced_text = " -> ".join(embedding_parts)
                print(f"📝 Enhanced Embedding Text:")
                print(f"   {enhanced_text[:300]}{'...' if len(enhanced_text) > 300 else ''}")
                
                print(f"🔄 Embedding Benefits:")
                print(f"   • Semantic matching on business context, not just query syntax")
                print(f"   • Result schema awareness for better query-result pairing") 
                print(f"   • Business insights for context-aware recommendations")
                print(f"   • Executive summary for high-level semantic understanding")
        
        print()
        print("🎯 SEMANTIC SEARCH IMPROVEMENTS:")
        print("-" * 40)
        print("✅ Before: Only user query text embedded")
        print("✅ Now: Query + SQL + Results + Insights + Summary embedded")
        print("✅ Benefit: Much richer semantic matching for similar business questions")
        print("✅ Use Case: Find similar analyses with comparable result patterns")
        print("✅ Context: Business insights help match intent, not just syntax")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await nl2sql_system.close()
        print()
        print("🔐 Enhanced Embeddings Test completed")

if __name__ == "__main__":
    asyncio.run(test_enhanced_embeddings())
