#!/usr/bin/env python3
"""
Quick test to verify Cosmos DB conversation logging with populated data
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

async def test_single_question():
    """Test a single question to verify Cosmos DB conversation logging works"""
    
    print("🧪 Quick Cosmos DB Test - Single Question")
    print("=" * 60)
    
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
    
    try:
        # Test question
        question = "Show me the top 5 customers by revenue in May 2025"
        user_id = "test_user"
        session_id = f"quick_test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"❓ Question: {question}")
        print(f"👤 User: {user_id}")
        print(f"📅 Session: {session_id}")
        print()
        
        # Process question
        print("🔄 Processing question...")
        result = await nl2sql_system.orchestrator_agent.process({
            "question": question,
            "user_id": user_id,
            "session_id": session_id,
            "execute": True,
            "limit": 100,
            "include_summary": True,
            "enable_conversation_logging": True
        })
        
        print()
        print("📊 RESULTS:")
        print("-" * 40)
        
        if result.get("success"):
            print("✅ Query executed successfully!")
            
            # Check if we have formatted results
            data = result.get("data", {})
            formatted_results = data.get("formatted_results", {})
            
            if formatted_results and formatted_results.get("rows"):
                print(f"📈 Rows returned: {len(formatted_results['rows'])}")
                print("🔍 Sample data:")
                for i, row in enumerate(formatted_results["rows"][:3]):
                    print(f"   {i+1}. {row}")
            else:
                print("⚠️ No formatted results found")
            
            # Check summary
            summary = data.get("summary", {})
            if summary:
                print(f"📋 Executive Summary: {summary.get('executive_summary', 'N/A')[:100]}...")
                insights = summary.get("key_insights", [])
                print(f"💡 Key Insights: {len(insights)} insights generated")
            else:
                print("⚠️ No summary found")
                
        else:
            print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
        
        print()
        print("🔍 Checking Cosmos DB Conversation Log...")
        print("-" * 50)
        
        # Check conversation history to verify data was stored correctly
        if hasattr(nl2sql_system.orchestrator_agent, 'memory_service'):
            memory_service = nl2sql_system.orchestrator_agent.memory_service
            conversations = await memory_service.get_user_conversation_history(
                user_id=user_id,
                session_id=session_id,
                limit=1
            )
            
            if conversations:
                conv = conversations[0]
                print(f"✅ Found conversation log: {conv.id}")
                print(f"📝 User Input: {conv.user_input}")
                
                # Check formatted results
                if conv.formatted_results:
                    print(f"📊 Formatted Results: ✅ Present ({len(conv.formatted_results.rows) if conv.formatted_results.rows else 0} rows)")
                else:
                    print("📊 Formatted Results: ❌ Missing")
                
                # Check agent response
                if conv.agent_response:
                    print(f"🤖 Agent Response: ✅ Present")
                    print(f"   Executive Summary: {'✅ Present' if conv.agent_response.executive_summary else '❌ Missing'}")
                    print(f"   Key Insights: {'✅ Present' if conv.agent_response.key_insights else '❌ Missing'} ({len(conv.agent_response.key_insights or [])} insights)")
                    print(f"   Recommendations: {'✅ Present' if conv.agent_response.recommendations else '❌ Missing'} ({len(conv.agent_response.recommendations or [])} recommendations)")
                else:
                    print("🤖 Agent Response: ❌ Missing")
                
                # Check performance data
                if conv.performance:
                    print(f"⚡ Performance: ✅ Present ({conv.performance.processing_time_ms}ms)")
                else:
                    print("⚡ Performance: ❌ Missing")
                    
            else:
                print("❌ No conversation logs found in Cosmos DB")
        else:
            print("❌ Memory service not available")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await nl2sql_system.close()
        print()
        print("🔐 System closed")

if __name__ == "__main__":
    asyncio.run(test_single_question())
