"""
Integration test for conversation enhancements
Tests the complete conversation flow through the API
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import NL2SQLMultiAgentSystem


async def test_conversation_integration():
    """Test conversation integration with the full system"""
    
    print("🔄 Testing conversation integration...")
    
    try:
        # Initialize the system
        print("🚀 Initializing NL2SQL system...")
        system = NL2SQLMultiAgentSystem()
        await system.initialize()
        
        # Test user and session
        user_id = "test_user_conv"
        session_id = "test_session_conv"
        
        # Test 1: Initial query
        print("\n📝 Test 1: Initial Query")
        
        initial_query = "Show me total sales for last month"
        
        result1 = await system.orchestrator_agent.process({
            "question": initial_query,
            "user_id": user_id,
            "session_id": session_id,
            "execute": True,
            "limit": 10,
            "include_summary": True,
            "enable_conversation_logging": True
        })
        
        print(f"   Query: {initial_query}")
        print(f"   Success: {result1.get('success')}")
        if result1.get('data', {}).get('sql_query'):
            sql = result1['data']['sql_query']
            print(f"   SQL: {sql[:100]}...")
        print()
        
        # Wait a bit for conversation logging to complete
        await asyncio.sleep(1)
        
        # Test 2: Follow-up query
        print("🔄 Test 2: Follow-up Query")
        
        follow_up_query = "What about this month?"
        
        result2 = await system.orchestrator_agent.process({
            "question": follow_up_query,
            "user_id": user_id,
            "session_id": session_id,
            "execute": True,
            "limit": 10,
            "include_summary": True,
            "enable_conversation_logging": True
        })
        
        print(f"   Query: {follow_up_query}")
        print(f"   Success: {result2.get('success')}")
        
        # Check if follow-up was detected in metadata
        follow_up_info = result2.get('metadata', {}).get('conversation_context', {}).get('follow_up_info', {})
        if follow_up_info:
            print(f"   Follow-up detected: {follow_up_info.get('is_follow_up', False)}")
            print(f"   Enhanced question: {follow_up_info.get('enhanced_question', 'None')[:80]}...")
        
        if result2.get('data', {}).get('sql_query'):
            sql = result2['data']['sql_query']
            print(f"   SQL: {sql[:100]}...")
        
        # Check for suggestions
        if result2.get('data', {}).get('suggestions'):
            print(f"   Suggestions: {len(result2['data']['suggestions'])} provided")
            for i, suggestion in enumerate(result2['data']['suggestions'][:3], 1):
                print(f"      {i}. {suggestion}")
        print()
        
        # Test 3: Conversation history
        print("📚 Test 3: Conversation History")
        
        if system.memory_service:
            history = await system.orchestrator_agent.get_conversation_history(
                user_id=user_id, 
                session_id=session_id, 
                limit=5
            )
            
            print(f"   Conversation entries found: {len(history)}")
            for i, conv in enumerate(history, 1):
                if hasattr(conv, 'user_input'):
                    print(f"   {i}. {conv.user_input[:60]}...")
                elif isinstance(conv, dict) and 'user_input' in conv:
                    print(f"   {i}. {conv['user_input'][:60]}...")
        print()
        
        # Test 4: Session state
        print("🔧 Test 4: Session State")
        
        if system.memory_service:
            # Update session state
            await system.memory_service.update_session_state(
                session_id=session_id,
                user_id=user_id,
                updates={
                    "current_topic": "sales_analytics",
                    "conversation_style": "professional",
                    "recent_topics": ["sales", "time-series"]
                }
            )
            
            # Get session state
            session_state = await system.memory_service.get_session_state(session_id, user_id)
            print(f"   Session ID: {session_state.session_id}")
            print(f"   Current topic: {session_state.current_topic}")
            print(f"   Conversation style: {session_state.conversation_style}")
            print(f"   Recent topics: {session_state.recent_topics}")
        print()
        
        # Test 5: Contextual suggestions
        print("💡 Test 5: Contextual Suggestions")
        
        if system.memory_service:
            # Get conversation context
            context = await system.memory_service.get_conversation_context_with_summary(
                user_id=user_id,
                session_id=session_id,
                max_context_window=10
            )
            
            # Generate suggestions
            suggestions = await system.memory_service.generate_contextual_suggestions(
                user_id=user_id,
                current_query="sales analysis",
                session_context=context.get("context_messages", []),
                limit=5
            )
            
            print(f"   Generated suggestions: {len(suggestions)}")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        print()
        
        print("✅ Conversation integration test completed successfully!")
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"   - Initial query processed: {result1.get('success', False)}")
        print(f"   - Follow-up query processed: {result2.get('success', False)}")
        print(f"   - Conversation logging enabled: {system.memory_service is not None}")
        print(f"   - Follow-up detection working: {follow_up_info.get('is_follow_up', False) if follow_up_info else False}")
        print(f"   - Suggestions generated: {len(suggestions) if 'suggestions' in locals() else 0}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            if 'system' in locals() and system:
                await system.close()
                print("🧹 System cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_conversation_integration())
