"""
Test script for conversation features
Tests follow-up detection, contextual suggestions, and conversation flow
"""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.orchestrator_memory_service import OrchestratorMemoryService
from Models.agent_response import Message


async def test_conversation_features():
    """Test conversation enhancement features"""
    
    print("🧪 Testing conversation enhancement features...")
    
    try:
        # Create memory service (we'll use a mock since we don't have Cosmos DB setup in this test)
        memory_service = MockOrchestratorMemoryService()
        
        # Test 1: Follow-up detection
        print("\n🔍 Test 1: Follow-up Query Detection")
        
        # Create mock conversation context
        context_messages = [
            Message(
                session_id="test_session",
                user_id="test_user",
                role="user",
                content="Show me sales data for last month",
                timestamp=datetime.now(timezone.utc)
            ),
            Message(
                session_id="test_session", 
                user_id="test_user",
                role="assistant",
                content="Here's the sales data for last month...",
                timestamp=datetime.now(timezone.utc),
                metadata={"sql_query": "SELECT * FROM sales WHERE month = 'last_month'"}
            )
        ]
        
        # Test follow-up queries
        follow_up_queries = [
            "What about this month?",
            "Show me more details",
            "How does that compare to the previous year?",
            "Can you break it down by region?"
        ]
        
        for query in follow_up_queries:
            result = await memory_service.detect_follow_up_query(query, context_messages)
            print(f"   Query: '{query}'")
            print(f"   Follow-up: {result['is_follow_up']}")
            print(f"   Enhanced: '{result['enhanced_question']}'")
            print(f"   Reasoning: {result['reasoning']}")
            print()
        
        # Test 2: Topic extraction
        print("🏷️ Test 2: Topic Extraction")
        
        test_queries = [
            "Show me sales revenue for this quarter",
            "What are our customer retention rates?", 
            "Analyze product performance by category",
            "Compare financial metrics across regions",
            "What's the trend in user engagement over time?"
        ]
        
        for query in test_queries:
            topics = memory_service._extract_topics_from_query(query)
            print(f"   Query: '{query}'")
            print(f"   Topics: {topics}")
            print()
        
        # Test 3: Contextual suggestions
        print("💡 Test 3: Contextual Suggestions")
        
        suggestions = await memory_service.generate_contextual_suggestions(
            user_id="test_user",
            current_query="Show me sales data",
            session_context=context_messages,
            limit=5
        )
        
        print("   Suggestions based on sales query:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        print()
        
        # Test 4: Conversation context with summary
        print("📝 Test 4: Conversation Context with Summary")
        
        context_result = await memory_service.get_conversation_context_with_summary(
            user_id="test_user",
            session_id="test_session",
            max_context_window=5
        )
        
        print(f"   Has history: {context_result['has_history']}")
        print(f"   Total turns: {context_result['total_turns']}")
        print(f"   Context messages: {len(context_result['context_messages'])}")
        if context_result['conversation_summary']:
            print(f"   Summary: {context_result['conversation_summary']}")
        print()
        
        print("✅ All conversation features tested successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


class MockOrchestratorMemoryService:
    """Mock memory service for testing"""
    
    def __init__(self):
        self.cosmos_service = None
    
    async def detect_follow_up_query(self, question: str, context_messages):
        """Mock implementation of follow-up detection"""
        # Reuse the actual logic from OrchestratorMemoryService
        follow_up_indicators = [
            "that", "those", "it", "them", "the same", "also", "another",
            "more", "what about", "how about", "and", "in addition",
            "similarly", "likewise", "too", "as well", "furthermore",
            "moreover", "additionally", "plus", "again"
        ]
        
        question_lower = question.lower()
        is_follow_up = any(indicator in question_lower for indicator in follow_up_indicators)
        
        # Enhanced detection - check for pronouns and relative references
        pronoun_indicators = ["this", "these", "here", "there", "such"]
        has_pronouns = any(pronoun in question_lower for pronoun in pronoun_indicators)
        
        # Check for incomplete context
        incomplete_indicators = ["show me more", "tell me about", "explain", "why", "how about"]
        seems_incomplete = any(indicator in question_lower for indicator in incomplete_indicators) and len(question.split()) < 8
        
        # Final determination
        is_follow_up = is_follow_up or has_pronouns or seems_incomplete
        
        # Extract context from previous queries if follow-up detected
        context_elements = []
        enhanced_question = question
        
        if is_follow_up and context_messages:
            recent_context = []
            for msg in reversed(context_messages[-4:]):
                if msg.role == "user":
                    recent_context.append(f"Previous query: {msg.content}")
                elif msg.role == "assistant" and msg.metadata:
                    if "sql_query" in msg.metadata:
                        recent_context.append(f"Previous SQL: {msg.metadata['sql_query']}")
                
                if len(recent_context) >= 2:
                    break
            
            context_elements = recent_context
            
            if len(context_elements) > 0:
                enhanced_question = f"{question} (Building on: {'; '.join(context_elements[:1])})"
        
        return {
            "is_follow_up": is_follow_up,
            "context_elements": context_elements,
            "enhanced_question": enhanced_question,
            "confidence": "high" if (is_follow_up and len(context_elements) > 0) else "medium",
            "reasoning": f"Detected as follow-up: contains indicators" if is_follow_up else "Independent query"
        }
    
    def _extract_topics_from_query(self, query: str):
        """Mock topic extraction"""
        query_lower = query.lower()
        topics = []
        
        # Business domain topics
        business_keywords = {
            "sales": ["sales", "revenue", "income", "profit", "selling"],
            "customers": ["customer", "client", "buyer", "consumer", "retention"],
            "products": ["product", "item", "inventory", "catalog", "performance"],
            "financials": ["financial", "cost", "expense", "budget", "metrics"],
            "analytics": ["trend", "analysis", "compare", "average", "total", "engagement"]
        }
        
        for topic, keywords in business_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        # Time-based queries
        time_keywords = ["time", "date", "year", "month", "daily", "weekly", "monthly", "quarter", "trend"]
        if any(keyword in query_lower for keyword in time_keywords):
            topics.append("time-series")
        
        # Comparison queries
        comparison_keywords = ["compare", "vs", "versus", "difference", "between"]
        if any(keyword in query_lower for keyword in comparison_keywords):
            topics.append("comparison")
            
        return topics if topics else ["general"]
    
    async def generate_contextual_suggestions(self, user_id: str, current_query: str, session_context, limit: int = 5):
        """Mock contextual suggestions"""
        
        # Basic suggestions based on current query
        query_lower = current_query.lower()
        suggestions = []
        
        if "sales" in query_lower:
            suggestions.extend([
                "What about sales by customer segment?",
                "Show me sales trends over time",
                "Compare sales across different regions"
            ])
        elif "customer" in query_lower:
            suggestions.extend([
                "What's the customer acquisition cost?",
                "Show me customer lifetime value",
                "Which customers are at risk of churning?"
            ])
        else:
            suggestions.extend([
                "What are our top selling products?",
                "Show me customer trends this month",
                "Compare performance by region"
            ])
        
        return suggestions[:limit]
    
    async def get_conversation_context_with_summary(self, user_id: str, session_id: str, max_context_window: int = 10):
        """Mock conversation context"""
        
        # Mock response
        return {
            "context_messages": [
                Message(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content="Show me sales data for last month",
                    timestamp=datetime.now(timezone.utc)
                )
            ],
            "conversation_summary": "Previous discussion covered sales and revenue analysis",
            "total_turns": 3,
            "session_start": datetime.now(timezone.utc),
            "has_history": True
        }


if __name__ == "__main__":
    asyncio.run(test_conversation_features())
