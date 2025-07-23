"""
NL2SQL Orchestrator Integration Example with Memory.

This example shows how to integrate the orchestrator memory service
with your NL2SQL application for intelligent query processing.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from services.orchestrator_memory_service import OrchestratorMemoryService, QueryContext, QueryResult
from Models.agent_response import AgentResponse

class NL2SQLOrchestratorWithMemory:
    """
    Enhanced NL2SQL Orchestrator with memory capabilities.
    
    This class demonstrates how to integrate the memory service
    with your existing NL2SQL orchestrator.
    """
    
    def __init__(self):
        self.memory_service: Optional[OrchestratorMemoryService] = None
        
    async def initialize(self):
        """Initialize the orchestrator with memory service."""
        # Initialize with the configuration we've been using
        from services.cosmos_db_service import CosmosDbConfig
        
        config = CosmosDbConfig(
            endpoint="https://cosmos-acrasalesanalytics2.documents.azure.com:443/",
            database_name="sales_analytics",
            chat_container_name="nl2sql_chatlogs",
            cache_container_name="nl2sql_cache"
        )
        
        self.memory_service = await OrchestratorMemoryService.create_from_config(config)
        print("✅ NL2SQL Orchestrator with Memory initialized")
    
    async def close(self):
        """Clean shutdown."""
        if self.memory_service:
            await self.memory_service.close()
    
    async def start_user_session(self, user_id: str, session_name: str = None):
        """Start a new session for a user."""
        session = await self.memory_service.create_session(
            user_id=user_id,
            session_title=session_name or f"NL2SQL Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"📋 Started session: {session.session_name} for user {user_id}")
        return session
    
    async def process_nl2sql_query(
        self, 
        user_id: str, 
        session_id: str, 
        natural_language_query: str,
        use_similarity_search: bool = True
    ) -> dict:
        """
        Process a natural language query with memory-enhanced capabilities.
        
        Args:
            user_id: User identifier
            session_id: Session identifier  
            natural_language_query: User's query in natural language
            use_similarity_search: Whether to check for similar past queries
            
        Returns:
            Dictionary with results and memory insights
        """
        
        # Step 1: Generate embedding (you would use your actual embedding model here)
        query_embedding = self._simulate_embedding(natural_language_query)
        
        # Step 2: Check for similar past queries if enabled
        similar_queries = []
        if use_similarity_search:
            similar_queries = await self.memory_service.find_similar_queries(
                query_embedding=query_embedding,
                user_id=user_id,  # Search user's history first
                limit=3,
                similarity_threshold=0.8
            )
            
            if similar_queries:
                print(f"🔍 Found {len(similar_queries)} similar past queries:")
                for sq in similar_queries:
                    print(f"   - {sq.original_query[:60]}... (similarity: {sq.similarity_score:.3f})")
        
        # Step 3: Store the current query context
        query_context = await self.memory_service.process_query(
            user_id=user_id,
            session_id=session_id,
            query=natural_language_query,
            query_embedding=query_embedding
        )
        
        # Step 4: Process the query (simulate your NL2SQL pipeline)
        processing_start = datetime.now()
        
        # Simulate the NL2SQL workflow
        sql_result = await self._simulate_nl2sql_processing(
            natural_language_query, 
            similar_queries
        )
        
        processing_time = (datetime.now() - processing_start).total_seconds() * 1000
        
        # Step 5: Create result object
        query_result = QueryResult(
            query_id=query_context.query_id,
            sql_query=sql_result["sql"],
            execution_result=sql_result["data"],
            agent_response=AgentResponse(
                agent_type="nl2sql_orchestrator",
                response=sql_result["response"],
                success=sql_result["success"],
                processing_time_ms=int(processing_time)
            ),
            execution_time_ms=processing_time,
            success=sql_result["success"],
            error_message=sql_result.get("error")
        )
        
        # Step 6: Store the result in memory
        await self.memory_service.store_query_result(query_context, query_result)
        
        return {
            "query_id": query_context.query_id,
            "sql_query": sql_result["sql"],
            "results": sql_result["data"],
            "response": sql_result["response"],
            "success": sql_result["success"],
            "processing_time_ms": processing_time,
            "similar_queries": [
                {
                    "query": sq.original_query,
                    "similarity": sq.similarity_score,
                    "timestamp": sq.timestamp
                }
                for sq in similar_queries
            ],
            "memory_context": {
                "query_context_id": query_context.query_id,
                "embedding_stored": query_embedding is not None,
                "similar_queries_found": len(similar_queries)
            }
        }
    
    async def get_session_insights(self, user_id: str, session_id: str) -> dict:
        """Get insights about the current session."""
        
        # Get session context
        messages = await self.memory_service.get_session_context(
            user_id=user_id,
            session_id=session_id,
            max_messages=20
        )
        
        # Get user stats
        user_stats = await self.memory_service.get_user_query_stats(user_id)
        
        # Analyze query patterns
        user_queries = [msg.content for msg in messages if msg.role == "user"]
        
        return {
            "session_summary": {
                "total_messages": len(messages),
                "user_queries": len(user_queries),
                "assistant_responses": len([msg for msg in messages if msg.role == "assistant"]),
                "recent_queries": user_queries[-5:] if user_queries else []
            },
            "user_statistics": user_stats,
            "insights": {
                "session_active": len(messages) > 0,
                "query_frequency": len(user_queries) / max(1, len(messages) / 2),
                "recent_activity": messages[-1].timestamp if messages else None
            }
        }
    
    async def suggest_next_queries(self, user_id: str, current_query: str) -> List[str]:
        """Suggest relevant follow-up queries based on memory."""
        
        # Generate embedding for current query
        query_embedding = self._simulate_embedding(current_query)
        
        # Find similar queries from all users for inspiration
        similar_queries = await self.memory_service.find_similar_queries(
            query_embedding=query_embedding,
            user_id=None,  # Search all users
            limit=10,
            similarity_threshold=0.6
        )
        
        # Generate suggestions based on patterns
        suggestions = []
        
        # Extract common patterns and suggest variations
        for sq in similar_queries[:5]:
            if "sales" in sq.original_query.lower():
                suggestions.extend([
                    "Show me sales trends over time",
                    "What are the best performing sales regions?",
                    "Compare sales by product category"
                ])
            elif "customer" in sq.original_query.lower():
                suggestions.extend([
                    "Analyze customer satisfaction scores",
                    "Show customer retention rates",
                    "Which customers have the highest lifetime value?"
                ])
            elif "revenue" in sq.original_query.lower():
                suggestions.extend([
                    "What factors drive revenue growth?",
                    "Show revenue by business segment",
                    "Forecast next quarter's revenue"
                ])
        
        # Remove duplicates and limit
        unique_suggestions = list(set(suggestions))
        return unique_suggestions[:3]
    
    def _simulate_embedding(self, text: str) -> List[float]:
        """
        Simulate embedding generation.
        In production, you would use your actual embedding model here.
        """
        # Simple hash-based simulation for testing
        import hashlib
        hash_obj = hashlib.md5(text.lower().encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hex to normalized floats
        embedding = []
        for i in range(0, min(32, len(hash_hex)), 4):
            hex_chunk = hash_hex[i:i+4]
            normalized = int(hex_chunk, 16) / 65535.0  # Normalize to 0-1
            embedding.append(normalized)
        
        # Pad to ensure consistent size
        while len(embedding) < 8:
            embedding.append(0.0)
        
        return embedding[:8]
    
    async def _simulate_nl2sql_processing(
        self, 
        natural_query: str, 
        similar_queries: List
    ) -> dict:
        """
        Simulate the NL2SQL processing pipeline.
        In production, this would call your actual agents.
        """
        
        # Simulate different query types
        query_lower = natural_query.lower()
        
        if "sales" in query_lower:
            return {
                "sql": "SELECT product_name, SUM(sales_amount) FROM sales_data GROUP BY product_name ORDER BY SUM(sales_amount) DESC LIMIT 10",
                "data": {
                    "rows": [
                        {"product_name": "Product A", "total_sales": 150000},
                        {"product_name": "Product B", "total_sales": 120000},
                        {"product_name": "Product C", "total_sales": 95000}
                    ],
                    "total_rows": 3
                },
                "response": f"Here are the sales results for your query: '{natural_query}'. Found 3 top-selling products with total sales ranging from $95K to $150K.",
                "success": True
            }
        elif "customer" in query_lower:
            return {
                "sql": "SELECT customer_segment, COUNT(*) as customer_count, AVG(purchase_frequency) FROM customers GROUP BY customer_segment",
                "data": {
                    "rows": [
                        {"customer_segment": "Premium", "customer_count": 150, "avg_frequency": 8.5},
                        {"customer_segment": "Standard", "customer_count": 300, "avg_frequency": 4.2},
                        {"customer_segment": "Basic", "customer_count": 450, "avg_frequency": 2.1}
                    ],
                    "total_rows": 3
                },
                "response": f"Customer analysis completed for: '{natural_query}'. Found 3 customer segments with varying purchase patterns.",
                "success": True
            }
        else:
            return {
                "sql": "SELECT * FROM data_table WHERE conditions_match_query LIMIT 10",
                "data": {
                    "rows": [{"id": 1, "value": "Sample data"}],
                    "total_rows": 1
                },
                "response": f"Query processed: '{natural_query}'. Results show general data matching your criteria.",
                "success": True
            }


async def demo_integration():
    """Demonstrate the integrated NL2SQL orchestrator with memory."""
    
    print("🚀 NL2SQL Orchestrator with Memory - Integration Demo")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = NL2SQLOrchestratorWithMemory()
    await orchestrator.initialize()
    
    try:
        # Demo user
        user_id = "demo_user_integration"
        
        # Start a session
        session = await orchestrator.start_user_session(
            user_id=user_id,
            session_name="Sales Analytics Demo Session"
        )
        session_id = session.session_id
        
        # Process several queries to build memory
        demo_queries = [
            "Show me the top 10 products by sales revenue",
            "What are the customer segments and their purchase patterns?", 
            "Analyze sales trends for the last quarter",
            "Which customers have the highest lifetime value?",
            "Show quarterly revenue trends by region"  # This should be similar to query 3
        ]
        
        print(f"\n💬 Processing {len(demo_queries)} demo queries...")
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            result = await orchestrator.process_nl2sql_query(
                user_id=user_id,
                session_id=session_id,
                natural_language_query=query,
                use_similarity_search=True
            )
            
            print(f"✅ SQL: {result['sql_query'][:60]}...")
            print(f"✅ Response: {result['response'][:80]}...")
            print(f"⏱️ Processing time: {result['processing_time_ms']:.0f}ms")
            
            if result['similar_queries']:
                print(f"🔍 Similar queries found: {len(result['similar_queries'])}")
                for sq in result['similar_queries'][:2]:
                    print(f"   - {sq['query'][:50]}... (similarity: {sq['similarity']:.3f})")
        
        # Get session insights
        print(f"\n📊 Session Insights")
        print("-" * 30)
        
        insights = await orchestrator.get_session_insights(user_id, session_id)
        
        print(f"Session Summary:")
        print(f"  - Total messages: {insights['session_summary']['total_messages']}")
        print(f"  - User queries: {insights['session_summary']['user_queries']}")
        print(f"  - Assistant responses: {insights['session_summary']['assistant_responses']}")
        
        print(f"\nUser Statistics:")
        for key, value in insights['user_statistics'].items():
            print(f"  - {key}: {value}")
        
        # Generate suggestions
        print(f"\n💡 Query Suggestions")
        print("-" * 30)
        
        suggestions = await orchestrator.suggest_next_queries(
            user_id=user_id,
            current_query="sales performance analysis"
        )
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        print(f"\n🎯 Integration Demo Summary")
        print("=" * 40)
        print("✅ Memory-enhanced query processing")
        print("✅ Semantic similarity detection")
        print("✅ Session context management")
        print("✅ User analytics and insights")
        print("✅ Intelligent query suggestions")
        print("✅ Complete conversation history")
        
        print(f"\n🚀 Your NL2SQL application now has enterprise-grade memory!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        await orchestrator.close()
        print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_integration())
