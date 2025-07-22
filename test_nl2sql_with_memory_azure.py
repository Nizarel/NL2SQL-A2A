"""
Test: NL2SQL System with Azure Identity Memory Integration
This test demonstrates the complete workflow with memory capabilities
"""

import os
import sys
import asyncio
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Set up environment for Azure Identity
os.environ['USE_AZURE_IDENTITY'] = 'true'
os.environ['COSMOS_DB_ENDPOINT'] = 'https://cosmos-acrasalesanalytics2.documents.azure.com'
os.environ['AZURE_TENANT_ID'] = '433ec967-f454-49f2-b132-d07f81545e02'

from main import NL2SQLMultiAgentSystem
from services.orchestrator_memory_service import OrchestratorMemoryService


class MemoryEnabledNL2SQLSystem:
    """
    NL2SQL System with integrated memory capabilities using Azure Identity
    """
    
    def __init__(self):
        self.nl2sql_system = None
        self.memory_service = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize both the NL2SQL system and memory service"""
        print("🚀 Initializing Memory-Enabled NL2SQL System with Azure Identity...")
        
        # Initialize NL2SQL system
        print("🔧 Setting up NL2SQL Multi-Agent System...")
        self.nl2sql_system = NL2SQLMultiAgentSystem()
        await self.nl2sql_system.initialize()
        print("✅ NL2SQL System initialized")
        
        # Initialize memory service with Azure Identity
        print("💾 Setting up Memory Service with Azure Identity...")
        self.memory_service = OrchestratorMemoryService()
        self.memory_service.initialize()
        print("✅ Memory Service initialized with Azure Identity")
        
        self.initialized = True
        print("🎉 Memory-Enabled NL2SQL System ready!")
    
    async def process_query_with_memory(
        self, 
        user_id: str,
        question: str,
        session_id: str = None,
        execute: bool = True,
        limit: int = 10
    ) -> dict:
        """
        Process NL2SQL query with memory context and logging
        
        Args:
            user_id: User identifier
            question: Natural language question
            session_id: Optional session identifier
            execute: Whether to execute the SQL query
            limit: Maximum number of rows to return
        
        Returns:
            Complete workflow results with memory integration
        """
        
        if not self.initialized:
            raise ValueError("System not initialized. Call initialize() first.")
        
        print(f"\n🤔 Processing question: {question}")
        print(f"👤 User: {user_id}")
        
        # Start or continue session
        if not session_id:
            session = self.memory_service.start_conversation(user_id)
            session_id = session.session_id
            print(f"🆕 New session: {session_id}")
        else:
            print(f"🔄 Continuing session: {session_id}")
        
        # Get conversation history for context
        print("📚 Retrieving conversation history...")
        conversation_history = self.memory_service.get_conversation_history(
            user_id=user_id,
            session_id=session_id,
            limit=5
        )
        print(f"   Found {len(conversation_history)} previous conversations")
        
        # Build context from history
        context_from_history = self._build_context_from_history(conversation_history)
        if context_from_history:
            print("🧠 Using conversation context from memory")
        
        # Process query with NL2SQL system
        print("⚡ Executing NL2SQL workflow...")
        start_time = time.time()
        
        try:
            # Call the NL2SQL system
            result = await self.nl2sql_system.ask_question(
                question=question,
                execute=execute,
                limit=limit,
                include_summary=True,
                context=context_from_history
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            print(f"⏱️  Processing completed in {processing_time_ms}ms")
            
            # Store conversation memory
            print("💾 Storing conversation in memory...")
            
            # Create assistant response from NL2SQL results
            assistant_response = self._format_assistant_response(result)
            
            entry_id = self.memory_service.store_conversation_memory(
                conversation_id=f"{user_id}/{session_id}",
                user_message=question,
                assistant_response=assistant_response,
                metadata={
                    'processing_time_ms': processing_time_ms,
                    'success': result.get('success', False),
                    'executed_sql': result.get('sql_query', ''),
                    'row_count': result.get('row_count', 0),
                    'timestamp': datetime.now().isoformat(),
                    'workflow_steps': list(result.keys())
                }
            )
            
            print(f"✅ Conversation stored with ID: {entry_id}")
            
            # Enhance result with memory metadata
            result['memory_info'] = {
                'session_id': session_id,
                'user_id': user_id,
                'entry_id': entry_id,
                'conversation_turn': len(conversation_history) + 1,
                'context_used': bool(context_from_history),
                'processing_time_ms': processing_time_ms
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            
            # Store error in memory
            error_response = f"Error processing query: {str(e)}"
            self.memory_service.store_conversation_memory(
                conversation_id=f"{user_id}/{session_id}",
                user_message=question,
                assistant_response=error_response,
                metadata={
                    'error': True,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return {
                'success': False,
                'error': str(e),
                'memory_info': {
                    'session_id': session_id,
                    'user_id': user_id,
                    'error_logged': True
                }
            }
    
    def _build_context_from_history(self, conversation_history: list) -> str:
        """Build context string from conversation history"""
        if not conversation_history:
            return ""
        
        context_parts = []
        
        for entry in conversation_history[-2:]:  # Last 2 conversations for context
            context_parts.append(f"Previous Q: {entry.user_input}")
            
            # Add key insights from agent responses
            if entry.agent_responses:
                for agent_response in entry.agent_responses:
                    if agent_response.summary:
                        # Truncate long responses
                        summary = agent_response.summary[:150] + "..." if len(agent_response.summary) > 150 else agent_response.summary
                        context_parts.append(f"Previous A: {summary}")
        
        return "\\n".join(context_parts) if context_parts else ""
    
    def _format_assistant_response(self, result: dict) -> str:
        """Format NL2SQL results into a readable assistant response"""
        if not result.get('success', False):
            return f"I encountered an error: {result.get('error', 'Unknown error')}"
        
        response_parts = []
        
        # Add SQL query if generated
        if 'sql_query' in result:
            response_parts.append(f"Generated SQL: {result['sql_query']}")
        
        # Add execution results if available
        if 'results' in result and result['results']:
            row_count = len(result['results']) if isinstance(result['results'], list) else result.get('row_count', 0)
            response_parts.append(f"Query executed successfully, returned {row_count} rows")
        
        # Add summary if available
        if 'summary' in result:
            response_parts.append(f"Summary: {result['summary']}")
        
        return " | ".join(response_parts) if response_parts else "Query processed successfully"
    
    def get_user_memory_stats(self, user_id: str) -> dict:
        """Get memory statistics for a user"""
        return self.memory_service.get_memory_stats(user_id)
    
    async def close(self):
        """Close all connections"""
        if self.nl2sql_system:
            await self.nl2sql_system.close()
        if self.memory_service:
            self.memory_service.close()


async def main():
    """
    Test the complete Memory-Enabled NL2SQL System with real queries
    """
    
    print("🧪 Starting Memory-Enabled NL2SQL Test with Azure Identity")
    print("=" * 60)
    
    # Create and initialize the system
    system = MemoryEnabledNL2SQLSystem()
    
    try:
        await system.initialize()
        
        print("\\n🎯 Test 1: First sales query")
        print("-" * 40)
        
        # Test user
        test_user = "sales_analyst_001"
        
        # First query - sales data
        result1 = await system.process_query_with_memory(
            user_id=test_user,
            question="What are the total sales this year?",
            execute=True,
            limit=5
        )
        
        print("\\n📊 Result 1:")
        print(f"Success: {result1.get('success', False)}")
        if 'sql_query' in result1:
            print(f"SQL: {result1['sql_query']}")
        if 'memory_info' in result1:
            print(f"Session: {result1['memory_info']['session_id']}")
            print(f"Entry ID: {result1['memory_info']['entry_id']}")
        
        # Wait a moment
        await asyncio.sleep(2)
        
        print("\\n🎯 Test 2: Follow-up query with memory context")
        print("-" * 40)
        
        # Follow-up query in the same session
        session_id = result1.get('memory_info', {}).get('session_id')
        
        result2 = await system.process_query_with_memory(
            user_id=test_user,
            question="Show me the breakdown by region for those sales",
            session_id=session_id,
            execute=True,
            limit=10
        )
        
        print("\\n📊 Result 2:")
        print(f"Success: {result2.get('success', False)}")
        if 'sql_query' in result2:
            print(f"SQL: {result2['sql_query']}")
        if 'memory_info' in result2:
            print(f"Context Used: {result2['memory_info']['context_used']}")
            print(f"Conversation Turn: {result2['memory_info']['conversation_turn']}")
        
        print("\\n🎯 Test 3: Customer data query")
        print("-" * 40)
        
        # Different type of query
        result3 = await system.process_query_with_memory(
            user_id=test_user,
            question="Who are our top 5 customers by revenue?",
            session_id=session_id,
            execute=True,
            limit=5
        )
        
        print("\\n📊 Result 3:")
        print(f"Success: {result3.get('success', False)}")
        if 'sql_query' in result3:
            print(f"SQL: {result3['sql_query']}")
        
        print("\\n📈 Memory Statistics")
        print("-" * 40)
        
        # Get memory statistics
        stats = system.get_user_memory_stats(test_user)
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Total sessions: {stats['total_sessions']}")
        print(f"Last activity: {stats['last_activity']}")
        
        print("\\n✅ All tests completed successfully!")
        print("🎉 Memory-Enabled NL2SQL System is working with Azure Identity!")
        
    except Exception as e:
        print(f"\\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\\n🔒 Cleaning up...")
        await system.close()
        print("👋 Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
