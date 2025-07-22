"""
Example: Integrating Orchestrator Agent with Memory Service
Compatible with Semantic Kernel 1.35.0

Updated for single container Cosmos DB design:
- Database: sales_analytics
- Container: nl2sql_chatlogs
- Partition Key: /user_id/session_id (hierarchical)
- Vector embedding support
"""

import os
import time
from typing import Dict, Any
from semantic_kernel import Kernel
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory

from agents.orchestrator_agent import OrchestratorAgent
from services.orchestrator_memory_service import OrchestratorMemoryService


class MemoryEnabledOrchestrator:
    """
    Enhanced orchestrator agent with Cosmos DB memory capabilities
    Uses single container design with hierarchical partitioning
    """
    
    def __init__(self, orchestrator_agent: OrchestratorAgent, cosmos_connection_string: str = None):
        self.orchestrator = orchestrator_agent
        
        # Initialize memory service with your Cosmos DB configuration
        self.memory_service = OrchestratorMemoryService(cosmos_connection_string)
        
        # Create semantic memory with our custom memory store
        self.semantic_memory = SemanticTextMemory(
            storage=self.memory_service,
            embeddings_generator=None  # Can be enhanced with embeddings
        )
        
        # Register memory with the kernel
        if hasattr(self.orchestrator.kernel, 'register_memory_store'):
            self.orchestrator.kernel.register_memory_store(self.memory_service)
    
    def initialize(self):
        """Initialize the memory-enabled orchestrator"""
        self.memory_service.initialize()
    
    def process_with_memory(
        self, 
        user_id: str,
        user_input: str,
        session_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process user input with memory context
        
        Args:
            user_id: User identifier
            user_input: User's natural language question
            session_id: Optional session identifier
            **kwargs: Additional parameters for orchestrator
        """
        
        # Start or get session
        if not session_id:
            session = self.memory_service.start_conversation(user_id)
            session_id = session.session_id
        
        # Get conversation history for context
        conversation_history = self.memory_service.get_conversation_history(
            user_id=user_id,
            session_id=session_id,
            limit=5  # Last 5 conversations for context
        )
        
        # Build context from history
        context_from_history = self._build_context_from_history(conversation_history)
        
        # Enhance input data with memory context
        input_data = {
            "question": user_input,
            "context": context_from_history,
            "execute": kwargs.get("execute", True),
            "limit": kwargs.get("limit", 100),
            "include_summary": kwargs.get("include_summary", True)
        }
        
        # Execute orchestrator workflow
        start_time = time.time()
        
        # Note: Assuming your orchestrator.process is synchronous
        # If it's async, you'd need to handle that differently
        result = self.orchestrator.process(input_data)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Log the complete workflow to memory
        if result.get("success"):
            self.memory_service.log_orchestrator_workflow(
                user_id=user_id,
                session_id=session_id,
                user_input=user_input,
                workflow_results=result.get("metadata", {}),
                processing_time_ms=processing_time_ms
            )
        
        # Enhance result with memory metadata
        if "metadata" not in result:
            result["metadata"] = {}
        
        result["metadata"].update({
            "session_id": session_id,
            "user_id": user_id,
            "memory_enabled": True,
            "conversation_turn": len(conversation_history) + 1,
            "context_used": bool(context_from_history)
        })
        
        return result
    
    def _build_context_from_history(self, conversation_history: list) -> str:
        """
        Build context string from conversation history
        """
        if not conversation_history:
            return ""
        
        context_parts = []
        
        for entry in conversation_history[-3:]:  # Last 3 conversations
            context_parts.append(f"Previous Question: {entry.user_input}")
            
            # Add key insights from agent responses
            for agent_response in entry.agent_responses:
                if agent_response.summary:
                    context_parts.append(f"Previous Answer: {agent_response.summary[:200]}...")

        
        return "\n".join(context_parts) if context_parts else ""
    
    def find_similar_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> list:
        """
        Find similar conversations from user's history
        Can be enhanced with semantic search
        """
        return self.memory_service.find_similar_queries(user_id, query, limit)
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user memory statistics"""
        return self.memory_service.get_memory_stats(user_id)
    
    def end_session(self, user_id: str, session_id: str):
        """End a conversation session"""
        self.memory_service.end_conversation(user_id, session_id)
    
    def close(self):
        """Close the memory-enabled orchestrator"""
        self.memory_service.close()


# Example usage
def main():
    """
    Example of using memory-enabled orchestrator
    """
    
    # Mock setup - replace with your actual kernel and agents
    kernel = Kernel()  # Configure with your AI service
    
    # This would be your actual orchestrator agent
    # orchestrator_agent = OrchestratorAgent(kernel, schema_analyst, sql_generator, executor, summarizer)
    
    # Cosmos DB connection string from environment
    cosmos_connection = None  # Will be built from environment variables
    
    try:
        # Create memory-enabled orchestrator
        # memory_orchestrator = MemoryEnabledOrchestrator(orchestrator_agent, cosmos_connection)
        # memory_orchestrator.initialize()
        
        # Example conversation
        user_id = "test_user_123"
        
        # First question
        # result1 = memory_orchestrator.process_with_memory(
        #     user_id=user_id,
        #     user_input="What are the total sales for this month?",
        #     execute=True
        # )
        
        # Follow-up question with memory context
        # result2 = memory_orchestrator.process_with_memory(
        #     user_id=user_id,
        #     user_input="Show me the breakdown by region",
        #     session_id=result1["metadata"]["session_id"],
        #     execute=True
        # )
        
        # Get user statistics
        # stats = memory_orchestrator.get_user_statistics(user_id)
        # print(f"User statistics: {stats}")
        
        # Clean up
        # memory_orchestrator.close()
        
        print("✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
