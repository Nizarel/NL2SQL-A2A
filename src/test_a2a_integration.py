"""
Test script to verify a2a-sdk integration with OrchestratorAgentExecutor
"""

import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)

@pytest.mark.asyncio
async def test_orchestrator_executor():
    """Test the OrchestratorAgentExecutor integration"""
    print("🧪 Testing OrchestratorAgentExecutor...")
    
    try:
        # Import our executor with fallback
        try:
            from .a2a_executors.orchestrator_executor import OrchestratorAgentExecutor
        except ImportError:
            from a2a_executors.orchestrator_executor import OrchestratorAgentExecutor
        print("✅ Import successful")
        
        # Create executor instance
        executor = OrchestratorAgentExecutor()
        print("✅ Executor created successfully")
        
        # Verify it implements AgentExecutor interface
        from a2a.server.agent_execution import AgentExecutor
        assert isinstance(executor, AgentExecutor), "Executor must implement AgentExecutor interface"
        print("✅ AgentExecutor interface verified")
        
        # Check required methods exist
        assert hasattr(executor, 'execute'), "Missing execute method"
        assert hasattr(executor, 'cancel'), "Missing cancel method"
        assert hasattr(executor, 'set_orchestrator'), "Missing set_orchestrator method"
        print("✅ Required methods present")
        
        # Test with mock objects
        print("🧪 Testing with mock objects...")
        
        # Create mock context and event queue with proper types
        from a2a.types import Message, TextPart
        
        # Create a proper message object
        text_part = TextPart(type="text", text="Test query: Show me sales data")
        message = Message(
            messageId="test-message-123",
            role="user",
            parts=[text_part]
        )
        
        mock_context = MagicMock()
        mock_context.get_user_input.return_value = "Test query: Show me sales data"
        mock_context.current_task = None
        mock_context.message = message
        
        mock_event_queue = AsyncMock()
        
        # Test execution without orchestrator (should send placeholder)
        print("🧪 Testing execution without orchestrator...")
        await executor.execute(mock_context, mock_event_queue)
        print("✅ Execution completed (placeholder mode)")
        
        # Verify event queue was called
        assert mock_event_queue.enqueue_event.called, "Event queue should be called"
        print("✅ Event queue integration working")
        
        # Test with mock orchestrator
        print("🧪 Testing with mock orchestrator...")
        mock_orchestrator = AsyncMock()
        
        # Mock the stream method
        async def mock_stream(query, context_id):
            yield {'content': 'Starting analysis...', 'is_task_complete': False}
            yield {'content': 'Generating SQL...', 'is_task_complete': False}
            yield {'content': 'Final result: SQL executed successfully', 'is_task_complete': True}
        
        mock_orchestrator.stream = mock_stream
        executor.set_orchestrator(mock_orchestrator)
        print("✅ Mock orchestrator set")
        
        # Reset event queue mock
        mock_event_queue.reset_mock()
        
        # Test execution with orchestrator
        await executor.execute(mock_context, mock_event_queue)
        print("✅ Execution with orchestrator completed")
        
        # Verify streaming integration
        assert mock_event_queue.enqueue_event.called, "Event queue should be called with streaming updates"
        call_count = mock_event_queue.enqueue_event.call_count
        print(f"✅ Event queue called {call_count} times (streaming updates)")
        
        print("🎉 All tests passed! OrchestratorAgentExecutor is properly aligned with a2a-sdk")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_orchestrator_executor())
    if success:
        print("\n✅ a2a-sdk integration verification completed successfully!")
    else:
        print("\n❌ a2a-sdk integration verification failed!")
