import logging
import asyncio
from typing import Dict, Any, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
    new_text_artifact,
)

# Import the OrchestratorAgent
try:
    from ..agents.orchestrator_agent import OrchestratorAgent
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.orchestrator_agent import OrchestratorAgent

logger = logging.getLogger(__name__)


class OrchestratorAgentExecutor(AgentExecutor):
    """OrchestratorAgent Executor for A2A Protocol"""

    def __init__(self, orchestrator_agent: Optional[OrchestratorAgent] = None):
        """Initialize the executor with optional orchestrator agent"""
        try:
            self.orchestrator = orchestrator_agent
            if self.orchestrator:
                logger.info("✅ OrchestratorAgentExecutor initialized with orchestrator agent")
            else:
                logger.info("✅ OrchestratorAgentExecutor initialized (orchestrator agent to be set later)")
        except Exception as e:
            logger.error(f"Failed to initialize OrchestratorAgentExecutor: {str(e)}")
            raise

    def set_orchestrator(self, orchestrator_agent: OrchestratorAgent):
        """Set the orchestrator agent after initialization"""
        self.orchestrator = orchestrator_agent
        logger.info("✅ Orchestrator agent set successfully")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute agent request with A2A protocol support
        
        Args:
            context: Request context containing user input and task info
            event_queue: Event queue for publishing task updates
        """
        try:
            query = context.get_user_input()
            task = context.current_task
            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)

            logger.info(f"Processing NL2SQL query: {query[:100]}...")

            # Check if orchestrator is available
            if not self.orchestrator:
                await self._send_placeholder_response(query, task, event_queue)
                return

            # Execute the actual NL2SQL workflow using the orchestrator's streaming method
            await self._execute_nl2sql_workflow_streaming(query, task, event_queue)
                
        except Exception as e:
            logger.error(f"A2A execution failed: {str(e)}")
            # Send error status to event queue
            if 'task' in locals() and task:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.completed,
                            message=new_agent_text_message(
                                f"Execution failed: {str(e)}",
                                task.contextId,
                                task.id,
                            ),
                        ),
                        final=True,
                        contextId=task.contextId,
                        taskId=task.id,
                    )
                )
            raise

    async def _execute_nl2sql_workflow_streaming(self, query: str, task, event_queue: EventQueue):
        """Execute the actual NL2SQL workflow using orchestrator's streaming method"""
        
        try:
            # Use the orchestrator's streaming method for real-time updates
            async for update in self.orchestrator.stream(query, task.contextId):
                content = update.get('content', '')
                is_complete = update.get('is_task_complete', False)
                require_input = update.get('require_user_input', False)
                raw_data = update.get('raw_data', None)
                
                # For A2A final result, use raw data to create better formatting
                if is_complete and raw_data:
                    content = self._format_a2a_final_result(raw_data)
                
                if content:
                    # Determine the appropriate task state based on the update
                    if require_input:
                        # Agent needs user input
                        await event_queue.enqueue_event(
                            TaskStatusUpdateEvent(
                                status=TaskStatus(
                                    state=TaskState.input_required,
                                    message=new_agent_text_message(
                                        content,
                                        task.contextId,
                                        task.id,
                                    ),
                                ),
                                final=True,
                                contextId=task.contextId,
                                taskId=task.id,
                            )
                        )
                    elif is_complete:
                        # Task is complete - send artifact and final status
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                append=False,
                                contextId=task.contextId,
                                taskId=task.id,
                                lastChunk=True,
                                artifact=new_text_artifact(
                                    name='nl2sql_result',
                                    description='Complete NL2SQL workflow result',
                                    text=content,
                                ),
                            )
                        )
                        await event_queue.enqueue_event(
                            TaskStatusUpdateEvent(
                                status=TaskStatus(state=TaskState.completed),
                                final=True,
                                contextId=task.contextId,
                                taskId=task.id,
                            )
                        )
                        break
                    else:
                        # Work in progress
                        await event_queue.enqueue_event(
                            TaskStatusUpdateEvent(
                                status=TaskStatus(
                                    state=TaskState.working,
                                    message=new_agent_text_message(
                                        content,
                                        task.contextId,
                                        task.id,
                                    ),
                                ),
                                final=False,
                                contextId=task.contextId,
                                taskId=task.id,
                            )
                        )
                    
            logger.info(f"A2A execution completed for task {task.id}")
            
        except Exception as e:
            logger.error(f"Streaming workflow failed: {str(e)}")
            # Send error message
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=new_agent_text_message(
                            f"❌ NL2SQL workflow failed: {str(e)}",
                            task.contextId,
                            task.id,
                        ),
                    ),
                    final=True,
                    contextId=task.contextId,
                    taskId=task.id,
                )
            )

    async def _send_placeholder_response(self, query: str, task, event_queue: EventQueue):
        """Send placeholder response when orchestrator is not available"""
        
        # Update status: No orchestrator available
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=TaskState.working,
                    message=new_agent_text_message(
                        "⚠️ Orchestrator agent not initialized",
                        task.contextId,
                        task.id,
                    ),
                ),
                final=False,
                contextId=task.contextId,
                taskId=task.id,
            )
        )
        
        placeholder_response = f"""🚧 NL2SQL A2A Integration Status

📋 **Query Received:** {query}

⚠️ **Status:** Orchestrator agent not connected

🔧 **To Complete Integration:**
1. Initialize the OrchestratorAgent with proper Kernel and agents
2. Set the orchestrator using: executor.set_orchestrator(orchestrator_agent)
3. Ensure all required agents are properly configured:
   - SQLGeneratorAgent
   - SQLExecutorAgent  
   - SummarizingAgent

📖 **Integration Example:**
```python
# Initialize the full NL2SQL system
kernel = setup_kernel()
sql_generator = SQLGeneratorAgent(kernel, schema_service)
executor = SQLExecutorAgent(kernel, database_plugin)
summarizer = SummarizingAgent(kernel)

# Create orchestrator
orchestrator = OrchestratorAgent(kernel, sql_generator, executor, summarizer)

# Set in A2A executor
orchestrator_executor.set_orchestrator(orchestrator)
```

🎯 **Current Capability:** A2A protocol integration ready, awaiting orchestrator connection
"""
        
        # Send final result
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                append=False,
                contextId=task.contextId,
                taskId=task.id,
                lastChunk=True,
                artifact=new_text_artifact(
                    name='nl2sql_integration_status',
                    description='NL2SQL A2A integration status and setup instructions',
                    text=placeholder_response,
                ),
            )
        )
        
        # Mark task as completed
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(state=TaskState.completed),
                final=True,
                contextId=task.contextId,
                taskId=task.id,
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel the current task execution"""
        logger.warning("Task cancellation requested")
        
        # Send cancellation notification
        if hasattr(context, 'current_task') and context.current_task:
            task = context.current_task
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=new_agent_text_message(
                            "🛑 Task cancelled by user request",
                            task.contextId,
                            task.id,
                        ),
                    ),
                    final=True,
                    contextId=task.contextId,
                    taskId=task.id,
                )
            )
        
        # TODO: If orchestrator supports cancellation, implement it here
        # if self.orchestrator and hasattr(self.orchestrator, 'cancel'):
        #     await self.orchestrator.cancel()
        
        logger.info("Task cancellation completed")

    def _format_a2a_final_result(self, raw_data: dict) -> str:
        """Format the final result specifically for A2A protocol with proper insight handling"""
        
        try:
            summary_result = raw_data.get('summary_result', {})
            sql_query = raw_data.get('sql_query', '')
            execution_result = raw_data.get('execution_result', {})
            row_count = raw_data.get('row_count', 0)
            
            # Extract structured data from summary result
            summary_data = summary_result.get('data', {})
            executive_summary = summary_data.get('executive_summary', 'No summary available')
            key_insights = summary_data.get('key_insights', [])
            recommendations = summary_data.get('recommendations', [])
            data_overview = summary_data.get('data_overview', '')
            
            # Format insights properly for A2A
            formatted_insights = []
            for insight in key_insights[:3]:  # Top 3 insights
                if isinstance(insight, dict):
                    finding = insight.get('finding', '')
                    significance = insight.get('business_significance', '')
                    if finding:
                        formatted_insights.append(f"• **{finding}** - {significance}")
                else:
                    formatted_insights.append(f"• {insight}")
            
            # Format recommendations for A2A  
            formatted_recommendations = []
            for rec in recommendations[:2]:  # Top 2 recommendations
                if isinstance(rec, dict):
                    action = rec.get('action', '')
                    priority = rec.get('priority', 'Medium')
                    if action:
                        formatted_recommendations.append(f"• **[{priority}]** {action}")
                else:
                    formatted_recommendations.append(f"• {rec}")
            
            # Create A2A-optimized final content
            final_content = f"""✅ NL2SQL Workflow Complete!

📋 **Executive Summary:**
{executive_summary}

🔍 **SQL Query:**
```sql
{sql_query}
```

📊 **Results:** {row_count} rows retrieved

📈 **Data Overview:**
{data_overview}

💡 **Key Business Insights:**
{chr(10).join(formatted_insights) if formatted_insights else "• No specific insights available"}

🎯 **Strategic Recommendations:**
{chr(10).join(formatted_recommendations) if formatted_recommendations else "• No specific recommendations available"}
"""
            
            return final_content
            
        except Exception as e:
            logger.error(f"A2A formatting failed: {str(e)}")
            # Fallback to original content
            return raw_data.get('content', f'❌ Formatting error: {str(e)}')