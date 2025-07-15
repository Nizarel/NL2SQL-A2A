import logging

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
from agents import OrchestratorAgent


logger = logging.getLogger(__name__)


class OrchestratorAgentExecutor(AgentExecutor):
    """OrchestratorAgent Executor for A2A Protocol"""

    def __init__(self):
        # Initialize with proper error handling
        try:
            self.agent = OrchestratorAgent()
        except Exception as e:
            logger.error(f"Failed to initialize OrchestratorAgent: {str(e)}")
            raise

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

            logger.info(f"Starting A2A execution for query: {query[:100]}...")

            async for partial in self.agent.stream(query, task.contextId):
                require_input = partial.get('require_user_input', False)
                is_done = partial.get('is_task_complete', False)
                text_content = partial.get('content', '')

                if require_input:
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            status=TaskStatus(
                                state=TaskState.input_required,
                                message=new_agent_text_message(
                                    text_content,
                                    task.contextId,
                                    task.id,
                                ),
                            ),
                            final=True,
                            contextId=task.contextId,
                            taskId=task.id,
                        )
                    )
                elif is_done:
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            append=False,
                            contextId=task.contextId,
                            taskId=task.id,
                            lastChunk=True,
                            artifact=new_text_artifact(
                                name='nl2sql_result',
                                description='Complete NL2SQL workflow result with SQL, execution, and insights.',
                                text=text_content,
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
                    logger.info(f"A2A execution completed successfully for task {task.id}")
                else:
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            status=TaskStatus(
                                state=TaskState.working,
                                message=new_agent_text_message(
                                    text_content,
                                    task.contextId,
                                    task.id,
                                ),
                            ),
                            final=False,
                            contextId=task.contextId,
                            taskId=task.id,
                        )
                    )

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

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel the current task execution"""
        logger.warning("Task cancellation requested but not implemented")
        raise Exception('cancel not supported')