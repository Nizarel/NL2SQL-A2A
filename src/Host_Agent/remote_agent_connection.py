from collections.abc import Callable

import httpx

from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    SendMessageRequest,
    SendMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from dotenv import load_dotenv


load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard | None, agent_url: str):
        print(f'agent_card: {agent_card}')
        print(f'agent_url: {agent_url}')
        self._httpx_client = httpx.AsyncClient(timeout=60)
        
        # Handle case where agent_card is None
        if agent_card is None:
            # Create a default agent card for connection
            capabilities = AgentCapabilities(
                streaming=False,
                pushNotifications=False
            )
            
            agent_card = AgentCard(
                name="Default NL2SQL Agent",
                description="Default agent card for NL2SQL connection",
                url=agent_url,
                version="1.0.0",
                defaultInputModes=["text"],
                defaultOutputModes=["text"],
                capabilities=capabilities,
                skills=[]
            )
            
        self.agent_client = A2AClient(
            self._httpx_client, agent_card, url=agent_url
        )
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(
        self, message_request: SendMessageRequest
    ) -> SendMessageResponse:
        return await self.agent_client.send_message(message_request)