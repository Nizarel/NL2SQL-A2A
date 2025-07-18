"""
Base Agent Class for NL2SQL Multi-Agent System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the NL2SQL system
    """
    
    def __init__(self, kernel: Kernel, agent_name: str):
        self.kernel = kernel
        self.agent_name = agent_name
        self.chat_history = ChatHistory()
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    async def _get_ai_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """
        Get response from AI service
        
        Args:
            prompt: The prompt to send to AI
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            
        Returns:
            AI response as string
        """
        # Get AI service from kernel
        ai_services = [service for service in self.kernel.services.values()]
        if not ai_services:
            raise RuntimeError(f"No AI service configured for {self.agent_name}")
        
        ai_service = ai_services[0]
        
        # Create chat history with prompt
        chat_history = ChatHistory()
        chat_history.add_user_message(prompt)
        
        # Create settings
        settings = OpenAIChatPromptExecutionSettings(
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Get response
        response = await ai_service.get_chat_message_content(
            chat_history=chat_history,
            settings=settings,
            kernel=self.kernel
        )
        
        return str(response).strip()
    
    def _create_result(self, success: bool, data: Any = None, error: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create standardized result dictionary
        
        Args:
            success: Whether the operation was successful
            data: The result data
            error: Error message if any
            metadata: Additional metadata
            
        Returns:
            Standardized result dictionary
        """
        return {
            "success": success,
            "agent": self.agent_name,
            "data": data,
            "error": error,
            "metadata": metadata or {}
        }
