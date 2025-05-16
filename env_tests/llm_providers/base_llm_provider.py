import abc
from typing import List, Dict, Any, Optional

class BaseLLMProvider(abc.ABC):
    """Base class for LLM providers.
    
    This abstract class defines the interface that all LLM providers must implement.
    """
    
    @abc.abstractmethod
    def __init__(self, api_key: Optional[str] = None, model: str = None, **kwargs):
        """Initialize the LLM provider.
        
        Args:
            api_key: API key for the LLM provider
            model: Model name to use
            **kwargs: Additional provider-specific arguments
        """
        pass
    
    @abc.abstractmethod
    def is_initialized(self) -> bool:
        """Check if the LLM provider is properly initialized.
        
        Returns:
            bool: True if the provider is initialized, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def generate_response(self, 
                         system_prompt: str, 
                         messages: List[Dict[str, Any]], 
                         temperature: float = 0.2, 
                         max_tokens: int = 1000,
                         **kwargs) -> str:
        """Generate a response from the LLM.
        
        Args:
            system_prompt: The system prompt to use
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific arguments
            
        Returns:
            str: The generated response text
        """
        pass
    
    @abc.abstractmethod
    def format_message_content(self, text_content: str, image_data: List[Dict[str, Any]] = None) -> Any:
        """Format message content for the specific LLM provider.
        
        Args:
            text_content: Text content of the message
            image_data: List of image data dictionaries, each containing at least 'base64' and 'type'
            
        Returns:
            Any: Formatted message content in the format expected by the provider
        """
        pass
    
    @abc.abstractmethod
    def add_history_to_message(self, content, chat_history):
        """
        Add chat history to the message content.
        
        Args:
            content: The content of the message
            chat_history: List of chat history dictionaries
            
        Returns:
            Any: Formatted message content in the format expected by the provider
        """
        pass
    
