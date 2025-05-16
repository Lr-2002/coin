import logging
from typing import Dict, Any, Optional, Type

from .base_llm_provider import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider


logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory class for creating LLM provider instances."""
    
    # Registry of available provider classes
    _providers = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]):
        """Register a new LLM provider.
        
        Args:
            name: Name of the provider
            provider_class: Provider class that implements BaseLLMProvider
        """
        cls._providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")
    
    @classmethod
    def create_provider(cls, provider_name: str, **kwargs) -> Optional[BaseLLMProvider]:
        """Create an instance of the specified LLM provider.
        
        Args:
            provider_name: Name of the provider to create
            **kwargs: Arguments to pass to the provider constructor
            
        Returns:
            BaseLLMProvider: Instance of the requested provider, or None if not available
        """
        if provider_name not in cls._providers:
            logger.error(f"LLM provider '{provider_name}' not found. Available providers: {list(cls._providers.keys())}")
            return None
        
        try:
            provider = cls._providers[provider_name](**kwargs)
            if not provider.is_initialized():
                logger.warning(f"LLM provider '{provider_name}' was created but not properly initialized.")
            return provider
        except Exception as e:
            logger.error(f"Error creating LLM provider '{provider_name}': {e}")
            return None
    
    @classmethod
    def list_available_providers(cls) -> list:
        """List all available LLM providers.
        
        Returns:
            list: List of available provider names
        """
        return list(cls._providers.keys())
