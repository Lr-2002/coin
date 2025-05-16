"""
OpenAI LLM provider implementation.
Input: 
    system_prompt: str
    messages: List[Dict[str, Any]]
        role: str
        content: List[Dict[str, Any]]
    temperature: float
    max_tokens: int
    **kwargs
Output: str
"""

import os
import logging
import base64
from typing import List, Dict, Any, Optional

from openai import OpenAI, AzureOpenAI
from env_tests.llm_providers.base_llm_provider import BaseLLMProvider
from env_tests.utils.image_utils import encode_image, extract_camera_image

logger = logging.getLogger(__name__)

REGION='eastus'
MODEL='gpt-4o-2024-11-20'

# REGION='eastus2'
# MODEL='o1-2024-12-17'
API_KEY="163eb119c0853717661df499f0fe06c7" 

API_BASE='http://123.127.249.51/proxy'
ENDPOINT=f'{API_BASE}/{REGION}'

class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                model: str = "gpt-4o", 
                base_url: str = "https://api.openai-proxy.org/v1",
                **kwargs):
        """Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o)
            base_url: Base URL for the OpenAI API
            **kwargs: Additional arguments
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        
        # if not self.api_key:
        #     logger.warning("No OpenAI API key provided. Please set OPENAI_API_KEY environment variable.")
        #     self.client = None
        # else:
        #     self.client = OpenAI(
        #         base_url=self.base_url,
        #         api_key=self.api_key
        #     )


        self.client = AzureOpenAI(
            api_key=API_KEY,
            api_version="2025-03-01-preview",
            azure_endpoint=ENDPOINT
        )
    
    def is_initialized(self) -> bool:
        """Check if the OpenAI client is properly initialized.
        
        Returns:
            bool: True if the client is initialized, False otherwise
        """
        return self.client is not None
    
    def generate_response(self, 
                         system_prompt: str, 
                         messages: List[Dict[str, Any]], 
                         temperature: float = 0.2, 
                         max_tokens: int = 1000,
                         **kwargs) -> str:
        """Generate a response from OpenAI.
        
        Args:
            system_prompt: The system prompt to use
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional OpenAI-specific arguments
            
        Returns:
            str: The generated response text
        """
        if not self.is_initialized():
            logger.error("OpenAI client not initialized. Please provide an API key.")
            return ""
        
        try:
            # Prepare messages with system prompt as plain text
            formatted_messages = [{"role": "system", "content": system_prompt}]
            
            # Process and add the user/assistant messages
            for msg in messages:
                if 'role' not in msg or 'content' not in msg:
                    formatted_messages.append({"role": "user", "content": [msg]})
                else:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role is None or content is None:
                        logger.error(f"Error generating response from OpenAI: role or content is None")
                        continue
                    # Add the message with appropriate formatting
                    formatted_messages.append({"role": role, "content": content})
            # For debugging
            print(f"Sending {len(formatted_messages)} messages to OpenAI API")
            # if len(formatted_messages) == 4:
            #     print(f"formatted_messages: {formatted_messages}")
            
            # Call the OpenAI API
            if 'o1' in MODEL:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=formatted_messages,
                    max_completion_tokens=max_tokens,
                    reasoning_effort="low",
                    **kwargs
                )
            else:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            
            # Extract and return the response text
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            return None

    def format_message_content(self, text_content: str, image_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Format message content for OpenAI.
        
        Args:
            text_content: Text content of the message
            image_data: List of image data dictionaries, each containing at least 'base64' and 'type'
            
        Returns:
            list: List of content items in OpenAI format
        """
        content = [{"type": "text", "text": str(text_content)}]
        
        if image_data:
            for img in image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img.get('type', 'jpeg')};base64,{img['base64']}",
                        "detail": img.get('detail', 'auto')
                    }
                })
                
                # Add image caption if provided
                if 'caption' in img and img['caption']:
                    content.append({"type": "text", "text": img['caption']})
        
        return content

    def add_history_to_message(self, content, chat_history):
        # Prepare the full messages list including recent chat history
        # after fromat message content, the content is a list dict
        messages = []
        if content:
            messages.append({"role": "user", "content": content})
        # Convert plain text to the correct structure for OpenAI
        if not chat_history:
            return messages
        messages.append({"role": "user", "content": [{"type": "text", "text": f'Below there are last {len(chat_history)} recent chat history:'}]})
            
        # Add recent chat history messages
        for chat in chat_history:
            chat_role = chat.get('role', 'user')
            chat_content = chat.get('content', '')
            chat_images = chat.get('images', None)
                
            # Format each chat message properly
            formatted_chat = self.format_message_content(
                str(chat_content) if not isinstance(chat_content, str) else chat_content,
                chat_images
            )
                
            messages.append({"role": chat_role, "content": formatted_chat})
        return messages
