"""
Gemini LLM provider implementation.
Input: 
    system_prompt: str
    messages: List[Dict[str, Any]]
        content: List[Dict[str, Any]]
    temperature: float
    max_tokens: int
    **kwargs
Output: str
"""

import os
import logging
import base64
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai
from google.generativeai import types

from .base_llm_provider import BaseLLMProvider
from env_tests.utils.image_utils import encode_image, extract_camera_image
from env_tests.llm_providers.google_api import GOOGLE_API_LIST

logger = logging.getLogger(__name__)

class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                model: str = "gemini-2.0-flash",
                **kwargs):
        """Initialize the Gemini provider.
        
        Args:
            api_key: Google API key (if None, will use GOOGLE_API_KEY env var)
            model: Gemini model to use (default: gemini-2.0-flash)
            **kwargs: Additional arguments
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        GOOGLE_API_LIST.reverse()
        api_keys = GOOGLE_API_LIST
        # breakpoint()

        self.client = None
        for key in api_keys:
            if not key:
                logger.warning("No Google API key provided in env. Use the default key in GOOGLE_API_LIST.")
                continue
            try:
                # Configure the Gemini API
                genai.configure(api_key=key)
                self.api_key = key # Update the api_key to the one that worked
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"Gemini client initialized successfully with API key: {key[:5]}...")
                break # Exit the loop if initialization is successful
            except Exception as e:
                logger.error(f"Error initializing Gemini client with API key {key[:5]}...: {e}")
        
        if self.client is None:
            logger.error("Failed to initialize Gemini client with any provided API keys.")
    
    def is_initialized(self) -> bool:
        """Check if the Gemini client is properly initialized.
        
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
        """Generate a response from Gemini.
        
        Args:
            system_prompt: The system prompt to use
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional Gemini-specific arguments
            
        Returns:
            str: The generated response text
        """
        if not self.is_initialized():
            logger.error("Gemini client not initialized. Please provide an API key and install the required package.")
            return ""
        
        try:
            # Set generation config directly on the model if client exists
            if self.client is not None:
                self.client.temperature = temperature
                self.client.max_output_tokens = max_tokens
            else:
                raise ValueError("Gemini client is not initialized")
            
            # Format messages for Gemini API
            gemini_messages = [system_prompt] + messages

            print(f"Sending {len(gemini_messages)} messages to Gemini API")
            # Send the message and get the response
            response = self.client.generate_content(
                gemini_messages,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    **kwargs
                }
            )
            
            # Return the text response
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            return ""
    
    def format_message_content(self, text_content: str, image_data: List[Dict[str, Any]] = None) -> List:
        """Format message content for Gemini.
        
        Args:
            text_content: Text content of the message
            image_data: List of image data dictionaries, each containing at least 'base64' and 'type'
            
        Returns:
            list: List of content items in Gemini format (text strings and Part objects)
        """
        # Ensure base64 is imported
        
        parts = [str(text_content)] if text_content else []
        
        if image_data:
            if isinstance(image_data[0], str):
                for img_data in image_data:
                    if isinstance(img_data, str) and os.path.exists(img_data):
                        # If image_data is a file path
                        with open(img_data, "rb") as img_file:
                            import base64
                            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            })
                    elif isinstance(img_data, np.ndarray):
                        # If image_data is a numpy array
                        import base64
                        from io import BytesIO
                        img = Image.fromarray(img_data)
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        parts.append({
                            "mime_type": "image/jpeg",
                            "data": encoded_image})
            else:
                for img in image_data:
                    # breakpoint()
                    try:
                        import base64
                        # Decode the base64 string to bytes
                        image_bytes = base64.b64decode(img['base64'])
                        mime_type = f"image/{img.get('type', 'jpeg')}"
                        
                        # Create the Part object correctly
                        image_part = {
                            "mime_type": mime_type,
                            "data": image_bytes
                        }
                        # Append the Part object directly
                        parts.append(image_part)
                        
                        # Add image caption as a separate text part if provided
                        if 'caption' in img and img['caption']:
                            parts.append(img['caption'])
                    except KeyError as e:
                        logger.error(f"Missing key {e} in image data dictionary: {img.keys()}")
                    except base64.binascii.Error as e:
                        logger.error(f"Error decoding base64 string for image: {e}")
                    except Exception as e:
                        logger.error(f"Error processing image data: {e}")
        
        return parts

    def add_history_to_message(self, content, chat_history):
        """Add chat history to message content.
        
        Args:
            content: List of content items in Gemini format
            chat_history: List of chat history entries
            
        Returns:
            list: List of content items with chat history added
        """
        if not chat_history:
            return content
        
        chat_history_prompt = f"Below there are last {len(chat_history)} recent chat history:'"
        content.append(chat_history_prompt)
        for chat_entry in chat_history:
            role = chat_entry['role']
            content.append(f"{role}: {chat_entry['content']}")

            if chat_entry.get('images'):
                # images: list dict for different cameras
                parts = self.format_message_content(None, chat_entry['images'])
                content.extend(parts)

        return content
                    

if __name__ == "__main__":
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
    provider = GeminiProvider(api_key=api_key, model="gemini-2.0-flash")
    response = provider.generate_response(
        system_prompt="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=0.2,
        max_tokens=1000
    )
    print(response)
