
import os
import json
import importlib.resources
import logging
import re
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import hashlib
import pickle

# Import LLM providers

# Add the current directory to the path to allow importing gpt_evaluator
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

sys.path.append("/media/raid/workspace/wangxianhao/project/reasoning/ManiSkill")
sys.path.append("./")
# print(sys.path)
from env_tests.llm_providers.gemini_provider import GeminiProvider
# Import GPTEvaluator
try:
    from gpt_evaluator import GPTEvaluator
except ImportError:
    print(f"Warning: Could not import GPTEvaluator. Current sys.path: {sys.path}")
    GPTEvaluator = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory class to create LLM providers."""
    
    @staticmethod
    def create_provider(provider_name, **kwargs):
        """Create an LLM provider instance based on the provider name."""
        print('using provider', provider_name)
        # input()
        if provider_name.lower() == "openai":
            return OpenAIProvider(**kwargs)
        elif provider_name.lower() == "azure":
            return AzureProvider(**kwargs)
        elif provider_name.lower() == "gemini":
            return GeminiProvider(**kwargs)
        elif provider_name.lower() == "mock":
            return MockLLMProvider(**kwargs)
        else:
            logger.error(f"Unsupported LLM provider: {provider_name}")
            return None

class BaseLLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, api_key=None, model=None, use_cache=True, cache_dir="llm_cache", **kwargs):
        self.api_key = api_key
        self.model = model
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
    def is_initialized(self):
        """Check if the provider is properly initialized."""
        return self.api_key is not None
        
    def format_message_content(self, text_content, image_data=None):
        """Format message content for the provider."""
        # Default implementation for text-only content
        return {"role": "user", "content": text_content}
        
    def add_history_to_message(self, current_message, chat_history):
        """Add chat history to the current message."""
        # Default implementation just returns the current message
        return [current_message]
        
    def generate_response(self, system_prompt, messages, temperature=0.7, max_tokens=1000):
        """Generate a response from the LLM."""
        raise NotImplementedError
        
    def _get_cache_key(self, system_prompt, messages, temperature, max_tokens):
        """Generate a unique cache key for the request."""
        # Create a string representation of the request
        request_str = f"{self.model}_{system_prompt}_{str(messages)}_{temperature}_{max_tokens}"
        
        # Generate a hash of the request string
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get the path to the cache file for the given key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _save_to_cache(self, cache_key, response):
        """Save the response to the cache."""
        if not self.use_cache:
            return
            
        cache_path = self._get_cache_path(cache_key)
        cache_data = {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "model": self.model
        }
        
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
    
    def _load_from_cache(self, cache_key):
        """Load the response from the cache if it exists."""
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                return cache_data["response"]
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")
        
        return None

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key=None, model="gpt-4o", **kwargs):
        super().__init__(api_key, model, **kwargs)
        import openai
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not provided and not found in environment variables.")
            return
            
        self.client = openai.OpenAI(api_key=self.api_key)
        
    def format_message_content(self, text_content, image_data=None):
        """Format message content for OpenAI."""
        content = []
        
        # Add text content
        if isinstance(text_content, dict):
            text_content = "\n".join([f"{k}: {v}" for k, v in text_content.items()])
        content.append({"type": "text", "text": text_content})
        
        # Add image data if provided
        if image_data:
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
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    })
        return {"role": "user", "content": content}
        
    def add_history_to_message(self, current_message, chat_history):
        """Add chat history to the current message."""
        messages = []
        
        # Add chat history
        for entry in chat_history:
            role = entry.get("role")
            content = entry.get("content")
            images = entry.get("images", [])
            
            if role in ["user", "assistant", "system"]:
                if role == "user" and images:
                    # Format user message with images
                    messages.append(self.format_message_content(content, images))
                else:
                    # Format message without images
                    messages.append({"role": role, "content": content})
                    
        # Add current message
        messages.append(current_message)
        
        return messages
        
    def generate_response(self, system_prompt, messages, temperature=0.7, max_tokens=1000):
        """Generate a response from OpenAI."""
        try:
            # Check cache first
            cache_key = self._get_cache_key(system_prompt, messages, temperature, max_tokens)
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response is not None:
                logger.info("Using cached response")
                return cached_response
            
            # Add system prompt
            from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
            
            # Convert messages to proper types
            typed_messages = []
            # Add system prompt
            typed_messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))
            
            # Process the rest of the messages
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                
                if role == "user":
                    typed_messages.append(ChatCompletionUserMessageParam(role="user", content=content))
                elif role == "assistant":
                    typed_messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=content))
                elif role == "system":
                    typed_messages.append(ChatCompletionSystemMessageParam(role="system", content=content))
            
            # Ensure model is not None to fix type error
            model_name = self.model if self.model is not None else "gpt-4"
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=typed_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            return None

class AzureProvider(BaseLLMProvider):
    """Azure OpenAI API provider."""
    
    def __init__(self, api_key=None, model="gpt-4", region="eastus2", 
                 api_base="https://api.tonggpt.mybigai.ac.cn/proxy", 
                 api_version="2025-03-01-preview", **kwargs):
        super().__init__(api_key, model, **kwargs)
        from openai import AzureOpenAI
        # breakpoint() 
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            logger.error("Azure API key not provided and not found in environment variables.")
            return
            
        self.endpoint = f"{api_base}/{region}"
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=api_version,
            azure_endpoint=self.endpoint,
        )
        
    def format_message_content(self, text_content, image_data=None):
        """Format message content for Azure OpenAI."""
        # Azure OpenAI has a different format for multimodal content
        content = []
        
        # Add text content
        if isinstance(text_content, dict):
            text_content = "\n".join([f"{k}: {v}" for k, v in text_content.items()])
        content.append({"type": "text", "text": text_content})
        
        # Add image data if provided
        if image_data:
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
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    })
                
        return {"role": "user", "content": content}
        
    def add_history_to_message(self, current_message, chat_history):
        """Add chat history to the current message."""
        messages = []
        
        # Add chat history
        for entry in chat_history:
            role = entry.get("role")
            content = entry.get("content")
            images = entry.get("images", [])
            
            if role in ["user", "assistant", "system"]:
                if role == "user" and images:
                    # Format user message with images
                    messages.append(self.format_message_content(content, images))
                else:
                    # Format message without images
                    messages.append({"role": role, "content": content})
                    
        # Add current message
        messages.append(current_message)
        
        return messages
        
    def generate_response(self, system_prompt, messages, temperature=0.7, max_tokens=1000):
        """Generate a response from Azure OpenAI."""
        try:
            # Check cache first
            # cache_key = self._get_cache_key(system_prompt, messages, temperature, max_tokens)
            # cached_response = self._load_from_cache(cache_key)
            
            # if cached_response is not None:
            #     logger.info("Using cached response")
            #     return cached_response
            
            # Add system prompt
            from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
            
            # Convert messages to proper types
            typed_messages = []
            # Add system prompt
            typed_messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))
            
            # Process the rest of the messages
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                
                if role == "user":
                    typed_messages.append(ChatCompletionUserMessageParam(role="user", content=content))
                elif role == "assistant":
                    typed_messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=content))
                elif role == "system":
                    typed_messages.append(ChatCompletionSystemMessageParam(role="system", content=content))
            
            # Ensure model is not None to fix type error
            model_name = self.model if self.model is not None else "gpt-4"
            if 'o1' in model_name:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=typed_messages,
                    # temperature=temperature,
                    max_completion_tokens=max_tokens ,
                    reasoning_effort='low'
                )
            else: 
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=typed_messages,
                    temperature=temperature,
                    max_tokens=max_tokens 
                )

            result = response.choices[0].message.content
            
            # Save to cache
            # self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            return None

class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing without an API key."""
    
    def __init__(self, api_key=None, model="mock-model", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.task_responses = {
            "Tabletop-Find-Book-FromShelf-v1": """
            Assessment: I can see a bookshelf with multiple books on it. There appears to be a task to find and retrieve a specific book from the shelf.
            
            Plan:
            1. Move the robot arm to face the bookshelf
            2. Scan the bookshelf to identify the highest book
            3. Position the gripper near the highest book on the shelf
            4. Grasp the book firmly with the gripper
            5. Carefully pull the book out from the shelf
            6. Move the book to the designated marker or target location
            7. Release the gripper to place the book down
            """,
            
            "Tabletop-Open-Cabinet-v1": """
            Assessment: I can see a closed cabinet on the tabletop. The task is to open the cabinet door.
            
            Plan:
            1. Position the robot arm in front of the cabinet door
            2. Align the gripper with the cabinet handle
            3. Close the gripper to grasp the handle firmly
            4. Pull the cabinet door open by moving the arm away from the cabinet
            5. Release the handle once the door is fully open
            """,
            
            "default": """
            Assessment: I can see a tabletop manipulation task, but I'm not entirely sure what the specific goal is from the image alone.
            
            Plan:
            1. Analyze the scene to identify key objects
            2. Determine the target object for manipulation
            3. Position the robot arm near the target object
            4. Grasp the object with the gripper
            5. Move the object to the desired location
            6. Release the object once properly positioned
            7. Verify the task has been completed successfully
            """
        }
        
    def is_initialized(self):
        """Check if the provider is properly initialized."""
        return True
        
    def format_message_content(self, text_content, image_data=None):
        """Format message content for the mock provider."""
        # For mock provider, we just return a simple format
        if isinstance(text_content, dict):
            text_content = "\n".join([f"{k}: {v}" for k, v in text_content.items()])
        return {"role": "user", "content": text_content}
        
    def add_history_to_message(self, current_message, chat_history):
        """Add chat history to the current message."""
        # For mock provider, we just return the current message
        return [current_message]
        
    def generate_response(self, system_prompt, messages, temperature=0.7, max_tokens=1000):
        """Generate a mock response based on the task name."""
        try:
            # Check cache first
            cache_key = self._get_cache_key(system_prompt, messages, temperature, max_tokens)
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response is not None:
                logger.info("Using cached response")
                return cached_response
            
            # For the mock provider, we'll just return a fixed response based on the task
            # Extract the task name from the messages
            task_name = ""
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str) and "instruction:" in content.lower():
                    # Extract the task name from the instruction
                    task_parts = content.split("instruction:")[1].strip().split()
                    task_name = " ".join(task_parts)
                    break
            
            # Generate a mock response based on the task
            if "find book" in task_name.lower():
                result = "Assessment: The task is to find a book from the shelf.\n\nPlan:\n    1. find and pick the book from the bookshelf and put it on the marker"
            elif "open door" in task_name.lower():
                result = "Assessment: The task is to open a door.\n\nPlan:\n    1. find the door\n    2. grasp the door handle\n    3. pull the door open"
            elif "pick cube" in task_name.lower():
                result = "Assessment: The task is to pick up a cube.\n\nPlan:\n    1. locate the cube\n    2. pick up the cube"
            else:
                result = "Assessment: The task is to manipulate an object.\n\nPlan:\n    1. locate the object\n    2. interact with the object"
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating mock response: {e}")
            return self.task_responses["default"]

class LLMAgentPlanner:
    """LLM Agent Planner class."""
    
    def __init__(self, llm_provider="azure", api_key=None, model="gpt-4o", 
                 checkpoint_dir="coin_videos/checkpoint", use_cache=True, **kwargs):
        """Initialize the LLM Agent Planner.
        
        Args:
            llm_provider (str): The LLM provider to use. Options: "openai", "azure", "mock".
            api_key (str): The API key for the LLM provider.
            model (str): The model to use.
            checkpoint_dir (str): The directory containing checkpoint files.
            use_cache (bool): Whether to use caching for LLM responses.
        """
        self.checkpoint_dir = checkpoint_dir
        self.use_cache = use_cache
        
        # Always use Azure OpenAI by default unless explicitly set to mock
        if llm_provider.lower() == "openai":
            logger.warning("OpenAI provider specified, but using Azure OpenAI instead for better compatibility.")
            llm_provider = "azure"
            
        self.llm_provider_name = llm_provider
        self.api_key = api_key
        self.model = model
        self.base_dirs = None
        self.step = 0
        self.chat_history = []
        self.chat_history_length = 5  # Number of recent messages to include in context
        self.subtasks = []
        self.current_subtask = None
        
        # Initialize the LLM provider
        self.llm_provider = self._initialize_llm_provider(llm_provider, api_key, model, use_cache=use_cache, **kwargs)
        
        if not self.llm_provider or not self.llm_provider.is_initialized():
            logger.warning(f"Failed to initialize {llm_provider} provider. Please check your API key and configuration.")
            
        # Initialize system prompt
        self.system_prompt = self.get_system_prompt()

        # Task state
        self.high_level_instruction = None
        self.subtasks = []
        self.max_subtasks = 10
        self.current_subtask = None

        # Chat history
        self.chat_history = []
        self.chat_history_length = 3
        
        # Load environment workflows (ground truth plans)
        self.env_workflows = self.load_env_workflows()
    
    def load_env_workflows(self):
        """Load environment workflows from JSON file."""
        try:
            with open('env_workflows.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading environment workflows: {e}")
            return {}

    def _initialize_llm_provider(self, provider_name, api_key=None, model=None, use_cache=True, **kwargs):
        """Initialize the LLM provider.
        
        Args:
            provider_name (str): The name of the provider.
            api_key (str): The API key for the provider.
            model (str): The model to use.
            use_cache (bool): Whether to use caching for LLM responses.
            
        Returns:
            BaseLLMProvider: The initialized provider.
        """
        # Ensure model is not None to fix type errors
        model_name = model if model is not None else "gpt-4"
        
        if provider_name.lower() == "openai":
            return OpenAIProvider(api_key, model_name, use_cache=use_cache, **kwargs)
        elif provider_name.lower() == "azure":
            return AzureProvider(api_key, model_name, use_cache=use_cache, **kwargs)
        elif provider_name.lower() == "gemini":
            return GeminiProvider(api_key, model_name, **kwargs)
        elif provider_name.lower() == "mock":
            return MockLLMProvider(api_key, model_name, use_cache=use_cache, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
            
    def get_available_instructions(self):
        """Get available instructions from JSON file."""
        available_instructions_file = os.path.join(os.path.dirname(__file__), "available_instructions.json")
        try:
            with open(available_instructions_file, 'r') as f:
                available_instructions = list(json.load(f).values())
            return available_instructions
        except Exception as e:
            logger.warning(f"Error loading available instructions: {e}")
            return []
    
    def get_system_prompt(self):
        """Get the system prompt for the LLM."""
        # Read the prompt from the prompt.txt file
        try:
            # Try to use importlib.resources if available
            try:
                prompt_file = importlib.resources.files("mani_skill.prompts") / "prompt.txt"
                with open(str(prompt_file), 'r') as f:
                    return f.read()
            except (ImportError, TypeError):
                # Fallback to direct file path
                prompt_path = os.path.join(os.path.dirname(__file__), "../../mani_skill/prompts", "prompt.txt")
                with open(prompt_path, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error reading prompt file: {e}")
            # Fallback to default prompt
            return "You are a robotic planning assistant. Given images of a scene and a high-level instruction, break down the task into a sequence of subtasks."
    
    def get_image_data(self, obs, camera_names=None):
        """Return the image data for the LLM from checkpoint JSON files.
        
        Args:
            obs: Observation dictionary or task name to load images for
            camera_names: List of camera names to get images for (not used with coin_videos format)
            
        Returns:
            List of image paths from the checkpoint directory
        """
        if not self.checkpoint_dir:
            logger.error("Checkpoint directory not set")
            return []
            
        # If obs is a string, treat it as a task name
        task_name = obs if isinstance(obs, str) else None
        
        if not task_name:
            logger.error("Task name not provided")
            return []
        
        # Convert task name format from Tabletop-Find-Book-FromShelf-v1 to Tabletop_Find_Book_FromShelf_v1
        task_name = task_name.replace('-', '_')
        
        # Find the checkpoint JSON file for this task
        checkpoint_file = os.path.join(self.checkpoint_dir, task_name, "checkpoint.json")
        
        # Load the checkpoint JSON
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading checkpoint file {checkpoint_file}: {e}")
            return []
        
        # Extract image paths from the checkpoint data
        # In coin_videos format, images are stored in the "saved_frames" list
        image_paths = []
        if "saved_frames" in checkpoint_data:
            # Use all available frames
            image_paths = checkpoint_data["saved_frames"]
            logger.info(f"Found {len(image_paths)} images for task {task_name}")
        else:
            logger.warning(f"No saved_frames found in checkpoint data for {task_name}")
        
        return image_paths
        
    def parse_llm_response(self, response):
        """Parse the LLM response to extract subtasks and the next instruction.
        
        Args:
            response: The LLM response string
        
        Returns:
            result: Dictionary containing 'assessment', 'subtasks', and 'next_instruction'
        """
        result = {
            'assessment': '',
            'subtasks': [],
            'next_instruction': ''
        }
    
        # Split the response into lines
        lines = response.split('\n')
        
        # Current section being parsed
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers (no space required after colon)
            if line.startswith('Assessment:'):
                current_section = 'assessment'
                result['assessment'] = line[len('Assessment:'):].strip()
            elif line.startswith('Plan:'):
                current_section = 'plan'
                result['subtasks'] = []
            # Parse nested subtasks in the plan section
            elif current_section == 'plan' and line:
                # Check if the line starts with a number followed by a period
                if line[0].isdigit() and '.' in line[:3]:
                    # Extract the subtask, removing the numbering
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        subtask = parts[1].strip()
                        if subtask:  # Only add non-empty subtasks
                            if len(result['subtasks']) < self.max_subtasks:
                                result['subtasks'].append(subtask)
                            else:
                                logger.warning(f"More than {self.max_subtasks} subtasks provided; ignoring excess.")
        
        # Check if subtasks list is not empty before accessing the first element
        if result['subtasks']:
            result['next_instruction'] = result['subtasks'][0]
        else:
            logger.warning("No valid subtasks found in LLM response.")
            result['next_instruction'] = self.current_subtask

        return result
    
    def add_to_chat_history(self, role, content, images=None) -> List[Dict[str, Any]]:
        """Add a chat interaction to the chat history.
        
        Args:
            role: The role of the speaker (e.g., 'user', 'assistant', 'system')
            content: The content of the message
            images: Optional list of image data to include with the message
        
        Returns:
            List[Dict[str, Any]]: Updated chat history
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_entry = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        # Add images if provided
        if images:
            chat_entry["images"] = images
            
        self.chat_history.append(chat_entry)
        # Keep only the last 10 interactions to prevent the history from growing too large
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        
        return self.chat_history

    def generate_plan_from_llm(self, obs, task_name=None):
        """Generate or update a plan from the LLM based on observations.

        Args:
            obs: Current observation from the environment or task name
            task_name: Optional task name to use for loading images and ground truth plans

        Returns:    
            bool: Whether planning was successful
            dict: The parsed response from the LLM
        """
        if not self.llm_provider or not self.llm_provider.is_initialized():
            logger.error(f"LLM provider '{self.llm_provider_name}' not initialized. Please check your API key and configuration.")
            return False, None

        try:
            # Store the original task name for evaluation purposes
            original_task_name = task_name
            
            # If task_name is provided, use it to set the high-level instruction
            if task_name :
                # Load high-level instruction from interactive_instruction_objects.pkl
                try:
                    instruction_file = 'interactive_instruction_objects.pkl'
                    if os.path.exists(instruction_file):
                        with open(instruction_file, 'rb') as f:
                            instruction_objects = pickle.load(f)
                        
                        # Try to find the instruction for this task
                        if task_name in instruction_objects:
                            self.high_level_instruction = instruction_objects[task_name]['ins']
                            logger.info(f"Loaded instruction for {task_name} from interactive_instruction_objects.pkl")
                        else:
                            # Fallback to deriving from task name if not found in pickle file
                            logger.warning(f"Task {task_name} not found in interactive_instruction_objects.pkl, using derived instruction")
                            self.high_level_instruction = task_name.replace('-', ' ').replace('_', ' ')
                    else:
                        logger.warning(f"interactive_instruction_objects.pkl not found at {instruction_file}, using derived instruction")
                        self.high_level_instruction = task_name.replace('-', ' ').replace('_', ' ')
                except Exception as e:
                    logger.error(f"Error loading interactive_instruction_objects.pkl: {e}")
                    # Fallback to deriving from task name
                    self.high_level_instruction = task_name.replace('-', ' ').replace('_', ' ')
       
            text_content = {}
            text_content['High-level instruction'] = self.high_level_instruction
            text_content['Step'] = self.step
            if self.step:
                text_content['Current subtask'] = self.current_subtask
                text_content['Current plan (remaining subtasks)'] = self.subtasks

            # Get images for the specified cameras
            camera_list = ["human_camera", "hand_camera", "base_front_camera"]  # Use the first two cameras by default
            image_data = self.get_image_data(task_name or obs, camera_list)

            recent_chat_history = self.chat_history[-self.chat_history_length:]

            # Add this interaction to chat history (before sending to LLM)
            self.add_to_chat_history('user', text_content, image_data)

            # Format the current message content without including chat history
            current_message = self.llm_provider.format_message_content(str(text_content), [image_data[-1]] if image_data  else image_data)  
            messages = self.llm_provider.add_history_to_message(current_message, recent_chat_history) 
            
            system_prompt = self.get_system_prompt() + "\n\n"
            
            # Store the image paths for recording
            image_paths = {}
            if self.base_dirs:
                for camera in ["human_camera", "hand_camera", "base_front_camera"]:
                    image_paths[camera] = os.path.join(self.base_dirs[camera], f"step_{self.step:04d}.png")

            print(f"{'-'*100}")
            logger.info(f"Generating plan at step {self.step} with instruction: {self.high_level_instruction}")
            print(f"{'-'*100}")
            
            # Get the LLM response with images
            response = self.llm_provider.generate_response(
                system_prompt=system_prompt,
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )
            
            if response is None:
                logger.error("LLM response is None.")
                return False, None
                
            print(f"response: {response}")
            # Parse the response
            parsed_response = self.parse_llm_response(response)
            
            # Add LLM response to chat history
            self.add_to_chat_history('assistant', response)
            
            # Update the agent's state with the parsed response
            if parsed_response:
                subtasks = parsed_response.get('subtasks', [])
                next_instruction = parsed_response.get('next_instruction', '')
                
                # Update subtasks if provided
                if subtasks:
                    self.subtasks = subtasks[:self.max_subtasks]
                
                # Set the next instruction as the current subtask
                if next_instruction:
                    self.current_subtask = next_instruction
                
                # Log the planning results
                logger.info(f"Assessment: {parsed_response.get('assessment', '')}")
                logger.info(f"Generated plan with {len(self.subtasks)} subtasks")
                for i, subtask in enumerate(self.subtasks):
                    logger.info(f"Subtask {i+1}: {subtask}")
                logger.info(f"Current subtask: {self.current_subtask}")
                
                return True, parsed_response
            else:
                logger.warning("Failed to parse LLM response.")
                return False, None
                
        except Exception as e:
            import traceback
            logger.error(f"Error in generate_plan_from_llm: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, None

    def save_evaluation_chat(self, task_name, ground_truth_plan, generated_plan, evaluation_result, response_text):
        """Save the evaluation chat to a file.
        
        Args:
            task_name: The task name
            ground_truth_plan: The ground truth plan
            generated_plan: The generated plan
            evaluation_result: The evaluation result dictionary
            response_text: The raw response from GPT
        """
        import datetime
        import os
        from pathlib import Path
        
        # Create the base directory if it doesn't exist
        base_chat_dir = "env_tests/agents/evaluate_chat_" + self.model.replace('-', '_')
        os.makedirs(base_chat_dir, exist_ok=True)
        
        # Create task-specific subdirectory
        task_folder_name = task_name.replace('-', '_')
        task_chat_dir = os.path.join(base_chat_dir, task_folder_name)
        os.makedirs(task_chat_dir, exist_ok=True)
        
        # Format the timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the number of images used (if available in the filename)
        num_images = "unknown"
        if hasattr(self, '_test_images'):
            num_images = len(self._test_images)
        
        # Create a filename with the task name, number of images, and timestamp
        filename = os.path.join(task_chat_dir, f"{num_images}_images_{timestamp}.md")
        
        # Format the ground truth and generated plans
        # Ensure ground_truth_plan is a list of strings, not individual characters
        if isinstance(ground_truth_plan, list) and all(isinstance(step, str) for step in ground_truth_plan):
            ground_truth_str = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(ground_truth_plan)])
        else:
            logger.warning(f"Ground truth plan for {task_name} is not properly formatted: {ground_truth_plan}")
            # Try to convert to a proper format if it's a string
            if isinstance(ground_truth_plan, str):
                # Split by newlines if it contains them
                if '\n' in ground_truth_plan:
                    steps = ground_truth_plan.strip().split('\n')
                    ground_truth_str = "\n".join([f"{step}" for i, step in enumerate(steps)])
                else:
                    # Just use the string as is
                    ground_truth_str = f"Step 1: {ground_truth_plan}"
            else:
                # Last resort: convert to string representation
                ground_truth_str = f"Invalid format: {str(ground_truth_plan)}"
        
        # Format the generated plan
        generated_plan_str = "\n".join([f"{step}" for i, step in enumerate(generated_plan.strip().split('\n'))])
        
        # Format the chat content
        chat_content = f"""# Plan Evaluation for {task_name} ({num_images} images)

## Ground Truth Plan
```
{ground_truth_str}
```

## Generated Plan
```
{generated_plan_str}
```

## Evaluation Results
- Completeness: {evaluation_result['completeness']}/100
- Correctness: {evaluation_result['correctness']}/100
- Clarity: {evaluation_result['clarity']}/100
- Mean Score: {evaluation_result['mean_score']}/100

## Justification
{evaluation_result['justification']}

## Raw GPT Response
```
{response_text}
```
"""
        
        # Write the chat content to the file
        with open(filename, "w") as f:
            f.write(chat_content)
            
        logger.info(f"Evaluation chat saved to {filename}")
        
        return filename
        
    def evaluate_plan(self, task_name, generated_plan):
        """Evaluate a generated plan against the ground truth plan.
        
        Args:
            task_name: The task name to get the ground truth plan for
            generated_plan: The generated plan to evaluate
            
        Returns:
            dict: Evaluation results with scores and justification
        """
        try:
            # Get the ground truth plan - ensure we're using the correct task name format with hyphens
            # as that's how it's stored in env_workflows.json
            normalized_task_name = task_name.replace('_', '-')
            ground_truth_plan = self.env_workflows.get(normalized_task_name, [])
            
            # If not found, try the original task name
            if not ground_truth_plan:
                ground_truth_plan = self.env_workflows.get(task_name, [])
                
            if not ground_truth_plan:
                logger.warning(f"No ground truth plan found for task {task_name} or {normalized_task_name}")
                return {"error": f"No ground truth plan found for task {task_name}"}
            
            # Ensure ground truth plan is properly formatted as a list of strings
            if not (isinstance(ground_truth_plan, list) and all(isinstance(step, str) for step in ground_truth_plan)):
                logger.warning(f"Ground truth plan for {task_name} is not properly formatted: {ground_truth_plan}")
                return {"error": f"Ground truth plan for {task_name} is not properly formatted"}
                
            # Format the plans for evaluation
            ground_truth_str = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(ground_truth_plan)])
            generated_plan_str = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(generated_plan)])
            
            # Check if GPTEvaluator was successfully imported
            if GPTEvaluator is None:
                logger.error("GPTEvaluator module could not be imported. Cannot evaluate plan.")
                return {
                    "completeness": 0,
                    "correctness": 0,
                    "clarity": 0,
                    "mean_score": 0,
                    "justification": "Evaluation failed: GPTEvaluator module not available",
                    "raw_response": ""
                }
                
            # Initialize the evaluator with Azure OpenAI
            # Set AZURE_OPENAI_API_KEY environment variable if not already set
            if self.api_key and "AZURE_OPENAI_API_KEY" not in os.environ:
                os.environ["AZURE_OPENAI_API_KEY"] = self.api_key
                
            evaluator = GPTEvaluator(
                model="gpt-4o-2024-11-20",
                region="eastus",
                api_base="http://123.127.249.51/proxy",
                api_version="2025-03-01-preview"
            )
            
            # Prepare the data for evaluation
            system_prompt = "You are an expert evaluator for robotic manipulation tasks."
            goal_instruction = f"The robot should perform the task: {task_name}"
            
            # Create the evaluation data
            evaluation_data = {
                "system_prompt": system_prompt,
                "goal_instruction": goal_instruction,
                "plans": {
                    "ground_truth": ground_truth_str,
                    "model_output": generated_plan_str
                }
            }
            
            # Build the messages for the evaluator
            messages = evaluator.build_instruction(evaluation_data)
            
            # Query the evaluator
            response_text = evaluator.query_gpt(messages)
            
            # Parse the scores
            scores = evaluator.parse_scores(response_text)
            
            # Define regex patterns as regular strings
            completeness_pattern = r'(?:\*\*)?Completeness(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/100'
            correctness_pattern = r'(?:\*\*)?Correctness(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/100'
            clarity_pattern = r'(?:\*\*)?Clarity(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/100'
            mean_pattern = r'(?:\*\*)?Mean Score(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/100'
            
            # Log the scores
            logger.info(f"Regex matches - Completeness: {re.search(completeness_pattern, response_text)}, "
                       f"Correctness: {re.search(correctness_pattern, response_text)}, "
                       f"Clarity: {re.search(clarity_pattern, response_text)}, "
                       f"Mean: {re.search(mean_pattern, response_text)}")
            logger.info(f"Raw response: {response_text[:100]}...")
            logger.info(f"Plan evaluation scores: Completeness: {scores['completeness']}, Correctness: {scores['correctness']}, Clarity: {scores['clarity']}, Mean: {scores['mean_score']}")
            logger.info(f"Justification: {scores['justification']}")
            
            # Save the evaluation chat
            self.save_evaluation_chat(task_name, ground_truth_str, generated_plan_str, scores, response_text)
            
            return scores
            
        except Exception as e:
            import traceback
            logger.error(f"Error in evaluate_plan: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

    def test_with_progressive_images(self, task_name, num_images=5, evaluate=False):
        """Test the planner with progressively more images.
        
        Args:
            task_name: The task name to test
            num_images: Maximum number of images to use
            evaluate: Whether to evaluate the generated plans
            
        Returns:
            dict: Test results
        """
        print('the evaluate is ', evaluate)
        # breakpoint()
        # Store the original task name for evaluation (with hyphens)
        original_task_name = task_name
        
        # Set the high-level instruction
        self.high_level_instruction = task_name.replace('-', ' ').replace('_', ' ')
        
        # Convert task name format from Tabletop-Find-Book-FromShelf-v1 to Tabletop_Find_Book_FromShelf_v1
        # for directory lookup only
        task_dir_name = task_name.replace('-', '_')
        
        # Load the checkpoint JSON
        checkpoint_file = os.path.join(self.checkpoint_dir, task_dir_name, "checkpoint.json")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading checkpoint file {checkpoint_file}: {e}")
            return {"error": f"Error loading checkpoint file: {e}"}
            
        # Get all available images
        available_images = checkpoint_data.get("saved_frames", [])
        if not available_images:
            raise FileNotFoundError('no image found')
            logger.error(f"No images found in checkpoint data for {task_name}")
            return {"error": "No images found in checkpoint data"}
            
        # Test with progressively more images
        results = {}
        for i in range(0, min(num_images + 1, len(available_images) + 1)):
            # Use i images
            test_images = available_images[:i]
            logger.info(f"Testing with {i} images")
            
            # Reset the planner state
            self.step = 0
            self.subtasks = []
            self.current_subtask = None
            self.chat_history = []
            
            # Store the test images to use
            self._test_images = test_images
            
            # Create a temporary method to override get_image_data
            def temp_get_image_data(self, obs, camera_names=None):
                return self._test_images
                
            # Save the original method and set the temporary one
            original_get_image_data = self.get_image_data
            self.get_image_data = temp_get_image_data.__get__(self, type(self))
            
            # Generate a plan
            # Pass the task_dir_name for image loading and original_task_name for instruction
            success, parsed_response = self.generate_plan_from_llm(task_dir_name, original_task_name)
            
            # Restore the original method
            self.get_image_data = original_get_image_data
            
            if success and parsed_response:
                # Store the results
                results[f"{i}_images"] = {
                    "images": test_images,
                    "subtasks": parsed_response.get('subtasks', []),
                    "assessment": parsed_response.get('assessment', '')
                }
                
                # Evaluate the plan if requested
                if evaluate:
                    # Always use the original task name (with hyphens) for evaluation
                    # to match the keys in env_workflows.json
                    evaluation = self.evaluate_plan(original_task_name, parsed_response.get('subtasks', []))
                    results[f"{i}_images"]["evaluation"] = evaluation
            
            # Increment the step
            self.step += 1
            
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the LLM Agent Planner')
    parser.add_argument('--task', type=str, required=True, help='Task name to test (e.g., Tabletop-Find-Book-FromShelf-v1)')
    parser.add_argument('--num-images', type=int, default=5, help='Maximum number of images to use')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the generated plans')
    parser.add_argument('--api-key', type=str, help='API key (for OpenAI)')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20', help='LLM model to use')
    parser.add_argument('--use-mock', action='store_true', help='Use mock LLM provider for testing without API key')
    parser.add_argument('--use-azure', action='store_true', help='Use Azure OpenAI instead of OpenAI')
    parser.add_argument('--azure-region', type=str, default='eastus', help='Azure region')
    parser.add_argument('--azure-api-base', type=str, default='https://api.tonggpt.mybigai.ac.cn/proxy', help='Azure API base URL')
    parser.add_argument('--azure-api-version', type=str, default='2025-03-01-preview', help='Azure API version')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of LLM responses')
    
    args = parser.parse_args()
    
    # Determine whether to use caching
    use_cache = not args.no_cache
    
    # Initialize the planner
    if args.use_mock:
        planner = LLMAgentPlanner(
            llm_provider="mock",
            model="mock-model",
            use_cache=use_cache
        )
        print("Using mock LLM provider for testing")
        if use_cache:
            print("LLM response caching is enabled")
    elif args.use_azure:
        # For Azure, we'll use the AZURE_OPENAI_API_KEY environment variable if no API key is provided
        azure_token = args.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not azure_token:
            print("Warning: No Azure token provided. Using AZURE_OPENAI_API_KEY environment variable.")
            
        planner = LLMAgentPlanner(
            llm_provider="azure",
            api_key=azure_token,
            model=args.model,
            region=args.azure_region,
            api_base=args.azure_api_base,
            api_version=args.azure_api_version,
            use_cache=use_cache
        )
        print(f"Using Azure OpenAI provider with model {args.model}")
        if use_cache:
            print("LLM response caching is enabled")
    else:
        # For OpenAI, we'll use the OPENAI_API_KEY environment variable if no API key is provided
        openai_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("Warning: No OpenAI API key provided. Using OPENAI_API_KEY environment variable.")
            
        planner = LLMAgentPlanner(
            llm_provider="openai",
            api_key=openai_key,
            model=args.model,
            use_cache=use_cache
        )
        print(f"Using OpenAI provider with model {args.model}")
        if use_cache:
            print("LLM response caching is enabled")
    
    # Test with progressive images
    results = planner.test_with_progressive_images(
        args.task,
        args.num_images,
        args.evaluate
    )
    
    # Print the results
    print(json.dumps(results, indent=2))
    
    # Save the results
    output_file = f"{args.task}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
