import os
import logging
import copy
import numpy as np
import cv2
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import base64
from datetime import datetime
import importlib.resources

import sys
from env_tests.agents.base_agent import BaseAgent
from env_tests.utils.image_utils import save_debug_image, extract_camera_image, encode_image
from env_tests.llm_providers import LLMFactory, BaseLLMProvider
logger = logging.getLogger(__name__)

class LLMAgent(BaseAgent):
    """Agent that uses an LLM (like GPT-4) as the high-level planner and a VLA agent for execution."""
    
    def __init__(
        self, 
        host="localhost", 
        port=8000, 
        vla_agent=None,
        llm_provider="openai",  # Name of the LLM provider to use
        api_key=None,
        model="gpt-4o",
        prompt_file="prompt.txt",
        available_instructions_file="available_instructions.json",
        observation_frequency=50,  # Get observations every N steps
        llm_provider_kwargs=None,  # Additional provider-specific arguments
        env=None,
        cameras: List[str] = ["human_camera", "hand_camera", "base_front_camera"],
    ):
        """Initialize the LLM agent.
        
        Args:
            host: VLA server host
            port: VLA server port
            vla_agent: The low-level VLA agent to use for execution
            llm_provider: Name of the LLM provider to use (e.g., 'openai', 'gemini')
            api_key: API key for the LLM provider (if None, will use environment variables)
            model: LLM model to use
            prompt_file: Path to the prompt file
            available_instructions_file: Path to the available instructions file
            observation_frequency: How often to get observations from the environment
            llm_provider_kwargs: Additional provider-specific arguments
        """
        super().__init__(host, port)
        self.vla_agent = vla_agent
        self.observation_frequency = observation_frequency
        self.use_which_external_camera = None
        self.base_dirs = None
        self.step = 0
        self.env = env # env for vla_recorder wrapper
        self.cameras = cameras
        
        # LLM provider setup
        self.llm_provider_name = llm_provider
        self.api_key = api_key
        self.model = model
        self.prompt_file = prompt_file
        self.available_instructions_file = available_instructions_file
        
        # Initialize the LLM provider
        provider_kwargs = llm_provider_kwargs or {}
        self.llm_provider = LLMFactory.create_provider(
            self.llm_provider_name,
            api_key=self.api_key,
            model=self.model,
            **provider_kwargs
        )
        
        if not self.llm_provider or not self.llm_provider.is_initialized():
            logger.warning(f"Failed to initialize {llm_provider} provider. Please check your API key and configuration.")
        
        # Task state
        self.high_level_instruction = None
        self.subtasks = []
        self.max_subtasks = 10
        self.current_subtask = None
        # self.executed_subtasks = []

        # Chat history
        self.chat_history = []
        self.chat_history_length = 3
    
    def reset(self):
        """Reset the agent's state for a new episode."""
        logger.info("Resetting LLMAgent state for new episode")

        self.subtasks = []
        self.current_subtask = None
        # self.executed_subtasks = []

        self.high_level_instruction = None
        self.step = 0
        self.use_which_external_camera = None
        self.base_dirs = None
        self._llm_plan_dir = None

        self.chat_history = []
    
    def connect(self):
        """Connect to the VLA agent."""
        if self.vla_agent is None:
            logger.error("No VLA agent provided for execution.")
            return False
        
        return self.vla_agent.connect()
    
    def is_connected(self):
        """Check if the VLA agent is connected."""
        if self.vla_agent is None:
            return False
        return self.vla_agent.is_connected()
    
    def get_available_instructions(self):
        available_instructions_file = os.path.join(os.path.dirname(__file__), self.available_instructions_file)
        with open(available_instructions_file, 'r') as f:
            available_instructions = list(json.load(f).values())
        return available_instructions
    
    def get_system_prompt_for_workflow(self):
        query_systom_prompt = """
        - You are an AI assistant controlling a robotic arm in a tabletop environment.
        - You are a Hierarchical Planner that uses a set of trained low-level VLA models to perform primitive manipulation tasks.
        - Your responsibilities:
            - For several steps VLA performs, you will receive a high-level instruction, visual observations, the groundtruth workflow of the task, current step(how many steps VLA has performed), and current subtask.
            - You need to get next subtask from the given workflow.
            - So analyze the robot and task status from the observations and decide if the robot need to do next subtask in the workflow or not.
        
        - Respond template:
        Assessment: <Your assessment of the current situation.>
                    < there is a \n after this sentence>
        Plan: <Your plan as a numbered list of subtasks, each subtask should be a simple and specific action>
            1. <Subtask 1>
            2. <Subtask 2>
            3. <Subtask 3>
            ...
            < there is a \n after this sentence>
        
        - Response example 1:
        Assessment: The current step is 0. From the image, I can see that there is a board on the way to close the draw, and based on the workflow, the subtask should be "open the drawer" for convience to move the board away.
        Plan:
            1. open the drawer
            2. put the board on the desk
            3. close the drawer

        - Response example 2:
        Assessment: From the image, I can see that the draw is opened, and based on the workflow, the subtask should be "pick the board and put it on the desk". Once it is down, we can close the drawer.    
        Plan:
            1. put the board on the desk
            2. close the drawer
        
        IMPORTANT: The plan you generate should only from the workflow, and you need to decide if the robot need to do next subtask in the workflow or not based on the observations and the workflow.
        """
        return query_systom_prompt
    def get_system_prompt(self):
        """Get the system prompt for the LLM."""
        # Read the prompt from the prompt.txt file
        prompt_file = importlib.resources.files("mani_skill.prompts") / self.prompt_file
        try:
            with open(prompt_file, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading prompt file: {e}")
            # Fallback to default prompt
            return None
    
    def get_image_data(self, obs, camera_name):
        """
        Return the image data for llm

        Args:
            obs: The observation from the environment
            camera_name: The name of the camera

        Returns:
            List of Dict: The image data for llm
        """
        image_data = []
        for image_name in camera_name:
            image = extract_camera_image(obs, image_name)
            if image is not None:
                base64_image = encode_image(image)
                if base64_image is None:
                    logger.error(f"Failed to encode image for camera: {image_name}")
                    continue
                image_data.append({
                    'base64': base64_image,
                    'type': 'jpeg',
                    'caption': f"Camera view ({image_name})"
                })
        return image_data
        
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

    def generate_plan_from_llm(self, obs):
        """Generate or update a plan from the LLM based on observations.

        Args:
            obs: Current observation from the environment

        Returns:    
            bool: Whether planning was successful
        """
        if not self.llm_provider or not self.llm_provider.is_initialized():
            logger.error(f"LLM provider '{self.llm_provider_name}' not initialized. Please check your API key and configuration.")
            return False

        try:
            # Prepare text content
            # text_content = f"High-level instruction: {self.high_level_instruction}\n\n"

            # Add step and subtask context
            # text_content += f"Step: {self.step}\n\n"
            # if self.step:
            #     current_subtask = self.current_subtask if self.current_subtask else f"Step {self.step}: No current subtask"
            #     text_content += f"Current subtask: {current_subtask}\n\n"
            #     if self.subtasks:
            #         text_content += "Current plan (remaining subtasks):\n"
            #         for i, subtask in enumerate(self.subtasks, 1):
            #             text_content += f"{i}. {subtask}\n"
            #     if self.executed_subtasks:
            #         text_content += "Executed subtasks so far:\n" + "\n".join([f"- {task}" for task in self.executed_subtasks])
        
            text_content = {}
            text_content['High-level instruction'] = self.high_level_instruction
            text_content['Workflow'] = self.env.workflow
            text_content['Step'] = self.step
            if self.step:
                text_content['Current subtask'] = self.current_subtask
                # text_content['Current plan (remaining subtasks)'] = self.subtasks
                # text_content['Executed subtasks so far'] = self.executed_subtasks

            camera_list = ["human_camera"]
            image_data = self.get_image_data(obs, camera_list)
            # # Get image data
            # if self.cameras:
            #     image_data = self.get_image_data(obs, camera_list) # list dict for different cameras, encode with base64
            # else:
            #     image_data = None

            # We'll handle the chat history separately rather than embedding it in text_content
            # This prevents issues with recursive chat history and image data formatting
            recent_chat_history = self.chat_history[-self.chat_history_length:] # 
            # Add this interaction to chat history (before sending to LLM)
            # Store images in the chat history
            self.add_to_chat_history('user', text_content)

            # Format the current message content without including chat history
            # This will be properly formatted for the LLM provider
            current_message = self.llm_provider.format_message_content(str(text_content), image_data)  
            messages = self.llm_provider.add_history_to_message(current_message, recent_chat_history) 
            

            system_prompt = self.get_system_prompt_for_workflow()+"\n\n"
            
            # Store the image paths for recording
            image_paths = {}
            if self.base_dirs:
                for camera in self.cameras:
                    image_paths[camera] = os.path.join(self.base_dirs[camera], f"step_{self.step:04d}.png")

            print(f"{'-'*100}")
            logger.info(f"Generating plan at step {self.step} with instruction: {self.high_level_instruction}")
            print(f"{'-'*100}")
            
            # breakpoint()
            # Get the LLM response with images
            response = self.llm_provider.generate_response(
                system_prompt=system_prompt,
                messages=messages,  # Use our custom formatted messages
                temperature=0.2,
                max_tokens=1000
            )
            
            if response is None:
                logger.error("LLM response is None. Stopping episode to prevent contaminated results.")
                logger.info("Episode marked as failed due to LLM/VLA agent error.")
                env.set_episode_metadata("failure_reason", "llm_response_none")
                return False
                
            print(f"response: {response}")
            # Parse the response
            parsed_response = self.parse_llm_response(response)
            
            # Add LLM response to chat history
            self.add_to_chat_history('assistant', response)
            
            # Record chat if available
            if hasattr(self, 'env') and hasattr(self.env, 'record_chat_func'):
                self.env.record_chat_func(self.step, text_content, image_paths, parsed_response)
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
                        
                    # if self.step % self.observation_frequency == 0:
                    #     self.executed_subtasks.append(next_instruction)
                        
                
                # Log the planning results
                logger.info(f"Assessment: {parsed_response.get('assessment', '')}")
                logger.info(f"Generated plan with {len(self.subtasks)} subtasks")
                for i, subtask in enumerate(self.subtasks):
                    logger.info(f"Subtask {i+1}: {subtask}")
                logger.info(f"Current subtask: {self.current_subtask}")
                # logger.info(f"Executed subtasks so far length: {len(self.executed_subtasks)}")
                
                return True
            else:
                logger.warning("Failed to parse LLM response.")
                return False
                
        except Exception as e:
            logger.error(f"Error in generate_plan_from_llm: {e}")
            return False
    
    def get_action(self, obs, description, step, use_which_external_camera, base_dirs=None):
        """Get an action from the agent.
        
        Args:
            obs: Environment observation
            description: Environment description
            step: Current step number
            use_which_external_camera: Which external camera to use
            base_dirs: Dictionary of directories for saving images and other data
            
        Returns:
            action: Action to take
        """
        self.step = step
        self.use_which_external_camera = use_which_external_camera
        self.base_dirs = base_dirs
        
        # Store high-level instruction from env description if not set yet
        if self.high_level_instruction is None:
            self.high_level_instruction = description
        
        # Generate/update plan if necessary
        if step % self.observation_frequency == 0:
            result = self.generate_plan_from_llm(obs)
            if not result:
                logger.warning("Failed to generate/update plan.")
                return None
        
        # Pass the current subtask to the VLA agent
        if self.vla_agent is not None:
            # Get action from VLA agent
            action = self.vla_agent.get_action(obs, self.current_subtask, step, self.use_which_external_camera)
            return action
        else:
            logger.error("No VLA agent available for execution")
            return None
            
    def save_debug_images(self, obs, step, image_dirs):
        """Save debug images.
        
        Args:
            obs: Environment observation
            step: Current step number
            image_dirs: Dictionary of image directories
        """
        try:
            for camera in self.cameras:
                if camera in image_dirs:
                    image = extract_camera_image(obs, camera)
                    save_debug_image(image, os.path.join(image_dirs[camera], f"step_{step:04d}.png"))
            
            # Store the llm_plan directory for saving interactions
            if 'llm_plan' in image_dirs:
                self._llm_plan_dir = image_dirs['llm_plan']
                
                # Also save the current plan and subtask
                plan_file = os.path.join(image_dirs['llm_plan'], f"{step:04d}.json")
                with open(plan_file, 'w') as f:
                    json.dump({
                        'high_level_instruction': self.high_level_instruction,
                        'subtasks': self.subtasks,
                        'current_subtask': self.current_subtask,
                        'step': self.step
                    }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving debug images: {e}")
    
    def get_system_prompt_for_VQA(self):
        """Get the system prompt for Visual Question Answering."""
        
        # Create a deep copy of chat history without the images
        chat_history = copy.deepcopy(self.chat_history[-4:])
        for entry in chat_history:
            entry.pop('images', None)
            
        query_system_prompt = f"""
        - You are an AI assistant controlling a robotic arm in a tabletop environment.
        - You are a Hierarchical Planner that uses a set of trained low-level VLA models to perform primitive manipulation tasks.
        - There are the last {len(chat_history)} chat history that you have interacted with the environment: {chat_history}
        
        Now the task is finished (whether successfully or not).
        You need to answer a multiple-choice question about what happened.
        Choose the most appropriate option from the choices provided.
        
        Please response with a valid JSON object with nothing else, because I will parse your response directly with: parsed_response = json.loads(response)
        The expected format is:
        {{
            "answer": "X",  // The letter of your chosen answer (A, B, C, etc.)
            "reasoning": "Your detailed reasoning for this choice based on the images and question"
        }}
        IMPORTANT: Think carefully about the question and all available choices. Do NOT copy this example format directly.
        Analyze the images provided, understand what happened in the task, and choose the most appropriate answer.
        """
        return query_system_prompt
    
    def get_system_prompt_for_VQA_with_videos(self):
        """Get the system prompt for Visual Question Answering."""
        
        query_system_prompt = f"""
        - You are an AI assistant controlling a robotic arm in a tabletop environment.
        - You are a Hierarchical Planner that uses a set of trained low-level VLA models to perform primitive manipulation tasks.

        Now the task is finished (whether successfully or not).
        You have the history of images of the trajectory with step number. 
        You need to answer a multiple-choice question about what happened.
        Choose the most appropriate option from the choices provided.
        
        Please response with a valid JSON object with nothing else, because I will parse your response directly with: parsed_response = json.loads(response)
        The expected format is:
        {{
            "answer": "X",  // The letter of your chosen answer (A, B, C, etc.)
            "reasoning": "Your detailed reasoning for this choice based on the images and question"
        }}

        IMPORTANT: Think carefully about the question and all available choices. Do NOT copy this example format directly.
        Analyze the images provided, understand what happened in the task, and choose the most appropriate answer.
        And the VLA agent might take wrong action which makes it difficult to answer the question, you need to report it and take a guess.
        """
        return query_system_prompt
    
    def get_query(self, env, args):
        """ 
        To get the query for VQA
        """
                
        # For interactive reasoning tasks, get VQA data
        try:
            # Set the task-specific image path
            image_path = os.path.join(
                "mani_skill/envs/tasks/coin_bench/interactive_reasoning/interactive_task_image",
                f"{args.env_id}.png"
            )
                
            # Get the base environment by unwrapping all layers
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env

            # Set the image path
            if hasattr(base_env, 'query_image_path_task'):
                base_env.query_image_path_task = image_path
                logger.info(f"Set query_image_path_task to: {image_path}")
            else:
                logger.info("query_image_path_task attribute not found")
                
            # Check if the environment has the data_for_VQA method
            if hasattr(base_env, 'data_for_VQA'):
                # Call the method
                vqa_data = base_env.data_for_VQA()
                # vqa_data['system_prompt'] = self.get_system_prompt_for_VQA()
                logger.info("Retrieved VQA data for interactive reasoning task")
                
                # # Save VQA data to a file
                # vqa_file = os.path.join(args.output_dir, f"{args.env_id}_vqa_data.json")
                # os.makedirs(os.path.dirname(vqa_file), exist_ok=True)
                
                # # Only save the instruction part to JSON (image is too large)
                # with open(vqa_file, 'w') as f:
                #     json.dump({
                #         'query_instruction': vqa_data['query_instruction'],
                #         'image_path': image_path
                #     }, f, indent=2)
                # logger.info(f"Saved VQA data to {vqa_file}")
        except Exception as e:
            logger.error(f"Error accessing data_for_VQA: {e}")
            logger.error(traceback.format_exc())

        return vqa_data
    
    def parse_VQA_response(self, response, vqa_data, env, idx):
        """ Parse  and save the json format response from LLM """

        # Clean the response and extract JSON
        try:
            # First attempt: Try to parse the raw response
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            # Log the original response for debugging
            logger.info(f"Original LLM response wasn't valid JSON: {response}")
            
            # Second attempt: Look for JSON-like content within curly braces
            import re
            json_match = re.search(r'\{.*"answer"\s*:\s*"[A-Z]".*"reasoning"\s*:\s*".*"\s*\}', response, re.DOTALL)
            
            if json_match:
                potential_json = json_match.group(0)
                try:
                    parsed_response = json.loads(potential_json)
                    logger.info(f"Successfully extracted JSON from response: {potential_json}")
                except json.JSONDecodeError as e:
                    # Third attempt: Clean up the extracted JSON
                    cleaned_json = potential_json.replace("\n", " ").replace("\\", "\\\\")
                    try:
                        parsed_response = json.loads(cleaned_json)
                        logger.info(f"Successfully parsed cleaned JSON: {cleaned_json}")
                    except json.JSONDecodeError:
                        # Final fallback: Manually create a response object
                        logger.error(f"Failed to parse JSON from response. Error: {e}")
                        parsed_response = None
            else:
                # If no JSON-like content found, extract just the answer if possible
                answer_match = re.search(r'"answer"\s*:\s*"([A-Z])"', response)
                if answer_match:
                    answer = answer_match.group(1)
                    logger.warning(f"Extracted only answer '{answer}' from malformed response")
                    parsed_response = {
                        "answer": answer,
                        "reasoning": "Answer extracted from malformed response. Original: " + response[:100] + "..."
                    }
                else:
                    # Absolute fallback
                    logger.error("Could not extract answer from LLM response.")
                    parsed_response = None
        
        vqa = {
            "system_prompt": vqa_data['system_prompt'],
            "query": vqa_data['query_instruction'],
            "answer": parsed_response,
            "idx": idx
        }
        
        env.save_VQA_answer(vqa)

    def answer_VQA(self, env, args):
        vqa_data = self.get_query(env, args)
        vqa_data['system_prompt'] = self.get_system_prompt_for_VQA()
        
        text_data = vqa_data['query_instruction']
        image_data_base64 = vqa_data['query_image']
        image_data = [{
            'base64': image_data_base64,
            'type': 'jpeg',
            'caption': 'Query image'
        }]
        
        content = self.llm_provider.format_message_content(str(text_data), image_data)
        # Get the LLM response with images
        response = self.llm_provider.generate_response(
            system_prompt=vqa_data['system_prompt'],
            messages=content,
            temperature=0.2,
            max_tokens=1000
        )
        
        self.parse_VQA_response(response, vqa_data, env, idx='initial')

    
    def answer_VQA_with_videos(self, env, args):
        
        # History images content:
        text_content = "There are the history images of the trajectory with step number: "

        # Extract video frames from episode video to send to LLM
        video_path = env.video_path
        freq = 100
        video_frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % freq == 0:
                # conver to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frames.append(frame)
            frame_count += 1
        cap.release()

        # Convert frames to base64 -> List of Dict
        history_image_data = []
        for i, image in enumerate(video_frames):
            if i == 0:
                continue
            base64_image = encode_image(image)
            history_image_data.append({
                'base64': base64_image,
                'type': 'jpeg',
                'caption': f"History image in step {i*freq}"
                })
        history_image_content = self.llm_provider.format_message_content(str(text_content), history_image_data)

        # Query content:
        vqa_data = self.get_query(env, args)
        vqa_data['system_prompt'] = self.get_system_prompt_for_VQA_with_videos()

        text_data = vqa_data['query_instruction']
        image_data_base64 = vqa_data['query_image']
        image_data = [{
            'base64': image_data_base64,
            'type': 'jpeg',
            'caption': 'Query image'
        }]
        
        query_content = self.llm_provider.format_message_content(str(text_data), image_data)

        # # Prepare messages
        # messages = []
        # messages.append({"role": "user", "content": query_content})
        # messages.append({"role": "user", "content": history_image_content})

        # List Dict, without role and content in each dict
        messages = query_content + history_image_content

        response = self.llm_provider.generate_response(
            system_prompt=vqa_data['system_prompt'],
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        
        self.parse_VQA_response(response, vqa_data, env, idx='end')

    def get_system_prompt_for_VQA_initial(self):
        
        query_system_prompt = f"""
        - You are an AI assistant controlling a robotic arm in a tabletop environment.
        - You are a Hierarchical Planner that uses a set of trained low-level VLA models to perform primitive manipulation tasks.

        Now the task is about beginning.
        Before the task, you need to answer a multiple-choice question.
        Choose the most appropriate option from the choices provided.
        
        Please response with a valid JSON object with nothing else, because I will parse your response directly with: parsed_response = json.loads(response)
        The expected format is:
        {{
            "answer": "X",  // The letter of your chosen answer (A, B, C, etc.)
            "reasoning": "Your detailed reasoning for this choice"
        }}
        IMPORTANT: Think carefully about the question and all available choices. Do NOT copy this example format directly.
        Analyze the images provided, understand what happened in the task, and choose the most appropriate answer.
        """
        return query_system_prompt

    def answer_VQA_initial(self, env, args):
        vqa_data = self.get_query(env, args)
        vqa_data['system_prompt'] = self.get_system_prompt_for_VQA_initial()
        
        text_data={}
        text_data['High-level task'] = env.description
        text_data['query'] = vqa_data['query_instruction']
        image_data_base64 = vqa_data['query_image']
        image_data = [{
            'base64': image_data_base64,
            'type': 'jpeg',
            'caption': 'Query image'
        }]
        
        content = self.llm_provider.format_message_content(str(text_data), image_data)
        # Get the LLM response with images
        response = self.llm_provider.generate_response(
            system_prompt=vqa_data['system_prompt'],
            messages=content,
            temperature=0.2,
            max_tokens=1000
        )
        
        self.parse_VQA_response(response, vqa_data, env, idx='initial')
        
        

        

        


if __name__ == "__main__":
    import traceback
    # import os
    # api_key = os.environ.get("OPENAI_API_KEY")

    # vla_agent = GR00TAgent()
    # agent = LLMAgent(
    #     vla_agent=vla_agent,
    #     api_key=api_key,
    #     model="gpt-4o",
    #     observation_frequency=5,
    # )
    # agent.connect()

    agent = LLMAgent(vla_agent=None)
#     test_response = """Assessment: The red block is on the left side of the table, and the blue platform is on the right.
# Plan:
#     1. Move the gripper above the red block
#     2. Close the gripper to grasp the red block
#     3. Move the gripper with the red block above the blue platform
#     4. Open the gripper to release the red block
# Next Instruction: Move the gripper above the red block"""
    prompt_file = "mani_skill/prompts/prompt.txt"
    test = agent.get_system_prompt()
    print(test)
    # result = agent.parse_llm_response(test_response)
    # print(result)
