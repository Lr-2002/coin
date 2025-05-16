#!/usr/bin/env python3
"""
This script is for evaluation VLMs on VQA tasks with checkpoints captured by human experts
"""

import os
import sys
import time
import json
import pickle
import torch
import logging
import argparse
import traceback
import numpy as np
import gymnasium as gym
from datetime import datetime
from pathlib import Path
import base64
import cv2
from tqdm import tqdm

from env_tests.llm_providers import LLMFactory, BaseLLMProvider
from env_tests.utils.image_utils import encode_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image(image_path):
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy.ndarray: The loaded image
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_query_image(env_name):
    query_image_path = "./mani_skill/envs/tasks/coin_bench/interactive_reasoning/interactive_task_image/"
    query_image_path += env_name + ".png"
    
    image = load_image(query_image_path)
    if image is None:
        logger.error(f"Failed to load query image: {query_image_path}")
        return None
    
    base64_img = encode_image(image)
    if base64_img:
        return [{"base64": base64_img, "type": "image/jpeg", "caption": "Query image"}]
    else:
        logger.error(f"Failed to encode query image: {query_image_path}")
    return None


def get_vqa_data(env_ins_objects_path):
    """
    Load VQA data from JSON or pickle file.
    
    Args:
        env_ins_objects_path: Path to the environment instructions and objects file
        
    Returns:
        dict: Dictionary of VQA data
    """
    if env_ins_objects_path.endswith('.json'):
        with open(env_ins_objects_path, 'r') as f:
            vqa_data = json.load(f)
    elif env_ins_objects_path.endswith('.pkl'):
        with open(env_ins_objects_path, 'rb') as f:
            vqa_data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {env_ins_objects_path}")
    
    return vqa_data

def get_system_prompt_for_VQA_initial():
    
    query_system_prompt = """
    - You are an AI assistant controlling a robotic arm in a tabletop environment.
    - You are a Hierarchical Planner that uses a set of trained low-level VLA models to perform primitive manipulation tasks.

    Now the task is about beginning.
    Before the task, you need to answer a multiple-choice question.
    Choose the most appropriate option from the choices provided.
    
    Please response with a valid JSON object with nothing else.
    The expected format is:
    {
        "answer": "X",  // The letter of your chosen answer (A, B, C, etc.)
        "reasoning": "Your detailed reasoning for this choice"
    }
    IMPORTANT: Think carefully about the question and all available choices. Do NOT copy this example format directly.
    """
    return query_system_prompt

def get_vqa_response(llm_provider, instruction, query, env_name):
    """
    Get VQA response from the LLM provider for a single image.
    
    Args:
        llm_provider: LLM provider instance
        system_prompt: System prompt for the LLM
        instruction: Task instruction
        query: VQA query with question and selection options
        image_paths: List of paths to the image files
        image_window: Image window for the image
        
    Returns:
        str: The LLM response
    """
    text_data={}
    text_data['High-level task'] = instruction
    text_data['query'] = query
    image_data = get_query_image(env_name)
    
    content = llm_provider.format_message_content(str(text_data), image_data)
    message = llm_provider.add_history_to_message(content, None)
    # breakpoint()
    # Get the LLM response with images
    try:
        response = llm_provider.generate_response(
            system_prompt=get_system_prompt_for_VQA_initial(),
            messages=message,
            temperature=0.2,
            max_tokens=1000
        )
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None

def get_llm_provider(args):
    # Auto-detect LLM provider based on model name
    if "gpt" in args.llm_model.lower() or "text-davinci" in args.llm_model.lower() or "o" in args.llm_model.lower():
        llm_provider = "openai"
    elif "gemini" in args.llm_model.lower() or "palm" in args.llm_model.lower():
        llm_provider = "gemini"
    else:
        # Default to OpenAI if unknown
        llm_provider = "openai"
        logger.warning(f"Could not determine provider for model {args.llm_model}, defaulting to OpenAI")
    return llm_provider

def eval_vqa(args):
    """
    Evaluate VQA on checkpoint images.
    
    Args:
        args: Command-line arguments
    """
    # Load VQA data
    vqa_data = get_vqa_data(args.env_ins_objects)
    
    # Create LLM provider
    llm_provider = get_llm_provider(args)
    llm_provider = LLMFactory.create_provider(
        llm_provider,
        api_key=args.api_key,
        model=args.llm_model
    )
    
    if not llm_provider:
        logger.error(f"Failed to create LLM provider: {llm_provider}")
        return
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get envs with query from vqa_data
    envs = []
    for env in vqa_data.keys():
        if vqa_data[env]["query"]["query"] is not None:
            envs.append(env)

    results = {}
    results["correct_num"] = 0
    results["correct_env"] = []
    results["wrong_env"] = []
    print(envs)
    print(f"Total environments: {len(envs)}")
    input("Press Enter to continue...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Process each environment
    for env_key in tqdm(envs, desc="Processing environments"):
        
        env_data = vqa_data[env_key]
        instruction = env_data["ins"]
        query = env_data["query"]
        answer = env_data["answer"]
        
        # Skip if no query or selection
        if not query["query"] or not query["selection"]:
            logger.warning(f"No query or selection for environment {env_key}, skipping...")
            continue
        
        # breakpoint()
        # Get VQA response for this single image
        response = get_vqa_response(
            llm_provider, 
            instruction, 
            query, 
            env_key
        )
            
        if not response:
            logger.warning(f"Failed to get response for environment {env_key}")
            continue
        # breakpoint()            
        # Parse response
        try:
            # Try to extract JSON from the response
            response_text = response.strip()
            
            # Find JSON content between curly braces
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                response_json = json.loads(json_str)
                
            else:
                logger.warning(f"Could not extract JSON from response for {env_key}")
                response_json={
                    "raw_response": response,
                    "error": "Could not extract JSON from response"
                }
                
        except Exception as e:
            logger.error(f"Error parsing response for {env_key}: {e}")
            response_json={
                "raw_response": response,
                "error": str(e)
            }
        
        if answer == response_json.get("answer", None):
            results["correct_num"] += 1
            results["correct_env"].append(env_key)
        else:
            results["wrong_env"].append(env_key)
            
        # Store results for this environment
        results[env_key] = {
            "instruction": instruction,
            "query": query,
            "ground_truth": answer,
            "correct": answer == response_json.get("answer", None),
            "response": response_json
        }
    
    # Save results
    results_file = os.path.join(args.output_dir, f"vqa_initial_{args.llm_model}_f{timestamp}.json")
    # make sure the directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {results_file}")
def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on VQA tasks with checkpoint images")
    parser.add_argument("--env-ins-objects", type=str,default="./env_ins_objects.json", help="Path to environment instructions and objects file")
    parser.add_argument("--output-dir", type=str, default="./env_tests/vqa_results/vqa_initial", help="Directory to save results")
    parser.add_argument("--llm-model", type=str, default="gpt-4o", help="LLM provider to use")
    parser.add_argument("--api-key", type=str, help="API key for the LLM provider")
    
    args = parser.parse_args()
    eval_vqa(args)

if __name__ == "__main__":
    main()

