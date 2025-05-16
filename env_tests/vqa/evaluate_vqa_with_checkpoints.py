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
import mani_skill
import base64
import cv2
from tqdm import tqdm

from env_tests.llm_providers import LLMFactory, BaseLLMProvider
from env_tests.utils.image_utils import encode_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
def get_checkpoint_images(checkpoint_dir):
    """
    Get all checkpoint images from a directory.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        dict: Dictionary of checkpoint images with filenames as keys and image paths as values
    """
    image_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.png') and not file.startswith('.'): 
            image_files.append(os.path.join(checkpoint_dir, file))
    
    # Sort images by frame number
    image_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    return image_files


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

def get_system_prompt():
    """
    Get the system prompt for the VQA task.
    
    Returns:
        str: The system prompt
    """
    system_prompt = """
    You are an AI assistant controlling a robotic arm in a tabletop environment.
    You will be shown a history of images that demonstrate the robot's manipulation process.
    
    Analyze the images carefully and answer the question based on what you see in the images.
    
    The expected format is:
    {
        "answer": "X",  // The letter of your chosen answer (A, B, C, etc.)
        "reasoning": "Your detailed reasoning for this choice based on the images and question"
    }
    
    You should only response with final answer, do not output other things.
    IMPORTANT: Think carefully about the question and all available choices. Do NOT copy this example format directly.
    Analyze the images provided, understand what happened in the task, and choose the most appropriate answer.
    """
    return system_prompt

def get_vqa_response(llm_provider, system_prompt, instruction, query, image_paths, env_name):
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
    # Checkpoint image content
    # Construct the query text
    checkpoint_image_text = f"High-level Task Instruction: {instruction}\n\n"
    # checkpoint_image_text += f"This is observation at step {os.path.basename(image_paths[-1])[:-4]} for this task.\n\n"
    checkpoint_image_text += f"There are the last {len(image_paths)} checkpoint observations for this task.\n\n"
    
    # get image steps
    history_step = []
    for image_path in image_paths:
        history_step.append(os.path.basename(image_path)[:-4])
    checkpoint_image_text += f"The images are at steps {history_step}.\n\n"
    
    # print(checkpoint_image_text)

    # Prepare the message content
    image_data = []
    for image_path in image_paths:
        image = load_image(image_path)
        base64_img = encode_image(image)
        if base64_img:
            image_data.append({"base64": base64_img, "type": "image/jpeg", "caption": "Observation at step " + os.path.basename(image_path)[:-4]})
    
    # Format the message content
    checkpoint_image_content = llm_provider.format_message_content(checkpoint_image_text, image_data)

    # Query image content
    # Prepare text
    query_text = "This is the query."
    if query["query"]:
        query_text = f"Question: {query['query']}\n\n"
        
        if query["selection"]:
            query_text += "Options:\n"
            for option, text in query["selection"].items():
                query_text += f"{option}: {text}\n"
    
    # Prepare query image
    query_image_data = get_query_image(env_name)

    # Format the message content
    query_image_content = llm_provider.format_message_content(query_text, query_image_data)

    messages = checkpoint_image_content + query_image_content
    # breakpoint()
    # Add no history, but format the message content
    messages = llm_provider.add_history_to_message(messages, None)
    # breakpoint()
    # Generate response
    try:
        response = llm_provider.generate_response(
            system_prompt=system_prompt,
            messages=messages,
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
    
    # Get system prompt
    system_prompt = get_system_prompt()
    
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
    
    # Get checkpoint directories
    checkpoint_base_dir = args.checkpoint_dir
    env_dirs = [d for d in os.listdir(checkpoint_base_dir) 
               if os.path.isdir(os.path.join(checkpoint_base_dir, d))]
    
    results = {}
    print(env_dirs)
    print(f"Total environments: {len(env_dirs)}")
    input("Press Enter to continue...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Process each environment
    for env_name in tqdm(env_dirs, desc="Processing environments"):
        env_key = env_name.replace('_', '-')
        
        # Skip if environment not in VQA data
        if env_key not in vqa_data or vqa_data[env_key]["query"]["query"] is None:
            logger.warning(f"Environment {env_key} not found in VQA data, skipping...")
            continue
        
        env_data = vqa_data[env_key]
        instruction = env_data["ins"]
        query = env_data["query"]
        answer = env_data["answer"]
        
        # Skip if no query or selection
        if not query["query"] or not query["selection"]:
            logger.warning(f"No query or selection for environment {env_key}, skipping...")
            continue
        
        # Get checkpoint images
        checkpoint_dir = os.path.join(checkpoint_base_dir, env_name)
        image_paths = get_checkpoint_images(checkpoint_dir)
        
        if not image_paths:
            logger.warning(f"No checkpoint images found for environment {env_key}, skipping...")
            continue
        
        # Process each image individually
        image_responses = []
        image_window = args.image_window
        for i, image_path in enumerate(image_paths):
            # Get image window paths
            image_window = min(image_window, len(image_paths))
            start = max(0, i-image_window+1)
            image_window_paths = image_paths[start:i+1]
            
            image_filename = os.path.basename(image_path)
            logger.info(f"Processing image {image_filename} ({i+1}/{len(image_paths)}) for environment {env_key}...")
            
            # breakpoint()
            # Get VQA response for this single image
            response = get_vqa_response(
                llm_provider, 
                system_prompt, 
                instruction, 
                query, 
                image_window_paths,
                env_key
            )
            
            if not response:
                logger.warning(f"Failed to get response for image {image_filename} in environment {env_key}")
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
                    
                    # Store image response
                    image_responses.append({
                        "image_filename": image_filename,
                        "response": response_json,
                        "answer": response_json.get("answer"),
                        "correct": response_json.get("answer") == answer
                    })
                else:
                    logger.warning(f"Could not extract JSON from response for image {image_filename} in {env_key}")
                    image_responses.append({
                        "image_filename": image_filename,
                        "raw_response": response,
                        "error": "Could not extract JSON from response"
                    })
                    
            except Exception as e:
                logger.error(f"Error parsing response for image {image_filename} in {env_key}: {e}")
                image_responses.append({
                    "image_filename": image_filename,
                    "raw_response": response,
                    "error": str(e)
                })
        
        # Store results for this environment
        results[env_key] = {
            "instruction": instruction,
            "query": query,
            "ground_truth": answer,
            "response_num": len(image_responses),
            "answers": [resp.get("answer", None) for resp in image_responses],
            "correct": [resp.get("correct", None) for resp in image_responses],
            "final_correct": image_responses[-1].get("correct", None) if image_responses else None,
            "image_responses": image_responses,
        }

        # save env results
        env_result_path = os.path.join(args.output_dir, f"{args.llm_model}", f"{timestamp}", "envs", f"{env_key}.json")
        # make sure the directory exists
        os.makedirs(os.path.dirname(env_result_path), exist_ok=True)
        with open(env_result_path, "w") as f:
            json.dump(results[env_key], f, indent=4)
    
    # Calculate accuracy metrics
    total_count = len([k for k in results.keys() if k != "summary"])
    
    # Calculate accuracy based on final image response
    final_correct_count = sum(1 for env_data in results.values() if env_data.get("final_correct", None))
    final_accuracy = final_correct_count / total_count if total_count > 0 else 0
    
    # # Calculate per-image accuracy
    # total_image_responses = sum(len(env_data.get("image_responses", [])) for env_data in results.values())
    # correct_image_responses = sum(
    #     sum(1 for img_resp in env_data.get("image_responses", []) if img_resp.get("correct", None))
    #     for env_data in results.values()
    # )
    # per_image_accuracy = correct_image_responses / total_image_responses if total_image_responses > 0 else 0
    
    # Add summary to results
    results["summary"] = {
        "total_environments": total_count,
        "final_correct": final_correct_count,
        "final_accuracy": final_accuracy,
        # "total_image_responses": total_image_responses,``
        # "correct_image_responses": correct_image_responses,
        # "per_image_accuracy": per_image_accuracy
    }
    
    # Save results
    results_file = os.path.join(args.output_dir, f"{args.llm_model}", f"{timestamp}", f"vqa_results_{args.llm_model}_{args.image_window}.json")
    # make sure the directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {results_file}")
    logger.info(f"Final Image Accuracy: {final_accuracy:.2f} ({final_correct_count}/{total_count})")
    # logger.info(f"Per-Image Accuracy: {per_image_accuracy:.2f} ({correct_image_responses}/{total_image_responses})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on VQA tasks with checkpoint images")
    parser.add_argument("--checkpoint-dir", type=str, default="coin_videos/checkpoint", help="Directory containing checkpoint images")
    parser.add_argument("--env-ins-objects", type=str,default="./env_ins_objects.json", help="Path to environment instructions and objects file")
    parser.add_argument("--output-dir", type=str, default="./env_tests/vqa_results/vqa_checkpoint_images", help="Directory to save results")
    parser.add_argument("--llm-model", type=str, default="gpt-4o", help="LLM provider to use")
    parser.add_argument("--api-key", type=str, help="API key for the LLM provider")
    parser.add_argument("--image-window", type=int, default=2, help="Image window for the image")
    
    args = parser.parse_args()
    eval_vqa(args)

if __name__ == "__main__":
    main()

