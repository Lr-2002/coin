#!/usr/bin/env python3
"""
Test script for VLA and hierarchical VLA on primitive tabletop tasks

This script provides an interface to test VLA agents on tabletop tasks
and record the results using the VLARecorderWrapper.

Usage:
    export DISPLAY=:0 && python run_primitive_task.py --env_id Tabletop-Open-Door-v1
"""

import os
import sys
import time
import json
import torch
import logging
import argparse
import traceback
import numpy as np
import gymnasium as gym
from datetime import datetime
from pathlib import Path

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mani_skill  # This registers all ManiSkill environments

# Import ManiSkill utilities
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.utils.wrappers.vla_recorder import VLARecorderWrapper, HVLARecorderWrapper
# Import VLA utilities and wrappers
from env_tests.utils.agent_utils import create_vla_agent, create_hierarchical_agent
from env_tests.utils.image_utils import setup_directories, setup_directories_hierarchical_vla
import pickle as pkl
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test VLA/HVLA on a primitive tabletop task")
    
    # Environment settings
    parser.add_argument("--robot", type=str, default="panda_wristcam",
                       help="Robot type (panda or panda_wristcam)")
    parser.add_argument("--camera-width", type=int, default=448,
                       help="Camera width resolution")
    parser.add_argument("--camera-height", type=int, default=448,
                       help="Camera height resolution")
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                       help="Render mode (human, rgb_array, none)")
    parser.add_argument("--obs-mode", type=str, default="rgbd",
                       help="Observation mode (state, rgb, rgbd, etc.)")
    
    # Agent settings
    parser.add_argument("--vla-agent", type=str, default="pi0", 
                       choices=["pi0", "gr00t", "cogact"],
                       help="VLA agent to use")
    parser.add_argument("--host", type=str, default="localhost",
                       help="VLA server host for cogact")
    parser.add_argument("--port", type=int, default=8000,
                       help="VLA server port for cogact")
    
    # Hierarchical VLA settings
    parser.add_argument("--hierarchical", action="store_true", default=False,
                       help="Use hierarchical VLA with LLM planner")
    parser.add_argument("--llm-model", type=str, default="gemini-2.0-flash",
                       help="LLM model to use (for hierarchical VLA) - provider will be auto-detected")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for LLM provider (optional)")
    parser.add_argument("--observation-frequency", type=int, default=50,
                       help="How often to get observations for hierarchical agent")
    
    # Run settings
    parser.add_argument("--num-episodes", type=int, default=1,
                       help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--output-dir", type=str, default="vla_results",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    
    # Debug and visualization settings
    parser.add_argument("--show-cameras", action="store_true", default=True,
                       help="Show camera views in an OpenCV window")
    parser.add_argument("--save-images", action="store_true", default=True,
                       help="Save debug images")
    parser.add_argument("--image-dir", type=str, default="debug_images",
                       help="Directory to save debug images")
    parser.add_argument("--external-camera", type=str, default="left_camera",
                       choices=["human_camera", "base_front_camera", "left_camera", "right_camera"],
                       help="Camera to use for external viewing")
    parser.add_argument("--cameras", type=str, nargs='+', default=["human_camera", "hand_camera", "base_front_camera"],
                       choices=["base_front_camera", "left_camera", "right_camera", "human_camera", "hand_camera"],
                       help="Camera to use for external viewing")
    parser.add_argument("--primitive", action="store_true",
                       help="Test primitive ")
    parser.add_argument("--interactive", action="store_true",
                       help="Test interactive ")
    parser.add_argument("--workflow", action="store_true",
                       help="Test if VLMs can change workflow ")
    
    return parser.parse_args()

def run_single_task(args, env_id):
    """Run the test on the specified environment."""
    
    # Set random seed
    args.env_id = env_id
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create base directories
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = env_id
    
    # # Include LLM model in directory name for hierarchical VLA
    # if args.hierarchical:
    #     run_dir = os.path.join(args.output_dir, f"{timestamp}_{env_name}_{args.vla_agent}_{args.llm_model}")
    # else:
    #     run_dir = os.path.join(args.output_dir, f"{timestamp}_{env_name}_{args.vla_agent}")
    
    # os.makedirs(run_dir, exist_ok=True)
    
    # Log run information
    logger.info(f"Starting run for {env_name} with {args.vla_agent} agent")
    # logger.info(f"Output directory: {run_dir}")
    
    # Get model path based on the agent type from Controllers/VLAs directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vla_dir = os.path.join(project_root, "Controllers", "VLAs")
    
    model_path = "unknown"
    if args.vla_agent == "pi0":
        model_path = os.path.join(vla_dir, "openpi")
    elif args.vla_agent == "gr00t":
        model_path = os.path.join(vla_dir, "Isaac-GR00T")
    elif args.vla_agent == "cogact":
        model_path = os.path.join(vla_dir, "CogACT")
        
    logger.info(f"Using VLA model from: {model_path}")
    
    # Create environment
    env_kwargs = {
        "robot_uids": args.robot,
        "obs_mode": args.obs_mode,
        "render_mode": args.render_mode,
        "control_mode": "pd_ee_delta_pose",
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
    }
    
    logger.info(f"Creating environment {env_name}")
    env = gym.make(env_name, **env_kwargs)
    
    # Get repository paths for commit IDs
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set dataset repository paths based on the VLA agent
    dataset_repo_path = ""
    if args.vla_agent == "pi0":
        dataset_repo_path = "/home/wangxianhao/data/project/reasoning/openpi/coin-dataset"
    elif args.vla_agent == "gr00t":
        dataset_repo_path = "/home/wangxianhao/data/project/reasoning/openpi/gr00t_dataset"
    elif args.vla_agent == "cogact":
        dataset_repo_path = "/home/wangxianhao/data/project/reasoning/rlds_dataset_builder/tabletop_dataset"
    
    logger.info(f"Using dataset repository path: {dataset_repo_path}")
    
    # Wrap environment with appropriate recorder
    if args.hierarchical:
        logger.info("Using Hierarchical VLA Recorder")
        env = HVLARecorderWrapper(
            env=env,
            output_dir=args.output_dir,
            model_class=args.vla_agent,
            model_path=model_path,
            llm_model=args.llm_model,
            dataset_repo_path=dataset_repo_path,
            env_repo_path=current_dir,
            save_trajectory=True,
            save_video=True,
            video_fps=30,
            external_camera=args.external_camera,
            cameras=args.cameras
        )
    else:
        logger.info("Using VLA Recorder")
        env = VLARecorderWrapper(
            env=env,
            output_dir=args.output_dir,
            model_class=args.vla_agent,
            model_path=model_path,
            dataset_repo_path=dataset_repo_path,
            env_repo_path=current_dir,
            is_hierarchical=False,
            save_trajectory=True,
            save_video=True,
            video_fps=30,
            external_camera=args.external_camera,
            cameras=args.cameras
        )
    
    # Print environment information
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Observation space: {env.observation_space}")
    
    # Create agent
    agent = None
    if args.hierarchical:
        # Auto-detect LLM provider based on model name
        if "gpt" in args.llm_model.lower() or "text-davinci" in args.llm_model.lower() or "o" in args.llm_model.lower():
            llm_provider = "openai"
        elif "gemini" in args.llm_model.lower() or "palm" in args.llm_model.lower():
            llm_provider = "gemini"
        else:
            # Default to OpenAI if unknown
            llm_provider = "openai"
            logger.warning(f"Could not determine provider for model {args.llm_model}, defaulting to OpenAI")
        
        # Add the detected provider to args
        args.llm_provider = llm_provider
        logger.info(f"Creating hierarchical agent with {args.llm_model} from {llm_provider} and {args.vla_agent} VLA agent")
        
        # Create and initialize the hierarchical LLM agent
        vla_agent = create_vla_agent(args)
        agent = create_hierarchical_agent(args, vla_agent, env)
    else:
        logger.info(f"Creating {args.vla_agent} VLA agent")
        agent = create_vla_agent(args)
    
    
    # Run episodes
    for episode in range(args.num_episodes):
        logger.info(f"\n========== Starting Episode {episode+1}/{args.num_episodes} ==========")
        
        # Reset environment
        try:
            obs, info = env.reset(seed=args.seed + episode)
            logger.info(f"Environment reset: {env_name}")
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            logger.error(traceback.format_exc())
            continue
        
        if args.hierarchical:
            agent.reset()

        # Additional directories for saving debug images if needed
        base_dirs = {}
        if args.save_images:
            if args.hierarchical:
                base_dirs = setup_directories_hierarchical_vla(args, {})
            else:
                base_dirs = setup_directories(args, {})
        
        # Initialize episode variables
        done = False
        truncated = False
        total_reward = 0
        step = 0    

        # Warm up
        for i in range(10):
            action = env.action_space.sample() * 0
            obs, _, _, _, _ =env.step(action)
        
        # # For hierarchical VLA, record task description
        # if args.hierarchical and hasattr(env, 'record_chat'):
        #     env.record_chat('system', f"Task description: {info['description']}")
        
        # Run episode
        while not (done or truncated) and step < args.max_steps:
            # Add a small delay for visualization
            time.sleep(0.1)
            
            # Get action from agent
            if agent:
                try:
                    if args.hierarchical:
                        # For hierarchical agent
                        action = agent.get_action(obs, info['description'], step, args.external_camera, base_dirs)
                    else:
                        # For regular VLA agent
                        action = agent.get_action(obs, info['description'], step, args.external_camera)
                    
                    # If action is None despite no exception, still treat it as a failure
                    if action is None:
                        logger.error("Agent returned None action. Stopping episode to prevent contaminated results.")
                        logger.info("Episode marked as failed due to LLM/VLA agent error.")
                        env.set_episode_metadata("failure_reason", "agent_returned_none")
                        break
                except Exception as e:
                    logger.error(f"Error getting action from agent: {e}")
                    logger.error(traceback.format_exc())
                    logger.info("Episode marked as failed due to LLM/VLA agent exception.")
                    env.set_episode_metadata("failure_reason", "agent_exception")
                    break
            else:
                logger.error("Agent is None. Stopping episode to prevent contaminated results.")
                break
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Save debug images if requested
            if args.save_images and hasattr(agent, 'save_debug_images'):
                agent.save_debug_images(obs, step, base_dirs)
            
            # Update episode variables
            reward_value = reward.item() if hasattr(reward, 'item') else reward
            total_reward += reward_value
            step += 1
            
            # Get success status
            success = info.get('success', False)
            if isinstance(success, torch.Tensor):
                success = success.item()
            
            # Print progress
            logger.info(f"Step {step}, Reward: {reward_value:.4f}, Success: {success}")
            
            # Display camera views if available
            if args.show_cameras:
                try:
                    # Collect available camera views
                    camera_views = {}
                    camera_views['left_camera'] = obs['sensor_data']['left_camera']['rgb'][0]
                    camera_views['hand_camera'] = obs['sensor_data']['hand_camera']['rgb'][0]
                    # camera_views['base_front_camera'] = obs['sensor_data']['base_front_camera']['rgb'][0]
                    
                    if camera_views:
                        display_camera_views(camera_views)
                except Exception as e:
                    logger.error(f"Error displaying camera views: {e}")
        
        # Record episode
        env.record()

        # Print episode summary
        logger.info(f"\nEpisode {episode+1} finished after {step} steps")
        logger.info(f"Total reward: {total_reward:.4f}")
        logger.info(f"Done: {done}, Truncated: {truncated}, Success: {info.get('success', False)}")

        if args.hierarchical:
            # agent.answer_VQA(env, args)
            agent.answer_VQA_with_videos(env, args)
    
    # Close environment
    env.close()
    logger.info("Environment closed and data recorded")
    return 0
def load_pkl_path(pkl_path):
    with open(pkl_path, 'rb') as f: 
        pkl_list = pkl.load(f) 
        pkl_list = [ x for x in pkl_list.keys()]

    return pkl_list
def test_all_envs(args):
    interactive_path = "/media/raid/workspace/wangxianhao/project/reasoning/ManiSkill/interactive_instruction_objects.pkl"
    primitive_path = "/media/raid/workspace/wangxianhao/project/reasoning/ManiSkill/primitive_instruction_objects.pkl"
    name_list = []
    if args.primitive:
        name_list += load_pkl_path(primitive_path)
    if args.interactive:
        name_list += load_pkl_path(interactive_path)
        # print(name_list)
        # print(f"num of envs={len(name_list)}")
        # name_list = name_list[::-1]

    print(name_list)
    print(f"num of envs={len(name_list)}")
    # breakpoint()
    # name_list = [x for x in name_list if 'Door' in x]
    for env_id in name_list:
        run_single_task(args, env_id)
if __name__ == "__main__":
    args = parse_args()

    test_all_envs(args)   
