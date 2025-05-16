#!/usr/bin/env python3
"""
Runner script for Hierarchical VLA with LLM as the high-level planner.
This script integrates an LLM (like GPT-4) for task planning with a VLA agent for execution.
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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mani_skill  # This registers all ManiSkill environments

from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.display_multi_camera import display_camera_views
from env_tests.utils.image_utils import setup_directories_hierarchical_vla
from env_tests.utils.agent_utils import create_vla_agent, create_hierarchical_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ManiSkill with Hierarchical VLA (LLM + VLA Agent)")
    
    # Environment settings
    parser.add_argument("--env-id", type=str, default="Tabletop-Close-Door-v1", help="Environment ID")
    parser.add_argument("--robot", type=str, default="panda_wristcam", help="Robot type (panda or panda_wristcam)")
    parser.add_argument("--camera-width", type=int, default=448, help="Camera width resolution")
    parser.add_argument("--camera-height", type=int, default=448, help="Camera height resolution")
    parser.add_argument("--render-mode", type=str, default="human", help="Render mode (human, rgb_array, none)")
    parser.add_argument("--obs-mode", type=str, default="rgbd", help="Observation mode (state, rgb, rgbd, etc.)")
    
    # Run settings
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--show-cameras", action="store_true", default=True, help="Show camera views in OpenCV window")
    parser.add_argument("--save-images", action="store_true", help="Save images for debugging")
    parser.add_argument("--save-actions", action="store_true", default=True, help="Save actions to a JSON file")
    parser.add_argument("--image-dir", type=str, default="debug_images", help="Directory to save debug images")
    parser.add_argument("--use-which-external-camera", type=str, default="base_camera", help="Camera to use", choices=["base_camera", "base_front_camera", "left_camera", "right_camera"])
    
    # LLM settings
    parser.add_argument("--llm-provider", type=str, default="gemini", help="LLM provider to use (openai, gemini)")
    parser.add_argument("--api-key", type=str, default=None, help="LLM API key (if not set in env var)")
    parser.add_argument("--llm-model", type=str, default="gemini-2.0-flash", help="LLM model to use")
    parser.add_argument("--observation-frequency", type=int, default=10, help="How often to get observations from the environment")
    
    # VLA Agent selection (choose one)
    parser.add_argument("--vla-agent", type=str, default="pi0", help="VLA agent to use (cogact, pi0, gr00t)")
    parser.add_argument("--host", type=str, default="localhost", help="VLA server host")
    parser.add_argument("--port", type=int, default=8000, help="VLA server port")
    
    return parser.parse_args()

def main():
    """Main function to run the environment with the hierarchical VLA agent."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env_kwargs = {
        "robot_uids": args.robot,
        "obs_mode": args.obs_mode,
        "render_mode": args.render_mode,
        "control_mode": "pd_ee_delta_pose",
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
    }
    
    try:
        env = gym.make(args.env_id, **env_kwargs)
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return
    
    # Create VLA agent
    vla_agent = create_vla_agent(args)
    
    # Create hierarchical agent
    agent = create_hierarchical_agent(args, vla_agent)
    
    # Run episodes
    for episode in range(args.num_episodes):
        logger.info(f"\nStarting episode {episode+1}/{args.num_episodes}")
        
        # Reset environment and agent
        obs, info = env.reset(seed=args.seed + episode)
        agent.reset()

        # Set up directories for saving images and videos
        base_dirs = setup_directories_hierarchical_vla(args, info)
        
        # Initialize episode variables
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        actions_log = []
        
        # Run episode
        while not (done or truncated) and step < args.max_steps:
            time.sleep(0.3)
            
            # Get action from agent or sample random action
            if agent:
                action = agent.get_action(obs, info['description'], step, args.use_which_external_camera, base_dirs)
                
                if action is None:
                    action = env.action_space.sample()
                    logger.warning("Using random action as fallback")
            else:
                action = env.action_space.sample()
            
            obs, reward, done, truncated, info = env.step(action)

            executed_subtasks = agent.executed_subtasks
            if args.save_images:
                agent.save_debug_images(obs, step, base_dirs)

            if args.save_actions:
                action_data = {
                    'step': step,
                    'action': action.tolist() if hasattr(action, 'tolist') else action,
                    'reward': float(reward) if hasattr(reward, 'item') else reward,
                    'done': bool(done),
                    'truncated': bool(truncated)
                }
                actions_log.append(action_data)

            
            reward_value = reward.item() if hasattr(reward, 'item') else reward
            total_reward += reward_value
            step += 1
            
            success = info.get('success', False)
            if isinstance(success, torch.Tensor):
                success = success.item()
            
            logger.info(f"Step {step}, Reward: {reward_value:.4f}, Success: {success}")
            
            if args.show_cameras:
                try:
                    camera_views = {}
                    camera_views['base_front_camera'] = obs['sensor_data']['base_front_camera']['rgb'][0]
                    camera_views['hand_camera'] = obs['sensor_data']['hand_camera']['rgb'][0]
                    camera_views['base_camera'] = obs['sensor_data']['base_camera']['rgb'][0]
                    
                    if camera_views:
                        display_camera_views(camera_views)
                except Exception as e:
                    logger.error(f"Error displaying camera views: {e}")
        
        # Save executed subtasks
        if executed_subtasks and 'run_folder' in base_dirs:
            run_dir = base_dirs['executed_subtasks']
            filename = os.path.join(run_dir, "executed_subtasks.json")
            with open(filename, 'w') as f:
                json.dump({
                    'env_id': args.env_id,
                    'agent': 'hierarchical_vla',
                    'vla_agent': args.vla_agent,
                    'llm_model': args.llm_model,
                    'executed_subtasks': executed_subtasks
                }, f, indent=2)
            logger.info(f"Saved executed subtasks to {filename}")
        
        # Save actions log
        if args.save_actions and actions_log and 'run_folder' in base_dirs:
            run_dir = base_dirs['action']
            filename = os.path.join(run_dir, "actions.json")
            with open(filename, 'w') as f:
                json.dump({
                    'env_id': args.env_id,
                    'agent': 'hierarchical_vla',
                    'vla_agent': args.vla_agent,
                    'llm_model': args.llm_model,
                    'total_reward': float(total_reward),
                    'steps': step,
                    'actions': actions_log
                }, f, indent=2)
            logger.info(f"Saved actions to {filename}")
        
        logger.info(f"\nEpisode {episode+1} finished after {step} steps")
        logger.info(f"Total reward: {total_reward:.4f}")
        logger.info(f"Done: {done}, Truncated: {truncated}")
    
    env.close()

if __name__ == "__main__":
    main()
