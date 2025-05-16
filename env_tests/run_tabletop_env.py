#!/usr/bin/env python3
"""
Main runner script for ManiSkill tabletop environments.
This script provides a modular approach to running different VLA agents.
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
from env_tests.utils.image_utils import setup_directories   
from env_tests.utils.agent_utils import create_vla_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ManiSkill Tabletop Environment with VLA Agents")
    
    # Environment settings
    parser.add_argument("--env-id", type=str, default="Tabletop-Close-Door-v1", help="Environment ID")
    parser.add_argument("--robot", type=str, default="panda_wristcam", help="Robot type (panda or panda_wristcam)")
    parser.add_argument("--camera-width", type=int, default=448, help="Camera width resolution")
    parser.add_argument("--camera-height", type=int, default=448, help="Camera height resolution")
    parser.add_argument("--render-mode", type=str, default="human", help="Render mode (human, rgb_array, none)")
    parser.add_argument("--obs-mode", type=str, default="rgbd", help="Observation mode (state, rgb, rgbd, etc.)")
    
    # Run settings
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--show-cameras", action="store_true", default=True, help="Show camera views in OpenCV window")
    parser.add_argument("--save-images", action="store_true", help="Save images for debugging")
    parser.add_argument("--image-dir", type=str, default="debug_images", help="Directory to save debug images")
    parser.add_argument("--save-actions", action="store_true", default=True, help="Save actions to a JSON file")
    parser.add_argument("--use-which-external-camera", type=str, default="base_camera", help="Camera to use", choices=["base_camera", "base_front_camera", "left_camera", "right_camera"])

    # Agent selection (choose one)
    parser.add_argument("--vla-agent", type=str, default="pi0", help="VLA agent to use (cogact, pi0, gr00t)")
    parser.add_argument("--host", type=str, default="localhost", help="VLA server host")
    parser.add_argument("--port", type=int, default=8000, help="VLA server port")
    
    return parser.parse_args()

def main():
    """Main function to run the environment with the specified agent."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create base image directory if needed
    if args.save_images:
        os.makedirs(args.image_dir, exist_ok=True)
    
    # Create environment
    env_kwargs = {
        "robot_uids": args.robot,
        "obs_mode": args.obs_mode,
        "render_mode": args.render_mode,
        "control_mode": "pd_ee_delta_pose",
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
    }
    
    # Create environment
    env = gym.make(args.env_id, **env_kwargs)
    
    # Initialize base_dirs
    base_dirs = {}
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Create and initialize agent
    agent = create_vla_agent(args)
    
    # Run episodes
    for episode in range(args.num_episodes):
        print(f"\nEpisode {episode+1}/{args.num_episodes}")
        
        # Reset environment
        obs, info = env.reset()
        
        # Set up directories for saving data
        if args.save_images or args.save_actions:
            base_dirs = setup_directories(args, info)
            
        # Display camera views if available
        if args.show_cameras:
            try:
                # Collect available camera views
                camera_views = {}
                camera_views['base_front_camera'] = obs['sensor_data']['base_front_camera']['rgb'][0]
                camera_views['hand_camera'] = obs['sensor_data']['hand_camera']['rgb'][0]
                camera_views['base_camera'] = obs['sensor_data']['base_camera']['rgb'][0]
                
                if camera_views:
                    display_camera_views(camera_views)
                else:
                    logger.warning("No camera views available to display")
            except Exception as e:
                logger.error(f"Error displaying camera views: {e}")
                logger.error(traceback.format_exc())
        
        # Initialize episode variables
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        # For saving actions
        actions_log = []
        
        # Run episode
        while not (done or truncated) and step < args.max_steps:
            # Add a small delay for visualization
            time.sleep(0.2)
            
            # Get action from agent or sample random action
            if agent:
                # Get action from agent
                action = agent.get_action(obs, info['description'], step, args.use_which_external_camera)
                
                # Fall back to random action if agent failed
                if action is None:
                    action = env.action_space.sample()
                    logger.warning("Using random action as fallback")
            else:
                # Sample random action
                action = env.action_space.sample()
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Save debug images if requested
            if args.save_images:
                agent.save_debug_images(obs, step, base_dirs)
            
            # Log action for saving
            if args.save_actions:
                action_data = {
                    'step': step,
                    'action': action.tolist() if hasattr(action, 'tolist') else action,
                    'reward': float(reward) if hasattr(reward, 'item') else reward,
                    'done': bool(done),
                    'truncated': bool(truncated)
                }
                actions_log.append(action_data)
            
            # Update episode variables
            reward_value = reward.item()
            total_reward += reward_value
            step += 1
            
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
                    camera_views['base_front_camera'] = obs['sensor_data']['base_front_camera']['rgb'][0]
                    camera_views['hand_camera'] = obs['sensor_data']['hand_camera']['rgb'][0]
                    camera_views['base_camera'] = obs['sensor_data']['base_camera']['rgb'][0]
                    
                    if camera_views:
                        display_camera_views(camera_views)
                except Exception as e:
                    logger.error(f"Error displaying camera views: {e}")
        
        # Save actions log
        if args.save_actions and actions_log and 'run_folder' in base_dirs:
            # Get the run directory from base_dirs
            agent_type = args.vla_agent
            
            # Use the run directory for saving actions
            run_dir = base_dirs['action']
            
            # Save actions in the run directory
            filename = os.path.join(run_dir, "actions.json")
            with open(filename, 'w') as f:
                json.dump({
                    'env_id': args.env_id,
                    'agent': agent_type,
                    'total_reward': float(total_reward),
                    'steps': step,
                    'actions': actions_log
                }, f, indent=2)
            logger.info(f"Saved actions to {filename}")
        
        # Print episode summary
        logger.info(f"\nEpisode {episode+1} finished after {step} steps")
        logger.info(f"Total reward: {total_reward:.4f}")
        logger.info(f"Done: {done}, Truncated: {truncated}")
        

    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
