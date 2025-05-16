import argparse
import os
import numpy as np
import imageio
import sapien
import gymnasium as gym
import torch
import cv2

import mani_skill
import mani_skill.envs
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.display_multi_camera import display_camera_views

def parse_args():
    parser = argparse.ArgumentParser(description="Test Universal Tabletop Environment")
    parser.add_argument("-e", "--env-id", type=str, default="Tabletop-Pick-Apple", help="Environment ID")
    parser.add_argument("--robot", type=str, default="panda_wristcam", help="Robot type (panda or panda_wristcam)")
    parser.add_argument("--object-config", type=str, default='configs/apple.json', help="Path to object config file")
    parser.add_argument("--camera-width", type=int, default=128, help="Camera width resolution")
    parser.add_argument("--camera-height", type=int, default=128, help="Camera height resolution")
    parser.add_argument("--render-mode", type=str, default="human", help="Render mode (human, rgb_array, none)")
    parser.add_argument("--obs-mode", type=str, default="rgbd", help="Observation mode (state, rgb, rgbd, etc.)")
    parser.add_argument("--save-video", action="store_true", help="Save video of the episode")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--show-cameras", action="store_true", default=True, help="Show camera views in OpenCV window")
    parser.add_argument("--debug-obs", action="store_true", help="Print observation structure for debugging")
    
    return parser.parse_args()



def print_observation_structure(obs, prefix=""):
    """Print the structure of the observation dictionary for debugging"""
    if isinstance(obs, dict):
        print(f"{prefix}Dict with keys: {list(obs.keys())}")
        for key, value in obs.items():
            print(f"{prefix}{key}:")
            print_observation_structure(value, prefix + "  ")
    elif isinstance(obs, (np.ndarray, torch.Tensor)):
        if isinstance(obs, torch.Tensor):
            shape = list(obs.shape)
            dtype = str(obs.dtype)
        else:
            shape = list(obs.shape)
            dtype = str(obs.dtype)
        print(f"{prefix}Array/Tensor with shape {shape}, dtype {dtype}")
    else:
        print(f"{prefix}Value: {obs}")


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    env_kwargs = {
        "robot_uids": args.robot,
        "obs_mode": args.obs_mode,
        "render_mode": args.render_mode,
        "control_mode": "pd_joint_delta_pos",
    }
    
    # Add object path if provided
    #     env_kwargs["object_path"] = args.usd_path
    # if args.usd_path:
    #     env_kwargs["object_scale"] = args.scale
    #     env_kwargs["object_mass"] = args.mass
    #     env_kwargs["object_friction"] = args.friction
    #
    # Add camera settings
    env_kwargs["camera_width"] = args.camera_width
    env_kwargs["camera_height"] = args.camera_height
    env_kwargs['object_config']  = args.object_config
    # Create environment
    env = gym.make(args.env_id, **env_kwargs)
    
    # Wrap environment for recording if needed
    if args.save_video:
        env = RecordEpisode(env, episode_dir="videos")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run episodes
    for episode in range(args.num_episodes):
        print(f"\nEpisode {episode+1}/{args.num_episodes}")
        
        # Reset environment
        obs, info = env.reset()
        
        # Debug observation structure if requested
        if args.debug_obs:
            print("\nObservation structure:")
            print_observation_structure(obs)
            print("\nInfo structure:")
            print_observation_structure(info)
        
        # Display camera views if available
        if args.show_cameras:
            env.display(obs)
        
        # Initialize episode variables
        done = False
        truncated = False
        total_reward = 0
        step = 0
        frames = []
        
        # Run episode
        while not (done or truncated) and step < args.max_steps:
            # Sample random action (for demonstration purposes)
            action = env.action_space.sample()
            import time 
            time.sleep(1)
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Display camera views if available
            if args.show_cameras:
                env.display(obs)
            
            # Render
            if args.render_mode == "rgb_array":
                frame = env.render()
                frames.append(frame)
            else:
                env.render()
            
            # Update episode variables
            total_reward += reward
            step += 1
            
            # Convert reward to float if it's a tensor
            if isinstance(reward, torch.Tensor):
                reward_value = reward.item()
            else:
                reward_value = reward
                
            # Convert success to bool if it's a tensor
            success = info.get('success', False)
            if isinstance(success, torch.Tensor):
                success = success.item()
            
            # Print step information
            print(f"Step {step}: Reward = {reward_value:.4f}, Success = {success}")
            
            # Print additional info
            if "distance_to_target" in info:
                distance = info['distance_to_target']
                if isinstance(distance, torch.Tensor):
                    distance = distance.item()
                print(f"  Distance to target: {distance:.4f}")
                
            if "is_grasped" in info:
                print(f"  Is grasped: {info['is_grasped']}")
        
        # Convert total_reward to float if it's a tensor
        if isinstance(total_reward, torch.Tensor):
            total_reward_value = total_reward.item()
        else:
            total_reward_value = total_reward
            
        # Print episode summary
        print(f"Episode {episode+1} finished: Steps = {step}, Total Reward = {total_reward_value:.4f}")
        
        # Save video if requested
        if args.save_video and frames:
            video_path = f"{args.env_id}-episode{episode+1}.mp4"
            imageio.mimsave(video_path, frames, fps=10)
            print(f"Saved video to {video_path}")
    
    # Close OpenCV windows
    if args.show_cameras:
        cv2.destroyAllWindows()
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
