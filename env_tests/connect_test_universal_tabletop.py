import argparse
import os
import numpy as np
import imageio
import sapien
import gymnasium as gym
import torch
import cv2
import logging
import sys
import time
import traceback
import json

from pathlib import Path
import mani_skill
import mani_skill.envs
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.display_multi_camera import display_camera_views

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the WebsocketClientPolicy class
# Note: You need to copy the socket_client directory from CogACT to ManiSkill
try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as pi0_client
    SOCKET_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("openpi_client module not found. Please installit from openpi.")
    logger.warning("Will run without policy integration.")
    SOCKET_CLIENT_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test Universal Tabletop Environment")
    parser.add_argument("--env-id", type=str, default="Tabletop-Pick-Apple", help="Environment ID")
    parser.add_argument("--robot", type=str, default="panda_wristcam", help="Robot type (panda or panda_wristcam)")
    # parser.add_argument("--object-config", type=str, default='configs/apple.json', help="Path to object config file")
    parser.add_argument("--camera-width", type=int, default=448, help="Camera width resolution")
    parser.add_argument("--camera-height", type=int, default=448, help="Camera height resolution")
    parser.add_argument("--render-mode", type=str, default="human", help="Render mode (human, rgb_array, none)")
    parser.add_argument("--obs-mode", type=str, default="rgbd", help="Observation mode (state, rgb, rgbd, etc.)")
    parser.add_argument("--save-video", action="store_true", help="Save video of the episode")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--show-cameras", action="store_true", default=True, help="Show camera views in OpenCV window")
    parser.add_argument("--debug-obs", action="store_true", help="Print observation structure for debugging")
    parser.add_argument("--use-camera", type=str, default="base_front_camera", help="Camera to use")
    
    # parser.add_argument("--prompt", type=str, default="pick up the apple", help="Prompt")
    # parser.add_argument("--action-scale", type=float, default=1, help="Scale factor")
    parser.add_argument("--save-images", action="store_true", help="Save images for debugging")
    parser.add_argument("--image-dir", type=str, default="debug_images", help="Directory to save debug images")
    parser.add_argument("--save-actions", action="store_true", default=True, help="Save actions to a JSON file")

    # Add arguments for CogACT server connection
    parser.add_argument("--cogact-host", type=str, default="localhost", help="CogACT server host")
    parser.add_argument("--cogact-port", type=int, default=8000, help="CogACT server port")
    parser.add_argument("--use-cogact", action="store_true", help="Use CogACT for action generation")
    parser.add_argument("--use-external-camera", action="store_true", help="Use external camera for CogACT")
    parser.add_argument("--use-hand-camera", action="store_true", help="Use hand camera for CogACT")

    # Add arguments for pi0
    parser.add_argument("--use-pi0", action="store_true", help="Use pi0 for action generation")
    parser.add_argument("--pi0-host", type=str, default="localhost", help="pi0 server host")
    parser.add_argument("--pi0-port", type=int, default=8000, help="pi0 server port")
    parser.add_argument("--pi0-resize", type=int, default=224, help="Resize images to this size for Pi0")

    # Add arguments for Gr00t
    parser.add_argument("--use-gr00t", action="store_true", help="Use Gr00t for action generation")
    parser.add_argument("--gr00t-host", type=str, default="localhost", help="Gr00t server host")
    parser.add_argument("--gr00t-port", type=int, default=8000, help="Gr00t server port")

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

def setup_image_directories(args, info):
    """
    Set up directory structure for saving images and videos.
    
    Structure:
    - image_dir/
      - cogact/
        - external_camera/
          - timestamp_task/
        - hand_camera/
          - timestamp_task/
      - pi0/
        - external_camera/
          - timestamp_task/
        - hand_camera/
          - timestamp_task/
      - ...

      - image_dir/
      - cogact/
        - timestamp_task/
          - external_camera/
          - hand_camera/
          - actions.json
          - video.mp4
      - pi0/
        - timestamp_task/
          - external_camera/
          - hand_camera/
          - actions.json
          - video.mp4
      - ...
    """
    # Get current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Clean task name for folder name (remove special characters)
    task_name = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in info['description'])
    task_name = task_name.replace(' ', '_')[:50]  # Limit length and replace spaces
    
    # Create run folder name
    run_folder = f"{timestamp}_{task_name}"
    
    # Base directories
    base_dirs = {}
    base_dirs['run_folder'] = run_folder
    base_dirs['timestamp'] = timestamp
    
    if args.use_cogact:
        # CogACT directories
        cogact_base = os.path.join(args.image_dir, "cogact")
        os.makedirs(cogact_base, exist_ok=True)
        
        # Create camera-specific directories based on which camera is used
        if args.use_external_camera:
            external_dir = os.path.join(cogact_base, "external_camera", run_folder)
            os.makedirs(external_dir, exist_ok=True)
            base_dirs['cogact_external'] = external_dir
            logger.info(f"CogACT external camera images will be saved to: {external_dir}")
        
        if args.use_hand_camera:
            hand_dir = os.path.join(cogact_base, "hand_camera", run_folder)
            os.makedirs(hand_dir, exist_ok=True)
            base_dirs['cogact_hand'] = hand_dir
            logger.info(f"CogACT hand camera images will be saved to: {hand_dir}")

        if args.save_video:
            video_dir = os.path.join(cogact_base, "videos", run_folder)
            os.makedirs(video_dir, exist_ok=True)
            base_dirs['video'] = video_dir
            logger.info(f"CogACT videos will be saved to: {video_dir}")
    
    if args.use_pi0:
        # Pi0 directories (always create both camera directories)
        pi0_base = os.path.join(args.image_dir, "pi0")
        os.makedirs(pi0_base, exist_ok=True)
        
        # Create both external and hand camera directories for Pi0
        external_dir = os.path.join(pi0_base, "external_camera", run_folder)
        os.makedirs(external_dir, exist_ok=True)
        base_dirs['pi0_external'] = external_dir
        logger.info(f"Pi0 external camera images will be saved to: {external_dir}")
        
        hand_dir = os.path.join(pi0_base, "hand_camera", run_folder)
        os.makedirs(hand_dir, exist_ok=True)
        base_dirs['pi0_hand'] = hand_dir
        logger.info(f"Pi0 hand camera images will be saved to: {hand_dir}")

        if args.save_video:
            video_dir = os.path.join(pi0_base, "videos", run_folder)
            os.makedirs(video_dir, exist_ok=True)
            base_dirs['video'] = video_dir
            logger.info(f"Pi0 videos will be saved to: {video_dir}")
    
    if args.use_gr00t:
        # Gr00t directories
        gr00t_base = os.path.join(args.image_dir, "gr00t")
        os.makedirs(gr00t_base, exist_ok=True)
        
        # Create both external and hand camera directories for Gr00t
        external_dir = os.path.join(gr00t_base, "external_camera", run_folder)
        os.makedirs(external_dir, exist_ok=True)
        base_dirs['gr00t_external'] = external_dir
        logger.info(f"Gr00t external camera images will be saved to: {external_dir}")
        
        hand_dir = os.path.join(gr00t_base, "hand_camera", run_folder)
        os.makedirs(hand_dir, exist_ok=True)
        base_dirs['gr00t_hand'] = hand_dir
        logger.info(f"Gr00t hand camera images will be saved to: {hand_dir}")

        if args.save_video:
            video_dir = os.path.join(gr00t_base, "videos", run_folder)
            os.makedirs(video_dir, exist_ok=True)
            base_dirs['video'] = video_dir
            logger.info(f"Gr00t videos will be saved to: {video_dir}")
    
    return base_dirs

def main():
    args = parse_args()
    
    # Check if socket_client is available when --use-cogact is specified
    if args.use_cogact and not SOCKET_CLIENT_AVAILABLE:
        logger.error("Cannot use CogACT: socket_client module not found.")
        logger.error("Please copy the socket_client directory from CogACT to ManiSkill.")
        return
    
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
    }
    
    # Add camera settings
    env_kwargs["camera_width"] = args.camera_width
    env_kwargs["camera_height"] = args.camera_height
    # env_kwargs['object_config'] = args.object_config
    
    # Create environment
    env = gym.make(args.env_id, **env_kwargs)
    
    # Wrap environment for recording if needed
    if args.save_video:
        env = RecordEpisode(env, output_dir="videos")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Initialize WebsocketClientPolicy if using CogACT
    client = None
    if args.use_cogact:
        try:
            logger.info(f"Connecting to CogACT server at {args.cogact_host}:{args.cogact_port}")
            client = WebsocketClientPolicy(host=args.cogact_host, port=args.cogact_port)
            logger.info("Connected to CogACT server successfully")
            logger.info(f"Server metadata: {client.get_server_metadata()}")
        except Exception as e:
            logger.error(f"Failed to connect to CogACT server: {e}")
            logger.info("Will use random actions instead")
            client = None
    
    # Initialize pi0 client if using pi0
    pi0_client = None
    if args.use_pi0:
        try:
            logger.info(f"Connecting to pi0 server at {args.pi0_host}:{args.pi0_port}")
            pi0_client = WebsocketClientPolicy(host=args.pi0_host, port=args.pi0_port)
            logger.info("Connected to pi0 server successfully")
            logger.info(f"Server metadata: {pi0_client.get_server_metadata()}")
        except Exception as e:
            logger.error(f"Failed to connect to pi0 server: {e}")
            logger.info("Will use random actions instead")
            pi0_client = None
    
    # Initialize Gr00t client if using Gr00t
    gr00t_client = None
    if args.use_gr00t:
        try:
            logger.info(f"Connecting to Gr00t server at {args.gr00t_host}:{args.gr00t_port}")
            gr00t_client = WebsocketClientPolicy(host=args.gr00t_host, port=args.gr00t_port)
            logger.info("Connected to Gr00t server successfully")
            logger.info(f"Server metadata: {gr00t_client.get_server_metadata()}")
        except Exception as e:
            logger.error(f"Failed to connect to Gr00t server: {e}")
            logger.info("Will use random actions instead")
            gr00t_client = None
    
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
        
        # Set up image directories after we have info
        image_dirs = {}
        if args.save_images or args.save_video:
            image_dirs = setup_image_directories(args, info)
        
        # Display camera views if available
        if args.show_cameras:
            try:
                # Use the imported display_camera_views function instead of env.display
                camera_views = {}
                if 'sensor_data' in obs:
                    if 'base_front_camera' in obs['sensor_data'] and 'rgb' in obs['sensor_data']['base_front_camera']:
                        camera_views['base_front_camera'] = obs['sensor_data']['base_front_camera']['rgb'][0]
                    if 'hand_camera' in obs['sensor_data'] and 'rgb' in obs['sensor_data']['hand_camera']:
                        camera_views['hand_camera'] = obs['sensor_data']['hand_camera']['rgb'][0]
                    if 'base_camera' in obs['sensor_data'] and 'rgb' in obs['sensor_data']['base_camera']:
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
        frames = []
        action_chunk = None
        current_chunk_index = 0
        
        # For saving actions
        actions_log = []
        
        # Run episode
        while not (done or truncated) and step < args.max_steps:
            time.sleep(0.5)
            # Get action from CogACT or sample random action
            if args.use_cogact and client:
                try:
                    # Extract image from observation based on camera preference
                    if args.use_hand_camera:
                        image = extract_hand_camera_image(obs)
                        camera_type = "hand_camera"
                    else:
                        image = extract_external_camera_image(obs, args)
                        camera_type = "external_camera"
                    
                    # Save image for debugging if requested
                    if args.save_images:
                        if camera_type == "external_camera" and 'cogact_external' in image_dirs:
                            save_debug_image(image, os.path.join(image_dirs['cogact_external'], f"step_{step:04d}.png"))
                        elif camera_type == "hand_camera" and 'cogact_hand' in image_dirs:
                            save_debug_image(image, os.path.join(image_dirs['cogact_hand'], f"step_{step:04d}.png"))
                    
                    # Create observation dictionary for CogACT server
                    cogact_obs = {
                        'image': image,
                        'prompt': info['description']
                    }
                    print(f"step {step+1}--------------------------------------")
                    logger.info(f"cogact_obs info: image shape->{cogact_obs['image'].shape}; prompt->{cogact_obs['prompt']}")
                    
                    # Send observation to CogACT server
                    logger.info(f"Sending observation to CogACT server (step {step+1})")
                    result = client.infer(cogact_obs)
                    logger.info(f"socket result: {result['status']}, result keys: {list(result.keys())}")
                    
                    # Print full result for debugging
                    logger.debug(f"Full result: {result}")

                    # Get action from result
                    if 'actions' in result and result['actions'] is not None and len(result['actions']) > 0:
                        action = result['actions'][0]
                        logger.info(f"Using CogACT action: {action}")
                    else:
                        # Fallback to random action if something went wrong
                        action = env.action_space.sample()
                        logger.warning(f"Using random action: no valid actions in result. Result keys: {list(result.keys())}")
                except Exception as e:
                    logger.error(f"Error getting actions from CogACT: {e}")
                    logger.error(f"Exception details: {str(e)}")
                    # Fallback to random action
                    action = env.action_space.sample()
                    logger.warning("Using random action due to exception")
            
            elif args.use_pi0 and pi0_client:
                # For pi0, we need to handle action chunks and timing differently
                # Only get new actions when we've used up our current chunk or at the start
                # action_delay = 0.02
                # action_horizon = 25
                try:
                    # Extract image from observation based on camera 
                    wrist_image = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(extract_hand_camera_image(obs), 224, 224))
                    image = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(extract_external_camera_image(obs, args), 224, 224))
                    print(f"wrist_image shape: {wrist_image.shape}, image shape: {image.shape}")
                    
                    # Save image for debugging if requested
                    if args.save_images:
                        if 'pi0_external' in image_dirs and image is not None:
                            save_debug_image(image, os.path.join(image_dirs['pi0_external'], f"step_{step:04d}.png"))
                        if 'pi0_hand' in image_dirs and wrist_image is not None:
                            save_debug_image(wrist_image, os.path.join(image_dirs['pi0_hand'], f"step_{step:04d}.png"))
                    
                    state = obs['agent']['qpos'][0]
                    state = state[:8].detach().cpu().numpy()
                    logger.info(f"state shape: {state.shape}, state example: {state}")

                    # Create observation dictionary for pi0 server
                    pi0_obs = {
                        'observation/image': image,
                        'observation/wrist_image': wrist_image,
                        'observation/state': state,
                        'prompt': info['description']
                    }
                    print(f"step {step+1}--------------------------------------")
                    logger.info(f"pi0_obs info: image shape->{pi0_obs['observation/image'].shape}; wrist_image shape->{pi0_obs['observation/wrist_image'].shape}; prompt->{pi0_obs['prompt']}, state->{pi0_obs['observation/state'].shape}")
                    
                    # Send observation to pi0 server
                    logger.info(f"Sending observation to pi0 server (step {step+1})")
                    start_time = time.time()
                    result = pi0_client.infer(pi0_obs)
                    logger.debug(f"Full result: {result.keys()}")
                    end_time = time.time()
                    logger.info(f"Time taken for pi0 inference: {end_time - start_time:.2f} seconds")
                    
                    # Get action from result
                    if 'actions' in result and result['actions'] is not None and len(result['actions']) > 0:
                        action_chunk = result['actions']
                        logger.info(f"Received action chunk with shape: {action_chunk.shape}")
                        
                        # Get the first action from the chunk
                        action = action_chunk[0]
                        logger.info(f"Using pi0 action: {action}")
                    else:
                        # Fallback to random action if something went wrongd
                        action = env.action_space.sample()
                        logger.warning(f"Using random action: no valid actions in result. Result keys: {list(result.keys())}")
                except Exception as e:
                    logger.error(f"Error getting actions from pi0: {e}")
                    logger.error(f"Exception details: {str(e)}")
                    # Fallback to random action
                    action = env.action_space.sample()
                    logger.warning("Using random action due to exception")
            
            elif args.use_gr00t and gr00t_client is not None:
                try:
                    # Extract required observation for Gr00t 
                    image = extract_external_camera_image(obs, args)
                    wrist_image = extract_hand_camera_image(obs)
                    joint_state = extract_joint_state(obs)
                    
                    if args.save_images:
                        if 'gr00t_external' in image_dirs:
                            save_debug_image(image, os.path.join(image_dirs['gr00t_external'], f"step_{step+1:04d}.png"))
                        if 'gr00t_hand' in image_dirs:
                            save_debug_image(wrist_image, os.path.join(image_dirs['gr00t_hand'], f"step_{step+1:04d}.png"))
                    gr00t_obs = {
                        'image': image,
                        'wrist_image': wrist_image,
                        'joint_state': joint_state,
                        'prompt': info['description']
                    }
                    print(f"step {step+1}--------------------------------------")
                    logger.info(f"gr00t_obs info: image shape->{gr00t_obs['image'].shape}; wrist_image shape->{gr00t_obs['wrist_image'].shape}; joint_state shape->{gr00t_obs['joint_state'].shape}; prompt->{gr00t_obs['prompt']}")
                    result = gr00t_client.infer(gr00t_obs)

                    # Send observation to Gr00t server
                    logger.info(f"Sending observation to Gr00t server (step {step+1})")
                    result = gr00t_client.infer(gr00t_obs)
                    logger.info(f"Gr00t socket result: {result['status']}, result keys: {list(result.keys())}")
                    
                    # Get action from result
                    if 'actions' in result and result['actions'] is not None and len(result['actions']) > 0:
                        action = result['actions'][0]
                        logger.info(f"Using Gr00t action: {action}")
                    else:
                        # Fallback to random action if something went wrong
                        action = env.action_space.sample()
                        logger.warning(f"Using random action: no valid actions in result. Result keys: {list(result.keys())}")
                except Exception as e:
                    logger.error(f"Error getting actions from Gr00t: {e}")
                    logger.error(f"Exception details: {str(e)}")
                    # Fallback to random action
                    action = env.action_space.sample()
                    logger.warning("Using random action due to exception")

            else:
                # Sample random action
                action = env.action_space.sample()
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
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
            
            # Display camera views if available
            if args.show_cameras:
                try:
                    # Use the imported display_camera_views function instead of env.display
                    camera_views = {}
                    if 'sensor_data' in obs:
                        if 'base_front_camera' in obs['sensor_data'] and 'rgb' in obs['sensor_data']['base_front_camera']:
                            camera_views['base_front_camera'] = obs['sensor_data']['base_front_camera']['rgb'][0]
                        if 'hand_camera' in obs['sensor_data'] and 'rgb' in obs['sensor_data']['hand_camera']:
                            camera_views['hand_camera'] = obs['sensor_data']['hand_camera']['rgb'][0]
                        if 'base_camera' in obs['sensor_data'] and 'rgb' in obs['sensor_data']['base_camera']:
                            camera_views['base_camera'] = obs['sensor_data']['base_camera']['rgb'][0]
                    
                    if camera_views:
                        display_camera_views(camera_views)
                    else:
                        logger.warning("No camera views available to display")
                except Exception as e:
                    logger.error(f"Error displaying camera views: {e}")
                    logger.error(traceback.format_exc())
            
            # Render
            if args.render_mode == "rgb_array":
                frame = env.render()
                frames.append(frame)
            elif args.render_mode == "human" and not os.environ.get('DISPLAY', '').startswith(':'):
                # Only try to render in human mode if we're not in a headless environment
                try:
                    env.render()
                except Exception as e:
                    logger.warning(f"Rendering failed: {e}. Continuing without rendering.")
            
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
        
        # Save actions log if requested
        if args.save_actions and actions_log:
            # Save actions directly in the image directories
            if args.use_pi0 and 'pi0_external' in image_dirs:
                actions_log_file = os.path.join(image_dirs['pi0_external'], "actions.json")
            elif args.use_cogact and 'cogact_external' in image_dirs:
                actions_log_file = os.path.join(image_dirs['cogact_external'], "actions.json")
            elif args.use_gr00t and 'gr00t_external' in image_dirs:
                actions_log_file = os.path.join(image_dirs['gr00t_external'], "actions.json")
            else:
                # Fallback to a generic location
                os.makedirs(os.path.join(args.image_dir, "actions"), exist_ok=True)
                actions_log_file = os.path.join(args.image_dir, "actions", f"{image_dirs['timestamp']}_{image_dirs['run_folder']}_actions.json")
            
            try:
                with open(actions_log_file, 'w') as f:
                    json.dump(actions_log, f, indent=2)
                logger.info(f"Saved actions log to {actions_log_file}")
            except Exception as e:
                logger.error(f"Failed to save actions log: {e}")
        
        # Save video if requested
        if args.save_video and frames:
            if 'video' in image_dirs:
                # Create a clean version of the description for the filename
                clean_desc = ''.join(c if c.isalnum() or c == '_' else '_' for c in info['description'])
                clean_desc = clean_desc[:30]  # Limit length to avoid very long filenames
                video_path = os.path.join(image_dirs['video'], f"{args.env_id}-{clean_desc}.mp4")
            else:
                video_path = f"{args.env_id}.mp4"
            
            # Convert frames to CPU if they are CUDA tensors
            cpu_frames = []
            for frame in frames:
                if isinstance(frame, torch.Tensor):
                    # Move to CPU and convert to numpy
                    print(f"Frame shape: {frame.shape}")
                    frame = frame.cpu().numpy()
                
                # Remove batch dimension if present (shape [1, H, W, C] -> [H, W, C])
                if len(frame.shape) == 4 and frame.shape[0] == 1:
                    frame = frame[0]  # Remove batch dimension
                
                cpu_frames.append(frame)
            
            logger.info(f"Saving video to {video_path}")
            imageio.mimsave(video_path, cpu_frames, fps=10)
            logger.info(f"Video saved successfully")
    
    # Close OpenCV windows
    if args.show_cameras:
        cv2.destroyAllWindows()
    
    # Close environment
    env.close()

def extract_external_camera_image(obs, args=None):
    """
    Extract RGB image from external camera in ManiSkill observation.
    
    Args:
        obs: The observation from ManiSkill environment
        args: Optional arguments containing camera settings, defaults to using 'base_front_camera'
        
    Returns:
        image: The RGB image as a numpy array
    """

    camera_name = args.use_camera
    
    # Based on the observation space, the RGB image is in 'sensor_data'/camera_name/'rgb'
    # Get the image
    try:
        image = obs['sensor_data'][camera_name]['rgb']
        
        # Remove batch dimension if present
        if len(image.shape) == 4 and image.shape[0] == 1:
            image = image[0]
        
        # Convert PyTorch tensor to NumPy array if needed
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        return image
    except KeyError as e:
        logger.error(f"Camera '{camera_name}' not found in observation. Available cameras: {list(obs['sensor_data'].keys())}")
        # Return a default empty image or raise the exception
        raise e
    

def extract_hand_camera_image(obs):
    """
    Extract RGB image from hand camera in ManiSkill observation.
    
    Args:
        obs: The observation from ManiSkill environment
        
    Returns:
        image: The RGB image as a numpy array
    """
    # Based on the observation space, the RGB image is in 'sensor_data'/'hand_camera'/'rgb'
    if 'sensor_data' in obs and 'hand_camera' in obs['sensor_data'] and 'rgb' in obs['sensor_data']['hand_camera']:
        # Get the image
        image = obs['sensor_data']['hand_camera']['rgb']
        
        # Remove batch dimension if present
        if len(image.shape) == 4 and image.shape[0] == 1:
            image = image[0]
        
        # Convert PyTorch tensor to NumPy array if needed
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        return image
    
    raise ValueError("Cannot find hand camera RGB image in observation")

def extract_joint_state(obs):
    """
    Extract joint state from ManiSkill observation.
    
    Args:
        obs: The observation from ManiSkill environment
        
    Returns:
        joint_state: The joint state as a numpy array
    """
    # Extract joint state from the observation
    if 'agent' in obs and 'qpos' in obs['agent']:
        # Get the joint state
        joint_state = obs['agent']['qpos'][0][:8]
        
        # Convert PyTorch tensor to NumPy array if needed
        if isinstance(joint_state, torch.Tensor):
            joint_state = joint_state.detach().cpu().numpy()
        
        return joint_state
    
    raise ValueError("Cannot find joint state in observation")

def save_debug_image(image, filepath):
    """
    Save an image for debugging purposes.
    
    Args:
        image: The image as a numpy array
        filepath: The path to save the image to
    """
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Save image
    try:
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved debug image to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save debug image: {e}")

if __name__ == "__main__":
    main()
