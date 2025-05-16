import os
import time
import logging
import numpy as np
from env_tests.agents.base_agent import BaseAgent
from env_tests.utils.image_utils import save_debug_image
from openpi_client import image_tools

logger = logging.getLogger(__name__)

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as pi0_client
    SOCKET_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("openpi_client module not found. Please installit from openpi.")
    logger.warning("Will run without policy integration.")
    SOCKET_CLIENT_AVAILABLE = False

class PI0Agent(BaseAgent):
    """Agent that uses Pi0 for action generation."""
    
    def __init__(self, host="localhost", port=8000, cameras=None):
        """Initialize the Pi0 agent.
        
        Args:
            host: Pi0 server host
            port: Pi0 server port
        """
        super().__init__(host, port)
        self.use_which_external_camera = None   
        self.cameras = cameras
        
    def connect(self):
        """Connect to the Pi0 server."""
        if not SOCKET_CLIENT_AVAILABLE:
            logger.error("Cannot use Pi0: socket_client module not found.")
            logger.error("Please copy the socket_client directory from Pi0 to ManiSkill.")
            return False
        
        try:
            logger.info(f"Connecting to Pi0 server at {self.host}:{self.port}")
            self.client = WebsocketClientPolicy(host=self.host, port=self.port)
            logger.info("Connected to Pi0 server successfully")
            logger.info(f"Server metadata: {self.client.get_server_metadata()}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Pi0 server: {e}")
            logger.info("Will use random actions instead")
            self.client = None
            return False
    
    def extract_camera_image(self, obs, camera_name):
        """Extract RGB image from external camera in ManiSkill observation.
        
        Args:
            obs: The observation from ManiSkill environment
            camera_name: Name of the camera to use
            
        Returns:
            image: The RGB image as a numpy array
        """
        try:
            # Get the image
            image = obs['sensor_data'][camera_name]['rgb']
            
            # Remove batch dimension if present
            if len(image.shape) == 4 and image.shape[0] == 1:
                image = image[0]
            
            # Convert PyTorch tensor to NumPy array if needed
            if hasattr(image, 'detach'):
                image = image.detach().cpu().numpy()
            
            return image
        except KeyError as e:
            logger.error(f"Camera '{camera_name}' not found in observation. Available cameras: {list(obs['sensor_data'].keys())}")
            # Return a default empty image or raise the exception
            raise e
    
    def extract_state(self, obs):
        """Extract state information from observation.
        
        Args:
            obs: The observation from ManiSkill environment
            
        Returns:
            state: The state as a numpy array
        """
        try:
            state = obs['agent']['qpos'][0]
            state = state[:8].detach().cpu().numpy()
            return state
        except Exception as e:
            logger.error(f"Error extracting state: {e}")
            return np.zeros(8)  # Return zeros as fallback
    
    def get_action(self, obs, description, step, use_which_external_camera):
        """Get an action from the Pi0 agent.
        
        Args:
            obs: Environment observation
            description: Description of the task
            step: Current step number
            
        Returns:
            action: Action to take
        """
        if not self.is_connected():
            return None
        
        self.use_which_external_camera = use_which_external_camera

        # TODO: fix pi0 camera connect
        pi0_obs = {}
        
        try:
            for camera in self.cameras:
                image = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(self.extract_camera_image(obs, camera), 224, 224))
                if camera == 'hand_camera':
                    pi0_obs['observation/wrist_image'] = image
                else:
                    obs_name = camera.replace('_camera', '_image')
                    pi0_obs['observation/' + obs_name] = image

            pi0_obs['prompt'] = description
            pi0_obs['observation/state'] = self.extract_state(obs)
            
            # # Extract images and resize them for Pi0 - using image_tools from openpi_client
            # wrist_image = image_tools.convert_to_uint8(
            #     image_tools.resize_with_pad(self.extract_camera_image(obs, 'hand_camera'), 224, 224))
            # left_image = image_tools.convert_to_uint8(
            #     image_tools.resize_with_pad(self.extract_camera_image(obs, 'left_camera'), 224, 224))
            # image = image_tools.convert_to_uint8(
            #     image_tools.resize_with_pad(self.extract_camera_image(obs, 'base_front_camera'), 224, 224))
            
            print(f"step {step+1}{'-'*20}")
            logger.info("pi0_obs info:")
            for key, value in pi0_obs.items():
                if key == "prompt":
                    logger.info(f"{key}: {value}")
                else:
                    logger.info(f"{key}: {value.shape}")
            
            start_time = time.time()
            logger.info(f"Sending observation to Pi0 server (step {step+1})")
            result = self.client.infer(pi0_obs)
            end_time = time.time()
            logger.info(f"Time taken for Pi0 inference: {end_time - start_time:.2f} seconds")
            
            # Get action from result
            if 'actions' in result and result['actions'] is not None and len(result['actions']) > 0:
                action_chunk = result['actions']
                logger.info(f"Received action chunk with shape: {action_chunk.shape}")
                
                # Get the first action from the chunk
                action = action_chunk[0]
                # # Create a copy of the action to avoid "assignment destination is read-only" error
                # if hasattr(action, 'flags') and not action.flags.writeable:
                #     action = np.array(action, dtype=np.float32)
                
                # # Check if action is 2D and extract first action
                # if len(action.shape) > 1 and action.shape[0] > 0:
                #     logger.warning(f"Action has shape {action.shape}, extracting first vector")
                #     action = action[0]
                
                logger.info(f"Using Pi0 action: {action}")
                return action
            else:
                # Log available keys in result for debugging
                logger.warning(f"No valid actions in result. Result keys: {list(result.keys())}")
                return None
        except Exception as e:
            logger.error(f"Error getting actions from Pi0: {e}")
            logger.error(f"Exception details: {str(e)}")
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
                image = self.extract_camera_image(obs, camera)
                save_debug_image(image, os.path.join(image_dirs[camera], f"step_{step:04d}.png"))
        except Exception as e:
            logger.error(f"Error saving debug images: {e}")
