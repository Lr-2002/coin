import os
import logging
import numpy as np
import cv2
from env_tests.agents.base_agent import BaseAgent
from env_tests.utils.image_utils import save_debug_image

logger = logging.getLogger(__name__)

# Check if socket_client is available
try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as pi0_client
    SOCKET_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("openpi_client module not found. Please installit from openpi.")
    logger.warning("Will run without policy integration.")
    SOCKET_CLIENT_AVAILABLE = False

class COGACTAgent(BaseAgent):
    """Agent that uses CogACT for action generation."""
    
    def __init__(self, host="localhost", port=8000, cameras=None):
        """Initialize the CogACT agent.
        
        Args:
            host: CogACT server host
            port: CogACT server port
        """
        super().__init__(host, port)
        self.use_which_external_camera = None
        self.cameras = cameras
        
    def connect(self):
        """Connect to the CogACT server."""
        if not SOCKET_CLIENT_AVAILABLE:
            logger.error("Cannot use CogACT: socket_client module not found.")
            logger.error("Please copy the socket_client directory from CogACT to ManiSkill.")
            return False
        
        try:
            logger.info(f"Connecting to CogACT server at {self.host}:{self.port}")
            self.client = WebsocketClientPolicy(host=self.host, port=self.port)
            logger.info("Connected to CogACT server successfully")
            logger.info(f"Server metadata: {self.client.get_server_metadata()}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to CogACT server: {e}")
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
    
    def get_action(self, obs, description, step, use_which_external_camera):
        """Get an action from the CogACT agent.
        
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
        
        cogact_obs = {}
        try:
            # Extract image from observation based on camera preference
            for camera in self.cameras:
                cogact_obs['image'] = self.extract_camera_image(obs, camera)
            
            # Create observation dictionary for CogACT server
            cogact_obs['prompt'] = description
            
            print(f"step {step+1}{'-'*20}")
            logger.info("cogact_obs info:")
            for key, value in cogact_obs.items():
                if key == "prompt":
                    logger.info(f"{key}: {value}")
                else:
                    logger.info(f"{key}: {value.shape}")
            
            # Send observation to CogACT server
            logger.info(f"Sending observation to CogACT server (step {step+1})")
            result = self.client.infer(cogact_obs)
            logger.info(f"socket result: {result['status']}, result keys: {list(result.keys())}")
            
            # Get action from result
            if 'actions' in result and result['actions'] is not None and len(result['actions']) > 0:
                action = result['actions'][0]
                # Create a copy of the action to avoid "assignment destination is read-only" error
                if hasattr(action, 'flags') and not action.flags.writeable:
                    action = np.array(action, dtype=np.float32)
                logger.info(f"Using CogACT action: {action}")
                return action
            else:
                # Log available keys in result for debugging
                logger.warning(f"No valid actions in result. Result keys: {list(result.keys())}")
                return None
        except Exception as e:
            logger.error(f"Error getting actions from CogACT: {e}")
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
                save_debug_image(self.extract_camera_image(obs, camera), os.path.join(
                    image_dirs[camera], f"step_{step:04d}.png"))
        except Exception as e:
            logger.error(f"Error saving debug images: {e}")
