import os
import logging
import numpy as np
from env_tests.agents.base_agent import BaseAgent
from env_tests.utils.image_utils import save_debug_image

logger = logging.getLogger(__name__)

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as pi0_client
    SOCKET_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning(
        "openpi_client module not found. Please install it from openpi.")
    logger.warning("Will run without policy integration.")
    SOCKET_CLIENT_AVAILABLE = False


class GR00TAgent(BaseAgent):
    """Agent that uses Gr00t for action generation."""

    def __init__(self, host="localhost", port=8000, cameras=None):
        """Initialize the Gr00t agent.

        Args:
            host: Gr00t server host
            port: Gr00t server port
        """
        super().__init__(host, port)
        self.use_which_external_camera = None
        self.cameras = cameras

    def connect(self):
        """Connect to the Gr00t server."""
        if not SOCKET_CLIENT_AVAILABLE:
            logger.error("Cannot use Gr00t: socket_client module not found.")
            logger.error(
                "Please copy the socket_client directory from Gr00t to ManiSkill.")
            return False

        try:
            logger.info(
                f"Connecting to Gr00t server at {self.host}:{self.port}")
            self.client = WebsocketClientPolicy(host=self.host, port=self.port)
            logger.info("Connected to Gr00t server successfully")
            logger.info(
                f"Server metadata: {self.client.get_server_metadata()}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Gr00t server: {e}")
            logger.info("Will use random actions instead")
            self.client = None
            return False

    def extract_camera_image(self, obs, camera_name):
        """Extract RGB image from camera in ManiSkill observation.

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
            logger.error(
                f"Camera '{camera_name}' not found in observation. Available cameras: {list(obs['sensor_data'].keys())}")
            # Return a default empty image or raise the exception
            raise e

    def extract_joint_state(self, obs):
        """Extract joint state from observation.

        Args:
            obs: The observation from ManiSkill environment

        Returns:
            joint_state: The joint state as a numpy array
        """
        # Extract joint state from observation
        # This will depend on the exact format of your observation
        joint_state = obs['agent']['qpos'][0][:8]
        joint_state = joint_state.detach().cpu().numpy()
        return joint_state

    def get_action(self, obs, description, step, use_which_external_camera):
        """Get an action from the Gr00t agent.

        Args:
            obs: Environment observation
            description: Description of the task
            step: Current step number
            use_which_external_camera: Camera to use for external viewing

        Returns:
            action: Action to take
        """
        if not self.is_connected():
            return None

        self.use_which_external_camera = use_which_external_camera
        gr00t_obs = {}
        try:
            for camera in self.cameras:
                image = self.extract_camera_image(obs, camera)
                if camera == "hand_camera":
                    gr00t_obs['wrist_image'] = image
                else:
                    obs_name = camera.replace('_camera', '_image')
                    gr00t_obs[obs_name] = image
            # Extract required observation for Gr00t
            joint_state = self.extract_joint_state(obs)

            # Create observation dictionary for Gr00t server
            gr00t_obs['joint_state'] = joint_state
            gr00t_obs['prompt'] = description
            print(f"step {step+1}{'-'*20}")
            logger.info("gr00t_obs info:")
            for key, value in gr00t_obs.items():
                if key == "prompt":
                    logger.info(f"{key}: {value}")
                else:
                    logger.info(f"{key}: {value.shape}")

            # Send observation to Gr00t server
            logger.info(f"Sending observation to Gr00t server (step {step+1})")
            result = self.client.infer(gr00t_obs)
            logger.info(f"Gr00t socket result: {result['status']}, result keys: {list(result.keys())}")

            # Get action from result
            if 'actions' in result and result['actions'] is not None and len(result['actions']) > 0:
                action = result['actions'][0]
                # Create a copy of the action to avoid "assignment destination is read-only" error
                if hasattr(action, 'flags') and not action.flags.writeable:
                    action = np.array(action, dtype=np.float32)

                # Check if action is 2D and extract first action
                if len(action.shape) > 1 and action.shape[0] > 0:
                    logger.warning(
                        f"Action has shape {action.shape}, extracting first vector")
                    action = action[0]

                logger.info(f"Using Gr00t action: {action}")
                return action
            else:
                # Log available keys in result for debugging
                logger.warning(
                    f"No valid actions in result. Result keys: {list(result.keys())}")
                return None
        except Exception as e:
            logger.error(f"Error getting actions from Gr00t: {e}")
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
                save_debug_image(image, os.path.join(
                    image_dirs[camera], f"step_{step:04d}.png"))
        except Exception as e:
            logger.error(f"Error saving debug images: {e}")
