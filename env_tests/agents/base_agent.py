import abc
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseAgent(abc.ABC):
    """Base class for all VLA agents."""
    
    def __init__(self, host="localhost", port=8000, cameras=None):
        """Initialize the agent.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.cameras = cameras
        self.client = None  
        
    @abc.abstractmethod
    def connect(self):
        """Connect to the agent server."""
        pass
    
    @abc.abstractmethod
    def get_action(self, obs, info, step):
        """Get an action from the agent.
        
        Args:
            obs: Environment observation
            info: Environment info
            step: Current step number
            
        Returns:
            action: Action to take
        """
        pass
    
    def save_debug_images(self, obs, step, image_dirs):
        """Save debug images.
        
        Args:
            obs: Environment observation
            step: Current step number
            image_dirs: Dictionary of image directories
        """
        pass
    
    def is_connected(self):
        """Check if the agent is connected to the server.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.client is not None
    
    def fallback_action(self, env):
        """Fallback action when the agent fails.
        
        Args:
            env: Environment
            
        Returns:
            action: Random action
        """
        action = env.action_space.sample()
        logger.warning(f"Using random action as fallback: {action}")
        return action
