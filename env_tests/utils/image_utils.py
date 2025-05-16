import cv2
import numpy as np
import logging
import os
from datetime import datetime
import base64

logger = logging.getLogger(__name__)

def extract_camera_image(obs, camera_name):
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
def encode_image(image_array):
    """Encode image array to base64 string for LLM input."""
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    # Convert to BGR for OpenCV
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Encode image to JPEG
    success, buffer = cv2.imencode(".jpg", image_array)
    if not success:
        logger.error("Failed to encode image")
        return None
    
    # Convert to base64
    base64_string = base64.b64encode(buffer).decode("utf-8")
    return base64_string

def setup_directories(args, info):
    """
    Set up directory structure for saving images and videos.
    
    Structure:
    - image_dir/
      - agent_name/
        - timestamp_task/
          - external_camera/
          - hand_camera/
          - actions.json
          - video.mp4
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get environment ID for folder name
    env_id = args.env_id.replace('-', '_')  # Replace hyphens with underscores for better folder naming
    cameras = args.cameras
    # Create run folder name
    run_folder = f"{timestamp}_{env_id}"
    
    # Base directories
    base_dirs = {
        'run_folder': run_folder,
        'timestamp': timestamp
    }
    
    # Determine agent type
    agent_type = args.vla_agent
    
    # Create agent directory
    agent_base = os.path.join(args.image_dir, agent_type)
    os.makedirs(agent_base, exist_ok=True)
    
    # Create run directory
    run_dir = os.path.join(agent_base, run_folder)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create camera directories
    for camera in cameras:
        camera_dir = os.path.join(run_dir, camera)
        os.makedirs(camera_dir, exist_ok=True)
        base_dirs[camera] = camera_dir
        logger.info(f"{camera} images will be saved to: {camera_dir}")
    base_dirs['action'] = run_dir
    logger.info(f"actions will be saved to: {run_dir}")
    
    return base_dirs

def setup_directories_hierarchical_vla(args, info):
    """
    Set up directory structure for saving images and videos.
    
    Structure:
    - image_dir/
      - agent_name/
        - timestamp_task/
          - external_camera/
          - hand_camera/
          - base_camera/
          - llm_plan/
          - actions.json
          - executed_subtasks.json
          - video.mp4
    """
    base_dirs = {}
    
    if args.save_images:
        # Create base directory
        os.makedirs(args.image_dir, exist_ok=True)
        
        # Determine agent type
        agent_type = "hierarchical_vla_" + args.vla_agent + "_" + args.llm_model
        
        # Create agent directory
        agent_dir = os.path.join(args.image_dir, agent_type)
        os.makedirs(agent_dir, exist_ok=True)
        
        # Create timestamp and task directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = args.env_id.replace("-", "_").lower()
        run_dir = os.path.join(agent_dir, f"{timestamp}_{task_name}")
        os.makedirs(run_dir, exist_ok=True)
        base_dirs['run_folder'] = run_dir
        
        # Create camera directories
        if args.save_images:
            for camera in args.cameras:
                camera_dir = os.path.join(run_dir, camera)
                os.makedirs(camera_dir, exist_ok=True)
                base_dirs[camera] = camera_dir
            
            # Create directory for LLM plans
            llm_plan_dir = os.path.join(run_dir, "llm_plan")
            os.makedirs(llm_plan_dir, exist_ok=True)
            base_dirs['llm_plan'] = llm_plan_dir
            
            base_dirs['action'] = run_dir
            base_dirs['executed_subtasks'] = run_dir
    
    return base_dirs

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
        # logger.info(f"Saved debug image to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save debug image: {e}")

# def resize_with_pad(image, target_width, target_height):
#     """
#     Resize an image to the target dimensions while maintaining aspect ratio with padding.
    
#     Args:
#         image: The image as a numpy array
#         target_width: The target width
#         target_height: The target height
        
#     Returns:
#         The resized image
#     """
#     # Get current dimensions
#     h, w = image.shape[:2]
    
#     # Calculate scaling factor
#     scale = min(target_width / w, target_height / h)
    
#     # Calculate new dimensions
#     new_w = int(w * scale)
#     new_h = int(h * scale)
    
#     # Resize image
#     resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
#     # Create padded image
#     padded = np.zeros((target_height, target_width, 3), dtype=resized.dtype)
    
#     # Calculate padding
#     pad_x = (target_width - new_w) // 2
#     pad_y = (target_height - new_h) // 2
    
#     # Copy resized image to padded image
#     padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
#     return padded

# def convert_to_uint8(image):
#     """
#     Convert an image to uint8 format.
    
#     Args:
#         image: The image as a numpy array
        
#     Returns:
#         The image as uint8
#     """
#     if image.dtype != np.uint8:
#         if image.max() <= 1.0:
#             image = (image * 255).astype(np.uint8)
#         else:
#             image = image.astype(np.uint8)
#     return image

# def display_camera_views(camera_views):
#     """
#     Display multiple camera views in a single window.
    
#     Args:
#         camera_views: Dictionary of camera views, where keys are camera names and values are images
#     """
#     if not camera_views:
#         return
    
#     # Convert all images to uint8
#     for key in camera_views:
#         camera_views[key] = convert_to_uint8(camera_views[key])
    
#     # Get dimensions of first image
#     first_key = list(camera_views.keys())[0]
#     h, w = camera_views[first_key].shape[:2]
    
#     # Create a grid of images
#     num_images = len(camera_views)
#     cols = min(3, num_images)
#     rows = (num_images + cols - 1) // cols
    
#     # Create a blank canvas
#     canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
#     # Place images on canvas
#     for i, (name, img) in enumerate(camera_views.items()):
#         r, c = i // cols, i % cols
#         canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
        
#         # Add camera name
#         cv2.putText(canvas, name, (c*w + 10, r*h + 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
#     # Display canvas
#     cv2.imshow('Camera Views', canvas)
#     cv2.waitKey(1)  # Update window
