import numpy as np
import torch
import cv2


def display_camera_views(obs, target_size=(2048, 2048)):
    """Display multiple camera views side by side using OpenCV

    Args:
        obs: Observation dictionary containing camera data
        target_size: Tuple of (width, height) to resize the final image to.
                    If None, the original size is preserved.
                    Default is (512, 512).
    """
    # For rgbd observation mode, camera data is in sensor_data
    # print(obs)
    if "sensor_data" in obs:
        cameras = {}
        # print(obs['sensor_data'].keys())
        # Process all available cameras
        camera_names = [
            "left_camera",
            "right_camera",
            "base_camera",
            "wrist_camera",
            "base_front_camera",
            "human_camera",
        ]

        # Check for legacy camera names
        legacy_camera_map = {
            "external_camera": "base_camera",
            "hand_camera": "wrist_camera",
        }

        # Process all cameras
        for camera_name in camera_names:
            # Check if this camera exists in the observation
            if (
                camera_name in obs["sensor_data"]
                and "rgb" in obs["sensor_data"][camera_name]
            ):
                rgb_img = obs["sensor_data"][camera_name]["rgb"]
                cameras[camera_name] = process_camera_image(rgb_img)

        # Check for legacy camera names for backward compatibility
        for legacy_name, new_name in legacy_camera_map.items():
            if (
                legacy_name in obs["sensor_data"]
                and "rgb" in obs["sensor_data"][legacy_name]
                and new_name not in cameras
            ):
                rgb_img = obs["sensor_data"][legacy_name]["rgb"]
                cameras[legacy_name] = process_camera_image(rgb_img)

        if not cameras:
            return

        # Create a combined view
        if len(cameras) == 1:
            # Only one camera available
            cam_name = list(cameras.keys())[0]
            combined_view = cameras[cam_name]
            title = f"Camera View: {cam_name}"
        else:
            # Multiple cameras - create side-by-side view
            # Get all camera images and resize to the same height
            cam_images = list(cameras.values())
            heights = [img.shape[0] for img in cam_images]

            # Use the maximum height among all cameras
            max_height = max(heights)

            # Resize all images to have the same height
            resized_images = []
            for i, img in enumerate(cam_images):
                if img.shape[0] != max_height:
                    # Calculate new width to maintain aspect ratio
                    new_width = int(img.shape[1] * (max_height / img.shape[0]))
                    resized_img = cv2.resize(img, (new_width, max_height))
                    resized_images.append(resized_img)
                else:
                    resized_images.append(img)

            # Concatenate images horizontally
            combined_view = cv2.hconcat(resized_images)

            # Create title with camera names
            cam_names = list(cameras.keys())
            title = "Camera Views: " + " | ".join(cam_names)

        # Resize the combined view to target size if specified
        if target_size is not None:
            # Preserve aspect ratio
            h, w = combined_view.shape[:2]
            aspect_ratio = w / h

            if aspect_ratio > 1:  # Width > Height
                new_width = target_size[0]
                new_height = int(new_width / aspect_ratio)
            else:  # Height >= Width
                new_height = target_size[1]
                new_width = int(new_height * aspect_ratio)

            # Ensure dimensions don't exceed target size
            new_width = min(new_width, target_size[0])
            new_height = min(new_height, target_size[1])

            # Resize the image
            combined_view = cv2.resize(combined_view, (new_width, new_height))

        # Display the combined view
        cv2.imshow(title, combined_view)
        return combined_view, title


def process_camera_image(rgb_img):
    """Process a camera image to convert it to BGR format for OpenCV display

    Args:
        rgb_img: RGB image from observation

    Returns:
        BGR image ready for OpenCV display
    """
    if isinstance(rgb_img, torch.Tensor):
        rgb_img = rgb_img.cpu().numpy()

    # Remove batch dimension if present
    if len(rgb_img.shape) == 4 and rgb_img.shape[0] == 1:
        rgb_img = rgb_img[0]

    # Make sure image is in the right format (H, W, C) with values in [0, 255]
    if rgb_img.max() <= 1.0:
        rgb_img = (rgb_img * 255).astype(np.uint8)

    # OpenCV uses BGR format
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return bgr_img
