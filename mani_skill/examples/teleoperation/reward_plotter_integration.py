"""
Reward Plotter Integration for MuJoCo AR Teleoperation

This file provides integration code for adding real-time reward plotting to the MuJoCo AR
teleoperation interface. It can be used to visualize rewards during task execution.

Usage:
1. Import the necessary functions from this file in mujoco_ar_teleop.py
2. Initialize the reward plotter in your main function
3. Update the reward plotter with each new reward
4. Render the reward plot on your camera views

Example:
```python
from reward_plotter_integration import initialize_reward_plotter, update_reward_plot, render_reward_plot

# In your main function:
reward_plotter = initialize_reward_plotter()

# After env.step():
update_reward_plot(reward_plotter, reward)

# Before displaying camera views:
camera_views[0] = render_reward_plot(reward_plotter, camera_views[0])
```
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import time
from typing import List, Optional, Tuple, Dict, Any, Union

class RewardPlotter:
    """
    A utility class for real-time plotting of rewards during teleoperation.
    Displays a continuously updating plot of rewards over time.
    """
    def __init__(
        self, 
        window_size: int = 100, 
        figsize: Tuple[int, int] = (4, 2),
        position: Tuple[int, int] = (20, 20),
        alpha: float = 0.7,
        font_scale: float = 0.6
    ):
        """
        Initialize the reward plotter.
        
        Args:
            window_size: Number of reward values to display in the plot
            figsize: Size of the matplotlib figure (width, height) in inches
            position: Position of the plot overlay (x, y) in pixels
            alpha: Transparency of the plot overlay (0-1)
            font_scale: Scale factor for font size
        """
        self.window_size = window_size
        self.rewards = []
        self.timestamps = []
        self.start_time = time.time()
        self.figsize = figsize
        self.position = position
        self.alpha = alpha
        self.font_scale = font_scale
        self.fig = Figure(figsize=self.figsize, dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setup_plot()
        
    def setup_plot(self):
        """Set up the plot appearance"""
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        self.fig.patch.set_alpha(0.0)  # Transparent background
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.set_title('Reward', color='white', fontsize=10)
        self.ax.set_ylabel('Value', color='white', fontsize=8)
        self.ax.set_xlabel('Time (s)', color='white', fontsize=8)
        self.ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
    def update(self, reward: Union[float, np.ndarray, List[float]]):
        """
        Update the reward plot with a new reward value.
        
        Args:
            reward: The reward value to add to the plot
        """
        # Handle different reward formats
        if isinstance(reward, (list, np.ndarray)) and len(reward) > 0:
            reward_value = reward[0] if isinstance(reward[0], (int, float)) else float(reward[0])
        else:
            reward_value = float(reward)
            
        current_time = time.time() - self.start_time
        self.rewards.append(reward_value)
        self.timestamps.append(current_time)
        
        # Keep only the most recent window_size values
        if len(self.rewards) > self.window_size:
            self.rewards = self.rewards[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
    
    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Render the reward plot as an overlay on the provided frame.
        
        Args:
            frame: The video frame to overlay the plot on
            
        Returns:
            The frame with the plot overlay
        """
        # Clear the plot and redraw
        self.ax.clear()
        self.setup_plot()
        
        if len(self.rewards) > 1:
            self.ax.plot(self.timestamps, self.rewards, 'g-', linewidth=2)
            
            # Add the latest reward value as text
            latest_reward = self.rewards[-1]
            self.ax.text(
                0.02, 0.92, f"Current: {latest_reward:.4f}", 
                transform=self.ax.transAxes, color='white', fontsize=8
            )
            
            # Add min/max/avg
            min_reward = min(self.rewards)
            max_reward = max(self.rewards)
            avg_reward = sum(self.rewards) / len(self.rewards)
            self.ax.text(
                0.02, 0.82, f"Min: {min_reward:.4f}", 
                transform=self.ax.transAxes, color='white', fontsize=8
            )
            self.ax.text(
                0.02, 0.72, f"Max: {max_reward:.4f}", 
                transform=self.ax.transAxes, color='white', fontsize=8
            )
            self.ax.text(
                0.02, 0.62, f"Avg: {avg_reward:.4f}", 
                transform=self.ax.transAxes, color='white', fontsize=8
            )
        
        # Draw the plot to the canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Convert to a numpy array
        plot_img = np.array(self.canvas.buffer_rgba())
        
        # Convert RGBA to BGR for OpenCV
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        
        # Resize the plot if needed
        h, w = frame.shape[:2]
        plot_h, plot_w = plot_img.shape[:2]
        
        # Create a mask for the plot (for transparency)
        mask = np.ones(plot_img.shape, dtype=np.uint8) * 255
        mask = mask * self.alpha
        
        # Overlay the plot on the frame
        x, y = self.position
        
        # Make sure the plot fits within the frame
        if x + plot_w > w:
            x = w - plot_w
        if y + plot_h > h:
            y = h - plot_h
            
        # Create a region of interest
        roi = frame[y:y+plot_h, x:x+plot_w]
        
        # Blend the plot with the ROI
        blended = cv2.addWeighted(
            plot_img, self.alpha, roi, 1 - self.alpha, 0
        )
        
        # Put the blended image back into the frame
        frame_with_plot = frame.copy()
        frame_with_plot[y:y+plot_h, x:x+plot_w] = blended
        
        return frame_with_plot


# Convenience functions for integration with mujoco_ar_teleop.py

def initialize_reward_plotter(window_size=100, position=(20, 20)):
    """
    Initialize a new reward plotter.
    
    Args:
        window_size: Number of reward values to display
        position: Position of the plot overlay (x, y) in pixels
        
    Returns:
        A new RewardPlotter instance
    """
    return RewardPlotter(window_size=window_size, position=position)


def update_reward_plot(plotter, reward):
    """
    Update the reward plotter with a new reward value.
    
    Args:
        plotter: The RewardPlotter instance
        reward: The new reward value
    """
    plotter.update(reward)


def render_reward_plot(plotter, frame):
    """
    Render the reward plot on a frame.
    
    Args:
        plotter: The RewardPlotter instance
        frame: The frame to overlay the plot on
        
    Returns:
        The frame with the plot overlay
    """
    return plotter.render(frame)


# Patch for display_camera_views to return images
def display_camera_views_with_return(obs, target_size=(512, 512), return_images=False):
    """
    Display camera views from observation dictionary and optionally return the images.
    
    Args:
        obs: Observation dictionary containing camera views
        target_size: Target size for the displayed images
        return_images: Whether to return the camera view images
        
    Returns:
        If return_images is True, returns a list of camera view images
        Otherwise, returns None
    """
    # Extract images from observations
    images = observations_to_images(obs)
    
    # Resize images to target size
    resized_images = []
    for img in images:
        if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
            img = cv2.resize(img, target_size)
        resized_images.append(img)
    
    # Display images
    if resized_images:
        tiled_img = tile_images(resized_images)
        cv2.imshow("Camera Views", tiled_img)
    
    # Return images if requested
    if return_images:
        return resized_images
    return None
