import cv2
import os
import numpy as np
from typing import Optional, Tuple, Union

class VideoWriter:
    """
    Video Writer class for ManiSkill environment.
    Used to save frames to a video file.
    """
    
    def __init__(self, frame: np.ndarray, path: str, fps: int = 30):
        """
        Initialize the video writer.
        
        Args:
            frame: Initial frame to determine video dimensions
            path: Path to save the video file
            fps: Frames per second (default: 30)
        """
        self.path = path
        self.fps = fps
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Fixed method name
        self.writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        
        # Write the first frame
        self.write(frame)
    
    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.
        
        Args:
            frame: Frame to write (numpy array in BGR format)
        """
        if frame.dtype != np.uint8:
            # Convert to uint8 if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # OpenCV expects BGR format
        if self.writer is not None:  # Check if writer exists
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                self.writer.write(frame)
            else:
                # Convert grayscale to BGR if needed
                self.writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    
    def close(self) -> None:
        """
        Close the video writer and save the video file.
        """
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Video saved to {self.path}")
    
    def __del__(self):
        """
        Destructor to ensure video is properly saved when object is deleted.
        """
        self.close()