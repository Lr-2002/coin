import cv2
import os
import csv
from collections import Counter
import numpy as np
import glob

def get_video_frame_count(video_path):
    """Get the number of frames in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    # Get frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return frame_count

def categorize_frames(frame_count):
    """Categorize video frame count into custom ranges."""
    if frame_count < 100:
        return "0-100"
    elif frame_count < 250:
        return "100-250"
    elif frame_count < 500:
        return "250-500"
    else:
        return "500+"

def main():
    # Path to the videos directory
    videos_dir = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/coin_videos/medias"
    
    # Find all video files
    video_files = glob.glob(os.path.join(videos_dir, "**/*.mp4"), recursive=True)
    
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Measure video frame counts
    video_categories = []
    video_data = []
    
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        frame_count = get_video_frame_count(video_path)
        category = categorize_frames(frame_count)
        
        video_categories.append(category)
        video_data.append((video_name, frame_count, category))
        
        print(f"Processed: {video_name} - Frames: {frame_count} - Category: {category}")
    
    # Count occurrences of each frame count category
    category_counts = Counter(video_categories)
    
    # Define the order of categories
    category_order = ["0-100", "100-250", "250-500", "500+"]
    
    # Sort categories in the predefined order
    sorted_categories = [(category, category_counts[category]) for category in category_order if category in category_counts]
    
    # Output path for CSV
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'github_page/static/video_frames.csv')
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_range', 'number'])
        writer.writerows(sorted_categories)
    
    print(f"\nCSV generated at: {csv_path}")
    print("Video frame count statistics:")
    for category, count in sorted_categories:
        print(f"{category} frames: {count} videos")
    
    # Calculate summary statistics
    frames_array = np.array([data[1] for data in video_data])
    print(f"\nSummary Statistics:")
    print(f"Total videos: {len(frames_array)}")
    print(f"Min frames: {np.min(frames_array)}")
    print(f"Max frames: {np.max(frames_array)}")
    print(f"Mean frames: {np.mean(frames_array):.2f}")
    print(f"Median frames: {np.median(frames_array):.2f}")

if __name__ == "__main__":
    main()