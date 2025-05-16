import h5py
import numpy as np
import os
import cv2
import glob
import argparse
from typing import Dict, List, Any, Union, Optional, Tuple, cast
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def explore_hdf5_structure(file_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Explore the structure of an HDF5 file and print detailed information.
    
    Args:
        file_path (str): Path to the HDF5 file
        verbose (bool): Whether to print detailed information
    
    Returns:
        dict: A dictionary containing information about the file structure
    """
    file_info = {
        'path': file_path,
        'groups': [],
        'datasets': [],
        'image_datasets': [],
        'attributes': {}
    }
    
    try:
        # Open the HDF5 file in read mode
        with h5py.File(file_path, 'r') as f:
            if verbose:
                print(f"File: {file_path}")
                print("Structure of HDF5 file:")
                print("-" * 50)
            
            # Check if file is empty
            if len(f.keys()) == 0:
                if verbose:
                    print("File is empty (no groups or datasets)")
                file_info['is_empty'] = True
                return file_info
            
            file_info['is_empty'] = False
            
            # Recursive function to explore the structure
            def explore_structure(name: str, obj: Union[h5py.Group, h5py.Dataset]) -> None:
                indent = "  " * name.count('/')
                
                if isinstance(obj, h5py.Group):
                    if verbose:
                        print(f"{indent}Group: {name}")
                        print(f"{indent}  Attributes: {list(obj.attrs.keys())}")
                    
                    group_info = {
                        'name': name,
                        'attributes': {k: str(v) for k, v in obj.attrs.items()}
                    }
                    file_info['groups'].append(group_info)
                    
                elif isinstance(obj, h5py.Dataset):
                    # Get shape and dtype safely
                    shape_str = str(obj.shape)
                    dtype_str = str(obj.dtype)
                    
                    if verbose:
                        print(f"{indent}Dataset: {name}")
                        print(f"{indent}  Shape: {shape_str}")
                        print(f"{indent}  Dtype: {dtype_str}")
                        print(f"{indent}  Attributes: {list(obj.attrs.keys())}")
                    
                    dataset_info = {
                        'name': name,
                        'shape': obj.shape,
                        'dtype': dtype_str,
                        'attributes': {k: str(v) for k, v in obj.attrs.items()}
                    }
                    file_info['datasets'].append(dataset_info)
                    
                    # Check if it looks like an image dataset
                    if len(obj.shape) >= 3 and np.issubdtype(obj.dtype, np.number):
                        # Specific patterns for our dataset
                        if 'rgb' in name.lower() or 'depth' in name.lower():
                            if verbose:
                                print(f"{indent}  Appears to be an image dataset")
                            dataset_info['is_image'] = True
                            file_info['image_datasets'].append(dataset_info)
            
            # Visit all objects in the file
            f.visititems(explore_structure)
            
            # Store file attributes
            for key, value in f.attrs.items():
                file_info['attributes'][key] = str(value)
            
            if verbose:
                print("-" * 50)
                print("File attributes:", list(f.attrs.keys()))
                
    except FileNotFoundError:
        if verbose:
            print(f"Error: File '{file_path}' not found")
        file_info['error'] = f"File not found: {file_path}"
    except Exception as e:
        if verbose:
            print(f"Error: {str(e)}")
        file_info['error'] = str(e)
    
    return file_info

def convert_h5_images_to_mp4(file_path: str, fps: int = 30, output_dir: Optional[str] = None) -> List[str]:
    """
    Convert image sequences in an HDF5 file to MP4 videos.
    
    Args:
        file_path (str): Path to the HDF5 file
        fps (int): Frames per second for the output video
        output_dir (str): Directory to save the output videos (default: same as input file)
    
    Returns:
        list: Paths to the created video files
    """
    created_videos = []
    
    try:
        # Get base name for output files
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the HDF5 file in read mode
        with h5py.File(file_path, 'r') as f:
            # Dictionary to store image sequences
            image_sequences = {}
            
            # Function to find image datasets
            def find_image_datasets(name: str, obj: Union[h5py.Group, h5py.Dataset]) -> None:
                # Only process datasets (not groups or other types)
                if not isinstance(obj, h5py.Dataset):
                    return
                
                # Check if it's a dataset with shape that looks like images
                if len(obj.shape) >= 3 and np.issubdtype(obj.dtype, np.number):
                    # Specifically look for rgb and depth datasets
                    if 'rgb' in name.lower() or 'depth' in name.lower():
                        # Extract view name from the dataset path
                        path_parts = name.split('/')
                        view_name = '_'.join(path_parts[-2:])  # camera_type + image_type
                        
                        # Store the dataset path
                        image_sequences[view_name] = name
            
            # Find all image datasets
            f.visititems(find_image_datasets)
            
            logger.info(f"Found {len(image_sequences)} potential image sequences")
            
            # Process each image sequence
            for view_name, dataset_path in image_sequences.items():
                logger.info(f"Processing view: {view_name}")
                if not ('base' in view_name and 'rgb' in view_name) :
                    continue
                
                try:
                    # Get the dataset
                    if dataset_path not in f:
                        logger.error(f"Dataset path {dataset_path} not found in file")
                        continue
                    
                    dataset = f[dataset_path]
                    
                    # Check if it's a dataset
                    if not isinstance(dataset, h5py.Dataset):
                        logger.error(f"{dataset_path} is not a dataset")
                        continue
                    
                    # Determine dimensions
                    if len(dataset.shape) == 3:
                        # Shape is [frames, height, width] - grayscale
                        frames, height, width = dataset.shape
                        is_grayscale = True
                        channels = 1
                    elif len(dataset.shape) == 4 and dataset.shape[-1] == 1:
                        # Shape is [frames, height, width, 1] - grayscale
                        frames, height, width, channels = dataset.shape
                        is_grayscale = True
                    elif len(dataset.shape) == 4 and dataset.shape[-1] in [3, 4]:
                        # Shape is [frames, height, width, channels] - RGB/RGBA
                        frames, height, width, channels = dataset.shape
                        is_grayscale = False
                    else:
                        logger.warning(f"Skipping {view_name}: unexpected dimensions {dataset.shape}")
                        continue
                    
                    logger.info(f"Creating video with {frames} frames at {width}x{height}, channels: {channels}")
                    
                    # Create output filename
                    output_file = os.path.join(output_dir, f"{base_name}_{view_name}.mp4")
                    
                    # Initialize video writer with fourcc code
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
                    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                    
                    # Write frames to video
                    for i in range(frames):
                        if i % 100 == 0:
                            logger.info(f"Processing frame {i}/{frames}")
                        
                        try:
                            # Get the frame as a numpy array
                            frame = np.array(dataset[i])
                            
                            # Handle different data types
                            if is_grayscale:
                                # For depth images, normalize to 0-255 range
                                if 'depth' in view_name.lower():
                                    # Normalize depth values
                                    depth_min = np.min(frame)
                                    depth_max = np.max(frame)
                                    if depth_max > depth_min:
                                        frame = ((frame - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                                    else:
                                        frame = np.zeros_like(frame, dtype=np.uint8)
                                else:
                                    # For other grayscale images
                                    if frame.dtype != np.uint8:
                                        if np.issubdtype(frame.dtype, np.floating) and np.max(frame) <= 1.0:
                                            frame = (frame * 255).astype(np.uint8)
                                        else:
                                            frame = frame.astype(np.uint8)
                                
                                # Reshape if needed
                                if len(frame.shape) == 3 and frame.shape[-1] == 1:
                                    frame = frame.reshape(height, width)
                                
                                # Convert grayscale to BGR for video
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            else:
                                # For RGB/RGBA images
                                if frame.dtype != np.uint8:
                                    if np.issubdtype(frame.dtype, np.floating) and np.max(frame) <= 1.0:
                                        frame = (frame * 255).astype(np.uint8)
                                    else:
                                        frame = frame.astype(np.uint8)
                                
                                # Make sure it's RGB (not RGBA)
                                if channels == 4:
                                    frame = frame[..., :3]
                                
                                # Ensure correct channel order (RGB to BGR for OpenCV)
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            
                            # Write the frame
                            video.write(frame)
                            
                        except Exception as e:
                            logger.error(f"Error processing frame {i}: {str(e)}")
                            continue
                    
                    # Release the video writer
                    video.release()
                    created_videos.append(output_file)
                    logger.info(f"Video saved to {output_file}")
                    
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset_path}: {str(e)}")
                    continue
                
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return created_videos

def process_directory(directory_path: str, output_dir: Optional[str] = None, fps: int = 30, recursive: bool = True) -> Dict[str, Any]:
    """
    Process all HDF5 files in a given directory.
    
    Args:
        directory_path (str): Path to the directory containing HDF5 files
        output_dir (str): Directory to save the output videos (default: same as input files)
        fps (int): Frames per second for the output videos
        recursive (bool): Whether to search for HDF5 files recursively
    
    Returns:
        dict: Summary of processing results
    """
    results = {
        'total_files': 0,
        'empty_files': 0,
        'files_with_images': 0,
        'videos_created': 0,
        'video_paths': []
    }
    
    # Ensure directory path exists
    if not os.path.exists(directory_path):
        logger.error(f"Error: Directory '{directory_path}' not found")
        return results
    
    # Find all HDF5 files in the directory
    pattern = "**/*.h5" if recursive else "*.h5"
    h5_files = glob.glob(os.path.join(directory_path, pattern), recursive=recursive)
    
    if not h5_files:
        logger.warning(f"No HDF5 files found in '{directory_path}'")
        return results
    
    results['total_files'] = len(h5_files)
    logger.info(f"Found {len(h5_files)} HDF5 files to process")
    
    # Process each file
    for i, file_path in enumerate(h5_files):
        print(f"\nProcessing file {i+1}/{len(h5_files)}: {file_path}")
        print("-" * 80)
        
        # Explore file structure
        file_info = explore_hdf5_structure(file_path)
        
        # Check if file is empty
        if file_info.get('is_empty', False):
            results['empty_files'] += 1
            continue
        
        # Check if file has image datasets
        if not file_info.get('image_datasets', []):
            logger.info("No image datasets found in this file")
            continue
        
        results['files_with_images'] += 1
        
        # Convert images to MP4
        videos = convert_h5_images_to_mp4(file_path, fps=fps, output_dir=output_dir)
        results['videos_created'] += len(videos)
        results['video_paths'].extend(videos)
        
        print("-" * 80)
    
    print(f"\nProcessing Summary:")
    print(f"  Total HDF5 files processed: {results['total_files']}")
    print(f"  Empty files: {results['empty_files']}")
    print(f"  Files with image datasets: {results['files_with_images']}")
    print(f"  Total videos created: {results['videos_created']}")
    
    return results

def main():
    # parser = argparse.ArgumentParser(description='Process HDF5 files and convert image sequences to MP4 videos')
    # parser.add_argument('directory', help='Directory containing HDF5 files')
    # parser.add_argument('--output-dir', help='Directory to save output videos')
    # parser.add_argument('--fps', type=int, default=30, help='Frames per second for output videos')
    # parser.add_argument('--no-recursive', action='store_true', help='Do not search directories recursively')
    # parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    # args = parser.parse_args()
    #
    # # Set logging level based on verbosity
    # if args.verbose:
    #     logger.setLevel(logging.DEBUG)
    # #
    # process_directory(
    #     args.directory, 
    #     output_dir=args.output_dir, 
    #     fps=args.fps, 
    #     recursive=not args.no_recursive
    # )
    # hdf_path = "/home/lr-2002/project/reasoning_manipulation/gello_software/teleoperation_dataset/Tabletop-Clean-For-Dinner-v1/trajectory_20250414_145217.h5"
    # hdf_path = "/home/lr-2002/project/reasoning_manipulation/gello_software/teleoperation_dataset/Tabletop-Clean-For-Dinner-v1/trajectory_20250414_145217.h5"
    hdf_path = "/home/lr-2002/project/reasoning_manipulation/gello_software/teleoperation_dataset/Tabletop-Find-Book-From-Shelf-v1/trajectory_20250414_160747.rgbd.pd_ee_delta_pose.physx_cpu.h5"
    convert_h5_images_to_mp4(hdf_path,output_dir='./')
    # explore_hdf5_structure(hdf_path)

if __name__ == '__main__':
    main()
