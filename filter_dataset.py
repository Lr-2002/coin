#!/usr/bin/env python3

import os
import json
import glob
import shutil
from pathlib import Path
import argparse
import logging
from huggingface_hub import HfApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def filter_dataset(source_dir: str, target_dir: str):
    """
    Filter dataset by copying only JSON files and their corresponding h5 files
    to a target directory, maintaining the same directory structure.
    
    Args:
        source_dir (str): Path to the source dataset directory
        target_dir (str): Path to the target directory where filtered dataset will be stored
    """
    logger.info(f"Filtering dataset from {source_dir} to {target_dir}")
    
    breakpoint()
    # Get all task directories in the source directory
    task_dirs = [d for d in glob.glob(os.path.join(source_dir, "*")) if os.path.isdir(d)]
    
    total_json_files = 0
    total_copied_pairs = 0
    
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        logger.info(f"Processing task: {task_name}")
        
        # Create corresponding task directory in the target directory
        target_task_dir = os.path.join(target_dir, task_name)
        os.makedirs(target_task_dir, exist_ok=True)
        
        # Find all JSON files in the task directory
        json_files = glob.glob(os.path.join(task_dir, "*.json"))
        total_json_files += len(json_files)
        
        # Process each JSON file
        for json_file in json_files:
            json_basename = os.path.basename(json_file)
            base_name = os.path.splitext(json_basename)[0]
            
            # Path to corresponding h5 file
            h5_file = os.path.join(task_dir, f"{base_name}.h5")
            
            # Check if corresponding h5 file exists
            if os.path.exists(h5_file):
                # Copy JSON file
                target_json_file = os.path.join(target_task_dir, json_basename)
                shutil.copy2(json_file, target_json_file)
                
                # Copy h5 file
                target_h5_file = os.path.join(target_task_dir, f"{base_name}.h5")
                shutil.copy2(h5_file, target_h5_file)
                
                logger.debug(f"Copied pair: {json_basename} and {base_name}.h5")
                total_copied_pairs += 1
            else:
                logger.warning(f"No corresponding h5 file found for {json_file}")
    
    logger.info(f"\nFiltering completed:")
    logger.info(f"Total JSON files found: {total_json_files}")
    logger.info(f"Total pairs (JSON + h5) copied: {total_copied_pairs}")

def upload_to_huggingface(folder_path, repo_id, repo_type="dataset"):
    """
    Upload a folder to Hugging Face Hub.
    
    Args:
        folder_path (str): Path to the local folder to upload
        repo_id (str): The repository ID to upload to (username/repo_name)
        repo_type (str): The repository type (default: "dataset")
    """
    logger.info(f"Uploading {folder_path} to Hugging Face Hub repository {repo_id}")
    
    try:
        api = HfApi()
        api.upload_large_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type=repo_type,
            print_report=True,
            private=True
        )
        logger.info(f"Upload to {repo_id} completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error uploading to Hugging Face Hub: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Filter dataset by copying JSON files and their corresponding h5 files')
    parser.add_argument('--source_dir', type=str, default='coin_dataset',
                        help='Path to the source dataset directory')
    parser.add_argument('--target_dir', type=str, default='cleaned_coin_dataset_put_fork',
                        help='Path to the target directory where filtered dataset will be stored')
    parser.add_argument('--upload', action='store_true',
                        help='Upload the filtered dataset to Hugging Face Hub')
    parser.add_argument('--repo_id', type=str, default="coin-dataset/coin_filtered",
                        help='Hugging Face Hub repository ID (username/repo_name) for upload')
    parser.add_argument('--repo_type', type=str, default="dataset",
                        help='Repository type (dataset, model, space)')
    parser.add_argument('--filter', action='store_true',
                        help='Filter the dataset')
    args = parser.parse_args()
    
    # Create target directory if it doesn't exist
    os.makedirs(args.target_dir, exist_ok=True)
    
    # Filter dataset
    if args.filter: 
        filter_dataset(args.source_dir, args.target_dir)
    
    # Upload to Hugging Face Hub if requested
    if args.upload:
        if args.repo_id is None:
            logger.error("Repository ID (--repo_id) must be provided for upload")
            return
        
        upload_to_huggingface(args.target_dir, args.repo_id, args.repo_type)

if __name__ == "__main__":
    main()
