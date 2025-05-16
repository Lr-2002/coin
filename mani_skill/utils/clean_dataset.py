#!/usr/bin/env python

import os
import json
import glob
from pathlib import Path


def clean_datasets(datasets_dir):
    """
    Clean datasets by checking JSON files and removing corresponding H5 files if they don't meet criteria.
    
    Criteria:
    - JSON file must exist
    - elapsed_steps must be >= 15
    - If these criteria are not met, the corresponding H5 file is deleted
    
    Args:
        datasets_dir (str): Path to the datasets directory
    """
    print(f"Cleaning datasets in {datasets_dir}")
    
    # Get all dataset directories
    dataset_dirs = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    
    total_h5_files = 0
    deleted_h5_files = 0
    
    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(datasets_dir, dataset_dir)
        print(f"Processing dataset: {dataset_dir}")
        
        # Find all subdirectories in the dataset directory
        for root, dirs, files in os.walk(dataset_path):
            # Get all H5 files in the current directory
            h5_files = [f for f in files if f.endswith('.h5')]
            total_h5_files += len(h5_files)
            
            for h5_file in h5_files:
                base_name = h5_file[:-3]  # Remove .h5 extension
                json_file = os.path.join(root, base_name + '.json')
                
                # Check if corresponding JSON file exists
                if not os.path.exists(json_file):
                    print(f"  Deleting {h5_file} - No corresponding JSON file")
                    os.remove(os.path.join(root, h5_file))
                    deleted_h5_files += 1
                    continue
                
                # Read JSON file and check criteria
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if the JSON has episodes and elapsed_steps >= 15
                    valid = False
                    if 'episodes' in data and len(data['episodes']) > 0:
                        for episode in data['episodes']:
                            if 'elapsed_steps' in episode and episode['elapsed_steps'] >= 15:
                                valid = True
                                break
                    
                    if not valid:
                        print(f"  Deleting {h5_file} - Failed criteria (elapsed_steps < 15)")
                        os.remove(os.path.join(root, h5_file))
                        deleted_h5_files += 1
                
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  Error processing {json_file}: {e}")
                    print(f"  Deleting {h5_file} - JSON parsing error")
                    os.remove(os.path.join(root, h5_file))
                    deleted_h5_files += 1
    
    print(f"\nCleaning completed:")
    print(f"Total H5 files: {total_h5_files}")
    print(f"Deleted H5 files: {deleted_h5_files}")
    print(f"Remaining H5 files: {total_h5_files - deleted_h5_files}")


if __name__ == "__main__":
    # Get the absolute path to the datasets directory
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datasets_dir = os.path.join(current_dir, "datasets")
    
    clean_datasets(datasets_dir)