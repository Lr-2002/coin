#!/usr/bin/env python3
"""
Script to generate JSON configuration files for all GLB files in the assets_glb directory.
Each configuration file will include the path to the GLB file, scale, mass, and friction.
"""

import os
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate object configuration files")
    parser.add_argument("--assets-dir", type=str, default="assets_glb", help="Directory containing GLB files")
    parser.add_argument("--output-dir", type=str, default="configs", help="Directory to save JSON files")
    parser.add_argument("--scale", type=float, default=0.01, help="Default scale for all objects")
    parser.add_argument("--mass", type=float, default=0.5, help="Default mass for all objects")
    parser.add_argument("--friction", type=float, default=1.0, help="Default friction for all objects")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files")
    return parser.parse_args()

def generate_config_for_file(glb_file, assets_dir, output_dir, scale, mass, friction, overwrite=False):
    """Generate a JSON configuration file for a GLB file."""
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(glb_file))[0]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output JSON file path
    json_file = os.path.join(output_dir, f"{filename}.json")
    
    # Check if the JSON file already exists
    if os.path.exists(json_file) and not overwrite:
        print(f"Skipping {filename} (JSON file already exists)")
        return False
    
    # Create the configuration dictionary
    config = {
        "usd-path": os.path.join(assets_dir, os.path.basename(glb_file)),
        "scale": scale,
        "mass": mass,
        "friction": friction
    }
    
    # Write the configuration to the JSON file
    with open(json_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated configuration for {filename}")
    return True

def main():
    args = parse_args()
    
    # Get the absolute path to the assets directory
    script_dir = Path(__file__).parent
    assets_dir = os.path.join(script_dir, args.assets_dir)
    output_dir = os.path.join(script_dir, args.output_dir)
    
    # Check if the assets directory exists
    if not os.path.exists(assets_dir):
        print(f"Error: Assets directory {assets_dir} not found")
        return
    
    # Get all GLB files in the assets directory
    glb_files = [f for f in os.listdir(assets_dir) if f.lower().endswith('.glb')]
    
    if not glb_files:
        print(f"No GLB files found in {assets_dir}")
        return
    
    # Generate configuration files for each GLB file
    generated_count = 0
    for glb_file in glb_files:
        if generate_config_for_file(
            glb_file, 
            args.assets_dir, 
            output_dir, 
            args.scale, 
            args.mass, 
            args.friction, 
            args.overwrite
        ):
            generated_count += 1
    
    print(f"\nGenerated {generated_count} configuration files in {output_dir}")
    print(f"Total GLB files processed: {len(glb_files)}")
    
    # List all generated JSON files
    json_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.json')]
    print(f"Total JSON files in {output_dir}: {len(json_files)}")
    
    # Print a sample of the generated JSON files
    if json_files:
        print("\nSample of generated JSON files:")
        for json_file in json_files[:5]:  # Show first 5 files
            print(f"- {json_file}")
        if len(json_files) > 5:
            print(f"... and {len(json_files) - 5} more")

if __name__ == "__main__":
    main()
