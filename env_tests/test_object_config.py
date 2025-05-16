#!/usr/bin/env python3
"""
Test script for object configuration loading in ManiSkill.
This script tests the configuration loading without initializing the full environment.
"""

import json
import os
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Test object configuration loading")
    parser.add_argument("--config", type=str, required=True, help="Path to object JSON config file")
    return parser.parse_args()

def load_object_config(config_path):
    """Simulate loading object configuration as done in PickPlaceTaskEnv"""
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found")
        return None
        
    # Default values
    object_path = None
    object_scale = 0.01
    object_mass = 0.5
    object_friction = 1.0
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded object configuration from {config_path}")
        print(f"Configuration contents: {config}")
        
        # Update parameters from config
        if "usd-path" in config:
            object_path = config["usd-path"]
            print(f"Using object path from config: {object_path}")
            
            # Check if the object file exists
            if object_path:
                # Try both absolute path and relative to ManiSkill root
                if os.path.exists(object_path):
                    print(f"Object file exists at absolute path: {object_path}")
                else:
                    # Try relative to ManiSkill root
                    mani_skill_root = Path(__file__).parent
                    rel_path = os.path.join(mani_skill_root, object_path)
                    if os.path.exists(rel_path):
                        print(f"Object file exists at relative path: {rel_path}")
                    else:
                        print(f"Warning: Object file not found at {object_path} or {rel_path}")
            
        if "scale" in config:
            object_scale = config["scale"]
            print(f"Using object scale from config: {object_scale}")
        if "mass" in config:
            object_mass = config["mass"]
            print(f"Using object mass from config: {object_mass}")
        if "friction" in config:
            object_friction = config["friction"]
            print(f"Using object friction from config: {object_friction}")
            
        # Return the configured values
        return {
            "object_path": object_path,
            "object_scale": object_scale,
            "object_mass": object_mass,
            "object_friction": object_friction
        }
            
    except json.JSONDecodeError:
        print(f"Error: Config file {config_path} is not valid JSON")
    except Exception as e:
        print(f"Error loading config file: {e}")
    
    return None

def main():
    args = parse_args()
    config = load_object_config(args.config)
    
    if config:
        print("\nSummary of object configuration:")
        print(f"Object path: {config['object_path']}")
        print(f"Object scale: {config['object_scale']}")
        print(f"Object mass: {config['object_mass']} kg")
        print(f"Object friction: {config['object_friction']}")
        
        # Check if assets directory exists
        assets_dir = os.path.join(Path(__file__).parent, "assets_glb")
        if os.path.exists(assets_dir):
            print(f"\nAssets directory exists: {assets_dir}")
            # List files in assets directory
            files = os.listdir(assets_dir)
            print(f"Files in assets directory: {files}")
        else:
            print(f"\nAssets directory not found: {assets_dir}")
            # Create the directory
            print("Creating assets directory...")
            os.makedirs(assets_dir, exist_ok=True)
            print(f"Created assets directory: {assets_dir}")

if __name__ == "__main__":
    main()
