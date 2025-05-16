#!/usr/bin/env python3
import json
import os
import sys

def load_config(config_path):
    """Load and print configuration from a JSON file"""
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found")
        return None
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Successfully loaded configuration from {config_path}")
        print(f"Configuration contents: {config}")
        
        # Print individual values
        if "usd-path" in config:
            print(f"usd-path: {config['usd-path']}")
        if "scale" in config:
            print(f"scale: {config['scale']}")
        if "mass" in config:
            print(f"mass: {config['mass']}")
        if "friction" in config:
            print(f"friction: {config['friction']}")
            
        return config
    except json.JSONDecodeError:
        print(f"Error: Config file {config_path} is not valid JSON")
    except Exception as e:
        print(f"Error loading config file: {e}")
    
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_config_loading.py <config_file_path>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    load_config(config_path)
