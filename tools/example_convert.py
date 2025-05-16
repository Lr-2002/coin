#!/usr/bin/env python
"""
Example script demonstrating how to use the USD to GLB converter.
"""

import os
import argparse
from pathlib import Path
from usd_to_glb_converter import find_blender, convert_with_blender, convert_with_usdcat

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Example of using the USD to GLB converter')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input USD file')
    parser.add_argument('--output', '-o', type=str, help='Path to save the output GLB file (default: same as input with .glb extension)')
    parser.add_argument('--blender-path', type=str, help='Path to Blender executable (optional, will try to find automatically)')
    return parser.parse_args()

def main():
    """Run the example conversion."""
    args = parse_arguments()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.glb')
    
    print(f"Converting {input_path} to {output_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Try with usdcat first
        print("Step 1: Attempting conversion with usdcat...")
        if convert_with_usdcat(input_path, output_path):
            print(f"Conversion completed successfully with usdcat. Output file: {output_path}")
            return
        
        print("usdcat conversion failed or not available, trying with Blender...")
        
        # Step 2: Find Blender
        print("Step 2: Finding Blender...")
        blender_path = args.blender_path
        if not blender_path:
            blender_path = find_blender()
            
        if not blender_path:
            print("Error: Blender not found. Please install Blender or specify the path with --blender-path")
            return
        
        print(f"Found Blender at: {blender_path}")
        
        # Step 3: Convert with Blender
        print("Step 3: Converting with Blender...")
        convert_with_blender(blender_path, input_path, output_path)
        
        print(f"Conversion completed successfully. Output file: {output_path}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
