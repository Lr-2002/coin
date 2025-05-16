#!/usr/bin/env python
"""
USD to GLB Converter

This script loads a USD file and converts it to GLB format using a direct approach with Blender.
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('usd_to_glb_converter')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert USD files to GLB format using Blender')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input USD file')
    parser.add_argument('--output', '-o', type=str, help='Path to save the output GLB file (default: same as input with .glb extension)')
    parser.add_argument('--blender-path', type=str, help='Path to Blender executable (optional, will try to find automatically)')
    return parser.parse_args()

def find_blender():
    """Find the Blender executable on the system."""
    # Common locations for Blender
    common_locations = [
        '/Applications/Blender.app/Contents/MacOS/Blender',  # macOS
        'blender',  # If in PATH
        'C:\\Program Files\\Blender Foundation\\Blender\\blender.exe',  # Windows
        '/usr/bin/blender'  # Linux
    ]
    
    for location in common_locations:
        try:
            result = subprocess.run([location, '--version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   check=False)
            if result.returncode == 0:
                logger.info(f"Found Blender at: {location}")
                return location
        except FileNotFoundError:
            continue
    
    logger.error("Could not find Blender. Please specify the path using --blender-path")
    return None

def create_blender_script(input_path, output_path):
    """Create a Python script for Blender to convert USD to GLB."""
    script = f"""
import bpy
import os
import sys
import math
import bmesh
from mathutils import Vector, Matrix

# Clear default scene
bpy.ops.wm.read_homefile(use_empty=True)

# Import USD
bpy.ops.wm.usd_import(filepath="{str(input_path)}")

# Process all mesh objects
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"Found {{len(mesh_objects)}} mesh objects")

# Process each object
for obj in mesh_objects:
    print(f"Processing object: {{obj.name}}")
    
    # Get original dimensions and scale
    original_dimensions = obj.dimensions.copy()
    original_scale = obj.scale.copy()
    print(f"Original dimensions: {{original_dimensions}}, Original scale: {{original_scale}}")
    
    # Apply a scale factor to normalize the object
    # If the original scale is around 100, we'll use 0.01 to get to 1.0
    if original_scale.x > 10:  # Assuming scale is uniform
        scale_factor = 0.01
    else:
        scale_factor = 1.0
    
    obj.scale = Vector((scale_factor, scale_factor, scale_factor))
    
    # Center the object at origin
    obj.location = Vector((0.0, 0.0, 0.0))
    
    # Apply all transformations
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    print(f"Final dimensions: {{obj.dimensions}}, Final scale: {{obj.scale}}")
    
    # Deselect the object
    obj.select_set(False)

# Make sure output directory exists
output_dir = os.path.dirname(os.path.abspath("{str(output_path)}"))
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Export as GLB
bpy.ops.export_scene.gltf(
    filepath="{str(output_path)}",
    export_format='GLB'
)

print(f"Exported GLB file to: {str(output_path)}")

# Exit Blender
bpy.ops.wm.quit_blender()
"""
    
    # Create a temporary file for the script
    script_file = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
    script_file.write(script.encode('utf-8'))
    script_file.close()
    
    return script_file.name

def convert_with_blender(blender_path, input_path, output_path):
    """Convert USD to GLB using Blender."""
    logger.info(f"Converting {input_path} to {output_path} using Blender")
    
    # Create the Blender script
    script_path = create_blender_script(input_path, output_path)
    logger.info(f"Created Blender script at: {script_path}")
    
    try:
        # Run Blender with the script
        cmd = [blender_path, '--background', '--python', script_path]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stream the output
        for line in process.stdout:
            logger.info(f"Blender: {line.strip()}")
        
        # Wait for the process to complete
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Blender conversion failed with code {process.returncode}: {stderr}")
            raise Exception(f"Blender conversion failed: {stderr}")
        
        logger.info("Blender conversion completed successfully")
        
        # Check if the output file was created
        if not os.path.exists(output_path):
            logger.error(f"Output file was not created: {output_path}")
            raise Exception(f"Output file was not created: {output_path}")
        
        return output_path
        
    finally:
        # Clean up the script file
        if os.path.exists(script_path):
            os.unlink(script_path)

def convert_with_usdcat(input_path, output_path):
    """Try to convert USD to GLB using usdcat if available."""
    try:
        logger.info(f"Attempting conversion with usdcat: {input_path} to {output_path}")
        
        # Check if usdcat is available
        subprocess.run(['which', 'usdcat'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Run usdcat to convert USD to GLB
        cmd = ['usdcat', str(input_path), '--out', str(output_path)]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if os.path.exists(output_path):
            logger.info(f"Successfully converted with usdcat: {output_path}")
            return True
        else:
            logger.warning("usdcat did not produce an output file")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"usdcat conversion failed: {str(e)}")
        return False

def main():
    """Main function to run the conversion process."""
    args = parse_arguments()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)
    
    if not input_path.is_file():
        logger.error(f"Input path is not a file: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.glb')
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # First try with usdcat if available
    if convert_with_usdcat(input_path, output_path):
        logger.info(f"Conversion completed successfully with usdcat. Output file: {output_path}")
        return
    
    # If usdcat fails, try with Blender
    try:
        # Find Blender executable
        blender_path = args.blender_path
        if not blender_path:
            blender_path = find_blender()
            
        if not blender_path:
            logger.error("Blender not found. Please install Blender or specify the path with --blender-path")
            sys.exit(1)
        
        # Convert the file
        convert_with_blender(blender_path, input_path, output_path)
        
        logger.info(f"Conversion completed successfully. Output file: {output_path}")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
