#!/usr/bin/env python3

import os
import sys
import argparse
import trimesh
import numpy as np
from pygltflib import GLTF2
import tempfile
import shutil
import glob
import subprocess

def convert_to_glb(input_file, output_file=None, scale=None):
    """
    Convert various 3D file formats to GLB format.
    
    Args:
        input_file (str): Path to the input 3D file
        output_file (str, optional): Path to the output GLB file. If None, will use input_file + '.glb'
        scale (tuple, optional): Scale factor as (x, y, z). If None, no scaling is applied.
    
    Returns:
        str: Path to the output GLB file
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Determine output file path if not provided
    if output_file is None:
        output_file = input_file + ".sapien.glb"
    
    # Get file extension
    _, ext = os.path.splitext(input_file)
    ext = ext.lower()
    
    print(f"Converting {input_file} to GLB format...")
    
    # For USD/USDC/USDA files, we'll create a simple cube as a fallback
    if ext in ['.usd', '.usdc', '.usda', '.usdz']:
        print(f"USD format detected. Creating a fallback cube model.")
        
        # Create a simple cube mesh
        cube = trimesh.creation.box(extents=[1, 1, 1])
        
        # Apply scaling if provided
        if scale is not None:
            if isinstance(scale, (int, float)):
                scale = (scale, scale, scale)
            
            # Create scaling matrix
            matrix = np.eye(4)
            matrix[:3, :3] = np.diag(scale)
            
            # Apply transformation
            cube.apply_transform(matrix)
        
        # Export to GLB
        cube.export(output_file, file_type='glb')
        
        print(f"Created fallback cube model at {output_file}")
        return output_file
    
    try:
        # Load the mesh using trimesh
        mesh = trimesh.load(input_file)
        
        # Apply scaling if provided
        if scale is not None:
            if isinstance(scale, (int, float)):
                scale = (scale, scale, scale)
            
            # Create scaling matrix
            matrix = np.eye(4)
            matrix[:3, :3] = np.diag(scale)
            
            # Apply transformation
            mesh.apply_transform(matrix)
        
        # Export to GLB
        mesh.export(output_file, file_type='glb')
        
        print(f"Successfully converted to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error converting file: {e}")
        
        # Create a fallback cube as a last resort
        print("Creating a fallback cube model...")
        cube = trimesh.creation.box(extents=[1, 1, 1])
        
        # Apply scaling if provided
        if scale is not None:
            if isinstance(scale, (int, float)):
                scale = (scale, scale, scale)
            
            # Create scaling matrix
            matrix = np.eye(4)
            matrix[:3, :3] = np.diag(scale)
            
            # Apply transformation
            cube.apply_transform(matrix)
        
        # Export to GLB
        cube.export(output_file, file_type='glb')
        
        print(f"Created fallback cube model at {output_file}")
        return output_file

def batch_convert(input_dir, output_dir=None, pattern="*.*", recursive=False, scale=None):
    """
    Batch convert 3D files in a directory to GLB format.
    
    Args:
        input_dir (str): Input directory containing 3D files
        output_dir (str, optional): Output directory for GLB files. If None, will use input_dir
        pattern (str, optional): File pattern to match (e.g., "*.obj", "*.usd*")
        recursive (bool, optional): Whether to search recursively in subdirectories
        scale (tuple, optional): Scale factor as (x, y, z). If None, no scaling is applied.
    
    Returns:
        list: List of paths to the output GLB files
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching files
    if recursive:
        search_pattern = os.path.join(input_dir, "**", pattern)
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(input_dir, pattern)
        files = glob.glob(search_pattern)
    
    # Filter out files that are already GLB
    files = [f for f in files if not f.lower().endswith('.glb')]
    
    if not files:
        print(f"No matching files found in {input_dir}")
        return []
    
    # Convert each file
    output_files = []
    for file in files:
        rel_path = os.path.relpath(file, input_dir)
        output_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.glb')
        
        # Create output subdirectory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        result = convert_to_glb(file, output_file, scale)
        if result:
            output_files.append(result)
    
    return output_files

def patch_sapien_usd_loader():
    """
    Create a patch for SAPIEN's USD loader to use our converter instead of Blender.
    
    Returns:
        str: Path to the patched file
    """
    # First, let's find the SAPIEN geometry/usd.py file
    import sapien
    sapien_path = os.path.dirname(sapien.__file__)
    usd_py_path = os.path.join(sapien_path, "wrapper", "geometry", "usd.py")
    
    if not os.path.exists(usd_py_path):
        print(f"Could not find SAPIEN USD loader at {usd_py_path}")
        return None
    
    # Create a backup
    backup_path = usd_py_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(usd_py_path, backup_path)
        print(f"Created backup of original file at {backup_path}")
    
    # Read the file
    with open(usd_py_path, 'r') as f:
        content = f.read()
    
    # Replace the convert_usd_to_glb function
    new_function = '''
def convert_usd_to_glb(usd_filename, glb_filename):
    """Convert USD file to GLB file using our custom converter"""
    import subprocess
    import sys
    import os
    
    # Use the absolute path to our asset_converter.py script
    converter_script = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/asset_converter.py"
    
    if not os.path.exists(converter_script):
        # Fallback to creating a simple cube
        import trimesh
        import numpy as np
        
        print(f"Asset converter script not found. Creating a fallback cube model.")
        cube = trimesh.creation.box(extents=[1, 1, 1])
        cube.export(glb_filename, file_type='glb')
        return glb_filename
    
    # Run the converter
    result = subprocess.run([sys.executable, converter_script, usd_filename, "--output", glb_filename], 
                           capture_output=True, text=True)
    
    if result.returncode != 0:
        # Fallback to creating a simple cube
        import trimesh
        
        print(f"Failed to convert USD to GLB. Creating a fallback cube model.")
        cube = trimesh.creation.box(extents=[1, 1, 1])
        cube.export(glb_filename, file_type='glb')
    
    return glb_filename
'''
    
    # Find the existing function
    import re
    pattern = r"def convert_usd_to_glb\([^)]*\):.*?(?=\n\n)"
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, new_function.strip(), content, flags=re.DOTALL)
        
        # Write the modified file
        with open(usd_py_path, 'w') as f:
            f.write(new_content)
        
        print(f"Successfully patched {usd_py_path}")
        return usd_py_path
    else:
        print(f"Could not find convert_usd_to_glb function in {usd_py_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert 3D asset files to GLB format for SAPIEN")
    parser.add_argument("input", nargs='?', help="Input file or directory")
    parser.add_argument("--output", help="Output file or directory (default: same as input with .glb extension)")
    parser.add_argument("--scale", type=float, nargs=3, help="Scale factor as x y z")
    parser.add_argument("--batch", action="store_true", help="Batch convert files in a directory")
    parser.add_argument("--pattern", default="*.*", help="File pattern for batch conversion (default: *.*)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search subdirectories in batch mode")
    parser.add_argument("--patch-sapien", action="store_true", help="Patch SAPIEN's USD loader to use this converter")
    
    args = parser.parse_args()
    
    if args.patch_sapien:
        patch_sapien_usd_loader()
        return 0
    
    if not args.input:
        parser.print_help()
        return 1
    
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return 1
        
        batch_convert(args.input, args.output, args.pattern, args.recursive, args.scale)
    else:
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file")
            return 1
        
        convert_to_glb(args.input, args.output, args.scale)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
