#!/usr/bin/env python3

import os
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
import trimesh
import logging

def move_object_to_origin(input_file, output_file=None, verbose=False):
    """
    Load a GLB file, move all objects to origin (0,0,0), and save the result.
    
    Args:
        input_file (str): Path to input GLB file
        output_file (str, optional): Path to output GLB file. If None, overwrites the input file.
        verbose (bool): Whether to print detailed information
    
    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        print(f"Processing {input_file}")
    
    try:
        # Load the mesh or scene
        loaded = trimesh.load(input_file)
        
        # Handle both single meshes and scenes
        if isinstance(loaded, trimesh.Scene):
            # It's a scene with multiple objects
            scene = loaded
            
            if verbose:
                print(f"Loaded scene with {len(scene.geometry)} geometries")
            
            # Try using scene graph approach first
            try:
                # Get the centroid of all objects combined
                all_vertices = []
                for name, geom in scene.geometry.items():
                    if hasattr(geom, 'vertices') and geom.vertices is not None:
                        # Transform vertices to world coordinates
                        try:
                            transform = scene.graph.get(name)[0]
                            vertices = trimesh.transformations.transform_points(geom.vertices, transform)
                            all_vertices.append(vertices)
                        except Exception as e:
                            if verbose:
                                print(f"Warning: Could not get transform for {name}: {e}")
                            continue
                
                if all_vertices:
                    all_vertices = np.vstack(all_vertices)
                    centroid = all_vertices.mean(axis=0)
                    
                    if verbose:
                        print(f"Original scene centroid: {centroid}")
                    
                    # Create translation matrix to move to origin
                    translation = trimesh.transformations.translation_matrix(-centroid)
                    
                    # Apply translation to the root node
                    scene.apply_transform(translation)
                    
                    if verbose:
                        # Verify the new centroid
                        all_vertices = []
                        for name, geom in scene.geometry.items():
                            if hasattr(geom, 'vertices') and geom.vertices is not None:
                                try:
                                    transform = scene.graph.get(name)[0]
                                    vertices = trimesh.transformations.transform_points(geom.vertices, transform)
                                    all_vertices.append(vertices)
                                except Exception:
                                    continue
                        
                        if all_vertices:
                            all_vertices = np.vstack(all_vertices)
                            new_centroid = all_vertices.mean(axis=0)
                            print(f"New scene centroid: {new_centroid}")
                else:
                    raise Exception("No valid vertices found in scene graph")
            
            except Exception as e:
                if verbose:
                    print(f"Scene graph approach failed: {e}. Trying direct geometry approach.")
                
                # Fallback: Use direct geometry approach
                # Get all vertices from all geometries without using the scene graph
                all_vertices = []
                for geom in scene.geometry.values():
                    if hasattr(geom, 'vertices') and geom.vertices is not None and len(geom.vertices) > 0:
                        all_vertices.append(geom.vertices)
                
                if not all_vertices:
                    raise Exception("No valid vertices found in scene geometries")
                
                all_vertices = np.vstack(all_vertices)
                centroid = all_vertices.mean(axis=0)
                
                if verbose:
                    print(f"Original scene centroid (direct approach): {centroid}")
                
                # Create a new scene with the same geometries but centered
                new_scene = trimesh.Scene()
                for name, geom in scene.geometry.items():
                    if hasattr(geom, 'vertices') and geom.vertices is not None:
                        # Create a copy of the geometry
                        geom_copy = geom.copy()
                        # Apply translation to the geometry directly
                        if hasattr(geom_copy, 'apply_translation'):
                            geom_copy.apply_translation(-centroid)
                        # Add to the new scene
                        new_scene.add_geometry(geom_copy, node_name=name)
                
                # Replace the scene
                scene = new_scene
                
                if verbose:
                    # Verify the new centroid
                    all_vertices = []
                    for geom in scene.geometry.values():
                        if hasattr(geom, 'vertices') and geom.vertices is not None:
                            all_vertices.append(geom.vertices)
                    
                    if all_vertices:
                        all_vertices = np.vstack(all_vertices)
                        new_centroid = all_vertices.mean(axis=0)
                        print(f"New scene centroid (direct approach): {new_centroid}")
            
            # Determine output file
            if output_file is None:
                output_file = input_file
            
            # Save the modified scene
            scene.export(output_file)
            
            if verbose:
                print(f"Saved scene to {output_file}")
            
            return True
        
        else:
            # It's a single mesh
            mesh = loaded
            
            if verbose:
                print(f"Loaded single mesh with {len(mesh.vertices)} vertices")
                print(f"Original mesh center: {mesh.vertices.mean(axis=0)}")
                print(f"Original bounds: {mesh.bounds}")
            
            # Calculate translation to move to origin
            centroid = mesh.vertices.mean(axis=0)
            translation = -centroid
            
            # Apply translation to move to origin
            mesh.apply_translation(translation)
            
            if verbose:
                print(f"New mesh center: {mesh.vertices.mean(axis=0)}")
                print(f"New bounds: {mesh.bounds}")
            
            # Determine output file
            if output_file is None:
                output_file = input_file
            
            # Save the modified mesh
            mesh.export(output_file)
            
            if verbose:
                print(f"Saved mesh to {output_file}")
            
            return True
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def process_multiple_files(input_pattern, output_dir=None, verbose=False):
    """
    Process multiple GLB files matching a pattern and move objects to origin.
    
    Args:
        input_pattern (str): Glob pattern to match input files (e.g., "*.glb")
        output_dir (str, optional): Directory to save output files. If None, overwrites input files.
        verbose (bool): Whether to print detailed information
    
    Returns:
        tuple: (success_count, total_count)
    """
    # Find all files matching the pattern
    files = glob.glob(input_pattern)
    
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return 0, 0
    
    success_count = 0
    total_count = len(files)
    
    if verbose:
        print(f"Found {total_count} files to process")
    
    for input_file in files:
        # Determine output file path
        if output_dir is not None:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the filename from the input path
            filename = os.path.basename(input_file)
            
            # Construct the output path
            output_file = os.path.join(output_dir, filename)
        else:
            output_file = None  # Will overwrite input file
        
        # Process the file
        if move_object_to_origin(input_file, output_file, verbose):
            success_count += 1
    
    return success_count, total_count

def main():
    parser = argparse.ArgumentParser(description="Move 3D objects in GLB files to origin (0,0,0)")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Single file command
    single_parser = subparsers.add_parser("single", help="Process a single GLB file")
    single_parser.add_argument("input", help="Input GLB file")
    single_parser.add_argument("-o", "--output", help="Output GLB file (default: overwrite input)")
    
    # Multiple files command
    multi_parser = subparsers.add_parser("multi", help="Process multiple GLB files")
    multi_parser.add_argument("pattern", help="Glob pattern to match input files (e.g., '*.glb')")
    multi_parser.add_argument("-o", "--output-dir", help="Output directory (default: overwrite input files)")
    
    # Common arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    if args.command == "single":
        success = move_object_to_origin(args.input, args.output, args.verbose)
        if success:
            print("Processing completed successfully")
        else:
            print("Processing failed")
            sys.exit(1)
    
    elif args.command == "multi":
        success_count, total_count = process_multiple_files(args.pattern, args.output_dir, args.verbose)
        print(f"Processed {success_count}/{total_count} files successfully")
        if success_count < total_count:
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()