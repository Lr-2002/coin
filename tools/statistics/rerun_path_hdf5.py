#!/usr/bin/env python
import rerun as rr
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import os
import sys
import json
import re
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Warm color palette
warm_palette_hex = [
    "#2E8B57", "#3CB371", "#43CD80", "#00FF7F", "#00FA9A",
    "#66CDAA", "#7FFFD4", "#00C957", "#32CD32", "#228B22",
    "#008000", "#006400", "#308014", "#7CFC00", "#00FF00",
    "#4CBB17", "#00A550", "#4F7942", "#29AB87", "#00A877"
]


# Convert hex colors to RGB tuples (0-255 range)me
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Visualize end-effector trajectories from HDF5 files')
parser.add_argument('--num-files', type=int, default=20000, help='Number of HDF5 files to process')
parser.add_argument('--data-dir', type=str, 
                    default='/ssd/gello_software/gello_pd_dataset/Tabletop-Close-Drawer-v1',
                    help='Directory containing HDF5 files')
parser.add_argument('--position-only', action='store_true', help='Visualize only positions without rotations')
parser.add_argument('--no-frames', action='store_true', help='Do not show any orientation frames')
parser.add_argument('--filter-below-zero', action='store_true', help='Filter out points with z < 0')
parser.add_argument('--color-by-task', action='store_true', help='Color trajectories by task type')
parser.add_argument('--save-to', type=str, default='', help='Save the visualization to a file (e.g., "visualization.rrd")')
parser.add_argument('--use-warm-palette', action='store_true', help='Use warm color palette with fading transparency')
parser.add_argument('--save-matplotlib', type=str, default='./action_space_coin.png', help='Save a static matplotlib visualization to a PNG file')
args = parser.parse_args()

# Initialize rerun with the application name
rr.init("panda_eef_all_trajs", spawn=True)

# Connect to the viewer or save to a file
if args.save_to:
    # Save to a file
    rr.save(args.save_to)
    print(f"Recording to file: {args.save_to}")
else:
    # Connect to the viewer
    rr.connect()
    print("Connecting to Rerun viewer")

# Add a grid at z=0 for reference
rr.log("floor/grid", rr.Points3D(
    positions=[[x, y, 0] for x in np.linspace(-2, 2, 20) for y in np.linspace(-2, 2, 20)],
    colors=[[200, 200, 200, 100] for _ in range(400)],
    radii=0.005
))

# Coordinate system at origin removed as requested

def get_color(i, total):
    """Get a color from the matplotlib colormap (range 0-255)"""
    cmap = plt.cm.get_cmap('tab10')
    rgba = cmap(i % 10)
    return [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)]

def rpy_to_quat(rpy):
    """Convert roll-pitch-yaw angles to quaternion"""
    # Create a rotation object from Euler angles
    rot = R.from_euler('xyz', rpy)
    # Convert to quaternion
    quat = rot.as_quat()  # Returns x, y, z, w
    # Return as [x, y, z, w] for rerun
    return [quat[0], quat[1], quat[2], quat[3]]

def load_trajectory_data(file_path):
    """Load trajectory data from an HDF5 file"""
    # Read the HDF5 file
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if the file has the expected structure using string paths
            if 'traj_0' in f and 'traj_0/obs/extra/tcp_pose' in f:
                # Extract TCP pose (end-effector pose) as numpy array
                tcp_poses = np.array(f['traj_0/obs/extra/tcp_pose'])
                
                if tcp_poses.size == 0:
                    print(f"Warning: File {file_path} has empty tcp_pose data")
                    return np.array([]), np.array([])
                
                # TCP pose is typically [x, y, z, qw, qx, qy, qz]
                positions = tcp_poses[:, :3]  # First 3 elements are xyz
                
                # Convert quaternion from [qw, qx, qy, qz] to [qx, qy, qz, qw] for scipy
                quats = np.zeros((tcp_poses.shape[0], 4))
                quats[:, 0] = tcp_poses[:, 4]  # qx
                quats[:, 1] = tcp_poses[:, 5]  # qy
                quats[:, 2] = tcp_poses[:, 6]  # qz
                quats[:, 3] = tcp_poses[:, 3]  # qw
                
                return positions, quats
            else:
                print(f"Warning: File {file_path} doesn't have the expected structure")
                return np.array([]), np.array([])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([]), np.array([])

# Store trajectory data for matplotlib visualization
all_trajectories = {}

def visualize_trajectory(idx, pos, quat, color, task_name=None, file_name=None, position_only=False):
    """Visualize a trajectory using Rerun and store data for matplotlib visualization"""
    # Create a label for the trajectory
    label = f"Trajectory {idx}"
    if task_name and file_name:
        label = f"{task_name} - {file_name}"
    elif task_name:
        label = f"{task_name} - {idx}"
    elif file_name:
        label = f"{file_name} - {idx}"
    
    # Store trajectory data for matplotlib visualization
    all_trajectories[idx] = {
        'positions': pos,
        'label': label,
        'task_name': task_name
    }
    
    rr.log(f"trajectory/{idx}/label", rr.TextLog(label))
    
    # Log start point with F0F1C5 color
    start_color = hex_to_rgb('#F0F1C5')
    rr.log(
        f"trajectory/{idx}/points/start",
        rr.Points3D(positions=[pos[0]], colors=[[start_color[0], start_color[1], start_color[2], 255]], radii=0.01)
    )
    
    # Log end point with BBD8A3 color
    end_color = hex_to_rgb('#BBD8A3')
    rr.log(
        f"trajectory/{idx}/points/end",
        rr.Points3D(positions=[pos[-1]], colors=[[end_color[0], end_color[1], end_color[2], 255]], radii=0.01)
    )
    
    # Get color from warm palette
    palette_idx = idx % len(warm_palette_hex)
    base_color_rgb = hex_to_rgb(warm_palette_hex[palette_idx])
    
    # Log all points along the trajectory with fading transparency
    points = []
    colors = []
    radii = []
    
    for i, p in enumerate(pos):
        points.append(p)
        # Calculate alpha that decreases from start to end (255 to 50)
        alpha = int(255 - (i / len(pos)) * 205)
        colors.append([base_color_rgb[0], base_color_rgb[1], base_color_rgb[2], alpha])
        radii.append(0.01)
    
    # Add points with the fading transparency
    rr.log(
        f"trajectory/{idx}/points/all",
        rr.Points3D(positions=points, colors=colors, radii=radii)
    )

# Find all HDF5 files in the directory and its subdirectories
hdf5_files = []
for root, dirs, files in os.walk(args.data_dir):
    for file in files:
        if file.endswith('.h5'):
            hdf5_files.append(os.path.join(root, file))
hdf5_files = sorted(hdf5_files)
print(f"Found {len(hdf5_files)} HDF5 files, processing {min(args.num_files, len(hdf5_files))}")

# Print task distribution
task_count = {}
for file_path in hdf5_files:
    task_name = os.path.basename(os.path.dirname(file_path))
    task_count[task_name] = task_count.get(task_name, 0) + 1
print("\nTask distribution:")
for task, count in sorted(task_count.items()):
    print(f"  {task}: {count} files")

# Limit the number of files to process
max_files = min(args.num_files, len(hdf5_files))
hdf5_files = hdf5_files[:max_files]

# Extract task names from file paths
task_names = set()
for file_path in hdf5_files:
    # Extract task name from file path (assuming directory name is the task name)
    task_name = os.path.basename(os.path.dirname(file_path))
    task_names.add(task_name)

# Create a colormap for tasks
task_colors = {}

# Extract unique task names from directory paths
task_names = set()
for file_path in hdf5_files:
    dir_name = os.path.basename(os.path.dirname(file_path))
    task_names.add(dir_name)

print(f"Found {len(task_names)} unique task types")

# Create color mappings for tasks
if args.use_warm_palette:
    # Use the warm palette for tasks
    for i, task_name in enumerate(sorted(task_names)):
        palette_idx = i % len(warm_palette_hex)
        rgb = hex_to_rgb(warm_palette_hex[palette_idx])
        task_colors[task_name] = [rgb[0], rgb[1], rgb[2]]
    print("Using warm color palette with fading transparency")
else:
    # Use the default tab20 colormap
    cmap = plt.cm.get_cmap('tab20')
    for i, task_name in enumerate(sorted(task_names)):
        # Convert matplotlib color (0-1 range) to rerun color (0-255 range)
        rgba = cmap(i % cmap.N)
        task_colors[task_name] = [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)]
    print("Using tab20 color palette")

# Try to load task mappings from tasks.jsonl if it exists (for additional metadata)
tasks_jsonl_path = os.path.join(args.data_dir, '../meta/tasks.jsonl')
if os.path.exists(tasks_jsonl_path):
    print(f"Loading additional task metadata from {tasks_jsonl_path}")
    with open(tasks_jsonl_path, 'r') as f:
        for line in f:
            try:
                task_data = json.loads(line)
                task_name = task_data.get('name', '')
                # We already have colors for tasks, this is just for additional metadata
            except json.JSONDecodeError:
                continue
else:
    print(f"Note: No tasks.jsonl found at {tasks_jsonl_path} (this is optional)")

# Process and visualize each trajectory
trajectories_loaded = 0
for i, file_path in enumerate(hdf5_files):
    # Extract task name and file name
    task_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    
    # Get color for this task
    color = task_colors[task_name] if args.color_by_task else [100, 100, 255]  # Default blue if not coloring by task
    
    # Load trajectory data
    positions, quats = load_trajectory_data(file_path)
    
    # Visualize if we have valid data
    if isinstance(positions, np.ndarray) and positions.size > 0:
        # Filter points if requested
        if args.filter_below_zero:
            # Filter points with z < 0
            valid_indices = [j for j, p in enumerate(positions) if p[2] >= 0]
            if not valid_indices:
                continue  # Skip if no valid points
            
            filtered_positions = positions[valid_indices]
            filtered_quats = quats[valid_indices] if len(quats) > 0 else np.array([])
        else:
            filtered_positions = positions
            filtered_quats = quats
        
        if len(filtered_positions) == 0:
            continue  # Skip if no valid points after filtering
        
        # Visualize the trajectory
        visualize_trajectory(i, filtered_positions, filtered_quats, color, task_name, file_name, args.position_only)
        
        trajectories_loaded += 1
        print(f"Loaded trajectory from {file_name} with {len(positions)} points")
    else:
        print(f"Skipping {file_path} - no trajectory data found")

if args.save_to:
    print(f"\nVisualization saved to file: {args.save_to}")
    print(f"To view the recording, run: rerun {args.save_to}")
else:
    print("\nVisualization ready! The rerun viewer should be open in a new window.")

print(f"Loaded {trajectories_loaded} trajectories from {len(hdf5_files)} HDF5 files")
print(f"Data directory: {args.data_dir}")
# Colors are now defined below

# Coordinate frames have been removed
print("Coordinate frames removed from visualization")
print("Light yellow points (#F0F1C5): Start positions")
print("Light green points (#BBD8A3): End positions")

if args.color_by_task:
    print(f"Coloring: Trajectories colored by task type ({len(task_names)} unique tasks)")
    for i, (task_name, color) in enumerate(sorted(task_colors.items())):
        print(f"  {task_name}: RGB({color[0]}, {color[1]}, {color[2]})")

def save_matplotlib_visualization(trajectories, task_colors, output_file='trajectory_visualization.png'):
    """Save the 3D visualization as a static image using matplotlib"""
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End-Effector Trajectories')
    
    # Add a grid at z=0 for reference
    x = np.linspace(-0.5, 0.5, 20)
    y = np.linspace(-0.5, 0.5, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    try:
        ax.plot_surface(X, Y, Z, alpha=0.2, color='gray')
    except AttributeError:
        print("Warning: Could not plot surface, falling back to scatter")
        ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), color='gray', alpha=0.2, s=0.2)
    
    # Coordinate system at origin removed as requested
    
    # Plot each trajectory
    for traj_id, traj_data in trajectories.items():
        positions = traj_data['positions']
        task_name = traj_data['task_name']
        
        # Get color for this task
        if task_name in task_colors:
            rgb_color = task_colors[task_name]
            # Convert to 0-1 range for matplotlib as a tuple
            color = (rgb_color[0]/255.0, rgb_color[1]/255.0, rgb_color[2]/255.0)
        else:
            color = (0.4, 0.4, 1.0)  # Default blue
        
        # Extract x, y, z coordinates
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]
        z = [p[2] for p in positions]
        
        # Plot the trajectory line
        ax.plot(x, y, z, '-', color=color, alpha=0.7, linewidth=1.5)
        
        # Plot start point with F0F1C5 color
        ax.scatter(x[0], y[0], z[0], color='#F0F1C5', marker='o', s=4)
        
        # Plot end point with BBD8A3 color
        ax.scatter(x[-1], y[-1], z[-1], color='#BBD8A3', marker='o', s=4)
    
    # Set equal aspect ratio
    try:
        # Use a tuple instead of a list for aspect ratio
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        print("Warning: Could not set box aspect")
    
    # Add a legend for task types
    from matplotlib.lines import Line2D
    legend_elements = []
    for task_name, rgb_color in sorted(task_colors.items()):
        color_tuple = (rgb_color[0]/255.0, rgb_color[1]/255.0, rgb_color[2]/255.0)
        legend_elements.append(Line2D([0], [0], color=color_tuple, lw=2, label=task_name))
    
    # Add legend for start and end points
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#F0F1C5', markersize=10, label='Start Point'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#BBD8A3', markersize=10, label='End Point'))
    
    # Add the legend
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Set the view angle
    try:
        ax.view_init(elev=30, azim=45)
    except AttributeError:
        print("Warning: Could not set view angle")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Matplotlib visualization saved to: {output_file}")

# Save matplotlib visualization if requested
if args.save_matplotlib:
    print(f"\nGenerating matplotlib visualization to {args.save_matplotlib}...")
    save_matplotlib_visualization(all_trajectories, task_colors, args.save_matplotlib)
