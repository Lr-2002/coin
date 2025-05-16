import rerun as rr
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import glob
import argparse
import json
from scipy.spatial.transform import Rotation as R

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Visualize end-effector trajectories from parquet files')
parser.add_argument('--num-files', type=int, default=20000, help='Number of parquet files to process')
parser.add_argument('--data-dir', type=str, 
                    default='/home/lr-2002/project/reasoning_manipulation/ManiSkill/data/gr00t_coin_primitive/data/chunk-000',
                    help='Directory containing parquet files')
parser.add_argument('--position-only', action='store_true', help='Visualize only positions without rotations')
parser.add_argument('--filter-below-zero', action='store_true', help='Filter out points with z < 0')
parser.add_argument('--color-by-task', action='store_true', help='Color trajectories by task type')
args = parser.parse_args()

# Initialize rerun with the application name
rr.init("panda_eef_all_trajs", spawn=True)
rr.connect()

# Add a grid at z=0 for reference
rr.log("floor/grid", rr.Points3D(
    positions=[[x, y, 0] for x in np.linspace(-2, 2, 20) for y in np.linspace(-2, 2, 20)],
    colors=[[200, 200, 200, 100] for _ in range(400)],
    radii=0.005
))

# Add coordinate system at origin
rr.log("world/origin", rr.Arrows3D(
    origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    radii=0.01
))


def get_color(i, total):
    """Get a color from the matplotlib colormap (range 0-255)"""
    cmap = plt.get_cmap("viridis")
    rgb = cmap(i / total)[:3]
    return [int(c * 255) for c in rgb]


def rpy_to_quat(rpy):
    """Convert roll-pitch-yaw angles to quaternion"""
    # Create a rotation object from Euler angles
    rot = R.from_euler('xyz', rpy)
    # Convert to quaternion
    quat = rot.as_quat()  # Returns x, y, z, w
    # Return as [x, y, z, w] for rerun
    return [quat[0], quat[1], quat[2], quat[3]]


def load_trajectory_data(file_path):
    """Load trajectory data from a parquet file"""
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Extract end-effector positions (xyz) and rotations (rpy) from actions
    positions = []
    quats = []
    
    # Initialize with zeros (starting position)
    current_pos = np.zeros(3)
    
    # Initialize with identity rotation
    current_rot = R.identity()
    
    # Store initial position and orientation
    positions.append(current_pos.copy())
    quats.append(current_rot.as_quat())  # [x, y, z, w] format
    
    # Accumulate delta poses
    for i in range(len(df)):
        action = np.array(df['action'].iloc[i])
        
        # Extract delta position (xyz) and delta rotation (rpy)
        delta_pos = action[:3]  # First 3 elements are delta xyz
        delta_rpy = action[3:6]  # Next 3 elements are delta roll, pitch, yaw
        
        # Create rotation object for the delta rotation
        delta_rot = R.from_euler('xyz', delta_rpy)
        
        # Properly compose rotations (current * delta)
        # This is the correct way to accumulate 3D rotations
        current_rot = current_rot * delta_rot
        
        # Accumulate position
        current_pos += delta_pos
        
        # Get the quaternion from the current rotation
        quat = current_rot.as_quat()  # [x, y, z, w] format
        
        positions.append(current_pos.copy())
        quats.append(quat)
    
    return np.array(positions), np.array(quats)


# Load trajectory data from parquet files
data_dir = args.data_dir

# Find all parquet files in the directory
parquet_files = sorted(glob.glob(os.path.join(data_dir, 'episode_*.parquet')))
print(f"Found {len(parquet_files)} parquet files, processing {min(args.num_files, len(parquet_files))}")

# Limit the number of files to process
max_files = min(args.num_files, len(parquet_files))
parquet_files = parquet_files[:max_files]

# Load task mapping if color-by-task is enabled
task_mapping = {}
task_names = []
if args.color_by_task:
    # Determine the path to the tasks.jsonl file
    dataset_root = os.path.dirname(os.path.dirname(data_dir))
    tasks_file = os.path.join(dataset_root, 'meta', 'tasks.jsonl')
    
    if os.path.exists(tasks_file):
        # Load task mapping
        with open(tasks_file, 'r') as f:
            for line in f:
                task_data = json.loads(line.strip())
                task_index = task_data['task_index']
                task_name = task_data['task']
                task_mapping[task_index] = task_name
                if task_name not in task_names:
                    task_names.append(task_name)
        print(f"Loaded {len(task_mapping)} task mappings")
    else:
        print(f"Warning: Task mapping file not found at {tasks_file}")
        args.color_by_task = False

# Store the loaded trajectories
all_positions = []
all_quats = []
file_names = []
task_indices = []

# Load trajectory data from each file
for file_path in parquet_files:
    try:
        # Extract file name for labeling
        file_name = os.path.basename(file_path)
        
        # Load the parquet file to get task index if color-by-task is enabled
        task_index = None
        if args.color_by_task:
            try:
                df = pd.read_parquet(file_path)
                if 'task_index' in df.columns:
                    task_index = df['task_index'].iloc[0]
            except Exception as e:
                print(f"Error reading task index from {file_name}: {e}")
        
        # Store task index for coloring
        task_indices.append(task_index)
        
        # Load trajectory data
        positions, quats = load_trajectory_data(file_path)
        
        # Skip empty trajectories
        if len(positions) == 0:
            continue
        
        all_positions.append(positions)
        all_quats.append(quats)
        file_names.append(file_name)
        
        print(f"Loaded trajectory from {file_name} with {len(positions)} points")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Add a world coordinate frame for reference
rr.log("world", rr.Points3D(positions=[[0, 0, 0]], colors=[255, 255, 255], radii=0.01))
rr.log("world/axes", rr.Arrows3D(
    origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    radii=0.005
))

# Function to visualize a trajectory
def visualize_trajectory(idx, pos, quat, color, file_name=None):
    # Filter points if requested
    if args.filter_below_zero:
        # Filter start point if needed
        start_point = pos[0]
        if start_point[2] >= 0:
            rr.log(
                f"trajectory/{idx}/points/start",
                rr.Points3D(positions=[start_point], colors=[0, 255, 0], radii=0.01)
            )
        
        # Filter end point if needed
        end_point = pos[-1]
        if end_point[2] >= 0:
            rr.log(
                f"trajectory/{idx}/points/end",
                rr.Points3D(positions=[end_point], colors=[255, 0, 0], radii=0.01)
            )
        
        # Log key points along the trajectory (only those with z >= 0)
        step_size = max(1, len(pos) // 50)  # Show about 15 points per trajectory
        key_indices = list(range(0, len(pos), step_size))
        key_positions = [pos[i] for i in key_indices if pos[i][2] >= 0]
        
        if key_positions:
            # Add points at key positions
            rr.log(
                f"trajectory/{idx}/points/key",
                rr.Points3D(positions=key_positions, colors=[color] * len(key_positions), radii=0.008)
            )
    else:
        # Log all points without filtering
        # Log start and end points with different colors
        rr.log(
            f"trajectory/{idx}/points/start",
            rr.Points3D(positions=[pos[0]], colors=[0, 255, 0], radii=0.01)
        )
        rr.log(
            f"trajectory/{idx}/points/end",
            rr.Points3D(positions=[pos[-1]], colors=[255, 0, 0], radii=0.01)
        )
        
        # Log key points along the trajectory
        step_size = max(1, len(pos) // 15)  # Show about 15 points per trajectory
        key_indices = list(range(0, len(pos), step_size))
        key_positions = [pos[i] for i in key_indices]
        
        # Add points at key positions
        rr.log(
            f"trajectory/{idx}/points/key",
            rr.Points3D(positions=key_positions, colors=[color] * len(key_positions), radii=0.008)
        )
    
    # Add coordinate frames at key points to show orientation (if not position-only mode)
    if not args.position_only:
        for i in range(0, len(pos), step_size * 2):  # Fewer frames to avoid clutter
            # Skip points with z < 0 if filtering is enabled
            if args.filter_below_zero and pos[i][2] < 0:
                continue
                
            # Visualize orientation using coordinate axes
            # Convert quaternion to rotation matrix
            rot_matrix = R.from_quat(quat[i]).as_matrix()
            
            # Calculate rotated axis vectors
            axis_length = 0.03
            x_axis = np.dot(rot_matrix, np.array([axis_length, 0, 0]))
            y_axis = np.dot(rot_matrix, np.array([0, axis_length, 0]))
            z_axis = np.dot(rot_matrix, np.array([0, 0, axis_length]))
            
            # Log arrows for each axis
            rr.log(
                f"trajectory/{idx}/axes/{i}",
                rr.Arrows3D(
                    origins=[pos[i], pos[i], pos[i]],
                    vectors=[x_axis, y_axis, z_axis],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    radii=0.002
                )
            )
    
    # Add file name as label if provided
    if file_name:
        rr.log(
            f"trajectory/{idx}/label",
            rr.TextLog(file_name)
        )

# Create a colormap for tasks if color-by-task is enabled
task_colors = {}
if args.color_by_task and task_names:
    task_cmap = plt.cm.get_cmap('tab20', len(task_names))
    for i, task_name in enumerate(task_names):
        # Convert matplotlib color (0-1 range) to rerun color (0-255 range)
        rgba = task_cmap(i)
        task_colors[task_name] = [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)]

# Log each trajectory with points and coordinate frames
for idx, (pos, quat) in enumerate(zip(all_positions, all_quats)):
    # Get color for this trajectory
    if args.color_by_task and idx < len(task_indices) and task_indices[idx] is not None:
        # Get task name from task index
        task_index = task_indices[idx]
        task_name = task_mapping.get(task_index, f"Unknown Task {task_index}")
        # Use task-based color
        color = task_colors.get(task_name, get_color(idx, len(all_positions)))
    else:
        # Use default color scheme
        color = get_color(idx, len(all_positions))
    
    # Visualize this trajectory
    file_name = file_names[idx] if idx < len(file_names) else None
    visualize_trajectory(idx, pos, quat, color, file_name)

# Add summary information
rr.log(
    "info/summary",
    rr.TextLog(f"Loaded {len(all_positions)} trajectories from gr00t_coin_primitive data")
)

print("Visualization ready! The rerun viewer should be open in a new window.")
print(f"Loaded {len(all_positions)} trajectories from {max_files} parquet files")
print(f"Data directory: {data_dir}")
print("Green points: Start positions")
print("Red points: End positions")
if args.position_only:
    print("Position-only mode: No orientation frames shown")
else:
    print("Coordinate frames show orientation at key points")
if args.filter_below_zero:
    print("Filtering: Points with z < 0 are hidden")
if args.color_by_task:
    print(f"Coloring: Trajectories colored by task type ({len(task_names)} unique tasks)")
