#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import glob
import pandas as pd
import json
from typing import Dict, List, Optional


def compute_derivatives(trajectory, dt=1.0):
    """Compute velocity, acceleration and jerk from trajectory data."""
    velocity = np.diff(trajectory, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    return velocity, acceleration, jerk


def mean_jerk(trajectory, dt=1.0):
    """Compute mean jerk (lower is smoother)."""
    _, _, jerk = compute_derivatives(trajectory, dt)
    return np.mean(np.abs(jerk))


def dimensionless_jerk(trajectory, dt=1.0, movement_duration=None):
    """Compute dimensionless jerk (lower is smoother)."""
    if movement_duration is None:
        movement_duration = (trajectory.shape[0] - 1) * dt
    
    _, _, jerk = compute_derivatives(trajectory, dt)
    
    # Calculate path length (amplitude)
    displacement = np.max(trajectory, axis=0) - np.min(trajectory, axis=0)
    path_length = np.sum(displacement**2)**0.5
    
    # Normalized jerk
    normalized_jerk = np.mean(np.sum(jerk**2, axis=1)) * (movement_duration**5 / path_length**2)
    
    return normalized_jerk


def velocity_smoothness(trajectory, dt=1.0):
    """Compute smoothness based on velocity coherence (higher is smoother)."""
    velocity, _, _ = compute_derivatives(trajectory, dt)
    
    # Compute variation in velocity direction
    velocity_norms = np.linalg.norm(velocity, axis=1, keepdims=True)
    normalized_velocity = np.divide(velocity, velocity_norms, where=velocity_norms!=0)
    
    # Compute dot product between consecutive velocity vectors
    velocity_coherence = np.sum(normalized_velocity[:-1] * normalized_velocity[1:], axis=1)
    
    # Higher values mean smoother motion (closer to 1)
    smoothness = np.mean(velocity_coherence)
    
    return smoothness


def spectral_arc_length(trajectory, dt=1.0, cutoff=20.0):
    """Compute Spectral Arc Length (closer to zero is smoother)."""
    velocity, _, _ = compute_derivatives(trajectory, dt)
    
    # Compute speed profile
    speed = np.linalg.norm(velocity, axis=1)
    
    # Normalize speed
    if np.max(speed) > 0:
        speed = speed / np.max(speed)
    
    # Compute FFT
    n_samples = len(speed)
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    spectrum = np.fft.rfft(speed)
    magnitude = np.abs(spectrum) / n_samples
    
    # Apply cutoff frequency
    valid_idx = freqs <= cutoff
    valid_freqs = freqs[valid_idx]
    valid_magnitude = magnitude[valid_idx]
    
    # Normalize magnitude within cutoff
    if np.max(valid_magnitude) > 0:
        valid_magnitude = valid_magnitude / np.max(valid_magnitude)
    
    # Compute spectral arc length
    if len(valid_freqs) <= 1:
        return 0
    
    # Compute derivative of magnitude w.r.t. frequency
    dmag_df = np.diff(valid_magnitude) / np.diff(valid_freqs)
    
    # Compute arc length
    integrand = np.sqrt(1 + (dmag_df**2))
    arc_length = -np.sum(integrand * np.diff(valid_freqs))
    
    return arc_length


def load_trajectory_from_pickle(filepath):
    """Load trajectory data from a pickle file as numpy array."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check if the loaded data is a numpy array
        if isinstance(data, np.ndarray):
            return data
        
        # If it's a dictionary, try to extract the actions
        if isinstance(data, dict) and 'actions' in data:
            return data['actions']
        
        raise ValueError(f"Could not extract trajectory data from {filepath}")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def truncate_unchanged_actions(trajectory, window_size=10):
    """Remove sequences of identical actions using a sliding window approach.
    
    Args:
        trajectory: Array of shape (n_steps, n_dims) containing action data
        window_size: Size of window to check for unchanged actions
        
    Returns:
        filtered_trajectory: Trajectory with unchanged action sequences removed
    """
    if len(trajectory) <= 1:
        return trajectory
    
    # Initialize with first action
    filtered_trajectory = [trajectory[0]]
    i = 1
    
    while i < len(trajectory):
        # If we're too close to the end for a full window, just process normally
        if i + window_size > len(trajectory):
            if not np.array_equal(trajectory[i], filtered_trajectory[-1]):
                filtered_trajectory.append(trajectory[i])
            i += 1
            continue
            
        # Check if all actions in the window are identical
        window_actions = trajectory[i:i+window_size]
        all_identical = True
        
        # Compare each action in window with the first action in the window
        reference_action = window_actions[0]
        for action in window_actions[1:]:
            if not np.array_equal(action, reference_action):
                all_identical = False
                break
        
        if all_identical:
            # All actions in window are identical, only keep the first one if it differs from last kept action
            if not np.array_equal(reference_action, filtered_trajectory[-1]):
                filtered_trajectory.append(reference_action)
            i += window_size  # Skip the entire window
        else:
            # Window contains different actions, process next action normally
            if not np.array_equal(trajectory[i], filtered_trajectory[-1]):
                filtered_trajectory.append(trajectory[i])
            i += 1
    
    return np.array(filtered_trajectory)


def preprocess_trajectory(trajectory, warmup_steps=10, exclude_gripper=True):
    """Preprocess trajectory: skip warmup, exclude gripper, truncate unchanged actions."""
    info = {}
    
    # Store original length
    original_length = len(trajectory)
    info['original_length'] = original_length
    
    # Skip warm-up steps
    if warmup_steps > 0:
        if warmup_steps >= len(trajectory):
            print(f"Warning: warmup_steps ({warmup_steps}) >= trajectory length ({len(trajectory)}). Using warmup=0.")
            warmup_steps = 0
        else:
            trajectory = trajectory[warmup_steps:]
    
    info['after_warmup_length'] = len(trajectory)
    
    # Check if we need to exclude the gripper dimension
    if exclude_gripper and trajectory.shape[1] > 1:
        # Store the shape before excluding gripper
        info['original_dims'] = trajectory.shape[1]
        
        # Exclude the last dimension (gripper)
        trajectory = trajectory[:, :-1]
        
        # Store the shape after excluding gripper
        info['processed_dims'] = trajectory.shape[1]
    
    # Store trajectory before removing unchanged actions
    trajectory_with_unchanged = trajectory.copy()
    info['before_truncation_length'] = len(trajectory_with_unchanged)
    
    # Truncate unchanged actions
    filtered_trajectory = truncate_unchanged_actions(trajectory)
    info['after_truncation_length'] = len(filtered_trajectory)
    
    # Calculate percentage of unchanged actions
    if len(trajectory) > 0:  # Avoid division by zero
        info['unchanged_action_percent'] = 100 * (len(trajectory) - len(filtered_trajectory)) / len(trajectory)
    else:
        info['unchanged_action_percent'] = 0
    
    return filtered_trajectory, info


def analyze_smoothness(trajectory, dt=1.0, warmup_steps=10, exclude_gripper=True, verbose=True):
    """Analyze trajectory smoothness after preprocessing."""
    # Preprocess trajectory
    processed_traj, info = preprocess_trajectory(
        trajectory, 
        warmup_steps=warmup_steps, 
        exclude_gripper=exclude_gripper
    )
    
    # Initialize metrics with preprocessing info
    metrics = {**info}
    
    if verbose:
        print(f"Original trajectory length: {info['original_length']} steps")
        print(f"After warmup ({warmup_steps} steps): {info['after_warmup_length']} steps")
        if exclude_gripper and 'original_dims' in info:
            print(f"Excluded gripper dimension: {info['original_dims']} dims -> {info['processed_dims']} dims")
        print(f"Removed {info['unchanged_action_percent']:.1f}% unchanged actions")
        print(f"Final trajectory length: {info['after_truncation_length']} steps")
    
    # If we have too few points after preprocessing, return NaN for metrics
    if len(processed_traj) < 4:  # Need at least 4 points for jerk calculation
        if verbose:
            print("Warning: Too few points after preprocessing for smoothness calculation")
        metrics['mean_jerk'] = float('nan')
        metrics['dimensionless_jerk'] = float('nan')
        metrics['velocity_smoothness'] = float('nan')
        metrics['spectral_arc_length'] = float('nan')
        return metrics
    
    # Compute metrics on processed trajectory
    metrics['mean_jerk'] = mean_jerk(processed_traj, dt)
    metrics['dimensionless_jerk'] = dimensionless_jerk(processed_traj, dt)
    metrics['velocity_smoothness'] = velocity_smoothness(processed_traj, dt)
    metrics['spectral_arc_length'] = spectral_arc_length(processed_traj, dt)
    
    if verbose:
        print("\nSmoothness Metrics:")
        print(f"  Mean Jerk: {metrics['mean_jerk']:.4f} (lower is smoother)")
        print(f"  Dimensionless Jerk: {metrics['dimensionless_jerk']:.4f} (lower is smoother)")
        print(f"  Velocity Smoothness: {metrics['velocity_smoothness']:.4f} (higher is smoother, max 1.0)")
        print(f"  Spectral Arc Length: {metrics['spectral_arc_length']:.4f} (closer to zero is smoother)")
    
    return metrics


def visualize_trajectories(all_metrics, save_path):
    """Create comparison visualizations of smoothness metrics."""
    # Extract key data
    task_names = [m['task_full_name'] for m in all_metrics]
    mean_jerks = [m['mean_jerk'] for m in all_metrics]
    dimensionless_jerks = [m['dimensionless_jerk'] for m in all_metrics]
    velocity_smoothnesses = [m['velocity_smoothness'] for m in all_metrics]
    spectral_arc_lengths = [m['spectral_arc_length'] for m in all_metrics]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Flatten axes for easier access
    axes = axes.flatten()
    
    # Plot metrics
    axes[0].bar(task_names, mean_jerks)
    axes[0].set_title('Mean Jerk (lower is smoother)')
    axes[0].set_ylabel('Mean Jerk')
    axes[0].tick_params(axis='x', rotation=90)
    
    axes[1].bar(task_names, dimensionless_jerks)
    axes[1].set_title('Dimensionless Jerk (lower is smoother)')
    axes[1].set_ylabel('Dimensionless Jerk')
    axes[1].tick_params(axis='x', rotation=90)
    
    axes[2].bar(task_names, velocity_smoothnesses)
    axes[2].set_title('Velocity Smoothness (higher is smoother)')
    axes[2].set_ylabel('Velocity Smoothness')
    axes[2].tick_params(axis='x', rotation=90)
    
    axes[3].bar(task_names, spectral_arc_lengths)
    axes[3].set_title('Spectral Arc Length (closer to zero is smoother)')
    axes[3].set_ylabel('Spectral Arc Length')
    axes[3].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def find_trajectory_files(base_dir):
    """Find all trajectory pickle files in the given directory structure."""
    traj_pattern = os.path.join(base_dir, "**", "*.pkl")
    traj_files = glob.glob(traj_pattern, recursive=True)
    
    if not traj_files:
        print(f"No pickle files found in {base_dir}")
    
    return traj_files


def extract_task_info(file_path):
    """Extract task name and other info from trajectory file path.
    
    Handles directory format like:
    - 20250507_214620_Tabletop-Close-Cabinet-v1_cogact
    - 20250507_215616_Tabletop-Close-Door-v1_cogact
    
    Returns standardized task info for grouping and analysis.
    """
    dirname = os.path.basename(os.path.dirname(file_path))
    parts = dirname.split('_')
    
    # Initialize task information
    task_full_name = None
    
    # Parse full task name from directory parts
    # Format is typically: DATE_TIME_TASKNAME_MODEL
    if len(parts) >= 3:
        # Look for Tabletop part which contains the task name
        for part in parts[2:]:
            if 'Tabletop' in part:
                task_full_name = part
                break
    
    return {
        'task_full_name': task_full_name,  # Full task ID (e.g., 'Tabletop-Close-Door-v1')
        'dirname': dirname,                # Full directory name
        'filename': os.path.basename(file_path) # Filename without path
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze robot trajectory smoothness")
    parser.add_argument(
        "dir", type=str, help="Directory containing trajectory data")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Time step between consecutive points")
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of initial steps to skip")
    parser.add_argument(
        "--include-gripper", action='store_true', help='Include gripper dimension in analysis')
    parser.add_argument(
        "--max-jerk-threshold", type=float, default=float('500'), 
        help="Maximum allowed jerk threshold. Episodes with mean jerk above this value will be ignored")
    parser.add_argument(
        '--verbose', action='store_true', help='Print detailed information for each trajectory')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # If no output directory specified, use the input directory name in tools/smothness
    if args.output_dir is None:
        # Get the base name of the input directory
        input_dirname = os.path.basename(os.path.normpath(args.dir))
        args.output_dir = os.path.join('env_tests/smothness', input_dirname)
        print(f"Using output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find trajectory files
    traj_files = find_trajectory_files(args.dir)
    
    if not traj_files:
        print("No trajectory files found")
        return
    
    print(f"Found {len(traj_files)} trajectory files")
    if args.max_jerk_threshold < float('inf'):
        print(f"Will filter episodes with mean jerk > {args.max_jerk_threshold}")
    
    # Store metrics for all trajectories
    all_metrics = []
    
    # Process each trajectory file
    filtered_count = 0
    for file_path in traj_files:
        # Extract task info
        info = extract_task_info(file_path)
        
        if args.verbose:
            print(f"\nProcessing: {file_path}")
            print(f"Task: {info['task_full_name']}")
            
        # Log jerk threshold if set
        if args.max_jerk_threshold < float('inf') and args.verbose:
            print(f"Using jerk threshold: {args.max_jerk_threshold}")
        
        # Load trajectory
        trajectory = load_trajectory_from_pickle(file_path)
        if trajectory is None:
            continue
        
        # Analyze smoothness
        metrics = analyze_smoothness(
            trajectory,
            dt=args.dt,
            warmup_steps=args.warmup,
            exclude_gripper=not args.include_gripper,
            verbose=args.verbose
        )
        
        # Check if mean jerk exceeds the threshold
        if metrics['mean_jerk'] > args.max_jerk_threshold:
            if args.verbose:
                print(f"  Filtering out trajectory: mean jerk = {metrics['mean_jerk']:.4f} > threshold {args.max_jerk_threshold}")
            filtered_count += 1
            continue
        
        # Add file and task info to metrics
        metrics.update(info)
        metrics['file_path'] = file_path
        
        # Add to collection
        all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid trajectories found")
        return
    
    # Create a DataFrame for easier analysis
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics to CSV
    csv_path = os.path.join(args.output_dir, 'smoothness_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")
    
    # Create comparison visualizations
    viz_path = os.path.join(args.output_dir, 'smoothness_comparison.png')
    visualize_trajectories(all_metrics, viz_path)
    print(f"Saved visualization to {viz_path}")
    
    # Generate summary by task
    summary_path = os.path.join(args.output_dir, 'summary_by_task.csv')
    task_summary = metrics_df.groupby('task_full_name').agg({
        'mean_jerk': ['mean', 'std'],
        'dimensionless_jerk': ['mean', 'std'],
        'velocity_smoothness': ['mean', 'std'],
        'spectral_arc_length': ['mean', 'std'],
        'original_length': ['mean', 'count'],
        'after_truncation_length': ['mean'],
        'unchanged_action_percent': ['mean']
    })
    task_summary.to_csv(summary_path)
    print(f"Saved task summary to {summary_path}")
    
    # Create overall summary
    summary_text = "\n=== Overall Summary ===\n"
    summary_text += f"Total trajectories processed: {len(all_metrics)}\n"
    if filtered_count > 0:
        summary_text += f"Episodes filtered out (jerk > {args.max_jerk_threshold}): {filtered_count}\n"
    summary_text += f"Average trajectory length: {metrics_df['original_length'].mean():.1f} steps\n"
    summary_text += f"Average unchanged action percentage: {metrics_df['unchanged_action_percent'].mean():.1f}%\n"
    summary_text += f"Average length after processing: {metrics_df['after_truncation_length'].mean():.1f} steps\n"
    summary_text += "\nAverage smoothness metrics:\n"
    summary_text += f"  Mean Jerk: {metrics_df['mean_jerk'].mean():.4f} ± {metrics_df['mean_jerk'].std():.4f}\n"
    summary_text += f"  Dimensionless Jerk: {metrics_df['dimensionless_jerk'].mean():.4f} ± {metrics_df['dimensionless_jerk'].std():.4f}\n"
    summary_text += f"  Velocity Smoothness: {metrics_df['velocity_smoothness'].mean():.4f} ± {metrics_df['velocity_smoothness'].std():.4f}\n"
    summary_text += f"  Spectral Arc Length: {metrics_df['spectral_arc_length'].mean():.4f} ± {metrics_df['spectral_arc_length'].std():.4f}\n"
    
    # Create JSON summary
    summary_dict = {
        "total_trajectories": len(all_metrics),
        "filtered_episodes": filtered_count,
        "jerk_threshold": args.max_jerk_threshold if args.max_jerk_threshold < float('inf') else "none",
        "average_trajectory_length": float(metrics_df['original_length'].mean()),
        "average_unchanged_action_percentage": float(metrics_df['unchanged_action_percent'].mean()),
        "average_processed_length": float(metrics_df['after_truncation_length'].mean()),
        "smoothness_metrics": {
            "mean_jerk": {
                "mean": float(metrics_df['mean_jerk'].mean()),
                "std": float(metrics_df['mean_jerk'].std())
            },
            "dimensionless_jerk": {
                "mean": float(metrics_df['dimensionless_jerk'].mean()),
                "std": float(metrics_df['dimensionless_jerk'].std())
            },
            "velocity_smoothness": {
                "mean": float(metrics_df['velocity_smoothness'].mean()),
                "std": float(metrics_df['velocity_smoothness'].std())
            },
            "spectral_arc_length": {
                "mean": float(metrics_df['spectral_arc_length'].mean()),
                "std": float(metrics_df['spectral_arc_length'].std())
            }
        }
    }
    
    # Save overall summary to text file
    overall_summary_path = os.path.join(args.output_dir, 'overall_summary.txt')
    with open(overall_summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Saved overall summary to {overall_summary_path}")
    
    # Save overall summary to JSON file
    overall_summary_json_path = os.path.join(args.output_dir, 'overall_summary.json')
    with open(overall_summary_json_path, 'w') as f:
        json.dump(summary_dict, f, indent=4)
    print(f"Saved JSON summary to {overall_summary_json_path}")
    
    # Print summary to console
    print(summary_text)


if __name__ == "__main__":
    main()
