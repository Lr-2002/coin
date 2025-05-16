#!/usr/bin/env python3

import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# Base paths
STATIC_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static"
NORMALIZED_DIR = os.path.join(STATIC_DIR, "normalized_scores")
TAGS_FILE = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/env_extended_tags.json"

def load_tags():
    """Load task categories from the tags JSON file"""
    with open(TAGS_FILE, 'r') as f:
        tags_data = json.load(f)
    return tags_data

def get_tasks_per_category(tags_data):
    """Count how many tasks belong to each category"""
    category_counts = defaultdict(int)
    
    for task_name, task_tags in tags_data.items():
        # Process object-centric categories
        if 'obj' in task_tags:
            for obj_tag in task_tags['obj']:
                category_counts[obj_tag] += 1
        
        # Process robot-centric categories
        if 'rob' in task_tags:
            for rob_tag in task_tags['rob']:
                category_counts[rob_tag] += 1
                
        # Process interactive categories
        if 'iter' in task_tags:
            for iter_tag in task_tags['iter']:
                category_counts[iter_tag] += 1
    
    return category_counts

def get_valid_categories(min_tasks=3):
    """Get categories that have at least the minimum number of tasks"""
    tags_data = load_tags()
    counts = get_tasks_per_category(tags_data)
    
    # Filter categories with at least min_tasks tasks
    valid_categories = {cat: count for cat, count in counts.items() if count >= min_tasks}
    return valid_categories

def plot_step_progression(model_name, metric='mean_score'):
    """
    Plot the progression of scores across normalized steps
    
    Args:
        model_name: Name of the model
        metric: Score metric to plot (default: 'mean_score')
    """
    # Get path to normalized scores
    clean_name = model_name.replace('-', '')
    input_path = os.path.join(NORMALIZED_DIR, f"{clean_name}-normalized-score.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Normalized score file not found: {input_path}")
    
    # Read normalized scores
    df = pd.read_csv(input_path)
    
    # Ensure we have the metric column
    if metric not in df.columns:
        available_metrics = [col for col in df.columns if col not in ['task', 'step']]
        if available_metrics:
            metric = available_metrics[0]
            print(f"Requested metric not found. Using {metric} instead.")
        else:
            raise ValueError(f"No valid metrics found in {input_path}")
    
    # Exclude first (0) and last step
    # Get the maximum step number
    max_step = df['step'].max()
    # Filter steps to exclude first and last
    df_filtered = df[(df['step'] > 0) & (df['step'] < max_step)]
    
    # Group by step and calculate average
    step_avg = df_filtered.groupby('step')[metric].mean().reset_index()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot average trend line
    plt.plot(step_avg['step'], step_avg[metric], 'r-', linewidth=3, label='Average Score')
    
    # Add confidence interval
    step_std = df_filtered.groupby('step')[metric].std().reset_index()
    plt.fill_between(
        step_avg['step'],
        step_avg[metric] - step_std[metric],
        step_avg[metric] + step_std[metric],
        alpha=0.2, color='red',
        label='± 1 std dev'
    )
    
    # Plot individual task lines with transparency
    for task_name, task_data in df_filtered.groupby('task'):
        plt.plot(task_data['step'], task_data[metric], 'o-', alpha=0.3, linewidth=1)
    
    # Set labels and title
    plt.xlabel('Normalized Step (1-8)', fontsize=14)  # Updated range
    plt.ylabel(f'{metric.replace("_", " ").title()} (1-10)', fontsize=14)
    plt.title(f'{model_name}: Progression of {metric.replace("_", " ").title()} (Excluding First & Last Steps)', fontsize=16)
    
    # Set axis limits (adjusted for excluded steps)
    plt.xlim(0.5, max_step - 0.5)
    plt.ylim(0, 10.5)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add score reference lines
    for score in range(1, 11, 2):
        plt.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Save figure
    output_path = os.path.join(STATIC_DIR, f"{clean_name}_{metric}_middle_steps_progression.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {metric} progression plot (middle steps only) to {output_path}")
    
    return output_path

def plot_all_metrics_progression(model_name):
    """
    Plot progression for all available metrics
    
    Args:
        model_name: Name of the model
    """
    # Get path to normalized scores
    clean_name = model_name.replace('-', '')
    input_path = os.path.join(NORMALIZED_DIR, f"{clean_name}-normalized-score.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Normalized score file not found: {input_path}")
    
    # Read normalized scores
    df = pd.read_csv(input_path)
    
    # Identify available metrics
    available_metrics = [col for col in df.columns if col not in ['task', 'step']]
    
    if not available_metrics:
        raise ValueError(f"No valid metrics found in {input_path}")
    
    # Exclude first (0) and last step
    max_step = df['step'].max()
    df_filtered = df[(df['step'] > 0) & (df['step'] < max_step)]
    
    # Create subplots for each metric
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 6 * n_metrics))
    
    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        # Group by step and calculate average
        step_avg = df_filtered.groupby('step')[metric].mean().reset_index()
        
        # Plot average trend line
        ax.plot(step_avg['step'], step_avg[metric], 'r-', linewidth=2, label='Average Score')
        
        # Add confidence interval
        step_std = df_filtered.groupby('step')[metric].std().reset_index()
        ax.fill_between(
            step_avg['step'],
            step_avg[metric] - step_std[metric],
            step_avg[metric] + step_std[metric],
            alpha=0.2, color='red',
            label='± 1 std dev'
        )
        
        # Set labels and title
        ax.set_xlabel(f'Normalized Step (1-{max_step-1})')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (1-10)')
        ax.set_title(f'{metric.replace("_", " ").title()} Progression (excluding first & last steps)')
        
        # Set axis limits
        ax.set_xlim(0.5, max_step - 0.5)
        ax.set_ylim(0, 10.5)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='upper right')
    
    # Set overall title
    plt.suptitle(f'{model_name}: Score Metrics Progression (Middle Steps Only)', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_path = os.path.join(STATIC_DIR, f"{clean_name}_all_metrics_middle_steps_progression.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved all metrics progression plot (middle steps only) to {output_path}")
    
    return output_path

def compare_models_progression(models, metric='mean_score'):
    """
    Compare progression across multiple models
    
    Args:
        models: List of model names
        metric: Score metric to compare (default: 'mean_score')
    """
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Line styles and colors for different models
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Store model data for legend
    model_lines = []
    model_names = []
    max_step = 9  # Default max step
    
    # Plot each model
    for i, model_name in enumerate(models):
        # Get path to normalized scores
        clean_name = model_name.replace('-', '')
        input_path = os.path.join(NORMALIZED_DIR, f"{clean_name}-normalized-score.csv")
        
        if not os.path.exists(input_path):
            print(f"Normalized score file not found: {input_path}")
            continue
        
        # Read normalized scores
        df = pd.read_csv(input_path)
        
        # Ensure we have the metric column
        if metric not in df.columns:
            print(f"Metric {metric} not found for model {model_name}")
            continue
        
        # Exclude first (0) and last step
        curr_max_step = df['step'].max()
        max_step = min(max_step, curr_max_step) if max_step != 9 else curr_max_step
        df_filtered = df[(df['step'] > 0) & (df['step'] < curr_max_step)]
        
        # Group by step and calculate average
        step_avg = df_filtered.groupby('step')[metric].mean().reset_index()
        
        # Plot average trend line
        line, = plt.plot(
            step_avg['step'], 
            step_avg[metric], 
            marker='o',
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i % len(colors)],
            linewidth=3, 
            label=model_name
        )
        
        model_lines.append(line)
        model_names.append(model_name)
        
        # Add confidence interval
        step_std = df_filtered.groupby('step')[metric].std().reset_index()
        plt.fill_between(
            step_avg['step'],
            step_avg[metric] - step_std[metric],
            step_avg[metric] + step_std[metric],
            alpha=0.1,
            color=colors[i % len(colors)]
        )
    
    # Set labels and title
    plt.xlabel(f'Normalized Step (1-{max_step-1})', fontsize=14)
    plt.ylabel(f'{metric.replace("_", " ").title()} (1-10)', fontsize=14)
    plt.title(f'Comparison of {metric.replace("_", " ").title()} Progression (Excluding First & Last Steps)', fontsize=16)
    
    # Set axis limits
    plt.xlim(0.5, max_step - 0.5)
    plt.ylim(0, 10.5)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add score reference lines
    for score in range(1, 11, 2):
        plt.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
    
    # Add legend
    if model_lines:
        plt.legend(model_lines, model_names, loc='upper right')
    
    # Save figure
    output_path = os.path.join(STATIC_DIR, f"model_comparison_{metric}_middle_steps_progression.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved model comparison plot (middle steps only) to {output_path}")
    
    return output_path

# Already defined earlier in the file

def main():
    parser = argparse.ArgumentParser(description='Plot progression of normalized scores.')
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=['gpt-4o-2024-05-13'], 
        help='Model names to plot'
    )
    parser.add_argument(
        '--metric', 
        default='mean_score', 
        help='Metric to plot (default: mean_score)'
    )
    parser.add_argument(
        '--all-metrics', 
        action='store_true', 
        help='Plot all available metrics'
    )
    parser.add_argument(
        '--compare', 
        action='store_true', 
        help='Compare multiple models'
    )
    parser.add_argument(
        '--min-tasks',
        type=int,
        default=3,
        help='Minimum number of tasks required for a category to be included (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(NORMALIZED_DIR, exist_ok=True)
    
    # Get valid categories (with min_tasks threshold)
    valid_categories = get_valid_categories(args.min_tasks)
    print(f"Found {len(valid_categories)} categories with at least {args.min_tasks} tasks each")
    for category, count in valid_categories.items():
        print(f"  - {category}: {count} tasks")
    
    # Process based on arguments
    if args.compare and len(args.models) > 1:
        compare_models_progression(args.models, args.metric)
    elif args.all_metrics:
        for model in args.models:
            plot_all_metrics_progression(model)
    else:
        for model in args.models:
            plot_step_progression(model, args.metric)
    
    print("Score progression visualization complete!")

if __name__ == "__main__":
    main()
