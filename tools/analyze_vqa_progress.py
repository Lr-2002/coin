#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import argparse
from pathlib import Path
from collections import defaultdict

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# Base paths
BASE_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill"
STATIC_DIR = os.path.join(BASE_DIR, "github_page/static")
VQA_DIR = os.path.join(BASE_DIR, "env_tests/vqa_results/vqa_checkpoint_images")
OUTPUT_DIR = os.path.join(STATIC_DIR, "vqa_normalized")

def load_vqa_results(model_name, date_folder=None):
    """
    Load VQA results for a specific model
    
    Args:
        model_name: Name of the model (e.g., 'gemini')
        date_folder: Optional specific date folder to use
    
    Returns:
        Loaded JSON data
    """
    model_dir = os.path.join(VQA_DIR, model_name)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # If date_folder not specified, use the most recent one
    if date_folder is None:
        date_folders = sorted(os.listdir(model_dir))
        if not date_folders:
            raise FileNotFoundError(f"No date folders found in {model_dir}")
        date_folder = date_folders[-1]  # Use the most recent folder
    
    date_path = os.path.join(model_dir, date_folder)
    
    # Find the vqa_results json file
    json_files = [f for f in os.listdir(date_path) if f.startswith('vqa_results') and f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No vqa_results JSON files found in {date_path}")
    
    # Use the first file found
    json_path = os.path.join(date_path, json_files[0])
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data, json_path

def extract_middle_steps(vqa_data):
    """
    Extract middle steps from VQA data (ignoring first and last step)
    
    Args:
        vqa_data: VQA results data
        
    Returns:
        Dictionary with task and middle step data
    """
    middle_steps = {}
    
    for task_name, task_data in vqa_data.items():
        answers = task_data.get('answers', [])
        correct = task_data.get('correct', [])
        
        # Only process tasks with at least 3 steps (so we have at least 1 middle step)
        if len(answers) < 3:
            continue
        
        # Skip first and last step
        middle_answers = answers[1:-1]
        middle_correct = correct[1:-1]
        
        # Store middle step data
        middle_steps[task_name] = {
            'answers': middle_answers,
            'correct': middle_correct,
            'num_steps': len(middle_answers)
        }
    
    return middle_steps

def normalize_steps(middle_steps, num_normalized_steps=4):
    """
    Normalize middle steps to a fixed number of steps
    
    Args:
        middle_steps: Dictionary of middle step data
        num_normalized_steps: Number of steps to normalize to
        
    Returns:
        DataFrame with normalized step data
    """
    # List to store normalized data
    normalized_data = []
    
    # Process each task
    for task_name, task_data in middle_steps.items():
        correct = task_data['correct']
        num_actual_steps = len(correct)
        
        # Skip tasks with no correct data
        if num_actual_steps == 0:
            continue
        
        # Skip normalization if there's only one middle step
        if num_actual_steps == 1:
            for step in range(num_normalized_steps):
                normalized_data.append({
                    'task': task_name,
                    'normalized_step': step,
                    'accuracy': 1 if correct[0] else 0
                })
            continue
        
        # Create original step indices (0 to num_actual_steps-1)
        x_orig = np.arange(num_actual_steps)
        
        # Convert boolean correct values to 1s and 0s
        y_orig = np.array([1 if c else 0 for c in correct])
        
        # Create target step indices (0 to num_normalized_steps-1)
        x_target = np.linspace(0, num_actual_steps - 1, num_normalized_steps)
        
        # Create interpolation function (nearest neighbor to preserve binary values)
        f = interp1d(x_orig, y_orig, kind='nearest', bounds_error=False, fill_value="extrapolate")
        
        # Generate interpolated values
        y_interp = f(x_target)
        
        # Round to integers (0 or 1)
        y_interp = np.round(y_interp).astype(int)
        
        # Add to normalized data
        for i, step in enumerate(range(num_normalized_steps)):
            normalized_data.append({
                'task': task_name,
                'normalized_step': step,
                'accuracy': y_interp[i]
            })
    
    # Convert to DataFrame
    normalized_df = pd.DataFrame(normalized_data)
    
    return normalized_df

def plot_normalized_accuracy(normalized_df, model_name, num_normalized_steps=4):
    """
    Plot normalized accuracy across steps
    
    Args:
        normalized_df: DataFrame with normalized step data
        model_name: Name of the model
        num_normalized_steps: Number of normalized steps
    """
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # Calculate average accuracy at each normalized step
    step_avg = normalized_df.groupby('normalized_step')['accuracy'].mean().reset_index()
    
    # Calculate confidence intervals
    task_counts = normalized_df.groupby('normalized_step')['task'].nunique()
    step_std = normalized_df.groupby('normalized_step')['accuracy'].std().reset_index()
    step_avg['ci'] = 1.96 * step_std['accuracy'] / np.sqrt(task_counts)
    
    # Plot average accuracy line
    plt.plot(
        step_avg['normalized_step'], 
        step_avg['accuracy'] * 100,  # Convert to percentage
        'r-o',
        linewidth=3,
        markersize=10,
        label='Average Accuracy'
    )
    
    # Add confidence interval
    plt.fill_between(
        step_avg['normalized_step'],
        (step_avg['accuracy'] - step_avg['ci']) * 100,
        (step_avg['accuracy'] + step_avg['ci']) * 100,
        alpha=0.2,
        color='red',
        label='95% Confidence Interval'
    )
    
    # Add scatter plot of individual task accuracies with jitter
    for task_name, task_data in normalized_df.groupby('task'):
        # Add small random jitter to avoid overlapping points
        jitter = np.random.uniform(-0.1, 0.1, size=len(task_data))
        plt.scatter(
            task_data['normalized_step'] + jitter,
            task_data['accuracy'] * 100,
            alpha=0.3,
            s=30,
            color='gray'
        )
    
    # Set labels and title
    plt.xlabel('Normalized Step', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title(f'{model_name}: VQA Accuracy Across Task Progress\n(Excluding First and Last Steps)', fontsize=16)
    
    # Set axis limits and ticks
    plt.xlim(-0.5, num_normalized_steps - 0.5)
    plt.ylim(40, 80)
    plt.xticks(range(num_normalized_steps))
    plt.yticks(range(40, 80, 10))
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add reference line at 50% (random guessing for binary questions)
    plt.axhline(y=50, color='blue', linestyle='--', alpha=0.5, label='50% (Random Guess)')
    
    # Add value labels on the line
    for i, row in step_avg.iterrows():
        plt.text(
            row['normalized_step'],
            row['accuracy'] * 100 + 3,
            f"{row['accuracy']*100:.1f}%",
            ha='center',
            fontsize=14,
            weight='bold'
        )
    
    # Add task count annotations
    for i, count in enumerate(task_counts):
        plt.text(
            i,
            -5,
            f"n={count}",
            ha='center',
            fontsize=12
        )
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save figure
    clean_name = model_name.replace('-', '').replace('.', '')
    output_path = os.path.join(OUTPUT_DIR, f"{clean_name}_vqa_accuracy_progression.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved accuracy progression plot to {output_path}")
    
    # Save average data to CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{clean_name}_vqa_accuracy_progression.csv")
    step_avg.to_csv(csv_path, index=False)
    print(f"Saved average accuracy data to {csv_path}")
    
    return output_path

def compare_models(model_names, date_folders=None, num_normalized_steps=4):
    """
    Compare multiple models on VQA accuracy progression
    
    Args:
        model_names: List of model names
        date_folders: Optional list of date folders (must match model_names)
        num_normalized_steps: Number of normalized steps
    """
    # Dictionary to store model data
    model_data = {}
    
    # Process each model
    for i, model_name in enumerate(model_names):
        date_folder = None if date_folders is None else date_folders[i]
        
        try:
            # Load VQA results
            vqa_data, json_path = load_vqa_results(model_name, date_folder)
            
            # Extract middle steps
            middle_steps = extract_middle_steps(vqa_data)
            
            # Normalize steps
            normalized_df = normalize_steps(middle_steps, num_normalized_steps)
            
            # Store normalized data
            model_data[model_name] = normalized_df
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Skip if no models processed
    if not model_data:
        print("No models processed successfully")
        return
    
    # Create comparison plot
    plt.figure(figsize=(14, 6))
    
    # Line styles and colors for different models
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Store legend data
    legend_lines = []
    legend_labels = []
    
    # Plot each model
    for i, (model_name, normalized_df) in enumerate(model_data.items()):
        # Calculate average accuracy at each normalized step
        step_avg = normalized_df.groupby('normalized_step')['accuracy'].mean().reset_index()
        
        # Calculate confidence intervals
        task_counts = normalized_df.groupby('normalized_step')['task'].nunique()
        step_std = normalized_df.groupby('normalized_step')['accuracy'].std().reset_index()
        step_avg['ci'] = 1.96 * step_std['accuracy'] / np.sqrt(task_counts)
        
        # Clean model name for display
        display_name = model_name.split('-')[0].capitalize()
        
        # Plot average accuracy line
        line, = plt.plot(
            step_avg['normalized_step'], 
            step_avg['accuracy'] * 100,  # Convert to percentage
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=3,
            markersize=10,
            label=f'{display_name} (Avg: {step_avg["accuracy"].mean()*100:.1f}%)'
        )
        
        legend_lines.append(line)
        legend_labels.append(f'{display_name} (Avg: {step_avg["accuracy"].mean()*100:.1f}%)')
        
        # Add confidence interval
        plt.fill_between(
            step_avg['normalized_step'],
            (step_avg['accuracy'] - step_avg['ci']) * 100,
            (step_avg['accuracy'] + step_avg['ci']) * 100,
            alpha=0.1,
            color=colors[i % len(colors)]
        )
        
        # Add value labels on the line
        for _, row in step_avg.iterrows():
            plt.text(
                row['normalized_step'],
                row['accuracy'] * 100 + 3 + (i * 2),  # Offset to avoid overlap
                f"{row['accuracy']*100:.1f}%",
                ha='center',
                fontsize=12,
                color=colors[i % len(colors)]
            )
    
    # Set labels and title
    plt.xlabel('Normalized Step', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.title('VQA Accuracy Across Task Progress\n(Excluding First and Last Steps)', fontsize=16)
    
    # Set axis limits and ticks
    plt.xlim(-0.5, num_normalized_steps - 0.5)
    plt.ylim(40, 80)
    plt.xticks(range(num_normalized_steps))
    plt.yticks(range(40, 80, 10))
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add reference line at 50% (random guessing for binary questions)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (Random Guess)')
    
    # Add legend
    plt.legend(legend_lines, legend_labels, loc='lower right', fontsize=14)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "model_comparison_vqa_accuracy.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved model comparison plot to {output_path}")
    
    # Save combined data to CSV
    combined_data = []
    for model_name, normalized_df in model_data.items():
        step_avg = normalized_df.groupby('normalized_step')['accuracy'].mean().reset_index()
        step_avg['model'] = model_name
        combined_data.append(step_avg)
    
    combined_df = pd.concat(combined_data)
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison_vqa_accuracy.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved combined accuracy data to {csv_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Analyze VQA results progression")
    parser.add_argument("--model", type=str, default="gemini",
                        help="Model name to analyze")
    parser.add_argument("--date", type=str,
                        help="Specific date folder to use")
    parser.add_argument("--models", type=str, nargs='+',
                        help="Multiple models to compare")
    parser.add_argument("--dates", type=str, nargs='+',
                        help="Date folders for multiple models")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of normalized steps")
    args = parser.parse_args()
    
    # If multiple models specified, compare them
    if args.models:
        compare_models(args.models, args.dates, args.steps)
    else:
        # Load VQA results
        vqa_data, json_path = load_vqa_results(args.model, args.date)
        print(f"Loaded VQA results from {json_path}")
        
        # Extract middle steps
        middle_steps = extract_middle_steps(vqa_data)
        print(f"Extracted middle steps from {len(middle_steps)} tasks")
        
        # Normalize steps
        normalized_df = normalize_steps(middle_steps, args.steps)
        
        # Plot normalized accuracy
        plot_normalized_accuracy(normalized_df, args.model, args.steps)
        
        # Save normalized data to CSV
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        clean_name = args.model.replace('-', '').replace('.', '')
        csv_path = os.path.join(OUTPUT_DIR, f"{clean_name}_vqa_normalized.csv")
        normalized_df.to_csv(csv_path, index=False)
        print(f"Saved normalized data to {csv_path}")
    
    print("VQA analysis complete!")

if __name__ == "__main__":
    main()
