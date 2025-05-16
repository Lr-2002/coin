#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Base paths
STATIC_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static"

def normalize_steps(df):
    """
    Normalize steps for each task to be between 0 and 1
    """
    result_df = df.copy()
    
    # Group by task and normalize steps
    for task_name, task_group in df.groupby('task'):
        # Get max step value for this task
        max_step = int(task_group['step'].max())
        
        # Convert steps to float to avoid integer division
        steps_normalized = task_group['step'].astype(float) / max_step
        
        # Update the normalized values in the result dataframe for this task
        task_indices = result_df[result_df['task'] == task_name].index
        result_df.loc[task_indices, 'normalized_step'] = steps_normalized.values
    
    return result_df

def plot_model_comparison(models, output_path):
    """
    Generate a plot comparing score progression across models
    
    Args:
        models: List of model names
        output_path: Path to save output plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for different models
    model_colors = {
        'gpt-4o-2024-11-20': 'royalblue',
        'gemini-20': 'forestgreen'
    }
    
    # Model line styles
    line_styles = ['-', '--']
    
    for i, model_name in enumerate(models):
        # Construct input path
        clean_name = model_name.replace('-', '')
        input_path = f"{STATIC_DIR}/{clean_name}-plan-score.csv"
        
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            continue
        
        # Read CSV
        df = pd.read_csv(input_path)
        
        # Add header if not present
        if 'task' not in df.columns:
            df.columns = ['task', 'step', 'completeness', 'objects_correct', 
                        'sequence_correct', 'clarity', 'mean_score']
        
        # Convert step column to numeric
        df['step'] = pd.to_numeric(df['step'])
        
        # Normalize steps
        df = normalize_steps(df)
        
        # Sort by normalized step
        df_sorted = df.assign(sort_key=df['normalized_step']).sort_values('sort_key')
        
        # Group data into bins for smoother visualization
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        df_sorted['bin'] = pd.cut(df_sorted['normalized_step'], bins, labels=bins[:-1])
        
        # Calculate mean scores by bin
        bin_means = df_sorted.groupby('bin')['mean_score'].mean().reset_index()
        
        # Convert bin to numeric for plotting
        bin_means['bin'] = bin_means['bin'].astype(float)
        
        # Plot the aggregate trend line
        line, = ax.plot(bin_means['bin'], bin_means['mean_score'], 
                    marker='o', 
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=3,
                    color=model_colors.get(model_name, f'C{i}'),
                    label=f"{model_name}")
        
        # Add scatter plot of all individual data points with lower opacity
        ax.scatter(df_sorted['normalized_step'], df_sorted['mean_score'], 
                alpha=0.1, 
                color=model_colors.get(model_name, f'C{i}'),
                s=20)
    
    # Set axis labels and title
    ax.set_xlabel('Task Progress (Normalized)', fontsize=14)
    ax.set_ylabel('Mean Score (1-10)', fontsize=14)
    ax.set_title('Model Performance Comparison Across Task Progress', fontsize=16)
    
    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 10.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12)
    
    # Add score reference lines
    for score in range(1, 11, 2):
        ax.axhline(y=score, color='gray', linestyle=':', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate model comparison plot")
    parser.add_argument("--output", type=str, 
                        default=f"{STATIC_DIR}/model_score_comparison.png",
                        help="Output plot file")
    args = parser.parse_args()
    
    # Use both models
    models = ['gpt-4o-2024-11-20', 'gemini-20']
    
    # Plot comparison
    plot_model_comparison(models, args.output)

if __name__ == "__main__":
    main()
