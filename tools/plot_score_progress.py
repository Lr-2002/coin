#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib
# Use non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
sns.set_context("talk")

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

def plot_scores_by_progress(input_csv, output_path=None, metrics=None, model_name=None):
    """
    Generate a plot of scores vs. normalized progress
    
    Args:
        input_csv: Path to input CSV file
        output_path: Path to save output plot
        metrics: List of metrics to plot (default: only mean_score)
    """
    # Default to mean_score if no metrics specified
    if metrics is None:
        metrics = ['mean_score']
    
    # Read CSV
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    # Add header if not present
    if 'task' not in df.columns:
        df.columns = ['task', 'step', 'completeness', 'objects_correct', 
                      'sequence_correct', 'clarity', 'mean_score']
    
    # Convert step column to numeric
    df['step'] = pd.to_numeric(df['step'])
    
    # Normalize steps
    df = normalize_steps(df)
    
    # Create figure with larger figsize for better visualization
    fig = plt.figure(figsize=(14, 10))
    
    # Extract the model name from the input file if not provided
    if model_name is None:
        model_name = os.path.basename(input_csv).replace('-plan-score.csv', '')
    
    # Create a plot with individual points and a regression line
    scatter_plot = sns.scatterplot(
        data=df,
        x="normalized_step",
        y="mean_score",
        hue="task",
        palette="deep",
        alpha=0.7,
        s=100  # Larger points
    )
    
    # Add a smooth trend line with confidence interval using polynomial fit (order=2) instead of lowess
    trend_line = sns.regplot(
        x="normalized_step",
        y="mean_score",
        data=df,
        scatter=False,
        color="red",
        line_kws={"linewidth": 3, "linestyle": "-"},
        order=2  # Polynomial order (quadratic fit)
    )
    
    # Add title and formatting
    plt.title(f'Performance Across Task Progress for {model_name}', fontsize=18)
    plt.xlabel('Task Progress (Normalized)', fontsize=16)
    plt.ylabel('Mean Score (1-10)', fontsize=16)
    
    # Set axis limits and add horizontal score reference lines
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, 10.5)
    
    # Add horizontal reference lines for score levels
    for score in range(1, 11, 2):
        plt.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
        
    # Enhance legend - position it outside the plot
    plt.legend(title="Tasks", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    # Add annotations for trend interpretation
    df_start = df[df['normalized_step'] < 0.3]['mean_score'].mean()
    df_end = df[df['normalized_step'] > 0.7]['mean_score'].mean()
    trend_direction = "increasing" if df_end > df_start else "decreasing"
    
    plt.annotate(
        f"Overall trend: {trend_direction}", 
        xy=(0.5, 0.05), 
        xycoords='figure fraction',
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

    # Improve spacing and layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)  # Make room for legend
    
    # Save figure if output path is specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    # Don't show plot, only save it
    # plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot scores vs. normalized progress")
    parser.add_argument("--input", type=str, 
                        default="/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static/gpt4o20241120-plan-score.csv",
                        help="Input CSV file")
    parser.add_argument("--output", type=str, 
                        default="/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static/score_progress_plot.png",
                        help="Output plot file")
    parser.add_argument("--model", type=str, help="Model name (without -plan-score.csv)")
    
    args = parser.parse_args()
    
    # If model is specified, construct input path
    if args.model:
        model_name = args.model
        input_path = f"/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static/{model_name.replace('-', '')}-plan-score.csv"
    else:
        model_name = os.path.basename(args.input).replace('-plan-score.csv', '')
        input_path = args.input
    
    # Plot scores with model name
    plot_scores_by_progress(input_path, args.output, model_name=model_name)

if __name__ == "__main__":
    main()
