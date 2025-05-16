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

def load_normalized_scores(model_name):
    """Load normalized scores for a model"""
    clean_name = model_name.replace('-', '')
    input_path = os.path.join(NORMALIZED_DIR, f"{clean_name}-normalized-score.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Normalized score file not found: {input_path}")
    
    # Read normalized scores
    df = pd.read_csv(input_path)
    return df

def create_task_category_mapping(tags_data):
    """
    Create a mapping between tasks and their categories
    
    Returns a dictionary with task names as keys and dictionaries of category types as values
    """
    task_categories = {}
    
    for task_name, task_tags in tags_data.items():
        # Convert from tag format (with hyphens) to score format (with underscores)
        score_task_name = task_name.replace('-', '_')
        
        task_categories[score_task_name] = {
            'obj': task_tags.get('obj', []),
            'rob': task_tags.get('rob', []),
            'iter': task_tags.get('iter', [])
        }
    
    return task_categories

def plot_avg_by_step(model_name, metric='mean_score'):
    """
    Plot average scores by step for object, robot, and interaction categories
    
    Args:
        model_name: Name of the model
        metric: Score metric to plot
    """
    # Load normalized scores
    scores_df = load_normalized_scores(model_name)
    
    # Load tags
    tags_data = load_tags()
    
    # Create task-category mapping
    task_categories = create_task_category_mapping(tags_data)
    
    # Category type full names
    category_type_names = {
        'obj': 'Object Properties',
        'rob': 'Robot Skills',
        'iter': 'Interaction Types'
    }
    
    # Create a dictionary to store average scores by step for each category type
    category_types = ['obj', 'rob', 'iter']
    avg_scores_by_step = {cat_type: defaultdict(list) for cat_type in category_types}
    
    # Process each task and step
    for task_name, task_group in scores_df.groupby('task'):
        # Try to find corresponding task in categories
        if task_name not in task_categories:
            continue
            
        # Process each step for this task
        for _, row in task_group.iterrows():
            step = row['step']
            score = row[metric]
            
            # Add score to respective category types
            for cat_type in category_types:
                task_cats = task_categories[task_name][cat_type]
                if task_cats:  # If task has categories of this type
                    avg_scores_by_step[cat_type][step].append(score)
    
    # Calculate averages
    avg_by_step = {}
    for cat_type in category_types:
        avg_by_step[cat_type] = {}
        for step, scores in avg_scores_by_step[cat_type].items():
            if scores:  # Check if there are any scores for this step
                avg_by_step[cat_type][step] = np.mean(scores)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Custom colors for each category type
    cat_colors = {
        'obj': '#1f77b4',  # Blue
        'rob': '#2ca02c',  # Green
        'iter': '#d62728'  # Red
    }
    
    # Plot average score by step for each category type
    for cat_type in category_types:
        # Convert to DataFrame for easier plotting
        if avg_by_step[cat_type]:
            steps = sorted(avg_by_step[cat_type].keys())
            scores = [avg_by_step[cat_type][step] for step in steps]
            
            # Plot with markers and lines
            plt.plot(
                steps, scores,
                marker='o',
                markersize=10,
                linewidth=3,
                label=category_type_names[cat_type],
                color=cat_colors[cat_type]
            )
    
    # Set labels and title
    ft_size = 16
    plt.xlabel("Normalized Step", fontsize=ft_size)
    plt.ylabel(f'{metric.replace("_", " ").title()} (1-10)', fontsize=ft_size)
    plt.title(f'{model_name}: Average Category Performance by Step', fontsize=ft_size)
    
    # Set axis limits
    plt.xlim(-0.5, 9.5)
    plt.ylim(3, 9.5)
    
    # Set ticks with larger font size
    plt.xticks(range(10), fontsize=ft_size)
    plt.yticks(range(3, 10), fontsize=ft_size)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add score reference lines
    for score in range(1, 11, 2):
        plt.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
    
    # Add legend
    plt.legend(
        loc='lower right',
        fontsize=14,
        title="Category Types"
    )
    # Set title font size for legend
    plt.setp(plt.gca().get_legend().get_title(), fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(
        STATIC_DIR, 
        f"{model_name.replace('-','')}_category_avg_by_step.png"
    )
    plt.savefig(output_path, dpi=300)
    print(f"Saved average category by step plot to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Plot average category scores by step")
    parser.add_argument("--model", type=str, help="Model name to plot")
    parser.add_argument("--models", type=str, nargs='+', help="Multiple models to plot")
    parser.add_argument("--metric", type=str, default='mean_score', 
                      help="Score metric to plot (default: mean_score)")
    args = parser.parse_args()
    
    # If no models specified, use default models
    if not args.model and not args.models:
        models = ['gpt-4o-2024-11-20', 'gemini-20']
    elif args.models:
        models = args.models
    else:
        models = [args.model]
    
    # Process each model
    for model_name in models:
        try:
            print(f"Processing average category by step plot for {model_name}")
            plot_avg_by_step(model_name, args.metric)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    print("Average category by step visualization complete!")

if __name__ == "__main__":
    main()
