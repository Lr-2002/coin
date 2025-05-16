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

def plot_category_progression(model_name, category_type, metric='mean_score'):
    """
    Plot progression of different categories across normalized steps
    
    Args:
        model_name: Name of the model
        category_type: Type of category ('obj', 'rob', or 'iter')
        metric: Score metric to plot
    """
    # Load normalized scores
    scores_df = load_normalized_scores(model_name)
    
    # Load tags
    tags_data = load_tags()
    
    # Create task-category mapping
    task_categories = create_task_category_mapping(tags_data)
    
    # List of all unique categories
    all_categories = set()
    for task_data in task_categories.values():
        all_categories.update(task_data[category_type])
    
    # Skip if no categories
    if not all_categories:
        print(f"No {category_type} categories found")
        return None
    
    # Create a dictionary to store category data
    category_data = {category: [] for category in all_categories}
    
    # Process each task and step
    for task_name, task_group in scores_df.groupby('task'):
        # Try to find corresponding task in categories
        matching_task = None
        if task_name in task_categories:
            matching_task = task_name
        
        # Skip if task not found
        if not matching_task:
            continue
        
        # Get categories for this task
        task_cats = task_categories[matching_task][category_type]
        
        # Skip if no categories
        if not task_cats:
            continue
        
        # Add data points for each category and step
        for _, row in task_group.iterrows():
            for cat in task_cats:
                category_data[cat].append({
                    'step': row['step'],
                    'score': row[metric],
                    'category': cat
                })
    
    # Convert to DataFrame
    all_data = []
    for cat, data_points in category_data.items():
        for point in data_points:
            all_data.append(point)
    
    if not all_data:
        print(f"No data points found for {category_type} categories")
        return None
        
    plot_df = pd.DataFrame(all_data)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Calculate average by step for each category
    category_avg = {}
    for cat in all_categories:
        cat_data = plot_df[plot_df['category'] == cat]
        if len(cat_data) > 0:
            # Group by step and calculate average
            avg_data = cat_data.groupby('step')['score'].mean().reset_index()
            category_avg[cat] = avg_data
    
    # Plot each category
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    
    # Custom colors
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    # Create category lines for legend
    category_lines = []
    category_names = []
    
    # Get a mapping of category names to their count for better legends
    task_counts = {}
    for task_name, task_map in task_categories.items():
        for cat in task_map[category_type]:
            if cat in task_counts:
                task_counts[cat] += 1
            else:
                task_counts[cat] = 1
    
    # Filter out categories with fewer than 3 tasks
    filtered_avg = {}
    filtered_counts = {}
    for cat, avg_data in category_avg.items():
        count = task_counts.get(cat, 0)
        if count >= 4 and len(avg_data) > 0:
            filtered_avg[cat] = avg_data
            filtered_counts[cat] = count
    
    # Print filtered categories
    print(f"Found {len(filtered_avg)} categories with at least 3 tasks each")
    for cat, count in filtered_counts.items():
        print(f"  - {cat}: {count} tasks")
    
    # Plot each category that has at least 3 tasks
    for i, (cat, avg_data) in enumerate(filtered_avg.items()):
        count = filtered_counts[cat]
        # Plot average line with marker and label
        line, = plt.plot(
            avg_data['step'], 
            avg_data['score'],
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2.5,
            markersize=8,
            label=f"{cat} ({count} tasks)"
        )
        
        category_lines.append(line)
        category_names.append(f"{cat} ({count} tasks)")
    
    # Category type full names
    category_type_names = {
        'obj': 'Object Properties',
        'rob': 'Robot Skills',
        'iter': 'Interaction Types'
    }
    
    # Set labels and title
    ft_size= 20
    plt.xlabel("Normalized Step", fontsize=ft_size)
    plt.ylabel(f'{metric.replace("_", " ").title()} (1-10)', fontsize=ft_size)
    plt.title(f'{model_name}: {category_type_names[category_type]} Performance Across Task Steps', fontsize=ft_size)
    
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
    
    # Add legend with sorting by performance
    # Sort categories by performance
    avg_perf = {}
    for cat, avg_data in filtered_avg.items():  
        if len(avg_data) > 0:
            try:
                avg_perf[cat] = avg_data['score'].mean()
            except Exception as e:
                print(f"Error calculating mean for {cat}: {e}")
                # Use a default value if mean calculation fails
                avg_perf[cat] = 0
    
    # Sort categories by performance (if we have any)
    try:
        if avg_perf:
            sorted_cats = sorted(avg_perf.items(), key=lambda x: x[1], reverse=True)
            # Using safer list comprehension with error checking
            sorted_indices = []
            for cat, _ in sorted_cats:
                try:
                    idx = list(avg_perf.keys()).index(cat)
                    sorted_indices.append(idx)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not find index for category {cat}: {e}")
        else:
            sorted_cats = []
            sorted_indices = []
    except Exception as e:
        print(f"Error sorting categories: {e}")
        sorted_cats = []
        sorted_indices = []
    
    # Create sorted legend items (only if we have categories)
    sorted_lines = []
    sorted_names = []
    if sorted_indices:  # Only proceed if we have categories
        sorted_lines = [category_lines[i] for i in sorted_indices]
        sorted_names = [category_names[i] for i in sorted_indices]
    
    # Add legend only if we have categories to show
    if sorted_lines:
        # Place legend in lower left for 'obj' category, upper right for others
        legend_position = 'lower left' if category_type == 'obj' and 'gpt' in model_name else 'upper right'
        plt.legend(
            sorted_lines, sorted_names,
            loc=legend_position, 
            fontsize=20,
            title=f"{category_type_names[category_type]} (sorted by avg. score)"
        )
        # Set title font size for legend
        plt.setp(plt.gca().get_legend().get_title(), fontsize=20)
    
    # Adjust layout
    plt.tight_layout()
    # plt.subplots_adjust(right=0.75)  # Make room for legend
    
    # Save figure
    output_path = os.path.join(
        STATIC_DIR, 
        f"{model_name.replace('-','')}_category_{category_type}_{metric}_progression.png"
    )
    plt.savefig(output_path, dpi=300)
    print(f"Saved {category_type} progression plot to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Plot progression of normalized scores by category")
    parser.add_argument("--model", type=str, help="Model name to plot")
    parser.add_argument("--models", type=str, nargs='+', help="Multiple models to plot")
    parser.add_argument("--category-type", type=str, choices=['obj', 'rob', 'iter'], 
                        default='all', help="Category type to plot")
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
    
    # Get category types to process
    category_types = ['obj', 'rob', 'iter'] if args.category_type == 'all' else [args.category_type]
    
    # Process each model and category type
    for model_name in models:
        try:
            print(f"Processing category progression plots for {model_name}")
            
            for cat_type in category_types:
                plot_category_progression(model_name, cat_type, args.metric)
                
        except Exception as e:
            print(f"Error processing {model_name} {cat_type}: {e}")
    
    print("Category progression visualization complete!")

if __name__ == "__main__":
    main()
