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
TAGS_FILE = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/env_extended_tags.json"

def load_tags():
    """Load task categories from the tags JSON file"""
    with open(TAGS_FILE, 'r') as f:
        tags_data = json.load(f)
    return tags_data

def load_scores(model_name):
    """Load model scores from the CSV file"""
    clean_name = model_name.replace('-', '')
    csv_path = f"{STATIC_DIR}/{clean_name}-plan-score.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Score file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Add header if not present
    if 'task' not in df.columns:
        df.columns = ['task', 'step', 'completeness', 'objects_correct', 
                      'sequence_correct', 'clarity', 'mean_score']
    
    return df

def calculate_category_scores(scores_df, tags_data):
    """
    Calculate mean scores for each task category
    
    Returns a dictionary with category types (obj, rob, iter) as keys,
    each containing dictionaries of {category: mean_score}
    """
    # Extract just the task names without the step information
    task_scores = scores_df.groupby('task')['mean_score'].mean().to_dict()
    
    # Initialize category scores dictionary
    category_scores = {
        'obj': defaultdict(list),
        'rob': defaultdict(list),
        'iter': defaultdict(list)
    }
    
    # Print task names for debugging
    print(f"Task names in scores: {list(task_scores.keys())[:3]}...")
    print(f"Task names in tags: {list(tags_data.keys())[:3]}...")
    
    # Create a mapping between the two formats (underscores vs hyphens)
    tag_to_score_map = {}
    for score_task in task_scores.keys():
        # Convert from score format (with underscores) to tag format (with hyphens)
        hyphen_task = score_task.replace('_', '-')
        if hyphen_task in tags_data:
            tag_to_score_map[hyphen_task] = score_task
        # Try additionally removing 'v1' suffix
        elif hyphen_task.rstrip('1').rstrip('v').rstrip('-') in tags_data:
            clean_name = hyphen_task.rstrip('1').rstrip('v').rstrip('-')
            tag_to_score_map[clean_name] = score_task
    
    print(f"Matched {len(tag_to_score_map)} tasks between scores and tags")
    
    # Collect scores for each category
    for tag_task_name, task_tags in tags_data.items():
        # Get corresponding score task name
        score_task_name = tag_to_score_map.get(tag_task_name)
        
        # Skip tasks that don't have scores
        if not score_task_name or score_task_name not in task_scores:
            continue
            
        task_mean_score = task_scores[score_task_name]
        
        # Add the task score to each of its categories
        for category_type in ['obj', 'rob', 'iter']:
            for category in task_tags.get(category_type, []):
                category_scores[category_type][category].append(task_mean_score)
    
    # Calculate mean scores for each category
    category_means = {
        'obj': {},
        'rob': {},
        'iter': {}
    }
    
    for category_type in ['obj', 'rob', 'iter']:
        for category, scores in category_scores[category_type].items():
            if scores:  # Only calculate mean if there are scores
                category_means[category_type][category] = np.mean(scores)
    
    return category_means

def plot_category_scores(category_means, model_name, output_dir=STATIC_DIR):
    """Generate bar plots for each category type"""
    os.makedirs(output_dir, exist_ok=True)
    
    for category_type in ['obj', 'rob', 'iter']:
        if not category_means[category_type]:
            continue  # Skip empty categories
            
        # Sort categories by score
        categories = [(cat, score) for cat, score in category_means[category_type].items()]
        categories.sort(key=lambda x: x[1], reverse=True)
        
        category_names = [cat for cat, _ in categories]
        scores = [score for _, score in categories]
        
        if not category_names:
            continue
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create bar plot with seaborn
        ax = sns.barplot(x=category_names, y=scores, palette="deep")
        
        # Add score labels on top of each bar
        for i, score in enumerate(scores):
            ax.text(i, score + 0.1, f"{score:.2f}", ha='center', fontsize=12)
        
        # Set title and labels
        category_type_names = {
            'obj': 'Object Properties',
            'rob': 'Robot Skills',
            'iter': 'Interaction Types'
        }
        
        plt.title(f"{model_name}: Mean Scores by {category_type_names[category_type]}", fontsize=16)
        plt.xlabel(f"{category_type_names[category_type]} Categories", fontsize=14)
        plt.ylabel("Mean Score (1-10)", fontsize=14)
        
        # Make sure y-axis goes from 0 to 10
        plt.ylim(0, 10.5)
        
        # Add reference lines
        for score in range(2, 10, 2):
            plt.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
            
        # Format x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add overall mean score annotation
        overall_mean = np.mean(list(category_means[category_type].values()))
        plt.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2)
        plt.annotate(f"Overall Mean: {overall_mean:.2f}", 
                     xy=(0.85, overall_mean + 0.2), 
                     xycoords=('axes fraction', 'data'),
                     color='red', fontsize=12)
        
        # Save plot
        output_path = os.path.join(output_dir, f"{model_name.replace('-','')}_scores_{category_type}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved {category_type} plot to {output_path}")
        plt.close()
    
    return True

def create_category_comparison(models, category_type, output_dir=STATIC_DIR):
    """
    Create a comparison plot showing how different models perform on the same categories
    """
    # Dictionary to store category scores for each model
    all_category_scores = {}
    
    # Collect data for all models
    for model_name in models:
        try:
            scores_df = load_scores(model_name)
            tags_data = load_tags()
            category_means = calculate_category_scores(scores_df, tags_data)
            
            if category_means[category_type]:
                all_category_scores[model_name] = category_means[category_type]
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    if not all_category_scores:
        print(f"No data available for category type: {category_type}")
        return False
    
    # Get all unique categories across all models
    all_categories = set()
    for model_name, categories in all_category_scores.items():
        all_categories.update(categories.keys())
    
    # Prepare data for plotting
    plot_data = []
    for model_name, categories in all_category_scores.items():
        for category, score in categories.items():
            plot_data.append({
                'model': model_name,
                'category': category,
                'score': score
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Map category types to full names
    category_type_names = {
        'obj': 'Object Properties',
        'rob': 'Robot Skills',
        'iter': 'Interaction Types'
    }
    
    # Create grouped bar chart
    ax = sns.catplot(
        data=df, 
        kind="bar",
        x="category", y="score", hue="model",
        palette="deep", alpha=.8, height=8, aspect=1.5
    )
    
    # Set title and labels
    ax.set_xlabels(f"{category_type_names[category_type]} Categories", fontsize=14)
    ax.set_ylabels("Mean Score (1-10)", fontsize=14)
    plt.title(f"Model Comparison: {category_type_names[category_type]}", fontsize=16)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set(ylim=(0, 10.5))
    
    # Add reference lines
    for score in range(2, 10, 2):
        ax.ax.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, f"model_comparison_{category_type}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison plot to {output_path}")
    plt.close()
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate category score plots")
    parser.add_argument("--model", type=str, help="Model name to analyze (e.g., 'gpt-4o-2024-11-20')")
    parser.add_argument("--models", type=str, nargs='+', help="Multiple models to compare")
    parser.add_argument("--category", type=str, choices=['obj', 'rob', 'iter'], 
                        help="Category type to compare across models")
    parser.add_argument("--output-dir", type=str, default=STATIC_DIR,
                        help="Directory to save output plots")
    args = parser.parse_args()
    
    # If no models specified, use default models
    if not args.model and not args.models:
        models = ['gpt-4o-2024-11-20', 'gemini-20']
    elif args.models:
        models = args.models
    else:
        models = [args.model]
    
    # Generate individual model plots
    for model_name in models:
        try:
            print(f"Processing category scores for {model_name}")
            scores_df = load_scores(model_name)
            tags_data = load_tags()
            
            category_means = calculate_category_scores(scores_df, tags_data)
            plot_category_scores(category_means, model_name, args.output_dir)
            
            # Save category means to CSV
            for category_type in ['obj', 'rob', 'iter']:
                if category_means[category_type]:
                    df = pd.DataFrame([
                        {'category': cat, 'mean_score': score} 
                        for cat, score in category_means[category_type].items()
                    ])
                    
                    csv_path = os.path.join(
                        args.output_dir, 
                        f"{model_name.replace('-','')}_category_{category_type}.csv"
                    )
                    df.to_csv(csv_path, index=False)
                    print(f"Saved {category_type} scores to {csv_path}")
                    
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Generate comparison plots if multiple models
    if len(models) > 1 and args.category:
        create_category_comparison(models, args.category, args.output_dir)
    elif len(models) > 1:
        # Generate comparisons for all category types
        for category_type in ['obj', 'rob', 'iter']:
            create_category_comparison(models, category_type, args.output_dir)
    
    print("Category score analysis complete!")

if __name__ == "__main__":
    main()
