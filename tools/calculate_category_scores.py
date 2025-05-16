#!/usr/bin/env python3
"""
Calculate aggregated scores by category type (object-centric, robot-centric, compositional reasoning)
for each model, where each task is counted only once per category type.

Usage: python calculate_category_scores.py
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation_results")
TAGS_FILE = os.path.join(BASE_DIR, "env_extended_tags.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "github_page/static/category_scores")

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tags():
    """Load environment tags from JSON file"""
    with open(TAGS_FILE, 'r') as f:
        tags_data = json.load(f)
    return tags_data

def load_results():
    """Load evaluation results from CSV file"""
    results_file = os.path.join(EVAL_DIR, "final_interactive.csv")
    df = pd.read_csv(results_file)
    return df

def calculate_category_scores(results_df, tags_data):
    """
    Calculate scores by category type for each model,
    counting each task only once per category type.
    
    Args:
        results_df: DataFrame with evaluation results
        tags_data: Dictionary of task tags
        
    Returns:
        DataFrame with category scores per model
    """
    # Map category types to more readable names
    category_names = {
        'obj': 'Object-Centric',
        'rob': 'Robot-Centric',
        'iter': 'Compositional'
    }
    
    # Initialize data structure to track tasks per category and model
    category_tasks = {
        'obj': set(),
        'rob': set(),
        'iter': set()
    }
    
    # Initialize counters and sum trackers
    category_counts = {
        'obj': 0,
        'rob': 0,
        'iter': 0
    }
    
    model_category_sums = {}
    models = results_df.columns[1:]  # Skip Task_name column
    
    for model in models:
        model_category_sums[model] = {
            'obj': 0.0,
            'rob': 0.0,
            'iter': 0.0
        }
    
    # Process each task
    for _, row in results_df.iterrows():
        task_name = row['Task_name']
        
        # Skip if task not in tags
        if task_name not in tags_data:
            print(f"Warning: {task_name} not found in tags data")
            continue
        
        # Get tags for this task
        task_tags = tags_data[task_name]
        
        # Track if this task belongs to each category
        task_in_category = {
            'obj': False,
            'rob': False,
            'iter': False
        }
        
        # Check if task has tags in each category
        for category in ['obj', 'rob', 'iter']:
            if category in task_tags and task_tags[category]:
                task_in_category[category] = True
                if task_name not in category_tasks[category]:
                    category_tasks[category].add(task_name)
                    category_counts[category] += 1
        
        # Process success rates for each model
        for model in models:
            success_rate_str = row[model]
            
            # Convert success rate from string to float
            if isinstance(success_rate_str, str) and success_rate_str.endswith('%'):
                success_rate = float(success_rate_str.strip('%')) / 100.0
            else:
                success_rate = float(success_rate_str) / 100.0 if success_rate_str else 0.0
            
            # Add success rate to appropriate categories
            for category in ['obj', 'rob', 'iter']:
                if task_in_category[category]:
                    model_category_sums[model][category] += success_rate
    
    # Calculate final scores (average success rate per category)
    results = []
    
    for model in models:
        row_data = {'Model': model}
        
        for category in ['obj', 'rob', 'iter']:
            if category_counts[category] > 0:
                avg_score = (model_category_sums[model][category] / category_counts[category]) * 100
                row_data[category_names[category]] = f"{avg_score:.2f}%"
            else:
                row_data[category_names[category]] = "0.00%"
        
        results.append(row_data)
    
    # Create DataFrame and return
    results_df = pd.DataFrame(results)
    
    # Add category task counts info
    print(f"Category task counts:")
    for category, count in category_counts.items():
        print(f"  {category_names[category]}: {count} tasks")
    
    return results_df

def main():
    # Load data
    tags_data = load_tags()
    results_df = load_results()
    
    # Calculate scores
    category_scores = calculate_category_scores(results_df, tags_data)
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, "model_category_scores.csv")
    category_scores.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print pretty table
    print("\nModel Category Scores:\n")
    print(category_scores.to_string(index=False))
    
    # Create a visualization version for markdown/display
    markdown_table = "| Model | Object-Centric | Robot-Centric | Compositional |\n"
    markdown_table += "|-------|---------------|--------------|---------------|\n"
    
    for _, row in category_scores.iterrows():
        markdown_table += f"| {row['Model']} | {row['Object-Centric']} | {row['Robot-Centric']} | {row['Compositional']} |\n"
    
    print("\nMarkdown Table:\n")
    print(markdown_table)

if __name__ == "__main__":
    main()
