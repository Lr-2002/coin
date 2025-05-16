#!/usr/bin/env python3
"""
Analyze success rates by tag categories.
Usage: python analyze_success_by_tag.py
"""

import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation_results")
TAGS_FILE = os.path.join(BASE_DIR, "env_extended_tags.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "github_page/static/tag_analysis")

# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tags():
    """Load environment tags"""
    with open(TAGS_FILE, 'r') as f:
        tags_data = json.load(f)
    return tags_data

def load_results():
    """Load evaluation results"""
    results_file = os.path.join(EVAL_DIR, "final_interactive.csv")
    df = pd.read_csv(results_file)
    return df

def get_tag_success_rates(df, tags_data):
    """Calculate success rates by tag category"""
    # Initialize dictionaries to store success rates
    category_results = {
        'obj': {},  # Object-centric
        'rob': {},  # Robot-centric
        'iter': {}  # Compositional reasoning (interactive)
    }
    
    # Initialize counters for categories
    tag_counts = {
        'obj': {},
        'rob': {},
        'iter': {}
    }
    
    # Process each task
    for task_name, row in df.iterrows():
        task_name = row['Task_name']  # Get the task name from each row
        
        # Skip if task not in tags
        if task_name not in tags_data:
            print(f"Warning: {task_name} not found in tags data")
            continue
        
        # Get tags for this task
        task_tags = tags_data[task_name]
        
        # Process success rates for each model (skipping the Task_name column)
        for model_name in df.columns[1:]:
            success_rate = float(row[model_name].strip('%')) / 100.0 if isinstance(row[model_name], str) else row[model_name] / 100.0
            
            # Process for each category
            for category in ['obj', 'rob', 'iter']:
                # Skip if no tags in this category
                if category not in task_tags or not task_tags[category]:
                    continue
                
                # Update tag counts
                for tag in task_tags[category]:
                    if tag not in tag_counts[category]:
                        tag_counts[category][tag] = 0
                    tag_counts[category][tag] += 1
                    
                    # Add model if not already tracked
                    if model_name not in category_results[category]:
                        category_results[category][model_name] = {}
                    
                    # Add tag if not already tracked
                    if tag not in category_results[category][model_name]:
                        category_results[category][model_name][tag] = []
                    
                    # Add success rate
                    category_results[category][model_name][tag].append(success_rate)
    
    # Calculate average success rates by tag
    for category in ['obj', 'rob', 'iter']:
        for model_name in category_results[category]:
            for tag in category_results[category][model_name]:
                category_results[category][model_name][tag] = np.mean(category_results[category][model_name][tag])
    
    return category_results, tag_counts

def calculate_aggregate_success_rates(category_results, tag_counts):
    """Calculate aggregate success rates for each category and model"""
    aggregate_results = {
        'obj': {},
        'rob': {},
        'iter': {}
    }
    
    for category in ['obj', 'rob', 'iter']:
        for model_name in category_results[category]:
            # Calculate weighted average for each model
            total_weighted_sum = 0
            total_count = 0
            
            for tag in category_results[category][model_name]:
                count = tag_counts[category][tag]
                total_weighted_sum += category_results[category][model_name][tag] * count
                total_count += count
            
            # Store aggregate success rate (avoid division by zero)
            if total_count > 0:
                aggregate_results[category][model_name] = total_weighted_sum / total_count
            else:
                aggregate_results[category][model_name] = 0
    
    return aggregate_results

def plot_results(category_results, tag_counts, aggregate_results):
    """Plot the results"""
    category_names = {
        'obj': 'Object-Centric',
        'rob': 'Robot-Centric',
        'iter': 'Compositional Reasoning'
    }
    
    # 1. Plot aggregate results
    fig, ax = plt.subplots(figsize=(12, 8))
    models = []
    obj_scores = []
    rob_scores = []
    iter_scores = []
    
    # Collect all models across all categories
    all_models = set()
    for category in aggregate_results:
        for model in aggregate_results[category]:
            all_models.add(model)
    
    # Sort models for consistent ordering
    all_models = sorted(list(all_models))
    
    for model in all_models:
        models.append(model)
        obj_scores.append(aggregate_results['obj'].get(model, 0) * 100)
        rob_scores.append(aggregate_results['rob'].get(model, 0) * 100)
        iter_scores.append(aggregate_results['iter'].get(model, 0) * 100)
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, obj_scores, width, label='Object-Centric')
    ax.bar(x, rob_scores, width, label='Robot-Centric')
    ax.bar(x + width, iter_scores, width, label='Compositional Reasoning')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Success Rate (%)')
    ax.set_title('Average Success Rates by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'aggregate_success_rates.png'), dpi=300)
    plt.close()
    
    # 2. Create detailed plots for each category with specific tags
    for category, category_name in category_names.items():
        # Skip if no data for this category
        if not category_results[category]:
            continue
        
        # Get all tags for this category
        all_tags = set()
        for model in category_results[category]:
            all_tags.update(category_results[category][model].keys())
        
        # Skip if no tags
        if not all_tags:
            continue
        
        # For each tag, create a plot showing model performance
        for tag in sorted(all_tags):
            fig, ax = plt.subplots(figsize=(12, 8))
            tag_models = []
            tag_scores = []
            
            for model in all_models:
                if model in category_results[category] and tag in category_results[category][model]:
                    tag_models.append(model)
                    tag_scores.append(category_results[category][model][tag] * 100)
            
            if not tag_models:
                continue
            
            y_pos = np.arange(len(tag_models))
            ax.barh(y_pos, tag_scores, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tag_models)
            ax.invert_yaxis()  # Top values at the top
            ax.set_xlabel('Success Rate (%)')
            ax.set_title(f'{category_name}: {tag} (Tasks: {tag_counts[category][tag]})')
            
            plt.tight_layout()
            tag_filename = tag.replace(' ', '_').lower()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{category}_{tag_filename}.png'), dpi=300)
            plt.close()
    
    # 3. Create radar plots for each model
    for model in all_models:
        # Collect all tags for this model across categories
        model_tags = {}
        for category in category_results:
            if model in category_results[category]:
                for tag, score in category_results[category][model].items():
                    model_tags[f"{category_names[category]}: {tag}"] = score * 100
        
        # Skip if no tags
        if not model_tags:
            continue
        
        # Sort tags by name
        sorted_tags = sorted(model_tags.keys())
        sorted_scores = [model_tags[tag] for tag in sorted_tags]
        
        # Create radar plot
        angles = np.linspace(0, 2*np.pi, len(sorted_tags), endpoint=False).tolist()
        
        # Close the polygon
        scores = sorted_scores + [sorted_scores[0]]
        angles = angles + [angles[0]]
        tags = sorted_tags + [sorted_tags[0]]
        
        fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))
        ax.plot(angles, scores, 'o-', linewidth=2)
        ax.fill(angles, scores, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), tags[:-1], fontsize=8)
        
        ax.set_ylim(0, 100)
        ax.set_title(f"{model} Performance by Tag", fontsize=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model}_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Create a detailed CSV report
    report_data = []
    
    for category, category_name in category_names.items():
        for tag in sorted(tag_counts[category].keys()):
            tag_count = tag_counts[category][tag]
            row = {'Category': category_name, 'Tag': tag, 'Task Count': tag_count}
            
            for model in all_models:
                if (model in category_results[category] and 
                    tag in category_results[category][model]):
                    row[model] = f"{category_results[category][model][tag]*100:.2f}%"
                else:
                    row[model] = "0.00%"
            
            report_data.append(row)
    
    # Add aggregate rows
    for category, category_name in category_names.items():
        row = {'Category': category_name, 'Tag': 'OVERALL', 'Task Count': sum(tag_counts[category].values())}
        
        for model in all_models:
            if model in aggregate_results[category]:
                row[model] = f"{aggregate_results[category][model]*100:.2f}%"
            else:
                row[model] = "0.00%"
        
        report_data.append(row)
    
    # Create and save the report
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(os.path.join(OUTPUT_DIR, 'tag_success_report.csv'), index=False)
    
    # Print summary
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")
    
    # Return aggregate results as a DataFrame for easy viewing
    agg_data = []
    for category, category_name in category_names.items():
        row = {'Category': category_name}
        for model in all_models:
            if model in aggregate_results[category]:
                row[model] = f"{aggregate_results[category][model]*100:.2f}%"
            else:
                row[model] = "0.00%"
        agg_data.append(row)
    
    return pd.DataFrame(agg_data)

def main():
    # Load data
    tags_data = load_tags()
    results_df = load_results()
    
    # Calculate success rates
    category_results, tag_counts = get_tag_success_rates(results_df, tags_data)
    aggregate_results = calculate_aggregate_success_rates(category_results, tag_counts)
    
    # Plot and get summary
    summary_df = plot_results(category_results, tag_counts, aggregate_results)
    
    # Print summary
    print("\nAggregate Success Rates by Category:")
    print(summary_df)

if __name__ == "__main__":
    main()
