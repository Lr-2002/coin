#!/usr/bin/env python3
"""
Analyze and visualize the distribution of tasks across reasoning subcategories.
This script calculates how many tasks fall into each subcategory of
object-centric, robot-centric, and compositional reasoning.

Usage: python analyze_subcategory_distribution.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS_FILE = os.path.join(BASE_DIR, "env_extended_tags.json")
CATEGORIES_FILE = os.path.join(BASE_DIR, "github_page/static/reasoning_categories.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "github_page/static/subcategory_analysis")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load tags and category mapping data"""
    # Load environment tags
    with open(TAGS_FILE, 'r') as f:
        env_tags = json.load(f)
    
    # Load category mapping
    categories_df = pd.read_csv(CATEGORIES_FILE)
    
    return env_tags, categories_df

def create_tag_to_subcategory_map(categories_df):
    """Create a mapping from tag names to their subcategories"""
    tag_to_subcategory = {}
    for _, row in categories_df.iterrows():
        subcategory = row['SubCategory']
        # Replace 'N/A' with actual subcategory name for Compositional reasoning
        if subcategory == 'N/A' and row['MainCategory'] == 'Compositional':
            subcategory = 'Compositional Reasoning'
            
        tag_to_subcategory[row['LongName']] = {
            'MainCategory': row['MainCategory'],
            'SubCategory': subcategory
        }
    return tag_to_subcategory

def analyze_subcategory_distribution(env_tags, tag_to_subcategory):
    """
    Count tasks for each subcategory and generate distribution data
    Returns a dict with subcategory counts and a dataframe for plotting
    """
    # Initialize counters for each subcategory
    subcategory_counts = defaultdict(int)
    category_subcategory_counts = defaultdict(lambda: defaultdict(int))
    task_subcategory_map = defaultdict(set)
    
    # Define explicit compositional subcategories for better organization
    compositional_subcategories = {
        'Tool-mediated problem solving': 'Tool Usage',
        'Failure-driven adaptation': 'Adaptation & Planning',
        'Hierarchical planning': 'Adaptation & Planning',
        'Experience utilization': 'Knowledge Utilization'
    }
    
    # Count tasks per subcategory
    for task_name, tags in env_tags.items():
        # For each category (obj, rob, iter)
        for category_key, tag_list in tags.items():
            category_name = {
                'obj': 'Object-Centric',
                'rob': 'Robot-Centric',
                'iter': 'Compositional'
            }.get(category_key)
            
            if not category_name or not tag_list:
                continue
            
            # Track which subcategories this task belongs to
            for tag in tag_list:
                if tag in tag_to_subcategory:
                    if category_name == 'Compositional':
                        # Use our explicit mapping for compositional reasoning
                        subcategory = compositional_subcategories.get(tag, 'Other')
                    else:
                        subcategory = tag_to_subcategory[tag]['SubCategory']
                        
                    if subcategory and subcategory != 'N/A':
                        subcategory_counts[subcategory] += 1
                        category_subcategory_counts[category_name][subcategory] += 1
                        task_subcategory_map[task_name].add(f"{category_name}:{subcategory}")
    
    # Convert to DataFrame for visualization
    subcategory_data = []
    for category, subcategory_dict in category_subcategory_counts.items():
        for subcategory, count in subcategory_dict.items():
            subcategory_data.append({
                'MainCategory': category,
                'SubCategory': subcategory,
                'Count': count
            })
    
    subcategory_df = pd.DataFrame(subcategory_data)
    
    # Calculate the number of unique tasks per subcategory
    unique_task_counts = {}
    for subcategory in subcategory_counts.keys():
        tasks_with_subcategory = sum(1 for task_subcats in task_subcategory_map.values() 
                                    if any(str(subcategory) in str(subcat) for subcat in task_subcats))
        unique_task_counts[subcategory] = tasks_with_subcategory
    
    return {
        'subcategory_counts': dict(subcategory_counts),
        'unique_task_counts': unique_task_counts,
        'category_subcategory_counts': dict(category_subcategory_counts),
        'subcategory_df': subcategory_df,
        'task_count': len(env_tags)
    }

def plot_subcategory_distribution(analysis_results):
    """Create visualizations for subcategory distribution"""
    subcategory_df = analysis_results['subcategory_df']
    task_count = analysis_results['task_count']
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.size'] = 11
    
    # Create color palette keyed by MainCategory
    palette = {
        'Object-Centric': '#8FB339',
        'Robot-Centric': '#3C8031',
        'Compositional': '#BBD8A3'
    }
    
    # 1. Bar chart of subcategory counts
    plt.figure(figsize=(14, 10))
    g = sns.barplot(
        data=subcategory_df, 
        x='Count', 
        y='SubCategory', 
        hue='MainCategory',
        palette=palette,
        dodge=False
    )
    
    # Add count annotations to bars
    for i, bar in enumerate(g.patches):
        g.text(
            bar.get_width() + 0.5, 
            bar.get_y() + bar.get_height()/2, 
            f"{int(bar.get_width())} tasks", 
            ha='left', 
            va='center', 
            fontsize=11
        )
    
    plt.title('Task Distribution by Reasoning Subcategory', fontsize=16)
    plt.xlabel('Number of Tasks', fontsize=14)
    plt.ylabel('Subcategory', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'subcategory_distribution.png'), dpi=300)
    
    # 2. Percentage breakdown by main category with subcategories as stacked bars
    pivoted = subcategory_df.pivot(index='MainCategory', columns='SubCategory', values='Count').fillna(0)
    proportions = pivoted.div(pivoted.sum(axis=1), axis=0)
    
    plt.figure(figsize=(14, 8))
    proportions.plot(
        kind='bar', 
        stacked=True, 
        colormap='viridis',
        figsize=(14, 8)
    )
    plt.title('Proportional Distribution of Subcategories within Each Main Category', fontsize=16)
    plt.xlabel('Main Category', fontsize=14)
    plt.ylabel('Proportion', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.legend(title='Subcategory', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'subcategory_proportions.png'), dpi=300)
    
    # 3. Treemap of subcategories
    try:
        import squarify
        
        # Prepare data for treemap
        categories = []
        sizes = []
        colors = []
        labels = []
        
        for _, row in subcategory_df.iterrows():
            categories.append(f"{row['MainCategory']}: {row['SubCategory']}")
            sizes.append(row['Count'])
            colors.append(palette.get(row['MainCategory'], '#333333'))
            labels.append(f"{row['SubCategory']}\n({row['Count']} tasks)")
        
        plt.figure(figsize=(16, 12))
        squarify.plot(
            sizes=sizes, 
            label=labels, 
            color=colors,
            alpha=0.8,
            text_kwargs={'fontsize': 12}
        )
        plt.axis('off')
        plt.title('Task Distribution by Reasoning Subcategory (Treemap)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'subcategory_treemap.png'), dpi=300)
    except ImportError:
        print("squarify package not available, skipping treemap visualization")
    
    # Save a CSV with the subcategory counts
    subcategory_df['Percentage'] = subcategory_df['Count'] / task_count * 100
    subcategory_df.to_csv(os.path.join(OUTPUT_DIR, 'subcategory_distribution.csv'), index=False)
    
    print(f"Visualizations saved to {OUTPUT_DIR}")

def generate_subcategory_report(analysis_results):
    """Generate a detailed report of subcategory distribution"""
    subcategory_counts = analysis_results['subcategory_counts']
    task_count = analysis_results['task_count']
    category_subcategory_counts = analysis_results['category_subcategory_counts']
    
    report = []
    report.append("# Reasoning Subcategory Distribution Report")
    report.append(f"\nTotal tasks analyzed: {task_count}\n")
    
    # Add summary section
    report.append("## Summary of Subcategory Distribution")
    report.append("\nTop subcategories by task count:")
    
    # Sort subcategories by count
    sorted_subcategories = sorted(subcategory_counts.items(), key=lambda x: x[1], reverse=True)
    for subcategory, count in sorted_subcategories[:5]:
        percentage = (count / task_count) * 100
        report.append(f"- {subcategory}: {count} tasks ({percentage:.1f}%)")
    
    # Add detailed breakdown by main category
    report.append("\n## Detailed Breakdown by Main Category")
    
    for category, subcategory_dict in category_subcategory_counts.items():
        report.append(f"\n### {category}")
        category_total = sum(subcategory_dict.values())
        
        # Sort subcategories within this category
        sorted_subcats = sorted(subcategory_dict.items(), key=lambda x: x[1], reverse=True)
        for subcategory, count in sorted_subcats:
            percentage = (count / category_total) * 100
            report.append(f"- {subcategory}: {count} tasks ({percentage:.1f}%)")
    
    # Write report to file
    report_path = os.path.join(OUTPUT_DIR, 'subcategory_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to {report_path}")

def main():
    """Main function to run the analysis"""
    # Load data
    env_tags, categories_df = load_data()
    
    # Create mapping from tag names to subcategories
    tag_to_subcategory = create_tag_to_subcategory_map(categories_df)
    
    # Analyze subcategory distribution
    analysis_results = analyze_subcategory_distribution(env_tags, tag_to_subcategory)
    
    # Plot subcategory distribution
    plot_subcategory_distribution(analysis_results)
    
    # Generate detailed report
    generate_subcategory_report(analysis_results)
    
    print("Subcategory distribution analysis complete.")

if __name__ == "__main__":
    main()
