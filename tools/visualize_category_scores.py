#!/usr/bin/env python3
"""
Create visualizations for model category scores.

Usage: python visualize_category_scores.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_DIR = os.path.join(BASE_DIR, "github_page/static/category_scores")
OUTPUT_DIR = SCORES_DIR

def load_category_scores():
    """Load the category scores"""
    scores_file = os.path.join(SCORES_DIR, "model_category_scores.csv")
    df = pd.read_csv(scores_file)
    return df

def clean_model_names(df):
    """Clean model names for better display"""
    # Create a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Create shortened model names for display
    model_map = {
        'Rekep': 'Rekep',
        'Voxposer_normal': 'Voxposer (N)',
        'Voxposer_topdown': 'Voxposer (TD)',
        '20250509_cogact_30000_gemini_10_400_no_history_image': 'CogAct-Gemini',
        '20250509_cogact_30000_gpt4o_10_400_no_history_image': 'CogAct-GPT4o',
        '20250509_gr00t_120000_gemini_10_400_no_history_image': 'Gr00t-Gemini',
        '20250511_gr00t_120000_gpt4o_10_400_no_history_image': 'Gr00t-GPT4o',
        '20250511_pi0_470000_gpt4o_10_400_no_history_image': 'Pi0-GPT4o',
    }
    
    # Create a new column with short names
    clean_df['ShortName'] = clean_df['Model'].map(model_map)
    
    # Convert string percentages to float
    for col in ['Object-Centric', 'Robot-Centric', 'Compositional']:
        clean_df[col] = clean_df[col].str.rstrip('%').astype(float)
    
    return clean_df

def create_bar_chart(df):
    """Create a grouped bar chart for category scores"""
    # Prepare the data for plotting
    categories = ['Object-Centric', 'Robot-Centric', 'Compositional']
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate bar positions
    x = np.arange(len(df['ShortName']))
    width = 0.25
    
    # Plot each category as a set of bars
    for i, category in enumerate(categories):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[category], width, label=category)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}%', ha='center', va='bottom', rotation=0)
    
    # Add labels and title
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Model Performance by Reasoning Category')
    ax.set_xticks(x)
    ax.set_xticklabels(df['ShortName'], rotation=45, ha='right')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    
    # Add legend
    ax.legend(title='Category Type')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'category_scores_bar.png')
    plt.savefig(output_path, dpi=300)
    print(f"Bar chart saved to {output_path}")
    
    return output_path

def create_heatmap(df):
    """Create a heatmap for category scores"""
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Create a pivot for the heatmap
    heatmap_data = df.pivot_table(index='ShortName', values=['Object-Centric', 'Robot-Centric', 'Compositional'])
    
    # Create the heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=5,
                    cbar_kws={'label': 'Success Rate (%)'})
    
    # Add title
    plt.title('Model Performance by Reasoning Category')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'category_scores_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    
    return output_path

def create_radar_chart(df):
    """Create a simple radar chart for category scores"""
    # Prepare data for the radar chart
    categories = ['Object-Centric', 'Robot-Centric', 'Compositional']
    N = len(categories)
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Set the y-axis limits
    max_value = df[categories].max().max() * 1.1  # 10% margin
    ax.set_ylim(0, max_value)
    
    # Add grid lines and set them to be circular
    ax.grid(True)
    
    # Plot each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[cat] for cat in categories]
        values += values[:1]  # Close the loop
        
        # Plot the values
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=row['ShortName'])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add title
    plt.title('Model Performance Across Reasoning Categories', size=15)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'category_scores_radar.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to {output_path}")
    
    return output_path

def main():
    # Load the data
    df = load_category_scores()
    
    # Clean model names
    clean_df = clean_model_names(df)
    
    # Create visualizations
    bar_chart_path = create_bar_chart(clean_df)
    heatmap_path = create_heatmap(clean_df)
    radar_chart_path = create_radar_chart(clean_df)
    
    print("\nVisualizations created:")
    print(f"1. Bar Chart: {bar_chart_path}")
    print(f"2. Heatmap: {heatmap_path}")
    print(f"3. Radar Chart: {radar_chart_path}")

if __name__ == "__main__":
    main()
