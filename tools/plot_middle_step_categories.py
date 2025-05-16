#!/usr/bin/env python3

import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")

# Radar chart functionality
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """
            Override fill so that line is closed by default
            """
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """
            Override plot so that line is closed by default
            """
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                     radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    # register the custom projection
    register_projection(RadarAxes)
    return theta
sns.set_context("talk")

# Base paths
STATIC_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static"
NORMALIZED_DIR = os.path.join(STATIC_DIR, "normalized_scores")
MIDDLE_STEP_DIR = os.path.join(STATIC_DIR, "middle_step_plan_compare")
TAGS_FILE = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/env_extended_tags.json"

# Ensure output directory exists
os.makedirs(MIDDLE_STEP_DIR, exist_ok=True)

def load_tags():
    """Load task categories from the tags JSON file"""
    with open(TAGS_FILE, 'r') as f:
        tags_data = json.load(f)
    return tags_data

def normalize_task_name(task_name, to_format='hyphen'):
    """Convert task name between formats with hyphens and underscores"""
    if to_format == 'hyphen':
        # Convert from underscore to hyphen
        return task_name.replace('_', '-')
    else:
        # Convert from hyphen to underscore
        return task_name.replace('-', '_')

def get_tasks_per_category(tags_data):
    """Count how many tasks belong to each category"""
    category_counts = defaultdict(int)
    task_categories = defaultdict(list)
    
    for task_name, task_tags in tags_data.items():
        # Also store underscore version of the task name for matching with normalized scores
        underscore_task_name = normalize_task_name(task_name, to_format='underscore')
        
        # Process object-centric categories
        if 'obj' in task_tags:
            for obj_tag in task_tags['obj']:
                category_counts[obj_tag] += 1
                task_categories[obj_tag].append(underscore_task_name)
        
        # Process robot-centric categories
        if 'rob' in task_tags:
            for rob_tag in task_tags['rob']:
                category_counts[rob_tag] += 1
                task_categories[rob_tag].append(underscore_task_name)
                
        # Process interactive categories
        if 'iter' in task_tags:
            for iter_tag in task_tags['iter']:
                category_counts[iter_tag] += 1
                task_categories[iter_tag].append(underscore_task_name)
    
    return category_counts, task_categories

def get_valid_categories(min_tasks=3):
    """Get categories that have at least the minimum number of tasks"""
    tags_data = load_tags()
    counts, task_categories = get_tasks_per_category(tags_data)
    
    # Filter categories with at least min_tasks tasks
    valid_categories = {cat: count for cat, count in counts.items() if count >= min_tasks}
    valid_task_categories = {cat: tasks for cat, tasks in task_categories.items() if cat in valid_categories}
    
    return valid_categories, valid_task_categories

def load_normalized_scores(model_name):
    """Load normalized scores for a model"""
    clean_name = model_name.replace('-', '')
    input_path = os.path.join(NORMALIZED_DIR, f"{clean_name}-normalized-score.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Normalized score file not found: {input_path}")
    
    # Read normalized scores
    df = pd.read_csv(input_path)
    return df

def compare_categories_middle_steps(models, categories, metric='mean_score', min_tasks=3):
    """
    Compare middle steps progression for specific categories across models
    
    Args:
        models: List of model names
        categories: List of category names to compare
        metric: Score metric to compare (default: 'mean_score')
    """
    # Get valid categories and their tasks
    valid_categories, task_categories = get_valid_categories(min_tasks)
    
    # Filter to only include specified categories that are valid
    valid_specified_categories = [cat for cat in categories if cat in valid_categories]
    
    if not valid_specified_categories:
        print(f"None of the specified categories have at least {min_tasks} tasks.")
        return
    
    # For each valid category, generate a comparison plot
    for category in valid_specified_categories:
        # Get tasks for this category
        category_tasks = task_categories[category]
        
        print(f"Generating middle step comparison for category: {category} ({len(category_tasks)} tasks)")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Line styles and colors for different models
        line_styles = ['-', '--', '-.', ':']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        # Store model data for legend
        model_lines = []
        model_names = []
        max_step = 9  # Default max step
        
        # Plot each model
        for i, model_name in enumerate(models):
            try:
                # Load normalized scores
                df = load_normalized_scores(model_name)
                
                # Ensure we have the metric column
                if metric not in df.columns:
                    print(f"Metric {metric} not found for model {model_name}")
                    continue
                
                # Filter to only include tasks for this category
                df_category = df[df['task'].isin(category_tasks)]
                
                if df_category.empty:
                    print(f"No data for category {category} in model {model_name}")
                    continue
                
                # Exclude first (0) and last step
                curr_max_step = df_category['step'].max()
                max_step = min(max_step, curr_max_step) if max_step != 9 else curr_max_step
                df_filtered = df_category[(df_category['step'] > 0) & (df_category['step'] < curr_max_step)]
                
                # Group by step and calculate average
                step_avg = df_filtered.groupby('step')[metric].mean().reset_index()
                
                # Plot average trend line
                line, = plt.plot(
                    step_avg['step'], 
                    step_avg[metric], 
                    marker='o',
                    linestyle=line_styles[i % len(line_styles)],
                    color=colors[i % len(colors)],
                    linewidth=3, 
                    label=model_name
                )
                
                model_lines.append(line)
                model_names.append(model_name)
                
                # Add confidence interval
                step_std = df_filtered.groupby('step')[metric].std().reset_index()
                plt.fill_between(
                    step_avg['step'],
                    step_avg[metric] - step_std[metric],
                    step_avg[metric] + step_std[metric],
                    alpha=0.1,
                    color=colors[i % len(colors)]
                )
            except Exception as e:
                print(f"Error processing {model_name} for category {category}: {e}")
        
        # Set labels and title
        plt.xlabel(f'Normalized Step (1-{max_step-1})', fontsize=14)
        plt.ylabel(f'{metric.replace("_", " ").title()} (1-10)', fontsize=14)
        plt.title(f'Category: {category}\n{metric.replace("_", " ").title()} Progression (Middle Steps Only)', fontsize=16)
        
        # Set axis limits
        plt.xlim(0.5, max_step - 0.5)
        plt.ylim(0, 10.5)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add score reference lines
        for score in range(1, 11, 2):
            plt.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
        
        # Add legend
        if model_lines:
            plt.legend(model_lines, model_names, loc='upper right')
        
        # Save figure
        category_filename = category.replace(' ', '_').lower()
        output_path = os.path.join(MIDDLE_STEP_DIR, f"category_{category_filename}_{metric}_middle_steps.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved category plot to {output_path}")

def compare_all_categories_middle_steps(models, metric='mean_score', min_tasks=3):
    """
    Compare middle steps progression for all valid categories across models
    
    Args:
        models: List of model names
        metric: Score metric to compare (default: 'mean_score')
        min_tasks: Minimum number of tasks required for a category
    """
    # Get valid categories and their tasks
    valid_categories, _ = get_valid_categories(min_tasks)
    
    # Compare all valid categories
    compare_categories_middle_steps(models, list(valid_categories.keys()), metric, min_tasks)

def plot_category_comparison(models, category_type=None, category=None, min_tasks=3):
    """
    Generate direct comparisons of category scores between models
    
    Args:
        models: List of model names to compare
        category_type: Category type ('obj', 'rob', or 'iter') to focus on
        category: Specific category to compare (if provided)
        min_tasks: Minimum number of tasks for a category to be included
    """
    # Valid category types
    valid_types = {'obj': 'Object-Centric', 'rob': 'Robot-Centric', 'iter': 'Interactive'}
    
    # Ensure valid category type
    if category_type and category_type not in valid_types:
        print(f"Invalid category type: {category_type}. Must be one of: {', '.join(valid_types.keys())}")
        return None
    
    # Line styles and colors for different models
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Title based on category
    if category:
        title = f"Model Comparison: {category}"
    elif category_type:
        title = f"Model Comparison: {valid_types[category_type]} Categories"
    else:
        title = "Model Comparison: All Categories"
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get valid categories
    valid_categories, _ = get_valid_categories(min_tasks)
    
    # Store model data for legend
    model_bars = []
    model_names = []
    
    # Process each model
    model_data = {}
    all_categories = set()
    
    for model_name in models:
        clean_name = model_name.replace('-', '')
        
        # If category specified, find which type it belongs to
        if category:
            # Check each category type file for the specified category
            found = False
            for cat_type in valid_types.keys():
                cat_file = os.path.join(STATIC_DIR, f"{clean_name}_category_{cat_type}.csv")
                if os.path.exists(cat_file):
                    cat_df = pd.read_csv(cat_file)
                    if category in cat_df['category'].values:
                        cat_df = cat_df[cat_df['category'] == category]
                        model_data[model_name] = cat_df
                        found = True
                        all_categories.add(category)
                        break
            
            if not found:
                print(f"Category '{category}' not found for model '{model_name}'")
        
        # If category type specified, load that file
        elif category_type:
            cat_file = os.path.join(STATIC_DIR, f"{clean_name}_category_{category_type}.csv")
            if os.path.exists(cat_file):
                cat_df = pd.read_csv(cat_file)
                # Filter to only valid categories (with min_tasks)
                cat_df = cat_df[cat_df['category'].isin(valid_categories)]
                if not cat_df.empty:
                    model_data[model_name] = cat_df
                    all_categories.update(cat_df['category'].tolist())
                else:
                    print(f"No valid categories for {valid_types[category_type]} in model {model_name}")
            else:
                print(f"Category file not found: {cat_file}")
        
        # Otherwise, combine all category files
        else:
            model_categories = []
            for cat_type in valid_types.keys():
                cat_file = os.path.join(STATIC_DIR, f"{clean_name}_category_{cat_type}.csv")
                if os.path.exists(cat_file):
                    cat_df = pd.read_csv(cat_file)
                    # Filter to only valid categories (with min_tasks)
                    cat_df = cat_df[cat_df['category'].isin(valid_categories)]
                    if not cat_df.empty:
                        model_categories.append(cat_df)
                        all_categories.update(cat_df['category'].tolist())
            
            if model_categories:
                model_data[model_name] = pd.concat(model_categories)
    
    # Convert to list and sort alphabetically
    all_categories = sorted(list(all_categories))
    
    if not all_categories:
        print("No categories found for the specified models")
        plt.close()
        return None
    
    # Set up bar positions
    x = np.arange(len(all_categories))
    width = 0.8 / len(models)  # Adjust width based on number of models
    
    # Plot bars for each model
    for i, (model_name, data) in enumerate(model_data.items()):
        # Get scores for each category
        scores = []
        for cat in all_categories:
            cat_scores = data[data['category'] == cat]['mean_score'].values
            scores.append(cat_scores[0] if len(cat_scores) > 0 else 0)
        
        # Plot bars
        pos = x - 0.4 + (i + 0.5) * width  # Position bars side by side
        bars = plt.bar(pos, scores, width, label=model_name, color=colors[i % len(colors)])
        model_bars.append(bars)
        model_names.append(model_name)
    
    # Set labels and title
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Mean Score (1-10)', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Set x-ticks at category positions
    plt.xticks(x, all_categories, rotation=45, ha='right')
    
    # Set y-axis limits
    plt.ylim(0, 10.5)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add score reference lines
    for score in range(1, 11, 2):
        plt.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output path
    if category:
        cat_filename = category.replace(' ', '_').lower()
        output_path = os.path.join(MIDDLE_STEP_DIR, f"category_{cat_filename}_comparison.png")
    elif category_type:
        output_path = os.path.join(MIDDLE_STEP_DIR, f"{category_type}_categories_comparison.png")
    else:
        output_path = os.path.join(MIDDLE_STEP_DIR, f"all_categories_comparison.png")
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    print(f"Saved category comparison to {output_path}")
    
    return output_path

def compare_category_types_box(models, metric='mean_score', min_tasks=3):
    """
    Generate box plots comparing models across each category type
    
    Args:
        models: List of model names
        metric: Score metric to compare (default: 'mean_score')
        min_tasks: Minimum number of tasks required for a category
    """
    # Get model data for each category type
    category_types = {
        'Object-Centric': [],
        'Robot-Centric': [],
        'Interactive': []
    }
    
    # Load category files for each model
    for model_name in models:
        clean_name = model_name.replace('-', '')
        
        # Try to load each category type file
        for cat_type, prefix in [
            ('Object-Centric', 'obj'), 
            ('Robot-Centric', 'rob'), 
            ('Interactive', 'iter')
        ]:
            file_path = os.path.join(STATIC_DIR, f"{clean_name}_category_{prefix}.csv")
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                # Add model name column
                df['model'] = model_name
                
                category_types[cat_type].append(df)
    
    # Generate comparison plots for each category type
    for cat_type, dataframes in category_types.items():
        if not dataframes:
            print(f"No data found for category type: {cat_type}")
            continue
        
        # Combine dataframes
        combined_df = pd.concat(dataframes)
        
        # Filter categories with at least min_tasks
        valid_cats, _ = get_valid_categories(min_tasks)
        combined_df = combined_df[combined_df['category'].isin(valid_cats)]
        
        if combined_df.empty:
            print(f"No valid categories with {min_tasks}+ tasks found for {cat_type}")
            continue
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Set color palette
        palette = sns.color_palette("husl", len(models))
        
        # Create boxplot
        sns.boxplot(
            x='model', 
            y='mean_score', 
            data=combined_df, 
            palette=palette,
            width=0.5
        )
        
        # Set labels and title
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Mean Score (1-10)', fontsize=14)
        plt.title(f'{cat_type} Categories Comparison\n(Only categories with {min_tasks}+ tasks)', fontsize=16)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # Set y-axis limits
        plt.ylim(0, 10.5)
        
        # Add reference lines
        for score in range(1, 11, 2):
            plt.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
        
        # Save figure
        cat_type_filename = cat_type.lower().replace('-', '_').replace(' ', '_')
        output_path = os.path.join(MIDDLE_STEP_DIR, f"{cat_type_filename}_boxplot.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved {cat_type} boxplot comparison to {output_path}")
        
        # Also generate swarm plot for better data visualization
        plt.figure(figsize=(14, 10))
        
        # Plot individual category points
        sns.swarmplot(
            x='model',
            y='mean_score',
            data=combined_df,
            hue='category',
            size=8,
            alpha=0.7
        )
        
        # Set labels and title
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Mean Score (1-10)', fontsize=14)
        plt.title(f'{cat_type} Individual Category Scores\n(Only categories with {min_tasks}+ tasks)', fontsize=16)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # Set y-axis limits
        plt.ylim(0, 10.5)
        
        # Adjust legend to be outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        # Add reference lines
        for score in range(1, 11, 2):
            plt.axhline(y=score, color='gray', linestyle=':', alpha=0.3)
        
        # Save figure
        output_path = os.path.join(MIDDLE_STEP_DIR, f"{cat_type_filename}_swarmplot.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {cat_type} swarmplot to {output_path}")

def plot_radar_chart(models, category_type=None, min_tasks=3):
    """
    Generate radar/spider chart comparing models across categories
    
    Args:
        models: List of model names to compare
        category_type: Category type ('obj', 'rob', or 'iter')
        min_tasks: Minimum number of tasks for a category to be included
    """
    # Valid category types
    valid_types = {'obj': 'Object-Centric', 'rob': 'Robot-Centric', 'iter': 'Interactive'}
    
    # Ensure valid category type
    if category_type and category_type not in valid_types:
        print(f"Invalid category type: {category_type}. Must be one of: {', '.join(valid_types.keys())}")
        return None
    
    # Get valid categories
    valid_categories, _ = get_valid_categories(min_tasks)
    
    # Colors for different models
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Process each model
    model_data = {}
    all_categories = set()
    
    for model_name in models:
        clean_name = model_name.replace('-', '')
        model_categories = []
        
        # If category type specified, load that file
        if category_type:
            cat_file = os.path.join(STATIC_DIR, f"{clean_name}_category_{category_type}.csv")
            if os.path.exists(cat_file):
                cat_df = pd.read_csv(cat_file)
                # Filter to only valid categories (with min_tasks)
                cat_df = cat_df[cat_df['category'].isin(valid_categories)]
                if not cat_df.empty:
                    model_categories.append(cat_df)
                    all_categories.update(cat_df['category'].tolist())
                else:
                    print(f"No valid categories for {valid_types[category_type]} in model {model_name}")
            else:
                print(f"Category file not found: {cat_file}")
        
        # Otherwise, combine all category files
        else:
            for cat_type in valid_types.keys():
                cat_file = os.path.join(STATIC_DIR, f"{clean_name}_category_{cat_type}.csv")
                if os.path.exists(cat_file):
                    cat_df = pd.read_csv(cat_file)
                    # Filter to only valid categories (with min_tasks)
                    cat_df = cat_df[cat_df['category'].isin(valid_categories)]
                    if not cat_df.empty:
                        model_categories.append(cat_df)
                        all_categories.update(cat_df['category'].tolist())
        
        if model_categories:
            # Combine all category data for this model
            model_df = pd.concat(model_categories)
            # Calculate average score across steps 1-8 (excluding first and last steps if there are more)
            model_data[model_name] = model_df
    
    # Convert to list and sort alphabetically
    all_categories = sorted(list(all_categories))
    
    if not all_categories:
        print("No categories found for the specified models")
        return None
    
    # Create radar chart
    theta = radar_factory(len(all_categories), frame='polygon')
    
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    
    # Title based on category type
    if category_type:
        title = f"Model Comparison: {valid_types[category_type]} Categories"
    else:
        title = "Model Comparison: All Categories"
    
    ax.set_title(title, weight='bold', size=18, position=(0.5, 1.1),
                horizontalalignment='center', verticalalignment='center')
    
    # Plot each model on the radar chart
    for i, (model_name, data) in enumerate(model_data.items()):
        # Get scores for each category
        scores = []
        for cat in all_categories:
            cat_scores = data[data['category'] == cat]['mean_score'].values
            scores.append(cat_scores[0] if len(cat_scores) > 0 else 0)
        
        # Plot the model data
        line = ax.plot(theta, scores, color=colors[i % len(colors)], label=model_name, linewidth=3)
        ax.fill(theta, scores, alpha=0.25, color=colors[i % len(colors)])
    
    # Set the labels and limits
    ax.set_varlabels(all_categories)
    ax.set_ylim(0, 10)  # Scores range from 0-10
    
    # Add cleaner reference circles - just a few key ones
    for score in [2, 4, 6, 8]:
        circle = plt.Circle((0, 0), score, fill=False, 
                           edgecolor='gray', linestyle='--', alpha=0.2)
        ax.add_artist(circle)
    
    # Add cleaner score labels
    for score in [2, 4, 6, 8]:
        ax.text(0, score, str(score), ha='center', va='bottom', 
                color='gray', alpha=0.5, fontsize=10)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Create output path
    if category_type:
        output_path = os.path.join(MIDDLE_STEP_DIR, f"{category_type}_radar_chart.png")
    else:
        output_path = os.path.join(MIDDLE_STEP_DIR, "all_categories_radar_chart.png")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar chart to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Compare model performance across categories.')
    parser.add_argument(
        '--models', nargs='+', required=True, help='List of models to compare'
    )
    parser.add_argument(
        '--category', help='Specific category to focus on'
    )
    parser.add_argument(
        '--obj', action='store_true', help='Compare object-centric categories'
    )
    parser.add_argument(
        '--rob', action='store_true', help='Compare robot-centric categories'
    )
    parser.add_argument(
        '--iter', action='store_true', help='Compare interactive categories'
    )
    parser.add_argument(
        '--all', action='store_true', help='Compare all category types'
    )
    parser.add_argument(
        '--box', action='store_true', help='Generate boxplots for category types'
    )
    parser.add_argument(
        '--swarm', action='store_true', help='Generate swarmplots for category values'
    )
    parser.add_argument(
        '--radar', action='store_true', help='Generate radar charts for category comparisons'
    )
    parser.add_argument(
        '--metric', default='mean_score', help='Metric to compare (default: mean_score)'
    )
    parser.add_argument(
        '--min-tasks', type=int, default=3, help='Minimum tasks for a category to be included'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(MIDDLE_STEP_DIR, exist_ok=True)
    
    # Get valid categories with task count
    valid_categories, _ = get_valid_categories(args.min_tasks)
    
    print(f"Found {len(valid_categories)} categories with at least {args.min_tasks} tasks each")
    for category, count in valid_categories.items():
        print(f"  - {category}: {count} tasks")
    
    # Process all arguments
    if args.all:
        args.obj = args.rob = args.iter = True
    
    # Generate specific visualizations based on arguments
    if args.radar:
        # Generate radar charts
        if args.obj:
            plot_radar_chart(args.models, category_type='obj', min_tasks=args.min_tasks)
        if args.rob:
            plot_radar_chart(args.models, category_type='rob', min_tasks=args.min_tasks)
        if args.iter:
            plot_radar_chart(args.models, category_type='iter', min_tasks=args.min_tasks)
        
        # If no specific category type, create an overall radar chart
        if not (args.obj or args.rob or args.iter):
            plot_radar_chart(args.models, min_tasks=args.min_tasks)
    
    elif args.box:
        # Box plots for each category type
        compare_category_types_box(args.models, args.metric, args.min_tasks)
    
    # swarm plotting option temporarily removed
    
    elif args.obj or args.rob or args.iter:
        # Generate specific category type bar plots
        if args.obj:
            plot_category_comparison(args.models, category_type='obj')
        if args.rob:
            plot_category_comparison(args.models, category_type='rob')
        if args.iter:
            plot_category_comparison(args.models, category_type='iter')
    
    elif args.category:
        # Generate plot for specific category
        plot_category_comparison(args.models, category=args.category)
    
    else:
        # Generate overall comparison
        plot_category_comparison(args.models)

if __name__ == "__main__":
    main()
