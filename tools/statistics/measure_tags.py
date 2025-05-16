#!/usr/bin/env python3

import json
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Path to the extended tags JSON file
EXTENDED_TAGS_FILE = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/env_extended_tags.json"

def load_extended_tags():
    """Load the extended tags from the JSON file"""
    with open(EXTENDED_TAGS_FILE, 'r') as f:
        return json.load(f)

def analyze_tag_distribution(extended_tags):
    """Analyze the distribution of tags across environments"""
    # Initialize counters for the three main categories
    category_counts = Counter()
    
    # Initialize dictionaries to track which environments have which tags
    env_tags = defaultdict(list)
    
    # Initialize counter for specific tags within each category
    specific_tag_counts = defaultdict(Counter)
    
    # Count occurrences of each tag category and specific tag
    for env_id, tags in extended_tags.items():
        for category, tag_list in tags.items():
            if tag_list:  # Only count if there are tags in this category
                category_counts[category] += 1
                env_tags[category].append(env_id)
                
                # Count each specific tag
                for tag in tag_list:
                    specific_tag_counts[category][tag] += 1
    
    return category_counts, specific_tag_counts, env_tags

def save_category_distribution(category_counts):
    """Save the distribution of main categories to a CSV file"""
    # Create a DataFrame with the category counts
    df = pd.DataFrame({
        'Category': list(category_counts.keys()),
        'Count': list(category_counts.values())
    })
    
    # Calculate the percentage
    total_envs = sum(category_counts.values())
    df['Percentage'] = df['Count'] / total_envs * 100
    
    # Sort by count in descending order
    df = df.sort_values('Count', ascending=False)
    
    # Save to CSV
    df.to_csv('github_page/static/tag_category_distribution.csv', index=False)
    print(f"Saved category distribution to tag_category_distribution.csv")
    
    return df

def save_specific_tag_distribution(specific_tag_counts):
    """Save the distribution of specific tags within each category to CSV files"""
    for category, tag_counter in specific_tag_counts.items():
        # Create a DataFrame with the tag counts for this category
        df = pd.DataFrame({
            'Tag': list(tag_counter.keys()),
            'Count': list(tag_counter.values())
        })
        
        # Calculate the percentage
        total_tags = sum(tag_counter.values())
        df['Percentage'] = df['Count'] / total_tags * 100
        
        # Sort by count in descending order
        df = df.sort_values('Count', ascending=False)
        
        # Save to CSV
        filename = f'github_page/static/tags/tag_{category}_distribution.csv'
        df.to_csv(filename, index=False)
        print(f"Saved {category} tag distribution to {filename}")

def create_visualizations(category_counts, specific_tag_counts):
    """Create visualizations of the tag distributions"""
    # Create a pie chart for the main categories
    plt.figure(figsize=(10, 6))
    plt.pie(
        list(category_counts.values()),
        labels=list(category_counts.keys()),
        autopct='%1.1f%%',
        startangle=90,
        shadow=True
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Tag Categories')
    plt.savefig('tag_category_distribution.png', bbox_inches='tight', transparent=True)
    plt.close()
    
    # Create bar charts for each category's specific tags
    for category, tag_counter in specific_tag_counts.items():
        # Sort the tags by count
        sorted_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)
        tags, counts = zip(*sorted_tags) if sorted_tags else ([], [])
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(tags, counts, color='#BF9264')
        
        # Add count labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}',
                    ha='center', va='bottom')
        
        plt.xlabel('Tags')
        plt.ylabel('Count')
        plt.title(f'Distribution of {category} Tags')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'github_page/static/plots/tag_{category}_distribution.png', bbox_inches='tight', transparent=True)
        plt.close()

def main():
    # Load the extended tags
    extended_tags = load_extended_tags()
    
    if not extended_tags:
        print("No extended tags found in the JSON file.")
        return
    
    # Analyze the tag distribution
    category_counts, specific_tag_counts, env_tags = analyze_tag_distribution(extended_tags)
    
    # Save the distributions to CSV files
    save_category_distribution(category_counts)
    save_specific_tag_distribution(specific_tag_counts)
    
    # Create visualizations
    create_visualizations(category_counts, specific_tag_counts)
    
    # Print summary
    print("\nSummary:")
    print(f"Total environments with tags: {len(extended_tags)}")
    print("\nCategory counts:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} environments")
    
    print("\nSpecific tag counts:")
    for category, tag_counter in specific_tag_counts.items():
        print(f"\n  {category} tags:")
        for tag, count in sorted(tag_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"    {tag}: {count} environments")

if __name__ == "__main__":
    main()
