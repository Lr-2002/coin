#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import argparse
from scipy.interpolate import interp1d
from pathlib import Path

# Base paths
STATIC_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static"
NORMALIZED_DIR = os.path.join(STATIC_DIR, "normalized_scores")

def interpolate_task_scores(task_data, num_steps=10):
    """
    Interpolate task scores to a fixed number of steps
    
    Args:
        task_data: DataFrame containing scores for a single task
        num_steps: Number of steps to interpolate to (default: 10)
    
    Returns:
        DataFrame with interpolated scores for the fixed number of steps
    """
    # Sort by step to ensure proper interpolation
    task_data = task_data.sort_values('step')
    
    # Get the actual number of steps
    actual_steps = len(task_data)
    
    # Skip interpolation if there's only one step
    if actual_steps == 1:
        # Duplicate the single row for all steps
        result = pd.DataFrame([task_data.iloc[0].to_dict()] * num_steps)
        result['step'] = np.arange(num_steps)
        return result
    
    # Create original step indices (0 to actual_steps-1)
    x_orig = task_data['step'].astype(float).values
    
    # Create target step indices (0 to num_steps-1)
    x_target = np.linspace(x_orig.min(), x_orig.max(), num_steps)
    
    # List to store interpolated data rows
    interpolated_rows = []
    
    # Get task name for reference
    task_name = task_data['task'].iloc[0]
    
    # Define columns to interpolate (excluding 'task' and 'step')
    score_columns = [col for col in task_data.columns if col not in ['task', 'step']]
    
    # Perform interpolation for each score column
    for col in score_columns:
        # Get original y values
        y_orig = task_data[col].values
        
        # Create interpolation function
        # Use 'linear' for simple linear interpolation between points
        f = interp1d(x_orig, y_orig, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # Generate interpolated values
        y_interp = f(x_target)
        
        # Round to integers since scores are typically whole numbers
        y_interp = np.round(y_interp).astype(int)
        
        # Clip to valid score range (1-10)
        y_interp = np.clip(y_interp, 1, 10)
        
        # Add to interpolated data
        for i, step in enumerate(range(num_steps)):
            if i >= len(interpolated_rows):
                interpolated_rows.append({
                    'task': task_name,
                    'step': step
                })
            interpolated_rows[i][col] = y_interp[i]
    
    # Convert to DataFrame
    result_df = pd.DataFrame(interpolated_rows)
    
    return result_df

def normalize_model_scores(model_name, num_steps=10):
    """
    Normalize and interpolate scores for a specific model
    
    Args:
        model_name: Name of the model
        num_steps: Number of steps to interpolate to (default: 10)
    
    Returns:
        Path to the normalized scores CSV file
    """
    # Get path to original scores
    clean_name = model_name.replace('-', '')
    input_path = os.path.join(STATIC_DIR, f"{clean_name}-plan-score.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Score file not found: {input_path}")
    
    # Read original scores
    df = pd.read_csv(input_path)
    
    # Make sure the dataframe has the right headers
    if 'task' not in df.columns:
        df.columns = ['task', 'step', 'completeness', 'objects_correct', 
                      'sequence_correct', 'clarity', 'mean_score']
    
    # Convert steps to numeric
    df['step'] = pd.to_numeric(df['step'])
    
    # Create normalized directory if it doesn't exist
    os.makedirs(NORMALIZED_DIR, exist_ok=True)
    
    # List to store all normalized data
    all_normalized = []
    
    # Process each task separately
    for task_name, task_data in df.groupby('task'):
        print(f"Normalizing task: {task_name}")
        
        # Interpolate scores
        normalized_task = interpolate_task_scores(task_data, num_steps)
        
        # Add to combined results
        all_normalized.append(normalized_task)
    
    # Combine all normalized data
    normalized_df = pd.concat(all_normalized)
    
    # Save to CSV
    output_path = os.path.join(NORMALIZED_DIR, f"{clean_name}-normalized-score.csv")
    normalized_df.to_csv(output_path, index=False)
    
    print(f"Saved normalized scores to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Normalize and interpolate model scores to a fixed number of steps")
    parser.add_argument("--model", type=str, help="Model name to normalize")
    parser.add_argument("--models", type=str, nargs='+', help="Multiple models to normalize")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to normalize to")
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
            print(f"Processing normalized scores for {model_name}")
            normalize_model_scores(model_name, args.steps)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    print("Score normalization complete!")

if __name__ == "__main__":
    main()
