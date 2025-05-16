#!/usr/bin/env python3

import os
import json
import glob
import pandas as pd
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base paths
EVAL_BASE_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/evaluation_results/score_plan_gpt_evaluate"
OUTPUT_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static"
SUMMARY_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/evaluation_results/score_plan_gpt_evaluate/plans_scores"

def get_available_models():
    """Get list of available models (directory names in EVAL_BASE_DIR)"""
    model_dirs = [d for d in os.listdir(EVAL_BASE_DIR) 
                if os.path.isdir(os.path.join(EVAL_BASE_DIR, d))]
    # Filter out the 'plans_scores' directory which isn't a model
    model_dirs = [d for d in model_dirs if d != 'plans_scores']
    model_dirs.sort()
    return model_dirs

def process_model_scores(model_name):
    """Process all task score files for a specific model and generate CSV files"""
    model_dir = os.path.join(EVAL_BASE_DIR, model_name)
    task_files = glob.glob(os.path.join(model_dir, "*.json"))
    
    if not task_files:
        logger.warning(f"No score files found for model {model_name}")
        return None
    
    # Create dataframes to store all task scores
    task_data = []
    
    # Process each task file
    for task_file in task_files:
        task_name = os.path.basename(task_file).replace(".json", "")
        
        with open(task_file, 'r') as f:
            try:
                scores = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading JSON from {task_file}")
                continue
        
        # Process each step in the task
        for step_key, step_scores in scores.items():
            # Extract the step index from the key (e.g., '0_images' -> '0')
            step_index = step_key.split('_')[0]
            
            # Check if it's a valid score object 
            if not isinstance(step_scores, dict):
                continue
                
            # Create a row for the dataframe
            row = {
                'task': task_name,
                'step': step_index,
                'completeness': step_scores.get('completeness', 1),
                'objects_correct': step_scores.get('objects_correct', 1),
                'sequence_correct': step_scores.get('sequence_correct', 1),
                'clarity': step_scores.get('clarity', 1),
                'mean_score': step_scores.get('mean_score', 1),
            }
            
            task_data.append(row)
    
    # Create a DataFrame from all task data
    if not task_data:
        logger.warning(f"No valid scores found for model {model_name}")
        return None
        
    df = pd.DataFrame(task_data)
    
    # Calculate summary statistics
    summary_df = df.groupby('task').agg({
        'completeness': 'mean',
        'objects_correct': 'mean',
        'sequence_correct': 'mean',
        'clarity': 'mean',
        'mean_score': 'mean'
    }).reset_index()
    
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    
    # Save detailed scores to CSV
    csv_filename = f"{model_name.replace('-', '')}-plan-score.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed scores to {csv_path}")
    
    # Save summary scores to CSV
    summary_csv_path = os.path.join(SUMMARY_DIR, f"{model_name}_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved summary scores to {summary_csv_path}")
    
    return df

def generate_overall_comparison(models):
    """Generate a comparison CSV across all models"""
    all_model_data = []
    
    for model_name in models:
        summary_path = os.path.join(SUMMARY_DIR, f"{model_name}_summary.csv")
        if not os.path.exists(summary_path):
            logger.warning(f"Summary file not found for model {model_name}")
            continue
            
        model_df = pd.read_csv(summary_path)
        # Add model column
        model_df['model'] = model_name
        all_model_data.append(model_df)
    
    if not all_model_data:
        logger.warning("No model data available for comparison")
        return
        
    # Combine all model data
    comparison_df = pd.concat(all_model_data)
    
    # Calculate average scores across all models and tasks
    avg_scores = comparison_df.groupby('model').agg({
        'completeness': 'mean',
        'objects_correct': 'mean',
        'sequence_correct': 'mean',
        'clarity': 'mean',
        'mean_score': 'mean'
    }).reset_index()
    
    # Sort by mean score descending
    avg_scores = avg_scores.sort_values('mean_score', ascending=False)
    
    # Save comparison to CSV
    comparison_path = os.path.join(SUMMARY_DIR, "model_comparison.csv")
    avg_scores.to_csv(comparison_path, index=False)
    logger.info(f"Saved model comparison to {comparison_path}")
    
    # Save detailed task comparison
    detailed_path = os.path.join(SUMMARY_DIR, "model_task_comparison.csv")
    comparison_df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed task comparison to {detailed_path}")
    
    return avg_scores

def main():
    parser = argparse.ArgumentParser(description="Generate CSV summaries of plan evaluation scores")
    parser.add_argument("--model", type=str, help="Specific model to process")
    parser.add_argument("--all", action="store_true", help="Process all available models")
    args = parser.parse_args()
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        logger.error("No model directories found.")
        return
    
    # Determine which models to process
    models_to_process = []
    if args.all:
        models_to_process = available_models
    elif args.model:
        if args.model in available_models:
            models_to_process = [args.model]
        else:
            logger.error(f"Model {args.model} not found. Available models: {', '.join(available_models)}")
            return
    else:
        # Default to all models if none specified
        models_to_process = available_models
        logger.info(f"No model specified. Processing all models: {', '.join(models_to_process)}")
    
    # Process each model
    processed_models = []
    for model_name in models_to_process:
        logger.info(f"Processing scores for model: {model_name}")
        df = process_model_scores(model_name)
        if df is not None:
            processed_models.append(model_name)
    
    # Generate comparison across all processed models
    if len(processed_models) > 1:
        logger.info("Generating model comparison")
        avg_scores = generate_overall_comparison(processed_models)
        if avg_scores is not None:
            logger.info("\nModel Rankings (by mean score):")
            for _, row in avg_scores.iterrows():
                logger.info(f"{row['model']}: {row['mean_score']:.2f}")
    
    logger.info("Score summarization complete!")

if __name__ == "__main__":
    main()
