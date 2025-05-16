#!/usr/bin/env python
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from llm_agent_planner_refactored import LLMAgentPlanner

def main():
    parser = argparse.ArgumentParser(description='Evaluate all environments in env_workflows.json')
    parser.add_argument('--num-images', type=int, default=5, help='Maximum number of images to use')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20', help='LLM model to use')
    parser.add_argument('--use_evaluate', type=bool, default=False, help='need to evaluate?')
    
    # LLM provider selection arguments
    provider_group = parser.add_argument_group('LLM Provider')
    provider_group.add_argument('--use-mock', action='store_true', help='Use mock LLM provider for testing without API key')
    provider_group.add_argument('--use-azure', action='store_true', help='Use Azure OpenAI instead of OpenAI')
    provider_group.add_argument('--use-gemini', action='store_true', help='Use Google Gemini instead of OpenAI')
    
    # Azure-specific arguments
    azure_group = parser.add_argument_group('Azure OpenAI Settings')
    azure_group.add_argument('--azure-region', type=str, default='eastus2', help='Azure region')
    azure_group.add_argument('--azure-api-base', type=str, default='https://api.tonggpt.mybigai.ac.cn/proxy', help='Azure API base URL')
    azure_group.add_argument('--azure-api-version', type=str, default='2025-03-01-preview', help='Azure API version')
    
    # Gemini-specific arguments
    gemini_group = parser.add_argument_group('Google Gemini Settings')
    gemini_group.add_argument('--gemini-model', type=str, default='gemini-2.0-flash', help='Gemini model to use (e.g., gemini-2.0-flash, gemini-2.0-pro)')
    
    # General arguments
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of LLM responses')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Load environment workflows
    try:
        with open('env_workflows.json', 'r') as f:
            env_workflows = json.load(f)
    except Exception as e:
        print(f"Error loading env_workflows.json: {e}")
        return
    
    # Determine which LLM provider to use based on command-line arguments
    if args.use_mock:
        llm_provider = "mock"
    elif args.use_gemini:
        llm_provider = "gemini"
    elif args.use_azure:
        llm_provider = "azure"
    else:
        llm_provider = "openai"  # Default to OpenAI if no provider is specified
    
    # Set up provider-specific parameters
    provider_params = {
        "use_cache": not args.no_cache,
        "checkpoint_dir": "coin_videos/checkpoint",  # Add default checkpoint directory
    }
    
    # Add provider-specific parameters
    if llm_provider == "azure":
        print({
            "model": args.model,
            "azure_region": args.azure_region,
            "azure_api_base": args.azure_api_base,
            "azure_api_version": args.azure_api_version
        })
        provider_params.update({
            "model": args.model,
            "azure_region": args.azure_region,
            "azure_api_base": args.azure_api_base,
            "azure_api_version": args.azure_api_version
        })
    elif llm_provider == "gemini":
        provider_params.update({
            "model": args.gemini_model,
        })
    else:  # openai or mock
        provider_params.update({
            "model": args.model,
        })
    
    # Initialize the LLM agent planner
    planner = LLMAgentPlanner(
        llm_provider=llm_provider,
        **provider_params
    )
    
    # Create output directory
    output_dir = Path(args.output_dir )
    output_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a CSV file to store evaluation results
    results_file = output_dir / f"evaluation_results_{timestamp}.csv"
    
    # Initialize results dictionary
    all_results = []
    
    # Check checkpoint directory
    checkpoint_dir = Path(planner.checkpoint_dir)
    
    # Create a directory to store evaluation results by task
    eval_results_dir = output_dir / "task_results" / args.model 
    eval_results_dir.mkdir(exist_ok=True)
    
    # Get list of available checkpoint directories
    available_checkpoints = [d.name for d in checkpoint_dir.iterdir() if d.is_dir()]
    print(f"Found {len(available_checkpoints)} checkpoint directories")
    
    # Count total tasks to evaluate
    tasks_to_evaluate = []
    for task_name, ground_truth_plan in env_workflows.items():
        task_dir_name = task_name.replace('-', '_')
        if task_dir_name in available_checkpoints:
            checkpoint_file = checkpoint_dir / task_dir_name / "checkpoint.json"
            if checkpoint_file.exists():
                tasks_to_evaluate.append(task_name)
    
    print(f"Will evaluate {len(tasks_to_evaluate)} tasks out of {len(env_workflows)} in env_workflows.json")
    
    # Evaluate each environment
    for i, task_name in enumerate(tasks_to_evaluate):
        # Convert task name format from Tabletop-Find-Book-FromShelf-v1 to Tabletop_Find_Book_FromShelf_v1
        task_dir_name = task_name.replace('-', '_')
        task_checkpoint_dir = checkpoint_dir / task_dir_name
        
        # Check if checkpoint.json exists and load it to see how many images are available
        checkpoint_file = task_checkpoint_dir / "checkpoint.json"
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                available_images = checkpoint_data.get("saved_frames", [])
                # Use all available images
                max_images = len(available_images)
                print(f"[{i+1}/{len(tasks_to_evaluate)}] Evaluating {task_name} with all {max_images} available images...")
        except Exception as e:
            print(f"Error loading checkpoint file for {task_name}: {e}")
            continue
        
        # Create task-specific results file
        task_results_file = eval_results_dir / f"{task_dir_name}_results.json"
        task_scores_file = eval_results_dir / f"{task_dir_name}_scores.csv"
        
        # Test with progressive images and evaluate
        try:
            task_results = planner.test_with_progressive_images(task_name, num_images=max_images, evaluate=args.use_evaluate)
            
            # Save task-specific results
            with open(task_results_file, 'w') as f:
                json.dump(task_results, f, indent=2)
            
            # Create a list to store task-specific results for CSV
            task_csv_results = []
            
            # Process results for each number of images
            for img_key, result in task_results.items():
                if "evaluation" in result:
                    eval_result = result.get("evaluation", {})
                    if "error" not in eval_result:
                        num_images = int(img_key.split('_')[0])
                        
                        # Verify scores are correctly parsed
                        raw_response = eval_result.get("raw_response", "")
                        
                        # Add to results
                        result_entry = {
                            "task_name": task_name,
                            "num_images": num_images,
                            "completeness": eval_result.get("completeness", 0),
                            "correctness": eval_result.get("correctness", 0),
                            "clarity": eval_result.get("clarity", 0),
                            "mean_score": eval_result.get("mean_score", 0),
                            "timestamp": timestamp
                        }
                        
                        all_results.append(result_entry)
                        task_csv_results.append(result_entry)
                        
                        print(f"  {img_key}: Completeness: {eval_result.get('completeness', 0)}, "
                              f"Correctness: {eval_result.get('correctness', 0)}, "
                              f"Clarity: {eval_result.get('clarity', 0)}, "
                              f"Mean: {eval_result.get('mean_score', 0)}")
                    else:
                        error_msg = eval_result.get("error", "Unknown error")
                        print(f"  {img_key}: Evaluation failed - {error_msg}")
                else:
                    print(f"  {img_key}: No evaluation data available")
            
            # Save task-specific scores to CSV
            if task_csv_results:
                task_df = pd.DataFrame(task_csv_results)
                task_df.to_csv(task_scores_file, index=False)
                print(f"  Task scores saved to {task_scores_file}")
                
                # Generate task-specific plots
                task_plots_dir = output_dir / "plots" / task_dir_name
                task_plots_dir.mkdir(exist_ok=True, parents=True)
                
                # Plot scores by number of images for this task
                plt.figure(figsize=(10, 6))
                for metric in ["completeness", "correctness", "clarity", "mean_score"]:
                    plt.plot(task_df["num_images"], task_df[metric], marker='o', label=metric.capitalize())
                
                plt.xlabel("Number of Images")
                plt.ylabel("Score")
                plt.title(f"Evaluation Scores for {task_name}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(task_plots_dir / f"{task_dir_name}_scores_{timestamp}.png")
                plt.close()
                
        except Exception as e:
            print(f"Error evaluating {task_name}: {e}")
            import traceback
            print(traceback.format_exc())
    
    # Save results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
        
        # Generate plots
        generate_plots(df, output_dir, timestamp)
        
        # Generate summary report
        summary_file = generate_summary_report(df, output_dir, timestamp, env_workflows)
        
        # Print final summary
        print("\nEvaluation complete!")
        print(f"Total tasks evaluated: {len(df['task_name'].unique())}")
        print(f"Average mean score: {df['mean_score'].mean():.2f}/100")
        print(f"Summary report: {summary_file}")
    else:
        print("No evaluation results to save.")

def generate_summary_report(df, output_dir, timestamp, env_workflows):
    """Generate a summary report of evaluation scores."""
    # Create summary directory
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    # Generate summary report
    summary_file = summary_dir / f"evaluation_summary_{timestamp}.md"
    
    # Calculate summary statistics
    task_summary = df.groupby("task_name")[["completeness", "correctness", "clarity", "mean_score"]].mean().reset_index()
    image_summary = df.groupby("num_images")[["completeness", "correctness", "clarity", "mean_score"]].mean().reset_index()
    
    # Sort tasks by mean score
    task_summary = task_summary.sort_values(by="mean_score", ascending=False)
    
    # Calculate standard deviations for each metric
    std_metrics = df.groupby("task_name")[["completeness", "correctness", "clarity", "mean_score"]].std().reset_index()
    std_metrics.fillna(0, inplace=True)  # Replace NaN with 0 for tasks with only one evaluation
    
    # Merge mean and std for comprehensive analysis
    task_analysis = pd.merge(task_summary, std_metrics, on="task_name", suffixes=('_mean', '_std'))
    
    # Generate markdown report
    with open(summary_file, "w") as f:
        f.write(f"# Plan Evaluation Summary Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Overall Statistics\n\n")
        f.write(f"- Total tasks evaluated: {len(task_summary)}\n")
        f.write(f"- Average completeness score: {df['completeness'].mean():.2f}/100 (σ={df['completeness'].std():.2f})\n")
        f.write(f"- Average correctness score: {df['correctness'].mean():.2f}/100 (σ={df['correctness'].std():.2f})\n")
        f.write(f"- Average clarity score: {df['clarity'].mean():.2f}/100 (σ={df['clarity'].std():.2f})\n")
        f.write(f"- Average mean score: {df['mean_score'].mean():.2f}/100 (σ={df['mean_score'].std():.2f})\n\n")
        
        # Add correlation analysis
        f.write(f"### Correlation Analysis\n\n")
        correlation_matrix = df[["completeness", "correctness", "clarity", "mean_score", "num_images"]].corr()
        f.write("Correlation between metrics and number of images:\n\n")
        f.write("| Metric | Correlation with Num Images |\n")
        f.write("|--------|-------------------------------|\n")
        for metric in ["completeness", "correctness", "clarity", "mean_score"]:
            f.write(f"| {metric.capitalize()} | {correlation_matrix.loc[metric, 'num_images']:.3f} |\n")
        f.write("\n")
        
        f.write(f"## Top 5 Tasks by Mean Score\n\n")
        f.write("| Task | Completeness | Correctness | Clarity | Mean Score | Std Dev |\n")
        f.write("|------|--------------|-------------|---------|------------|---------|\n")
        for _, row in task_analysis.head(5).iterrows():
            f.write(f"| {row['task_name']} | {row['completeness_mean']:.2f} | {row['correctness_mean']:.2f} | "
                   f"{row['clarity_mean']:.2f} | {row['mean_score_mean']:.2f} | {row['mean_score_std']:.2f} |\n")
        
        f.write(f"\n## Bottom 5 Tasks by Mean Score\n\n")
        f.write("| Task | Completeness | Correctness | Clarity | Mean Score | Std Dev |\n")
        f.write("|------|--------------|-------------|---------|------------|---------|\n")
        for _, row in task_analysis.tail(5).iterrows():
            f.write(f"| {row['task_name']} | {row['completeness_mean']:.2f} | {row['correctness_mean']:.2f} | "
                   f"{row['clarity_mean']:.2f} | {row['mean_score_mean']:.2f} | {row['mean_score_std']:.2f} |\n")
        
        f.write(f"\n## Scores by Number of Images\n\n")
        f.write("| Number of Images | Completeness | Correctness | Clarity | Mean Score |\n")
        f.write("|-----------------|--------------|-------------|---------|------------|\n")
        for _, row in image_summary.iterrows():
            f.write(f"| {int(row['num_images'])} | {row['completeness']:.2f} | {row['correctness']:.2f} | {row['clarity']:.2f} | {row['mean_score']:.2f} |\n")
        
        # Add analysis of score improvement with more images
        if len(image_summary) > 1:
            f.write(f"\n### Score Improvement Analysis\n\n")
            min_images = image_summary['num_images'].min()
            max_images = image_summary['num_images'].max()
            min_row = image_summary[image_summary['num_images'] == min_images].iloc[0]
            max_row = image_summary[image_summary['num_images'] == max_images].iloc[0]
            
            f.write(f"Comparison between {min_images} and {max_images} images:\n\n")
            f.write("| Metric | {min_images} Images | {max_images} Images | Improvement | % Change |\n".format(min_images=min_images, max_images=max_images))
            f.write("|--------|--------------|--------------|-------------|----------|\n")
            
            for metric in ["completeness", "correctness", "clarity", "mean_score"]:
                improvement = max_row[metric] - min_row[metric]
                pct_change = (improvement / min_row[metric] * 100) if min_row[metric] > 0 else float('inf')
                f.write(f"| {metric.capitalize()} | {min_row[metric]:.2f} | {max_row[metric]:.2f} | {improvement:.2f} | {pct_change:.2f}% |\n")
        
        # Add skill distribution analysis based on task names and ground truth plans
        f.write(f"\n## Skill Distribution Analysis\n\n")
        f.write("This section analyzes the distribution of skills across tasks based on the task names and ground truth plans.\n\n")
        
        # Define common skills to look for in task names and plans
        skills = ["pick", "place", "open", "close", "move", "align", "rotate", "put", "push", "pull", "insert", "lift", "squeeze"]
        skill_counts = {skill: 0 for skill in skills}
        skill_scores = {skill: [] for skill in skills}
        
        # Count skills in task names and ground truth plans
        for task_name in df["task_name"].unique():
            task_name_lower = task_name.lower()
            task_mean_score = task_summary[task_summary["task_name"] == task_name]["mean_score"].values[0]
            
            # Check ground truth plan if available
            ground_truth_plan = env_workflows.get(task_name, "")
            ground_truth_lower = ground_truth_plan.lower() if ground_truth_plan else ""
            
            for skill in skills:
                if skill in task_name_lower or skill in ground_truth_lower:
                    skill_counts[skill] += 1
                    skill_scores[skill].append(task_mean_score)
        
        # Calculate average score for each skill
        skill_avg_scores = {}
        for skill, scores in skill_scores.items():
            if scores:
                skill_avg_scores[skill] = sum(scores) / len(scores)
            else:
                skill_avg_scores[skill] = 0
        
        # Sort skills by count
        sorted_skills = sorted([(skill, count, skill_avg_scores[skill]) 
                               for skill, count in skill_counts.items() if count > 0], 
                              key=lambda x: x[1], reverse=True)
        
        f.write("### Skill Frequency and Performance\n\n")
        f.write("| Skill | Count | Average Score |\n")
        f.write("|-------|-------|---------------|\n")
        for skill, count, avg_score in sorted_skills:
            f.write(f"| {skill.capitalize()} | {count} | {avg_score:.2f} |\n")
        
        # Add task complexity analysis
        f.write(f"\n## Task Complexity Analysis\n\n")
        
        # Count number of skills per task
        task_complexity = {}
        for task_name in df["task_name"].unique():
            task_name_lower = task_name.lower()
            ground_truth_plan = env_workflows.get(task_name, "")
            ground_truth_lower = ground_truth_plan.lower() if ground_truth_plan else ""
            
            skill_count = 0
            for skill in skills:
                if skill in task_name_lower or skill in ground_truth_lower:
                    skill_count += 1
            
            task_complexity[task_name] = skill_count
        
        # Add complexity to task summary
        task_summary["complexity"] = task_summary["task_name"].map(task_complexity)
        
        # Calculate correlation between complexity and scores
        complexity_corr = task_summary[["complexity", "completeness", "correctness", "clarity", "mean_score"]].corr()
        
        f.write("### Correlation between Task Complexity and Scores\n\n")
        f.write("| Metric | Correlation with Complexity |\n")
        f.write("|--------|------------------------------|\n")
        for metric in ["completeness", "correctness", "clarity", "mean_score"]:
            f.write(f"| {metric.capitalize()} | {complexity_corr.loc[metric, 'complexity']:.3f} |\n")
    
    print(f"Enhanced summary report saved to {summary_file}")
    return summary_file

def generate_plots(df, output_dir, timestamp):
    """Generate plots from evaluation results."""
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot mean scores by task and number of images
    plt.figure(figsize=(12, 8))
    for task_name, task_df in df.groupby("task_name"):
        plt.plot(task_df["num_images"], task_df["mean_score"], marker='o', label=task_name)
    
    plt.xlabel("Number of Images")
    plt.ylabel("Mean Score")
    plt.title("Mean Evaluation Scores by Task and Number of Images")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / f"mean_scores_by_task_{timestamp}.png")
    
    # Plot average scores across all tasks
    avg_by_images = df.groupby("num_images")[["completeness", "correctness", "clarity", "mean_score"]].mean()
    
    plt.figure(figsize=(10, 6))
    for metric in ["completeness", "correctness", "clarity", "mean_score"]:
        plt.plot(avg_by_images.index, avg_by_images[metric], marker='o', label=metric.capitalize())
    
    plt.xlabel("Number of Images")
    plt.ylabel("Average Score")
    plt.title("Average Evaluation Scores by Number of Images")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / f"average_scores_by_images_{timestamp}.png")
    
    # Plot distribution of scores
    plt.figure(figsize=(10, 6))
    for metric in ["completeness", "correctness", "clarity", "mean_score"]:
        plt.hist(df[metric], bins=10, alpha=0.5, label=metric.capitalize())
    
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Evaluation Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / f"score_distribution_{timestamp}.png")
    
    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    main()
