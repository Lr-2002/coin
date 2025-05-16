#!/usr/bin/env python3
import os
import glob
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_task_directories(dataset_path: str) -> List[str]:
    """
    Get all task directories in the dataset path.

    Args:
        dataset_path (str): Path to the dataset directory

    Returns:
        List[str]: List of task directory paths
    """
    task_dirs = [
        d for d in glob.glob(os.path.join(dataset_path, "*")) if os.path.isdir(d)
    ]
    return task_dirs


def analyze_task_directory(task_dir: str) -> Dict[str, Any]:
    """
    Analyze a task directory to get statistics about finished and unfinished episodes.

    Args:
        task_dir (str): Path to the task directory

    Returns:
        Dict[str, Any]: Dictionary containing task analysis results
    """
    task_name = os.path.basename(task_dir)
    logger.info(f"Analyzing task: {task_name}")

    # Get all h5 files and json files
    h5_files = glob.glob(os.path.join(task_dir, "*.h5"))
    json_files = glob.glob(os.path.join(task_dir, "*.json"))

    # Map json files to their base names (without extension)
    json_base_names = {os.path.splitext(os.path.basename(f))[0]: f for f in json_files}

    finished_episodes = []
    unfinished_episodes = []

    # Analyze each h5 file
    for h5_file in h5_files:
        base_name = os.path.splitext(os.path.basename(h5_file))[0]

        # Check if there's a corresponding json file
        is_finished = base_name in json_base_names

        try:
            # Open the h5 file to get episode steps
            with h5py.File(h5_file, "r") as f:
                # Check if file is empty or doesn't have the expected structure
                if "traj_0" not in f or len(f.keys()) == 0:
                    logger.warning(f"Skipping empty or invalid file: {h5_file}")
                    continue

                # Get episode data
                traj = f["traj_0"]

                # Check if the trajectory has the necessary data
                if "actions" not in traj:
                    logger.warning(f"File missing actions data: {h5_file}")
                    continue

                # Get number of steps from the actions dataset shape
                steps = traj["actions"].shape[0]

                # Get success status if available
                success = None
                if "success" in traj:
                    success_data = traj["success"][()]
                    if isinstance(success_data, np.ndarray) and success_data.size > 0:
                        success = bool(success_data[-1])  # Use the last value
                    else:
                        success = bool(success_data)

                episode_info = {"file": h5_file, "steps": steps, "success": success}

                # Add json data if available
                if is_finished:
                    json_file = json_base_names[base_name]
                    with open(json_file, "r") as jf:
                        try:
                            json_data = json.load(jf)
                            # Extract relevant information from json
                            if (
                                "episodes" in json_data
                                and len(json_data["episodes"]) > 0
                            ):
                                episode = json_data["episodes"][0]
                                episode_info["elapsed_steps"] = episode.get(
                                    "elapsed_steps"
                                )
                                episode_info["success_json"] = episode.get("success")
                            episode_info["json_data"] = json_data
                        except json.JSONDecodeError:
                            logger.error(f"Error parsing JSON file: {json_file}")

                    finished_episodes.append(episode_info)
                else:
                    unfinished_episodes.append(episode_info)

        except Exception as e:
            logger.error(f"Error processing file {h5_file}: {str(e)}")

    # Calculate statistics
    finished_steps = [ep["steps"] for ep in finished_episodes]
    unfinished_steps = [ep["steps"] for ep in unfinished_episodes]
    all_steps = finished_steps + unfinished_steps

    # Count successful episodes
    successful_episodes = sum(
        1 for ep in finished_episodes if ep.get("success_json", False)
    )

    result = {
        "task_name": task_name,
        "total_episodes": len(finished_episodes) + len(unfinished_episodes),
        "finished_episodes": len(finished_episodes),
        "unfinished_episodes": len(unfinished_episodes),
        "successful_episodes": successful_episodes,
        "avg_steps_finished": np.mean(finished_steps) if finished_steps else 0,
        "avg_steps_unfinished": np.mean(unfinished_steps) if unfinished_steps else 0,
        "avg_steps_all": np.mean(all_steps) if all_steps else 0,
        "min_steps": min(all_steps) if all_steps else 0,
        "max_steps": max(all_steps) if all_steps else 0,
        "finished_episodes_data": finished_episodes,
        "unfinished_episodes_data": unfinished_episodes,
    }

    return result


def analyze_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Analyze all tasks in the dataset.

    Args:
        dataset_path (str): Path to the dataset directory

    Returns:
        Dict[str, Any]: Dictionary containing analysis results for all tasks
    """
    task_dirs = get_task_directories(dataset_path)

    results = {}
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        results[task_name] = analyze_task_directory(task_dir)

    return results


def create_visualizations(
    analysis_results: Dict[str, Any], output_dir: Optional[str] = None
):
    """
    Create visualizations from the analysis results.

    Args:
        analysis_results (Dict[str, Any]): Analysis results
        output_dir (Optional[str]): Directory to save visualizations
    """
    if output_dir is None:
        output_dir = "analysis_results"

    os.makedirs(output_dir, exist_ok=True)

    # Extract data for plotting
    task_names = list(analysis_results.keys())
    finished_counts = [
        results["finished_episodes"] for results in analysis_results.values()
    ]
    unfinished_counts = [
        results["unfinished_episodes"] for results in analysis_results.values()
    ]
    avg_steps_finished = [
        results["avg_steps_finished"] for results in analysis_results.values()
    ]
    avg_steps_unfinished = [
        results["avg_steps_unfinished"] for results in analysis_results.values()
    ]
    avg_steps_all = [results["avg_steps_all"] for results in analysis_results.values()]
    success_rates = [
        results["successful_episodes"] / results["finished_episodes"]
        if results["finished_episodes"] > 0
        else 0
        for results in analysis_results.values()
    ]

    # Set style
    sns.set(style="whitegrid")

    # 1. Finished vs Unfinished Episodes
    plt.figure(figsize=(12, 6))
    x = np.arange(len(task_names))
    width = 0.35

    plt.bar(x - width / 2, finished_counts, width, label="Finished")
    plt.bar(x + width / 2, unfinished_counts, width, label="Unfinished")

    plt.xlabel("Tasks")
    plt.ylabel("Number of Episodes")
    plt.title("Finished vs Unfinished Episodes by Task")
    plt.xticks(x, task_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "finished_vs_unfinished.png"))
    plt.close()

    # 2. Average Steps Comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(task_names))
    width = 0.25

    plt.bar(x - width, avg_steps_finished, width, label="Finished Episodes")
    plt.bar(x, avg_steps_unfinished, width, label="Unfinished Episodes")
    plt.bar(x + width, avg_steps_all, width, label="All Episodes")

    plt.xlabel("Tasks")
    plt.ylabel("Average Steps")
    plt.title("Average Steps by Task and Episode Type")
    plt.xticks(x, task_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_steps.png"))
    plt.close()

    # 3. Success Rate
    plt.figure(figsize=(12, 6))
    plt.bar(task_names, success_rates, color="green")
    plt.xlabel("Tasks")
    plt.ylabel("Success Rate")
    plt.title("Success Rate by Task")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate.png"))
    plt.close()

    # 4. Task Completion Pie Chart
    total_finished = sum(finished_counts)
    total_unfinished = sum(unfinished_counts)

    plt.figure(figsize=(8, 8))
    plt.pie(
        [total_finished, total_unfinished],
        labels=["Finished", "Unfinished"],
        autopct="%1.1f%%",
        colors=["#66b3ff", "#ff9999"],
    )
    plt.title("Overall Task Completion Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "completion_pie.png"))
    plt.close()

    # 5. Task Distribution
    plt.figure(figsize=(10, 6))
    total_episodes = [
        results["total_episodes"] for results in analysis_results.values()
    ]
    plt.pie(total_episodes, labels=task_names, autopct="%1.1f%%")
    plt.title("Distribution of Episodes Across Tasks")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_distribution.png"))
    plt.close()

    # Save summary as text
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("Dataset Analysis Summary\n")
        f.write("=======================\n\n")

        # Overall statistics
        total_episodes = sum(
            results["total_episodes"] for results in analysis_results.values()
        )
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write(
            f"Finished Episodes: {sum(finished_counts)} ({sum(finished_counts)/total_episodes*100:.1f}%)\n"
        )
        f.write(
            f"Unfinished Episodes: {sum(unfinished_counts)} ({sum(unfinished_counts)/total_episodes*100:.1f}%)\n"
        )
        f.write(
            f"Successful Episodes: {sum(results['successful_episodes'] for results in analysis_results.values())}\n"
        )
        f.write(
            f"Overall Success Rate: {sum(results['successful_episodes'] for results in analysis_results.values())/sum(finished_counts)*100:.1f}%\n\n"
        )

        # Per-task statistics
        f.write("Per-Task Statistics\n")
        f.write("-----------------\n\n")

        for task_name, results in analysis_results.items():
            f.write(f"Task: {task_name}\n")
            f.write(f"  Total Episodes: {results['total_episodes']}\n")
            f.write(f"  Finished Episodes: {results['finished_episodes']}\n")
            f.write(f"  Unfinished Episodes: {results['unfinished_episodes']}\n")
            f.write(f"  Successful Episodes: {results['successful_episodes']}\n")

            success_rate = (
                results["successful_episodes"] / results["finished_episodes"] * 100
                if results["finished_episodes"] > 0
                else 0
            )
            f.write(f"  Success Rate: {success_rate:.1f}%\n")

            f.write(
                f"  Average Steps (Finished): {results['avg_steps_finished']:.1f}\n"
            )
            f.write(
                f"  Average Steps (Unfinished): {results['avg_steps_unfinished']:.1f}\n"
            )
            f.write(f"  Average Steps (All): {results['avg_steps_all']:.1f}\n")
            f.write(f"  Min Steps: {results['min_steps']}\n")
            f.write(f"  Max Steps: {results['max_steps']}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze ManiSkill dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="coin_dataset",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results and visualizations",
    )
    args = parser.parse_args()

    logger.info(f"Analyzing dataset at: {args.dataset_path}")

    # Analyze dataset
    results = analyze_dataset(args.dataset_path)

    # Create visualizations
    create_visualizations(results, args.output_dir)

    logger.info(f"Analysis complete. Results saved to: {args.output_dir}")

    # Print summary
    total_episodes = sum(results[task]["total_episodes"] for task in results)
    finished_episodes = sum(results[task]["finished_episodes"] for task in results)
    unfinished_episodes = sum(results[task]["unfinished_episodes"] for task in results)

    print("\nDataset Summary:")
    print(f"Total Tasks: {len(results)}")
    print(f"Total Episodes: {total_episodes}")
    print(
        f"Finished Episodes: {finished_episodes} ({finished_episodes/total_episodes*100:.1f}%)"
    )
    print(
        f"Unfinished Episodes: {unfinished_episodes} ({unfinished_episodes/total_episodes*100:.1f}%)"
    )


if __name__ == "__main__":
    main()

