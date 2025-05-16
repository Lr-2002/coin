import os
import csv
from collections import defaultdict
import subprocess


def find_git_tracked_csv_files(root_dir):
    # Use git to list all tracked CSV files under root_dir
    repo_root = os.path.abspath(os.path.join(root_dir, ".."))
    result = subprocess.run(
        ["git", "ls-files",
            os.path.relpath(root_dir, repo_root) + "/**/*.csv"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    files = [os.path.join(repo_root, f)
             for f in result.stdout.strip().split("\n") if f]
    return files


def merge_success_rates_pivot(csv_files, output_path):
    # task_name -> {csv_name: success_rate}
    merged = defaultdict(dict)
    all_csv_names = []
    for csv_file in csv_files:
        csv_name = os.path.splitext(os.path.basename(csv_file))[0]
        all_csv_names.append(csv_name)
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = row.get("Task", "")
                rate = row.get("Success Rate", "")
                if task:
                    merged[task][csv_name] = rate
    all_tasks = sorted(merged.keys())
    all_csv_names = sorted(set(all_csv_names))
    # Write merged CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Task"] + all_csv_names)
        for task in all_tasks:
            row = [task] + [
                merged[task].get(csv_name, "") for csv_name in all_csv_names
            ]
            writer.writerow(row)


def main():
    input_dir = os.path.join(os.path.dirname(
        __file__), "../../evaluation_results")
    output_csv = os.path.join(input_dir, "merged_success_rates.csv")
    csv_files = find_git_tracked_csv_files(input_dir)
    print(csv_files)
    merge_success_rates_pivot(csv_files, output_csv)
    print(
        f"Pivot-merged {len(csv_files)} git-tracked CSV files. Output: {output_csv}")


if __name__ == "__main__":
    main()
