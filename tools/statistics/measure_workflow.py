import json
import csv
import os
from collections import Counter


def main():
    # Path to the JSON file
    json_path = "env_workflows.json"

    # Read the JSON file
    with open(json_path, "r") as f:
        workflows = json.load(f)

    # Calculate the length of each workflow
    flow_lengths = [len(steps) for steps in workflows.values()]

    # Count occurrences of each flow length
    length_counts = Counter(flow_lengths)

    # Sort by flow length
    sorted_counts = sorted(length_counts.items())

    # Output path for CSV

    csv_path = "github_page/static/workflow_lengths.csv"

    # Write to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flow_length", "number"])
        writer.writerows(sorted_counts)

    print(f"CSV generated at: {csv_path}")
    print("Flow length statistics:")
    for length, count in sorted_counts:
        print(f"Length {length}: {count} workflows")


if __name__ == "__main__":
    main()

