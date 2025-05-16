#!/usr/bin/env python3

import csv
import argparse
import pickle as pkl

def merge_csv_files(target_file, source_file, output_file=None):
    """
    Merge environments from source CSV file into target CSV file.
    If environment exists in both files, combine results and recalculate accuracy.
    If environment only exists in source file, add it to target file.
    
    Args:
        target_file: Path to the target CSV file that will receive new entries
        source_file: Path to the source CSV file containing entries to be added
        output_file: Path to the output file (if None, will overwrite target_file)
    """
    # Read target file
    target_data = {}
    with open(target_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:  # Skip empty rows
                task_name = row[0]
                target_data[task_name] = {
                    'accuracy': row[1],
                    'success': int(row[2]),
                    'total': int(row[3])
                }
    
    # Read source file
    source_data = {}
    with open(source_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:  # Skip empty rows
                task_name = row[0]
                source_data[task_name] = {
                    'accuracy': row[1],
                    'success': int(row[2]),
                    'total': int(row[3])
                }
    
    # Merge data
    new_tasks = []
    combined_tasks = []
    
    for task_name, data in source_data.items():
        if task_name in target_data:
            # Combine results for existing tasks
            old_success = target_data[task_name]['success']
            old_total = target_data[task_name]['total']
            new_success = data['success']
            new_total = data['total']
            
            combined_success = old_success + new_success
            combined_total = old_total + new_total
            combined_accuracy = f"{(combined_success / combined_total * 100):.2f}%" if combined_total > 0 else "0.00%"
            
            target_data[task_name] = {
                'accuracy': combined_accuracy,
                'success': combined_success,
                'total': combined_total
            }
            
            combined_tasks.append({
                'name': task_name,
                'old_accuracy': f"{(old_success / old_total * 100):.2f}%" if old_total > 0 else "0.00%",
                'old_success': old_success,
                'old_total': old_total,
                'new_accuracy': data['accuracy'],
                'new_success': new_success,
                'new_total': new_total,
                'combined_accuracy': combined_accuracy,
                'combined_success': combined_success,
                'combined_total': combined_total
            })
        else:
            # Add new tasks
            target_data[task_name] = data
            new_tasks.append({
                'name': task_name,
                'accuracy': data['accuracy'],
                'success': data['success'],
                'total': data['total']
            })
    
    # Determine output file
    if output_file is None:
        output_file = target_file
    
    # Write merged data
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for task_name, data in target_data.items():
            writer.writerow([task_name, data['accuracy'], data['success'], data['total']])
    
    # Print summary
    print(f"Merged results from {source_file} into {output_file}")
    
    if new_tasks:
        print(f"\nAdded {len(new_tasks)} new environments:")
        for task in new_tasks:
            print(f"  {task['name']}: Accuracy={task['accuracy']}, Success={task['success']}, Total={task['total']}")
    
    if combined_tasks:
        print(f"\nCombined results for {len(combined_tasks)} existing environments:")
        for task in combined_tasks:
            print(f"  {task['name']}:")
            print(f"    Original: Accuracy={task['old_accuracy']}, Success={task['old_success']}, Total={task['old_total']}")
            print(f"    Added: Accuracy={task['new_accuracy']}, Success={task['new_success']}, Total={task['new_total']}")
            print(f"    Combined: Accuracy={task['combined_accuracy']}, Success={task['combined_success']}, Total={task['combined_total']}")

def load_pkl_path(pkl_path):
    """
    Load environment names from a pickle file.
    
    Args:
        pkl_path: Path to the pickle file containing environment names
        
    Returns:
        List of environment names
    """
    with open(pkl_path, 'rb') as f: 
        pkl_list = pkl.load(f) 
        pkl_list = [x for x in pkl_list.keys()]

    return pkl_list

def check_missing_envs(csv_file):
    """
    Check which environments are missing from a CSV file compared to the full environment list.
    
    Args:
        csv_file: Path to the CSV file to check
        interactive: Whether to include interactive environments
        primitive: Whether to include primitive environments
        
    Returns:
        List of missing environment names
    """
    # Paths to the pickle files containing the full environment lists
    interactive_path = "/home/wangxianhao/data/project/reasoning/ManiSkill/interactive_instruction_objects.pkl"
    
    # Load the full environment list
    full_env_list = []
    try:
        full_env_list += load_pkl_path(interactive_path)
    except Exception as e:
        print(f"Warning: Could not load interactive environments: {e}")

    # Read the CSV file
    csv_envs = set()
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:  # Skip empty rows
                    csv_envs.add(row[0])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    # Find missing environments
    missing_envs = [env for env in full_env_list if env not in csv_envs]
    
    return missing_envs

def main():
    parser = argparse.ArgumentParser(description='Merge missing environments from source CSV to target CSV or check for missing environments')
    parser.add_argument('csv_file', help='Path to the CSV file to check or merge into')
    parser.add_argument('source_file', nargs='?', help='Path to the source CSV file (required for merging, not for checking)')
    parser.add_argument('--output', '-o', help='Path to the output file (default: overwrite target file)')
    parser.add_argument('--check-missing', '-c', action='store_true', help='Check for missing environments in the CSV file')
    args = parser.parse_args()
    
    if args.check_missing:
        missing_envs = check_missing_envs(args.csv_file)
        if missing_envs:
            print(f"Found {len(missing_envs)} missing environments in {args.csv_file}:")
            for env in missing_envs:
                print(f"  {env}")
        else:
            print(f"No missing environments found in {args.csv_file}")
    else:
        if not args.source_file:
            parser.error("source_file is required when not using --check-missing")
        merge_csv_files(args.csv_file, args.source_file, args.output)

if __name__ == '__main__':
    main()
