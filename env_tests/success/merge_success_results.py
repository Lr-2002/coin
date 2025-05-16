#!/usr/bin/env python3
import os
import csv
import pandas as pd
import argparse
from datetime import datetime

def merge_csv_results(csv_file1, csv_file2, output_dir):
    """
    Merge two CSV files containing task success rates.
    For tasks that appear in both files, add the success counts and total counts,
    then recalculate the success rate.
    If csv_file2 is None, just copy csv_file1 to the output directory with the target name.
    
    Args:
        csv_file1: Path to the first CSV file
        csv_file2: Path to the second CSV file or None
        output_dir: Directory to save the merged results
    """
    # Read the first CSV file
    df1 = pd.read_csv(csv_file1)
    
    # If csv_file2 is not provided, just use df1 as the result
    if csv_file2 is None:
        output_df = df1
    else:
        # Read the second CSV file
        df2 = pd.read_csv(csv_file2)
        
        # Create a dictionary to store the merged results
        merged_results = {}
        
        # Process the first CSV file
        for _, row in df1.iterrows():
            task = row['Task']
            success_count = int(row['Success Count'])
            total_count = int(row['Total Count'])
            
            merged_results[task] = {
                'Success Count': success_count,
                'Total Count': total_count
            }
        
        # Process the second CSV file and merge with the first
        for _, row in df2.iterrows():
            task = row['Task']
            success_count = int(row['Success Count'])
            total_count = int(row['Total Count'])
            
            if task in merged_results:
                # Task exists in both files, add the counts
                merged_results[task]['Success Count'] += success_count
                merged_results[task]['Total Count'] += total_count
            else:
                # Task only exists in the second file
                merged_results[task] = {
                    'Success Count': success_count,
                    'Total Count': total_count
                }
        
        # Calculate success rates and create the output dataframe
        output_data = []
        for task, data in merged_results.items():
            success_count = data['Success Count']
            total_count = data['Total Count']
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            output_data.append({
                'Task': task,
                'Success Rate': f'{success_rate:.2f}%',
                'Success Count': success_count,
                'Total Count': total_count
            })
        
        # Convert to DataFrame and sort by task name
        output_df = pd.DataFrame(output_data)
        output_df = output_df.sort_values('Task')
    
    # Use the penultimate name (directory name) of the first CSV file
    # For example, from '/path/to/20250511_pi0_470000_gpt4o_10_400_no_history_image/file.csv'
    # Extract '20250511_pi0_470000_gpt4o_10_400_no_history_image'
    dir_path = os.path.dirname(csv_file1)
    penult_name = os.path.basename(dir_path)
    output_filename = f'{penult_name}.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f'Merged results saved to: {output_path}')
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Merge two CSV files with task success rates')
    parser.add_argument('--csv1', type=str, help='Path to the first CSV file')
    parser.add_argument('--csv2', type=str, help='Path to the second CSV file (optional)')
    parser.add_argument('--output', type=str, default='env_tests/success/results',
                        help='Directory to save the merged results')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # # Merge the CSV files (or just copy if csv2 is not provided)
    # output_path = merge_csv_results(args.csv1, args.csv2, args.output)
    
    # if args.csv2:
    #     print(f'Successfully merged {args.csv1} and {args.csv2}')
    # else:
    #     print(f'Successfully copied {args.csv1} to output directory')
    # print(f'Results saved to {output_path}')

    csv1 = "evaluation/202505010_pi0_470000_gemini_10_400_no_history_image/pi0_20250515_123437.csv"
    csv2 = "evaluation/202505010_pi0_470000_gemini_10_400_no_history_image_reverse/pi0_20250515_123430.csv"
    merge_csv_results(csv1, csv2, args.output)

if __name__ == '__main__':
    main()
