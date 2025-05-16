#!/usr/bin/env python
import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

def load_initial_vqa_results(file_path):
    """Load the initial VQA results from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract model name from filename
    model_name = os.path.basename(file_path).split('_f')[0].replace('vqa_initial_', '')
    
    correct_envs = set(data['correct_env'])
    wrong_envs = set(data['wrong_env'])
    
    return {
        'model': model_name,
        'correct_envs': correct_envs,
        'wrong_envs': wrong_envs,
        'correct_num': data['correct_num'],
        'total': len(correct_envs) + len(wrong_envs)
    }

def load_end_vqa_results(file_path):
    """Load the end VQA results from CSV file"""
    env_results = {}
    model_name = extract_model_from_csv_path(file_path)
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            env_id = row['Task']
            if 'Accuracy' in row and '%' in row['Accuracy']:
                accuracy = float(row['Accuracy'].strip('%')) / 100
            else:
                # Handle potential formatting issues
                success_correct = int(row.get('Success correct num', 0))
                total = int(row.get('Total num', 0))
                accuracy = success_correct / total if total > 0 else 0
            
            env_results[env_id] = {
                'accuracy': accuracy,
                'success_correct_num': int(row.get('Success correct num', 0)),
                'total_num': int(row.get('Total num', 0))
            }
    
    correct_envs = {env for env, result in env_results.items() 
                   if result['accuracy'] >= 0.5 and result['total_num'] > 0}
    wrong_envs = {env for env, result in env_results.items() 
                 if result['accuracy'] < 0.5 and result['total_num'] > 0}
    
    return {
        'model': model_name,
        'correct_envs': correct_envs,
        'wrong_envs': wrong_envs,
        'env_results': env_results,
        'total': len(correct_envs) + len(wrong_envs)
    }

def extract_model_from_csv_path(file_path):
    """Extract model name from CSV filename"""
    basename = os.path.basename(file_path)
    # The model is typically between the date and the config
    # Examples: 20250509_cogact_30000_gemini_10_400_no_history_image.csv
    #           20250511_pi0_470000_gpt4o_10_400_no_history_image_reverse.csv
    
    # Try to find the model name using patterns
    pattern = r'_(\w+)_\d+_\d+_'
    match = re.search(pattern, basename)
    if match:
        return match.group(1)
    
    # Fallback: return the whole filename without extension
    return os.path.splitext(basename)[0]

def compare_results(initial_results, end_results):
    """Compare initial and end VQA results for a specific model"""
    initial_model = initial_results['model']
    end_model = end_results['model']
    
    # Find matching models based on partial string match (e.g., gemini in initial and gemini in end)
    initial_model_key = initial_model.lower()
    end_model_key = end_model.lower()
    
    # Check if models are related (simplistic approach)
    models_match = False
    if 'gemini' in initial_model_key and 'gemini' in end_model_key:
        models_match = True
    elif 'gpt-4o' in initial_model_key and 'gpt4o' in end_model_key:
        models_match = True
    elif 'gpt-o1' in initial_model_key and 'gpt-o1' in end_model_key:
        models_match = True
    
    if not models_match:
        print(f"WARNING: Models may not match: {initial_model} vs {end_model}")
    
    # Calculate sets for analysis
    initial_correct = initial_results['correct_envs']
    initial_wrong = initial_results['wrong_envs']
    end_correct = end_results['correct_envs']
    end_wrong = end_results['wrong_envs']
    
    # Initialize result dictionary
    result = {
        'initial_model': initial_model,
        'end_model': end_model,
        'improved_envs': end_correct.intersection(initial_wrong),  # Wrong -> Correct
        'regressed_envs': end_wrong.intersection(initial_correct), # Correct -> Wrong
        'consistently_correct': end_correct.intersection(initial_correct),
        'consistently_wrong': end_wrong.intersection(initial_wrong),
        'initial_accuracy': len(initial_correct) / initial_results['total'] if initial_results['total'] > 0 else 0,
        'end_accuracy': len(end_correct) / end_results['total'] if end_results['total'] > 0 else 0,
        'improvement_rate': None  # Will be calculated
    }
    
    # Calculate improvement rate (changed from wrong to correct as a percentage of initially wrong)
    if len(initial_wrong) > 0:
        result['improvement_rate'] = len(result['improved_envs']) / len(initial_wrong)
    else:
        result['improvement_rate'] = 0
    
    # Calculate regression rate (changed from correct to wrong as a percentage of initially correct)
    if len(initial_correct) > 0:
        result['regression_rate'] = len(result['regressed_envs']) / len(initial_correct)
    else:
        result['regression_rate'] = 0
    
    return result

def find_matching_end_file(initial_file, end_dir):
    """Find a matching end results file for a given initial results file"""
    initial_model = load_initial_vqa_results(initial_file)['model'].lower()
    
    # Map initial model names to potential matching patterns in end files
    model_patterns = {
        'gemini': 'gemini',
        'gpt-4o': 'gpt4o',
        'gpt-o1': 'gpt-o1',
    }
    
    # Find the right pattern to look for
    matching_pattern = None
    for pattern_key, pattern_value in model_patterns.items():
        if pattern_key in initial_model:
            matching_pattern = pattern_value
            break
    
    if matching_pattern is None:
        return None
    
    # Get all end files
    end_files = [f for f in os.listdir(end_dir) if f.endswith('.csv')]
    
    # Find any files matching the pattern
    candidates = [os.path.join(end_dir, f) for f in end_files if matching_pattern.lower() in f.lower()]
    
    if candidates:
        return candidates[0]  # Return the first matching file
    
    return None

def analyze_vqa_results():
    """Analyze VQA results and generate a comprehensive report"""
    # Define paths
    initial_dir = "/home/wangxianhao/data/project/reasoning/ManiSkill/env_tests/vqa_results/vqa_initial"
    end_dir = "/home/wangxianhao/data/project/reasoning/ManiSkill/env_tests/vqa_results/results"
    
    # Get all initial result files
    initial_files = [os.path.join(initial_dir, f) for f in os.listdir(initial_dir) 
                    if f.endswith('.json') and f.startswith('vqa_initial_')]
    
    # Collect comparison results for each model
    all_comparisons = []
    
    for initial_file in initial_files:
        # Load initial results
        initial_results = load_initial_vqa_results(initial_file)
        model_name = initial_results['model']
        
        print(f"\nAnalyzing results for model: {model_name}")
        
        # Find matching end file
        matching_end_file = find_matching_end_file(initial_file, end_dir)
        
        if matching_end_file:
            print(f"Found matching end file: {os.path.basename(matching_end_file)}")
            end_results = load_end_vqa_results(matching_end_file)
            
            # Compare results
            comparison = compare_results(initial_results, end_results)
            all_comparisons.append(comparison)
            
            # Print summary
            print(f"Initial accuracy: {comparison['initial_accuracy']:.2%}")
            print(f"End accuracy: {comparison['end_accuracy']:.2%}")
            print(f"Improvement rate (wrong -> correct): {comparison['improvement_rate']:.2%}")
            print(f"Regression rate (correct -> wrong): {comparison['regression_rate']:.2%}")
            print(f"Improved environments: {len(comparison['improved_envs'])}")
            print(f"Regressed environments: {len(comparison['regressed_envs'])}")
            print(f"Consistently correct: {len(comparison['consistently_correct'])}")
            print(f"Consistently wrong: {len(comparison['consistently_wrong'])}")
            
            # Print detailed environment changes
            if comparison['improved_envs']:
                print("\nEnvironments that improved (wrong -> correct):")
                for env in sorted(comparison['improved_envs']):
                    print(f"  - {env}")
            
            if comparison['regressed_envs']:
                print("\nEnvironments that regressed (correct -> wrong):")
                for env in sorted(comparison['regressed_envs']):
                    print(f"  - {env}")
        else:
            print(f"No matching end file found for {model_name}")
    
    # Generate comparison visualization
    if all_comparisons:
        generate_comparison_chart(all_comparisons)

def generate_comparison_chart(comparisons):
    """Generate charts to visualize comparisons between initial and end results"""
    models = [c['initial_model'] for c in comparisons]
    initial_accuracies = [c['initial_accuracy'] * 100 for c in comparisons]
    end_accuracies = [c['end_accuracy'] * 100 for c in comparisons]
    improvement_rates = [c['improvement_rate'] * 100 for c in comparisons]
    regression_rates = [c['regression_rate'] * 100 for c in comparisons]
    
    # Set up the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model colors
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Plot accuracy comparison
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, initial_accuracies, width, label='Initial Accuracy', color=[colors[i % len(colors)] for i in range(len(models))])
    ax1.bar(x + width/2, end_accuracies, width, label='End Accuracy', color=[colors[i % len(colors)] for i in range(len(models))], alpha=0.7)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('VQA Accuracy: Initial vs. End')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot improvement and regression rates
    ax2.bar(x - width/2, improvement_rates, width, label='Improvement Rate', color='green')
    ax2.bar(x + width/2, regression_rates, width, label='Regression Rate', color='red')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Rate (%)')
    ax2.set_title('Improvement and Regression Rates')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the chart
    output_path = "/home/wangxianhao/data/project/reasoning/ManiSkill/env_tests/vqa_results/vqa_comparison_chart.png"
    plt.savefig(output_path)
    print(f"\nComparison chart saved to: {output_path}")
    
    # Generate detailed table
    generate_detailed_table(comparisons)

def generate_detailed_table(comparisons):
    """Generate a detailed comparison table as a CSV file and JSON file"""
    output_csv = "/home/wangxianhao/data/project/reasoning/ManiSkill/env_tests/vqa_results/vqa_comparison_details.csv"
    output_json = "/home/wangxianhao/data/project/reasoning/ManiSkill/env_tests/vqa_results/vqa_comparison_details.json"
    
    # Generate CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Initial Accuracy', 'End Accuracy', 'Change', 
                     'Improvement Rate', 'Regression Rate', 
                     'Consistently Correct', 'Consistently Wrong',
                     'Improved Envs', 'Improved Envs List',
                     'Regressed Envs', 'Regressed Envs List']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Store the rows for JSON
        json_data = []
        
        for c in comparisons:
            row = {
                'Model': c['initial_model'],
                'Initial Accuracy': f"{c['initial_accuracy']:.2%}",
                'End Accuracy': f"{c['end_accuracy']:.2%}",
                'Change': f"{(c['end_accuracy'] - c['initial_accuracy']):.2%}",
                'Improvement Rate': f"{c['improvement_rate']:.2%}",
                'Regression Rate': f"{c['regression_rate']:.2%}",
                'Consistently Correct': len(c['consistently_correct']),
                'Consistently Wrong': len(c['consistently_wrong']),
                'Improved Envs': len(c['improved_envs']),
                'Improved Envs List': ', '.join(sorted(c['improved_envs'])),
                'Regressed Envs': len(c['regressed_envs']),
                'Regressed Envs List': ', '.join(sorted(c['regressed_envs']))
            }
            writer.writerow(row)
            
            # Create a more structured JSON version with actual lists instead of comma-separated strings
            json_row = {
                'model': c['initial_model'],
                'initial_accuracy': c['initial_accuracy'],
                'end_accuracy': c['end_accuracy'],
                'change': c['end_accuracy'] - c['initial_accuracy'],
                'improvement_rate': c['improvement_rate'],
                'regression_rate': c['regression_rate'],
                'consistently_correct_count': len(c['consistently_correct']),
                'consistently_wrong_count': len(c['consistently_wrong']),
                'improved_envs_count': len(c['improved_envs']),
                'regressed_envs_count': len(c['regressed_envs']),
                'consistently_correct': sorted(list(c['consistently_correct'])),
                'consistently_wrong': sorted(list(c['consistently_wrong'])),
                'improved_envs': sorted(list(c['improved_envs'])),
                'regressed_envs': sorted(list(c['regressed_envs']))
            }
            json_data.append(json_row)
    
    # Generate JSON file
    with open(output_json, 'w') as jsonfile:
        json.dump(json_data, jsonfile, indent=2)
    
    print(f"Detailed comparison table saved to: {output_csv}")
    print(f"Detailed comparison JSON saved to: {output_json}")

if __name__ == "__main__":
    analyze_vqa_results()
