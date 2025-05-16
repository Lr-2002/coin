#!/usr/bin/env python
import os
import re
import csv
import glob
from pathlib import Path

# Base directory containing all evaluation chat files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_CHAT_DIR = os.path.join(BASE_DIR, 'evaluate_chat')

# Regular expressions to extract information from markdown files
GT_PLAN_PATTERN = r'## Ground Truth Plan\s*```([\s\S]*?)```'
GEN_PLAN_PATTERN = r'## Generated Plan\s*```([\s\S]*?)```'
SCORE_PATTERN = r'- (Completeness|Correctness|Clarity|Mean Score): (\d+)/100'
JUSTIFICATION_PATTERN = r'## Justification\s*([\s\S]*?)(?=## Raw|$)'
RAW_RESPONSE_PATTERN = r'## Raw GPT Response\s*```([\s\S]*?)```'

def extract_plan_text(content, pattern):
    """Extract plan text from markdown content using regex pattern"""
    match = re.search(pattern, content, re.DOTALL)
    if match:
        raw_text = match.group(1).strip()
        
        # Check if the text is in the character-by-character format with step numbers
        if re.search(r'^Step \d+: [a-zA-Z0-9]$', raw_text, re.MULTILINE):
            # This is a character-by-character format, let's reconstruct it
            lines = raw_text.split('\n')
            reconstructed_steps = {}
            current_step = None
            current_text = ""
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Check if this is a new step indicator
                step_match = re.match(r'^Step (\d+): (.*)$', line)
                if step_match:
                    step_num = int(step_match.group(1))
                    char = step_match.group(2)
                    
                    # If we encounter "Step 1:", it's the start of a new step
                    if step_num == 1 and current_step is not None:
                        reconstructed_steps[current_step] = current_text.strip()
                        current_text = ""
                        current_step = None
                    
                    # If we don't have a current step yet, this is the first character of a step number
                    if current_step is None:
                        # Try to determine the step number from consecutive lines
                        step_number_chars = ""
                        for i in range(10):  # Look ahead up to 10 lines to find the step number
                            if i < len(lines) - lines.index(line):
                                next_line = lines[lines.index(line) + i]
                                next_match = re.match(r'^Step \d+: (.*)$', next_line)
                                if next_match:
                                    step_number_chars += next_match.group(1)
                                    if step_number_chars.strip() and re.match(r'^Step \d+:$', step_number_chars.strip()):
                                        step_match = re.search(r'Step (\d+):', step_number_chars.strip())
                                        if step_match:
                                            current_step = int(step_match.group(1))
                                            break
                    
                    # Add the character to the current text
                    current_text += char
            
            # Add the last step
            if current_step is not None and current_text:
                reconstructed_steps[current_step] = current_text.strip()
            
            # If we couldn't reconstruct steps properly, just return the original text
            if not reconstructed_steps:
                return raw_text
            
            # Format the reconstructed steps
            result = []
            for step_num in sorted(reconstructed_steps.keys()):
                result.append(f"Step {step_num}: {reconstructed_steps[step_num]}")
            
            return "\n".join(result)
        else:
            # This is already in a proper format
            return raw_text
    return ""

def extract_scores(content):
    """Extract all scores from markdown content"""
    scores = {}
    for match in re.finditer(SCORE_PATTERN, content):
        score_type = match.group(1)
        score_value = float(match.group(2))
        scores[score_type.lower()] = score_value
    return scores

def extract_justification(content):
    """Extract justification text from markdown content"""
    match = re.search(JUSTIFICATION_PATTERN, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_raw_response(content):
    """Extract raw GPT response from markdown content"""
    match = re.search(RAW_RESPONSE_PATTERN, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def get_steps_from_filename(filename):
    """Extract steps number from filename (e.g., 3_images_20250510_004353.md -> 3)"""
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    if parts and parts[0].isdigit():
        return int(parts[0])
    return 0

def process_task_directory(task_dir):
    """Process all markdown files in a task directory and create a CSV file"""
    task_name = os.path.basename(task_dir)
    print(f"Processing task: {task_name}")
    
    # CSV file path
    csv_path = os.path.join(task_dir, f"{task_name}.csv")
    
    # Find all markdown files in the directory
    md_files = glob.glob(os.path.join(task_dir, "*.md"))
    
    # Prepare data for CSV
    rows = []
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract information
            steps = get_steps_from_filename(md_file)
            gt_plan = extract_plan_text(content, GT_PLAN_PATTERN)
            gen_plan = extract_plan_text(content, GEN_PLAN_PATTERN)
            scores = extract_scores(content)
            justification = extract_justification(content)
            raw_response = extract_raw_response(content)
            
            # Add row to data
            rows.append({
                'steps': steps,
                'gt_plan': gt_plan,
                'generated_plan': gen_plan,
                'completeness': scores.get('completeness', 0),
                'correctness': scores.get('correctness', 0),
                'clarity': scores.get('clarity', 0),
                'mean_score': scores.get('mean score', 0),
                'justification': justification,
                'raw_gpt_response': raw_response
            })
        except Exception as e:
            print(f"Error processing file {md_file}: {e}")
    
    # Sort rows by steps
    rows.sort(key=lambda x: x['steps'])
    
    # Write to CSV
    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['steps', 'gt_plan', 'generated_plan', 'completeness', 
                         'correctness', 'clarity', 'mean_score', 'justification', 
                         'raw_gpt_response']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Created CSV file: {csv_path}")
    else:
        print(f"No data found for task: {task_name}")

def main():
    """Main function to process all task directories"""
    # Get all task directories
    task_dirs = [d for d in glob.glob(os.path.join(EVAL_CHAT_DIR, "*")) if os.path.isdir(d)]
    
    for task_dir in task_dirs:
        process_task_directory(task_dir)
    
    print(f"Processed {len(task_dirs)} task directories")

if __name__ == "__main__":
    main()
