import json
import os
import csv
import re
from collections import Counter

# Define the skills to look for
SKILLS = [
    "pick", "place", "open", "close", "move", "align", "rotate", "put", 
    "push", "pull", "insert", "lift", "squeeze"
]

def extract_skills_from_step(step):
    """Extract skills from a workflow step."""
    step = step.lower()
    found_skills = []
    
    for skill in SKILLS:
        if skill in step:
            found_skills.append(skill)
    
    return found_skills

def main():
    # Path to the JSON file
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env_workflows.json')
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        workflows = json.load(f)
    
    # Dictionary to store skills per task
    task_skills = {}
    # Counter for overall skill distribution
    all_skills = Counter()
    
    # Process each task
    for task_name, steps in workflows.items():
        task_skills[task_name] = []
        
        # Process each step in the workflow
        for step in steps:
            skills = extract_skills_from_step(step)
            task_skills[task_name].extend(skills)
        
        # Count unique skills for this task
        unique_skills = set(task_skills[task_name])
        task_skills[task_name] = list(unique_skills)
        
        # Update overall skill counter
        all_skills.update(unique_skills)
    
    # Output path for CSV
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'github_page/static/task_skills.csv')
    
    # Write task-skill mapping to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_name'] + SKILLS)
        
        for task_name, skills in task_skills.items():
            # Create a row with 1/0 indicating if the skill is used
            row = [task_name] + [1 if skill in skills else 0 for skill in SKILLS]
            writer.writerow(row)
    
    # Output path for skill distribution CSV
    dist_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'github_page/static/skill_distribution.csv')
    
    # Write skill distribution to CSV
    with open(dist_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['skill', 'count'])
        
        # Sort skills by frequency
        for skill, count in all_skills.most_common():
            writer.writerow([skill, count])
    
    # Print summary
    print(f"Analyzed {len(workflows)} tasks")
    print(f"Task-skill mapping CSV generated at: {csv_path}")
    print(f"Skill distribution CSV generated at: {dist_csv_path}")
    print("\nSkill distribution:")
    for skill, count in all_skills.most_common():
        print(f"{skill}: {count} tasks")
    
    # Calculate average number of skills per task
    skill_counts = [len(skills) for skills in task_skills.values()]
    avg_skills = sum(skill_counts) / len(skill_counts) if skill_counts else 0
    print(f"\nAverage number of skills per task: {avg_skills:.2f}")
    
    # Find tasks with the most skills
    max_skills = max(skill_counts) if skill_counts else 0
    tasks_with_max_skills = [task for task, skills in task_skills.items() if len(skills) == max_skills]
    print(f"\nTasks with the most skills ({max_skills}):")
    for task in tasks_with_max_skills:
        print(f"- {task}: {', '.join(task_skills[task])}")

if __name__ == "__main__":
    main()
