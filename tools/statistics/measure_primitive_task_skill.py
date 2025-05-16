import csv
import os
import pickle
from collections import Counter

# Define the skills to look for
SKILLS = [
    "pick", "place", "open", "close", "move", "align", "rotate", "put", 
    "push", "pull", "insert", "lift", "squeeze", "turn", "stack", "find"
]

# Load primitive tasks from pickle file
def load_primitive_tasks():
    pickle_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'primitive_instruction_objects.pkl')
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create a mapping of task instructions to task names (env IDs)
    task_mapping = {v['ins']: k for k, v in data.items()}
    
    # Extract instructions from the pickle file
    primitive_tasks = [v['ins'] for v in data.values()]
    
    return primitive_tasks, task_mapping

def extract_skills_from_task(task):
    """Extract skills from a task description."""
    task = task.lower()
    found_skills = []
    
    for skill in SKILLS:
        if skill in task:
            found_skills.append(skill)
    
    return found_skills

def main():
    # Load tasks from pickle file
    primitive_tasks, task_mapping = load_primitive_tasks()
    
    # Dictionary to store skills per task
    task_skills = {}
    # Counter for overall skill distribution
    all_skills = Counter()
    
    # Process each task
    for task in primitive_tasks:
        skills = extract_skills_from_task(task)
        task_skills[task] = list(set(skills))  # Store unique skills
        
        # Update overall skill counter
        all_skills.update(set(skills))
    
    # Output directory
    output_dir =  'github_page/static'
    os.makedirs(output_dir, exist_ok=True)
    
    # Output path for CSV
    csv_path = os.path.join(output_dir, 'primitive_task_skills.csv')
    
    # Write task-skill mapping to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_name'] + SKILLS)
        
        for task_instruction, skills in task_skills.items():
            # Get the task name (env ID) from the mapping
            task_name = task_mapping[task_instruction]
            # Create a row with 1/0 indicating if the skill is used
            row = [task_name] + [1 if skill in skills else 0 for skill in SKILLS]
            writer.writerow(row)
    
    # Output path for skill distribution CSV
    dist_csv_path = os.path.join(output_dir, 'primitive_skill_distribution.csv')
    
    # Write skill distribution to CSV
    with open(dist_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['skill', 'count'])
        
        # Sort skills by frequency
        for skill, count in all_skills.most_common():
            writer.writerow([skill, count])
    
    # Print summary
    print(f"Analyzed {len(primitive_tasks)} primitive tasks")
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