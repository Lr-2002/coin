#!/usr/bin/env python3

import os
import json
import glob
import streamlit as st
from PIL import Image
import pandas as pd
from pathlib import Path
import re

# Base path to evaluation results
EVAL_BASE_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/evaluation_results/task_results"
# Path to save evaluation scores
SCORES_BASE_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/evaluation_results/score_plan"
# Path to ground truth workflow data
WORKFLOW_FILE = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/env_workflows.json"

def get_available_models():
    """Get list of available models (directory names in EVAL_BASE_DIR)"""
    model_dirs = [d for d in os.listdir(EVAL_BASE_DIR) 
                if os.path.isdir(os.path.join(EVAL_BASE_DIR, d))]
    model_dirs.sort()
    return model_dirs

def load_task_files(model_name):
    """Load all task JSON files from the results directory for a given model"""
    results_dir = os.path.join(EVAL_BASE_DIR, model_name)
    task_files = glob.glob(os.path.join(results_dir, "*.json"))
    task_files.sort()
    return task_files

def load_task_data(task_file):
    """Load data from a task JSON file"""
    with open(task_file, 'r') as f:
        return json.load(f)
        
def load_workflows():
    """Load workflow data from env_workflows.json"""
    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_scores_path(model_name, task_name):
    """Get path to scores file for a specific model and task"""
    # Create directory for model if it doesn't exist
    model_dir = os.path.join(SCORES_BASE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{task_name}.json")

def load_scores(model_name, task_name):
    """Load previously saved scores for a specific task"""
    scores_path = get_scores_path(model_name, task_name)
    if os.path.exists(scores_path):
        with open(scores_path, 'r') as f:
            return json.load(f)
    return {}

def save_scores(model_name, task_name, scores):
    """Save scores to a JSON file for a specific task"""
    scores_path = get_scores_path(model_name, task_name)
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)

def main():
    st.set_page_config(layout="wide", page_title="Task Plan Evaluation")
    
    # Initialize session state for navigation
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'task_index' not in st.session_state:
        st.session_state.task_index = 0
    if 'step_index' not in st.session_state:
        st.session_state.step_index = 0
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        st.error("No model directories found.")
        return
    
    # Model selection in sidebar
    st.sidebar.title("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=available_models.index(st.session_state.model_name) if st.session_state.model_name in available_models else 0
    )
    
    # Update model name in session state
    if selected_model != st.session_state.model_name:
        st.session_state.model_name = selected_model
        st.session_state.task_index = 0
        st.session_state.step_index = 0
    
    # Load task files for the selected model
    task_files = load_task_files(selected_model)
    if not task_files:
        st.error(f"No task files found for model {selected_model}.")
        return
    
    # Load workflow data for ground truth
    workflows = load_workflows()
    
    # Sidebar for task selection
    st.sidebar.title("Task Selection")
    task_names = [os.path.basename(f).replace("_results.json", "") for f in task_files]
    
    # Use session state for task index
    if st.session_state.task_index >= len(task_names):
        st.session_state.task_index = 0
    
    selected_task_index = st.sidebar.selectbox(
        "Select Task",
        range(len(task_names)),
        index=st.session_state.task_index,
        format_func=lambda i: task_names[i]
    )
    
    # Update task index in session state if changed via selectbox
    if selected_task_index != st.session_state.task_index:
        st.session_state.task_index = selected_task_index
        st.session_state.step_index = 0
    
    # Load selected task data
    selected_task_file = task_files[selected_task_index]
    task_data = load_task_data(selected_task_file)
    task_name = task_names[selected_task_index]
    
    # Load previously saved scores for this task
    scores = load_scores(selected_model, task_name)
    
    # Extract env_id from task_name to match with workflows
    # Convert "Tabletop_Pick_Apple_v1" to "Tabletop-Pick-Apple-v1"
    env_id = task_name.replace("_", "-")
    
    # Display task information with model name
    st.title(f"Model: {selected_model} - Task: {task_name}")
    
    # Get ground truth workflow for this env_id
    ground_truth_workflow = workflows.get(env_id, [])
    
    # Determine available steps
    steps = [k for k in task_data.keys() if k.endswith('_images')]
    
    # Check if step_index is valid for this task's steps
    if st.session_state.step_index >= len(steps):
        st.session_state.step_index = 0
    
    # Step selection
    step_index = st.sidebar.selectbox(
        "Select Step",
        range(len(steps)),
        index=st.session_state.step_index,
        format_func=lambda i: steps[i]
    )
    
    # Update session state if changed via selectbox
    if step_index != st.session_state.step_index:
        st.session_state.step_index = step_index
    
    current_step = steps[step_index]
    step_data = task_data[current_step]
    
    # Display images if available
    st.subheader(f"Step: {current_step}")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if step_data["images"]:
            # Display the last image
            image_path = step_data["images"][-1]
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=f"Image from step {current_step}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.warning(f"Image file not found: {image_path}")
        else:
            st.info("No images available for this step.")
    
    with col2:
        # Use ground truth workflow from env_workflows.json
        
        # Create a dataframe to display generated and ground truth subtasks side by side
        generated_subtasks = step_data["subtasks"]
        max_len = max(len(generated_subtasks), len(ground_truth_workflow))
        
        # Pad shorter list with empty strings
        generated_padded = generated_subtasks + [''] * (max_len - len(generated_subtasks))
        ground_truth_padded = ground_truth_workflow + [''] * (max_len - len(ground_truth_workflow))
        
        # Create DataFrame for display
        df = pd.DataFrame({
            "Generated Subtasks": generated_padded,
            "Ground Truth Subtasks": ground_truth_padded
        })
        
        st.dataframe(df, use_container_width=True)
        
        # Display assessment if available
        if "assessment" in step_data:
            st.subheader("Assessment")
            st.write(step_data["assessment"])
    
    # Evaluation form
    st.subheader("Evaluation")
    
    # Initialize score dictionary for this task and step if it doesn't exist
    if task_name not in scores:
        scores[task_name] = {}
    if current_step not in scores[task_name]:
        scores[task_name][current_step] = {
            "completeness": 1,
            "objects_correct": 1,
            "sequence_correct": 1,
            "clarity": 1,
            "image_path": step_data["images"][-1] if step_data["images"] else "",
            "notes": ""
        }
    
    # Create scoring form with number inputs (0-100) instead of sliders
    col1, col2 = st.columns(2)
    
    with col1:
        completeness = st.number_input(
            "Completeness (1-10)", 
            min_value=1, max_value=10, 
            value=int(scores[task_name][current_step]["completeness"]),
            step=1
        )
        
        objects_correct = st.number_input(
            "Objects Correctness (1-10)", 
            min_value=1, max_value=10, 
            value=int(scores[task_name][current_step]["objects_correct"]),
            step=1
        )
    
    with col2:
        sequence_correct = st.number_input(
            "Sequence Correctness (1-10)", 
            min_value=1, max_value=10, 
            value=int(scores[task_name][current_step]["sequence_correct"]),
            step=1
        )
        
        clarity = st.number_input(
            "Clarity (1-10)", 
            min_value=1, max_value=10, 
            value=int(scores[task_name][current_step]["clarity"]),
            step=1
        )
    
    notes = st.text_area(
        "Notes", 
        value=scores[task_name][current_step]["notes"],
        height=100
    )
    
    # Update scores
    scores[task_name][current_step] = {
        "completeness": completeness,
        "objects_correct": objects_correct,
        "sequence_correct": sequence_correct,
        "clarity": clarity,
        "image_path": step_data["images"][-1] if step_data["images"] else "",
        "notes": notes
    }
    
    # Navigation buttons for steps (on first line)
    st.write("### Step Navigation")
    step_col1, step_col2 = st.columns(2)
    
    with step_col1:
        if st.button("⬅️ Previous Step", key="prev_step"):
            if step_index > 0:
                # Auto-save current step scores before navigating
                save_scores(selected_model, task_name, scores)
                # Update session state
                st.session_state.step_index = step_index - 1
                st.rerun()
    
    with step_col2:
        if st.button("Next Step ➡️", key="next_step"):
            if step_index < len(steps) - 1:
                # Auto-save current step scores before navigating
                save_scores(selected_model, task_name, scores)
                # Update session state
                st.session_state.step_index = step_index + 1
                st.rerun()
    
    # Navigation buttons for tasks (on second line)
    st.write("### Task Navigation")
    task_col1, task_col2 = st.columns(2)
    
    with task_col1:
        if st.button("⬅️ Previous Task", key="prev_task"):
            if selected_task_index > 0:
                # Auto-save current task scores before navigating
                save_scores(selected_model, task_name, scores)
                # Update session state
                st.session_state.task_index = selected_task_index - 1
                st.session_state.step_index = 0  # Reset step index for new task
                st.rerun()
    
    with task_col2:
        if st.button("Next Task ➡️", key="next_task"):
            if selected_task_index < len(task_files) - 1:
                # Auto-save current task scores before navigating
                save_scores(selected_model, task_name, scores)
                # Update session state
                st.session_state.task_index = selected_task_index + 1
                st.session_state.step_index = 0  # Reset step index for new task
                st.rerun()
    
    # Save button
    if st.button("Save Evaluation", type="primary"):
        save_scores(selected_model, task_name, scores)
        st.success(f"Evaluation saved to {get_scores_path(selected_model, task_name)}")
    
    # Display overall statistics
    if st.sidebar.checkbox("Show Overall Statistics"):
        st.sidebar.subheader("Overall Task Statistics")
        
        # Get all score files for this model
        model_dir = os.path.join(SCORES_BASE_DIR, selected_model)
        if os.path.exists(model_dir):
            all_scores = []
            tasks_evaluated = 0
            steps_evaluated = 0
            
            # Loop through all task files in the model directory
            for task_file in glob.glob(os.path.join(model_dir, "*.json")):
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    tasks_evaluated += 1
                    steps_evaluated += len(task_data)
                    
                    # Extract scores from each step
                    for step in task_data.values():
                        all_scores.append([
                            step.get("completeness", 1),
                            step.get("objects_correct", 1),
                            step.get("sequence_correct", 1),
                            step.get("clarity", 1)
                        ])
            
            if all_scores:
                # Create a DataFrame for statistics
                stats_df = pd.DataFrame({
                    "Completeness": [score[0] for score in all_scores],
                    "Objects Correct": [score[1] for score in all_scores],
                    "Sequence Correct": [score[2] for score in all_scores],
                    "Clarity": [score[3] for score in all_scores]
                })
                
                st.sidebar.write("Average Scores:")
                st.sidebar.write(stats_df.mean())
                
                st.sidebar.write("Tasks Evaluated:", tasks_evaluated)
                st.sidebar.write("Steps Evaluated:", steps_evaluated)

if __name__ == "__main__":
    main()
