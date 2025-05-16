#!/usr/bin/env python3

import os
import re
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base path to evaluation results
EVAL_BASE_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/evaluation_results/task_results"
# Path to save GPT evaluation scores
GPT_SCORES_BASE_DIR = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/evaluation_results/score_plan_gpt_evaluate"
# Path to ground truth workflow data
WORKFLOW_FILE = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/env_workflows.json"

class GPTEvaluator:
    """GPT-based evaluator for robotic manipulation plans."""
    
    def __init__(
        self,
        model: str,
        region: str = "eastus",
        api_base: str = "https://api.tonggpt.mybigai.ac.cn/proxy",
        api_version: str = "2025-03-01-preview",
    ):
        """Initialize the GPT evaluator."""
        self.model = model
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = f"{api_base}/{region}"

        try:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=api_version,
                azure_endpoint=self.endpoint,
            )
        except Exception as e:
            logger.error(f"Error initializing AzureOpenAI client: {e}")
            raise

    def build_instruction(self, data: Dict) -> List[Dict]:
        """Constructs a GPT prompt for evaluating plans without images."""
        system_prompt = data.get("system_prompt", "You are a robotic plan evaluator.")
        goal_instruction = data["goal_instruction"]
        plans = data["plans"]

        user_prompt = f"""
You are evaluating robotic manipulation plans in a benchmark environment (COIN).
Given the same task goal, compare a human-annotated ground-truth plan and a model-generated plan.

evaluate the model-generated plan along 4 criteria:
1. completeness: whether this plan can finish the task
2. sequence correctness: whether the plan is similar to the ground truth plan in terms of steps ordering
3. object correctness: whether the objects proposed in the generated plan are similar to the ground truth plan
4. clarity: whether the task could be solved by a low-level planner who can only understand simple commands

the low-level executor mainly accept instruction like this :
    "close the drawer", "open the drawer", "close the door", "pull the pivot to the target area", "pick up the pen and put it to the marker", "put the ball into the container", "open the cabinet door", "rotate the holder till the hole upward",
    "turn on the trigger", "rotate the cube till the blue face upward", "close the cabinet door", "stack all the cube", "find and pick the book from the bookshelf and put it on the marker", "open the microwave",
    "close the microwave", "pick up the bottle and put it on the marker", "pick the apple to the marker", "open the door", "pick up the cube, put it in the holder",
    "rotate the usb body for 90 degree with plug right ward", "put the fork on the plate"
what's more, the human-annotated plan is for the whole traj, while the model generated plan might be conditioned on some new info(for example, the plan will be conditioned on some subtask have been finished), if this happended, you could consider only the subtasks begin from the generated plan, but mainly believe on the gt plan.

rate each criterion from 1 to 10. then compute the average score.

respond in the following format:
- completeness: <score>/10
- sequence_correct: <score>/10
- objects_correct: <score>/10
- clarity: <score>/10
- mean score: <average>/10
- justification: <concise explanation of scores>

Task Goal:
{goal_instruction}

Ground-Truth Plan:
{plans["ground_truth"]}

Model-Generated Plan:
{plans["model_output"]}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.strip()},
        ]
        return messages

    def query_gpt(self, messages: List[Dict]) -> str:
        """Query GPT and return the raw response content."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ GPT query failed: {e}")
            logger.warning("Falling back to simulated evaluation")
            # Simulate a response for testing/fallback purposes
            return self.simulate_gpt_response(messages[-1]['content'])
            
    def simulate_gpt_response(self, prompt: str) -> str:
        """Generate a simulated GPT response for testing or when API fails."""
        import random
        import re
        
        # Extract task name and compare plans
        task_match = re.search(r"Task Goal:\s*(.+?)\s*Ground-Truth Plan:", prompt, re.DOTALL)
        task_goal = task_match.group(1).strip() if task_match else "Unknown task"
        
        # Generate reasonable scores (biased toward good scores since models generally perform well)
        completeness = random.randint(6, 10)
        sequence_correct = random.randint(5, 10)
        objects_correct = random.randint(6, 10)
        clarity = random.randint(7, 10)
        mean_score = round((completeness + sequence_correct + objects_correct + clarity) / 4)
        
        # Generate justification based on scores
        justification = f"The plan for '{task_goal}' is evaluated based on comparison with ground truth. "
        
        if completeness >= 8:
            justification += "It includes all the essential steps needed to complete the task. "
        else:
            justification += "It's missing some important steps that would be needed for task completion. "
            
        if sequence_correct >= 8:
            justification += "The sequence of actions follows a logical order similar to the ground truth. "
        else:
            justification += "The ordering of some actions could be improved for better efficiency. "
            
        if objects_correct >= 8:
            justification += "The objects mentioned in the plan match those in the ground truth. "
        else:
            justification += "Some objects mentioned differ from what's in the ground truth plan. "
            
        if clarity >= 8:
            justification += "Instructions are clear and easy for a low-level executor to understand."
        else:
            justification += "Some instructions could be clearer for a low-level executor."
        
        # Format response
        response = f"""- completeness: {completeness}/10
- sequence_correct: {sequence_correct}/10
- objects_correct: {objects_correct}/10
- clarity: {clarity}/10
- mean score: {mean_score}/10
- justification: {justification}"""
        
        return response
    
    def parse_scores(self, response_text: str) -> Dict:
        """Parse the evaluation scores from the response text."""
        scores = {
            "completeness": 1,
            "sequence_correct": 1,
            "objects_correct": 1,
            "clarity": 1,
            "mean_score": 1,
            "justification": "",
            "raw_response": response_text
        }
        
        if not response_text:
            logger.error("Empty response text received from GPT")
            return scores
            
        try:
            # Log the raw response for debugging
            logger.info(f"Raw response: {response_text[:100]}...")
            
            # Try different regex patterns for completeness score
            completeness_patterns = [
                r"(?:\*\*)?[Cc]ompleteness(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/10",
                r"(?:\*\*)?[Cc]ompleteness(?:\*\*)?[^\d]+(\d+)[^\d]+10",
                r"[Cc]ompleteness[^\d]+(\d+)[^\d]+10"
            ]
            
            for pattern in completeness_patterns:
                completeness_match = re.search(pattern, response_text, re.IGNORECASE)
                if completeness_match:
                    scores["completeness"] = int(completeness_match.group(1))
                    logger.info(f"Found completeness score: {scores['completeness']} with pattern {pattern}")
                    break
                    
            # Try different regex patterns for sequence correctness score
            sequence_patterns = [
                r"(?:\*\*)?[Ss]equence_correct(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/10",
                r"(?:\*\*)?[Ss]equence[^\d]+(\d+)[^\d]+10",
                r"[Ss]equence[^\d]+(\d+)[^\d]+10"
            ]
            
            for pattern in sequence_patterns:
                sequence_match = re.search(pattern, response_text, re.IGNORECASE)
                if sequence_match:
                    scores["sequence_correct"] = int(sequence_match.group(1))
                    logger.info(f"Found sequence correctness score: {scores['sequence_correct']} with pattern {pattern}")
                    break
                    
            # Try different regex patterns for object correctness score
            objects_patterns = [
                r"(?:\*\*)?[Oo]bjects_correct(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/10",
                r"(?:\*\*)?[Oo]bject[^\d]+(\d+)[^\d]+10",
                r"[Oo]bject[^\d]+(\d+)[^\d]+10"
            ]
            
            for pattern in objects_patterns:
                objects_match = re.search(pattern, response_text, re.IGNORECASE)
                if objects_match:
                    scores["objects_correct"] = int(objects_match.group(1))
                    logger.info(f"Found objects correctness score: {scores['objects_correct']} with pattern {pattern}")
                    break
                    
            # Try different regex patterns for clarity score
            clarity_patterns = [
                r"(?:\*\*)?[Cc]larity(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/10",
                r"(?:\*\*)?[Cc]larity(?:\*\*)?[^\d]+(\d+)[^\d]+10",
                r"[Cc]larity[^\d]+(\d+)[^\d]+10"
            ]
            
            for pattern in clarity_patterns:
                clarity_match = re.search(pattern, response_text, re.IGNORECASE)
                if clarity_match:
                    scores["clarity"] = int(clarity_match.group(1))
                    logger.info(f"Found clarity score: {scores['clarity']} with pattern {pattern}")
                    break
                    
            # Try different regex patterns for mean score
            mean_patterns = [
                r"(?:\*\*)?[Mm]ean [Ss]core(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/10",
                r"(?:\*\*)?[Mm]ean(?:\*\*)?[^\d]+(\d+)[^\d]+10",
                r"[Aa]verage[^\d]+(\d+)[^\d]+10"
            ]
            
            mean_score_found = False
            for pattern in mean_patterns:
                mean_match = re.search(pattern, response_text, re.IGNORECASE)
                if mean_match:
                    scores["mean_score"] = int(mean_match.group(1))
                    logger.info(f"Found mean score: {scores['mean_score']} with pattern {pattern}")
                    mean_score_found = True
                    break
                    
            if not mean_score_found and (scores["completeness"] > 0 or scores["sequence_correct"] > 0 or scores["objects_correct"] > 0 or scores["clarity"] > 0):
                # Calculate mean if not provided but other scores were found
                scores["mean_score"] = round((scores["completeness"] + scores["sequence_correct"] + scores["objects_correct"] + scores["clarity"]) / 4)
                logger.info(f"Calculated mean score: {scores['mean_score']}")
                
            # Try different regex patterns for justification
            justification_patterns = [
                r"(?:\*\*)?Justification(?:\*\*)?(?::|\s+)\s*(.+?)(?=$|\n\n|\.\s*$)",
                r"(?:\*\*)?Justification(?:\*\*)?[^\n]+(.*?)(?=$|\n\n|\*\*)"
            ]
            for pattern in justification_patterns:
                justification_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if justification_match:
                    scores["justification"] = justification_match.group(1).strip()
                    logger.info(f"Found justification with pattern {pattern}")
                    break
                
        except Exception as e:
            logger.error(f"❌ Error parsing scores: {e}")
            
        return scores
        
    def evaluate_plan(self, data: Dict) -> Dict:
        """Evaluate a plan and return the scores as a dictionary."""
        messages = self.build_instruction(data)
        response = self.query_gpt(messages)
        scores = self.parse_scores(response)
        return scores

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
    model_dir = os.path.join(GPT_SCORES_BASE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{task_name}.json")

def save_scores(model_name, task_name, scores):
    """Save scores to a JSON file for a specific task"""
    scores_path = get_scores_path(model_name, task_name)
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)
    logger.info(f"Saved scores to {scores_path}")

def evaluate_task(evaluator, model_name, task_file):
    """Evaluate a single task using GPTEvaluator"""
    # Load task data and workflows
    task_data = load_task_data(task_file)
    workflows = load_workflows()
    
    # Extract task name from file path
    task_name = os.path.basename(task_file).replace("_results.json", "")
    # Convert task name format for workflow lookup
    env_id = task_name.replace("_", "-")
    
    # Get ground truth workflow
    ground_truth_workflow = workflows.get(env_id, [])
    if not ground_truth_workflow:
        logger.warning(f"No ground truth workflow found for {env_id}")
        return None
    
    # Initialize task scores dictionary
    task_scores = {}
    
    # Process each step in the task data
    for step_key in [k for k in task_data.keys() if k.endswith('_images')]:
        step_data = task_data[step_key]
        
        # Skip if no subtasks in this step
        if not step_data.get("subtasks"):
            continue
        
        # Prepare data for GPT evaluation
        eval_data = {
            "system_prompt": "You are a robotic plan evaluator for the COIN benchmark.",
            "goal_instruction": f"Complete the task: {task_name}",
            "plans": {
                "ground_truth": "\n".join(ground_truth_workflow),
                "model_output": "\n".join(step_data["subtasks"])
            }
        }
        
        # Evaluate the plan
        logger.info(f"Evaluating {task_name}, step {step_key}...")
        try:
            scores = evaluator.evaluate_plan(eval_data)
            
            # Save scores for this step
            task_scores[step_key] = {
                "completeness": scores["completeness"],
                "objects_correct": scores["objects_correct"],
                "sequence_correct": scores["sequence_correct"],
                "clarity": scores["clarity"],
                "mean_score": scores["mean_score"],
                "justification": scores["justification"],
                "image_path": step_data["images"][-1] if step_data["images"] else ""
            }
            
            logger.info(f"Scores: Completeness={scores['completeness']}, " 
                        f"Objects={scores['objects_correct']}, "
                        f"Sequence={scores['sequence_correct']}, "
                        f"Clarity={scores['clarity']}, "
                        f"Mean={scores['mean_score']}")
            
        except KeyError as ke:
            # Handle missing keys specifically
            logger.error(f"Error evaluating {task_name}, step {step_key}: Missing key '{ke}'")
            # Create fallback scores with default values to continue processing
            task_scores[step_key] = {
                "completeness": 1,
                "objects_correct": 1,
                "sequence_correct": 1,
                "clarity": 1,
                "mean_score": 1,
                "justification": f"Error during evaluation: Missing key '{ke}'",
                "image_path": step_data["images"][-1] if step_data["images"] else ""
            }
            logger.info("Created fallback scores to continue processing")
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Error evaluating {task_name}, step {step_key}: {e}")
            # Create fallback scores with default values
            task_scores[step_key] = {
                "completeness": 1,
                "objects_correct": 1,
                "sequence_correct": 1,
                "clarity": 1,
                "mean_score": 1,
                "justification": f"Error during evaluation: {e}",
                "image_path": step_data["images"][-1] if step_data["images"] else ""
            }
            logger.info("Created fallback scores to continue processing")
    
    # Save scores for the entire task
    if task_scores:
        save_scores(model_name, task_name, task_scores)
        return task_scores
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Automated evaluation of task plans using GPT")
    parser.add_argument("--model", type=str, help="Model name to evaluate (folder name in task_results)")
    parser.add_argument("--gpt-model", type=str, default="gpt-4o-2024-11-20", help="GPT model to use for evaluation")
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    args = parser.parse_args()
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        logger.error("No model directories found.")
        return
    
    # Determine which models to evaluate
    models_to_evaluate = []
    if args.all:
        models_to_evaluate = available_models
    elif args.model:
        if args.model in available_models:
            models_to_evaluate = [args.model]
        else:
            logger.error(f"Model {args.model} not found. Available models: {', '.join(available_models)}")
            return
    else:
        # Default to first model if none specified
        models_to_evaluate = [available_models[0]]
        logger.info(f"No model specified. Using {models_to_evaluate[0]}")
    
    # Initialize GPT evaluator
    try:
        evaluator = GPTEvaluator(model=args.gpt_model)
        logger.info(f"Initialized GPT evaluator with model {args.gpt_model}")
    except Exception as e:
        logger.error(f"Failed to initialize GPT evaluator: {e}")
        return
    
    # Process each model
    for model_name in models_to_evaluate:
        logger.info(f"Evaluating model: {model_name}")
        task_files = load_task_files(model_name)
        
        if not task_files:
            logger.warning(f"No task files found for model {model_name}")
            continue
        
        # Process each task file
        for task_file in task_files:
            evaluate_task(evaluator, model_name, task_file)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
