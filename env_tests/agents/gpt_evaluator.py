import os
import re
import logging
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class GPTEvaluator:
    """GPT-based evaluator for robotic manipulation plans."""
    
    def __init__(
        self,
        model: str,
        region: str = "eastus",
        api_base: str = "https://api.tonggpt.mybigai.ac.cn/proxy",
        api_version: str = "2025-03-01-preview",
    ):
        """Initialize the GPT evaluator.
        
        Args:
            model: The model to use for evaluation
            region: Azure OpenAI region
            api_base: API base URL
            api_version: API version
        """
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
        """Constructs a GPT prompt for evaluating plans without images.
        
        Args:
            data: Dictionary containing system_prompt, goal_instruction, and plans
            
        Returns:
            List of message dictionaries for the GPT API
        """
        system_prompt = data.get("system_prompt", "You are a robotic plan evaluator.")
        goal_instruction = data["goal_instruction"]
        plans = data["plans"]

        user_prompt = f"""
You are evaluating robotic manipulation plans in a benchmark environment (COIN).
Given the same task goal, compare a human-annotated ground-truth plan and a model-generated plan.

Evaluate the model-generated plan along 3 criteria:
1. Completeness: Does it include all essential steps?
2. Correctness: Are the steps logically and sequentially correct? Especially the orider
3. Clarity: Are the steps described in a clear and understandable way? And the task should be understandable for low-level executor who could accept only <action> the <object>

The low-level executor mainly accept instruction like this :
    "close the drawer", "open the drawer", "close the door", "pull the pivot to the target area", "pick up the pen and put it to the marker", "put the ball into the container", "open the cabinet door", "rotate the holder till the hole upward",
    "turn on the trigger", "rotate the cube till the blue face upward", "close the cabinet door", "stack all the cube", "Find and pick the book from the bookshelf and put it on the marker", "open the microwave",
    "close the microwave", "pick up the bottle and put it on the marker", "pick the apple to the marker", "open the door", "pick up the cube, put it in the holder",
    "Rotate the USB body for 90 degree with plug right ward", "put the fork on the plate"
What's more, the human-annotated plan is for the whole traj, while the model generated plan might be conditioned on some new info(for example, the plan will be conditioned on some subtask have been finished), if this happended, you could consider only the subtasks begin from the generated plan, but mainly believe on the gt plan.

Rate each criterion from 0 to 100. Then compute the average score.

Respond in the following format:
- Completeness: <score>/100
- Correctness: <score>/100
- Clarity: <score>/100
- Mean Score: <average>/100
- Justification: <concise explanation of scores>

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
        """Query GPT and return the raw response content.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Raw response content from GPT
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ GPT query failed: {e}")
            return f"Error: {str(e)}"
            
    def parse_scores(self, response_text: str) -> Dict:
        """Parse the evaluation scores from the response text.
        
        Args:
            response_text: Raw response text from GPT
            
        Returns:
            Dictionary containing parsed scores and justification
        """
        scores = {
            "completeness": 0,
            "correctness": 0,
            "clarity": 0,
            "mean_score": 0,
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
                r"(?:\*\*)?Completeness(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/100",
                r"(?:\*\*)?Completeness(?:\*\*)?[^\d]+(\d+)[^\d]+100",
                r"completeness[^\d]+(\d+)[^\d]+100"
            ]
            
            for pattern in completeness_patterns:
                completeness_match = re.search(pattern, response_text, re.IGNORECASE)
                if completeness_match:
                    scores["completeness"] = int(completeness_match.group(1))
                    logger.info(f"Found completeness score: {scores['completeness']} with pattern {pattern}")
                    break
                    
            # Try different regex patterns for correctness score
            correctness_patterns = [
                r"(?:\*\*)?Correctness(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/100",
                r"(?:\*\*)?Correctness(?:\*\*)?[^\d]+(\d+)[^\d]+100",
                r"correctness[^\d]+(\d+)[^\d]+100"
            ]
            
            for pattern in correctness_patterns:
                correctness_match = re.search(pattern, response_text, re.IGNORECASE)
                if correctness_match:
                    scores["correctness"] = int(correctness_match.group(1))
                    logger.info(f"Found correctness score: {scores['correctness']} with pattern {pattern}")
                    break
                    
            # Try different regex patterns for clarity score
            clarity_patterns = [
                r"(?:\*\*)?Clarity(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/100",
                r"(?:\*\*)?Clarity(?:\*\*)?[^\d]+(\d+)[^\d]+100",
                r"clarity[^\d]+(\d+)[^\d]+100"
            ]
            
            for pattern in clarity_patterns:
                clarity_match = re.search(pattern, response_text, re.IGNORECASE)
                if clarity_match:
                    scores["clarity"] = int(clarity_match.group(1))
                    logger.info(f"Found clarity score: {scores['clarity']} with pattern {pattern}")
                    break
                    
            # Try different regex patterns for mean score
            mean_patterns = [
                r"(?:\*\*)?Mean Score(?:\*\*)?(?::|:\s+|\s+)\s*(\d+)/100",
                r"(?:\*\*)?Mean(?:\*\*)?[^\d]+(\d+)[^\d]+100",
                r"average[^\d]+(\d+)[^\d]+100"
            ]
            
            mean_score_found = False
            for pattern in mean_patterns:
                mean_match = re.search(pattern, response_text, re.IGNORECASE)
                if mean_match:
                    scores["mean_score"] = int(mean_match.group(1))
                    logger.info(f"Found mean score: {scores['mean_score']} with pattern {pattern}")
                    mean_score_found = True
                    break
                    
            if not mean_score_found and (scores["completeness"] > 0 or scores["correctness"] > 0 or scores["clarity"] > 0):
                # Calculate mean if not provided but other scores were found
                scores["mean_score"] = round((scores["completeness"] + scores["correctness"] + scores["clarity"]) / 3)
                logger.info(f"Calculated mean score: {scores['mean_score']}")
                
            # Log the parsed scores for debugging
            logger.info(f"Parsed scores: {scores}")
                
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
        """Evaluate a plan and return the scores as a dictionary.
        
        Args:
            data: Dictionary containing system_prompt, goal_instruction, and plans
            
        Returns:
            Dictionary containing parsed scores and justification
        """
        messages = self.build_instruction(data)
        response = self.query_gpt(messages)
        scores = self.parse_scores(response)
        return scores
