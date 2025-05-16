import os
import json
import re
from openai import AzureOpenAI
from typing import List, Dict, Optional, Union, Any


class GPTEvaluator:
    def __init__(
        self,
        model: str,
        region: str = "eastus",
        api_base: str = "https://api.tonggpt.mybigai.ac.cn/proxy",
        api_version: str = "2025-03-01-preview",
    ):
        self.model = model
        self.api_key = os.environ["Azure_Token"]
        self.endpoint = f"{api_base}/{region}"

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=api_version,
            azure_endpoint=self.endpoint,
        )

    def build_instruction(self, data: Dict) -> List[Dict]:
        """
        Constructs a GPT prompt for evaluating plans without images.
        """
        system_prompt = data.get("system_prompt", "You are a robotic plan evaluator.")
        goal_instruction = data["goal_instruction"]
        plans = data["plans"]

        user_prompt = f"""
You are evaluating robotic manipulation plans in a benchmark environment (COIN).
Given the same task goal, compare a human-annotated ground-truth plan and a model-generated plan.

Evaluate the model-generated plan along 3 criteria:
1. Completeness: Does it include all essential steps?
2. Correctness: Are the steps logically and sequentially correct?
3. Clarity: Are the steps described in a clear and understandable way?

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
        """Query GPT and return the raw response content."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ GPT query failed: {e}")
            return ""
            
    def parse_scores(self, response_text: str) -> Dict:
        """Parse the evaluation scores from the response text."""
        scores = {
            "completeness": 0,
            "correctness": 0,
            "clarity": 0,
            "mean_score": 0,
            "justification": "",
            "raw_response": response_text
        }
        
        try:
            # Extract scores using regular expressions
            import re
            
            # Find completeness score
            completeness_match = re.search(r"Completeness:\s*(\d+)/100", response_text)
            if completeness_match:
                scores["completeness"] = int(completeness_match.group(1))
                
            # Find correctness score
            correctness_match = re.search(r"Correctness:\s*(\d+)/100", response_text)
            if correctness_match:
                scores["correctness"] = int(correctness_match.group(1))
                
            # Find clarity score
            clarity_match = re.search(r"Clarity:\s*(\d+)/100", response_text)
            if clarity_match:
                scores["clarity"] = int(clarity_match.group(1))
                
            # Find mean score
            mean_match = re.search(r"Mean Score:\s*(\d+)/100", response_text)
            if mean_match:
                scores["mean_score"] = int(mean_match.group(1))
            else:
                # Calculate mean if not provided
                scores["mean_score"] = round((scores["completeness"] + scores["correctness"] + scores["clarity"]) / 3)
                
            # Extract justification
            justification_match = re.search(r"Justification:\s*(.+?)(?=$|\n\n|\.\s*$)", response_text, re.DOTALL)
            if justification_match:
                scores["justification"] = justification_match.group(1).strip()
                
        except Exception as e:
            print(f"❌ Error parsing scores: {e}")
            
        return scores
        
    def evaluate_plan(self, data: Dict) -> Dict:
        """Evaluate a plan and return the scores as a dictionary."""
        messages = self.build_instruction(data)
        response = self.query_gpt(messages)
        scores = self.parse_scores(response)
        return scores


# === One-shot Example ===
if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    evaluator = GPTEvaluator(model="gpt-4o-mini-2024-07-18")

    sample_data = {
        "system_prompt": "You are an expert evaluator for robotic manipulation tasks.",
        "goal_instruction": "The robot should open the microwave, place the cup inside, and close the microwave.",
        "plans": {
            "ground_truth": "Step 1: Open the microwave door.\nStep 2: Pick up the cup.\nStep 3: Place the cup inside the microwave.\nStep 4: Close the microwave door.",
            "model_output": "Step 1: Pick up the cup.\nStep 2: Put it in the microwave.\nStep 3: Close the door.",
        },
    }

    # Method 1: Get raw response
    messages = evaluator.build_instruction(sample_data)
    response = evaluator.query_gpt(messages)
    print("\n💬 GPT Evaluation Raw Result:\n", response)
    
    # Method 2: Get parsed scores as dictionary
    scores = evaluator.evaluate_plan(sample_data)
    print("\n📊 GPT Evaluation Parsed Scores:\n", json.dumps(scores, indent=2))
