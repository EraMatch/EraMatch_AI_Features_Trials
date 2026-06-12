# import json
# import random

# def generate_hiring_prompt(record):
#     """
#     Generate a diverse natural language prompt based on the record's metadata.
#     """
#     difficulty = record.get("difficulty", "Medium")
#     tags = record.get("tags", [])
    
#     # Handle cases where tags might be empty
#     tags_str = ", ".join(tags) if tags else "general algorithmic concepts"
    
#     # Advanced: Diverse Prompts with Personas and Constraints
#     prompt_templates = [
#         # Standard Request
#         f"I need a complete interview question. It should be of {difficulty} difficulty and focus on {tags_str}. Include the problem description, starter code, a solution, and test cases.",
        
#         # Recruiter Persona
#         f"As a technical recruiter screening candidates, I need a {difficulty} coding challenge for {tags_str}. Please provide the full package: description, initial code, optimal solution, and tests. Ensure the format is strictly markdown.",
        
#         # Hiring Manager Persona
#         f"I'm a Hiring Manager putting together a technical assessment. Create a {difficulty} task evaluating {tags_str}. I need the problem statement, starting function signature, complete working solution, and assertions.",
        
#         # Candidate Practice Persona (Reverse Perspective)
#         f"Generate a {difficulty} level mock interview problem focusing on {tags_str}. Provide the problem formulation, base code, the solution I should aim for, and test assertions.",
        
#         # Urgent/Short Request Constraint
#         f"Quick! {difficulty} question for {tags_str}. Need problem, starter code, solution, tests."
#     ]
    
#     user_content = random.choice(prompt_templates)
#     assistant_content = f"""### Problem Statement
# {record.get('problem_description', '')}

# ### Starter Code
# ```python
# {record.get('starter_code', '')}
# ```

# ### Optimal Solution
# ```python
# {record.get('completion', '')}
# ```

# ### Test Cases
# ```python
# {record.get('test', '')}
# ```"""

#     # 3. ChatML / ShareGPT Format (Standard for instruction tuning)
#     transformed_record = {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are an expert AI technical hiring assistant. Your role is to generate comprehensive coding interview assessments in a strictly formatted Markdown structure."
#             },
#             {
#                 "role": "user",
#                 "content": user_content
#             },
#             {
#                 "role": "assistant",
#                 "content": assistant_content
#             }
#         ]
#     }
    
#     return transformed_record


# dummy_record = {
#     "task_id": "two-sum",
#     "difficulty": "Easy",
#     "tags": ["Array", "Hash Table"],
#     "problem_description": "Given an array of integers nums and an integer target...",
#     "starter_code": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        pass",
#     "completion": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        d = {}\n        for i, x in enumerate(nums):\n            if (y := target - x) in d:\n                return [d[y], i]\n            d[x] = i",
#     "test": "def check(candidate):\n    assert candidate(nums = [3, 3], target = 6) == [0, 1]"
# }

# transformed = generate_hiring_prompt(dummy_record)
# print(json.dumps(transformed, indent=2))

# """
# # Real Usage: Loop through your dataset
# transformed_dataset = []
# for record in original_dataset: # Replace with your actual loaded dataset
#     transformed_dataset.append(generate_hiring_prompt(record))

# # Save to JSONL for fine-tuning frameworks like Axolotl, LLaMA-Factory, or HuggingFace
# with open('hiring_dataset_qwen.jsonl', 'w') as f:
#     for item in transformed_dataset:
#         f.write(json.dumps(item) + '\\n')
# """
