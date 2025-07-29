from typing import List, Optional, Dict
import tiktoken
import os
import json
import re
import sys
import time
from datetime import datetime
from copy import deepcopy

from .Base import BaseStrategy
from models.Base import BaseModel

from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset

from results.Results import Results
from evaluations.func_evaluate import evaluate_io
import numpy as np
from typing import Optional, Dict

class ReasoningTrajectory:
    def __init__(self, t: int):
        self.t = t  
        self.historical_data: Dict[int, Dict] = {}  

    def generate_prompt_for_plan_update(
        self,
        iteration: int,
        error_analysis: Dict,
        problem: str,
        problem_understanding: str,
        plan: str,
        code: Optional[str] = None,
        historical_logs: Optional[Dict] = None
    ) -> str:
        def sanitize(text: Optional[str]) -> str:
            if not isinstance(text, str):
                text = str(text)
            return text.replace('%', '%%')

        test_results = sanitize(error_analysis.get('test_results', ''))
        analysis = sanitize(error_analysis.get('analysis', ''))
        success_rate = sanitize(str(error_analysis.get('success_rate', '0')))
        problem = sanitize(problem)
        problem_understanding = sanitize(problem_understanding)
        plan = sanitize(plan)

        if iteration == 1:
            prompt = f"""
You are the Reasoning Trajectory module (R_traj) in the CoEvolve framework, specializing in iterative plan refinement for competitive programming.

## Context
Problem: {problem}
Problem Understanding: {problem_understanding}
Initial Plan: {plan}
Test Results: {test_results}
Error Analysis: {analysis}
Success Rate: {success_rate}%

## Task
Create the initial reasoning prompt for plan revision in iteration 1. With no historical data, focus on:
1. Anticipating potential weaknesses in the plan based on the problem's details and understanding.
2. Drawing on planning strategies and common pitfalls, considering edge cases and special cases.
3. Setting forward-looking priorities for a thorough plan.

## Instructions
Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

## Predicted Plan Vulnerabilities
[Your vulnerabilities]

## Domain Expertise Synthesis
[Your synthesis]

## Strategic Prioritization
[Your prioritization]

## Reasoning Prompt
[Your prompt]
"""
        else:
            prev = historical_logs or {}
            prev_iteration = prev.get('previous_iteration', iteration - 1)
            prev_success_rate = sanitize(str(prev.get('previous_success_rate', '0')))
            previous_plan = sanitize(prev.get('previous_plan', ''))

            prompt = f"""
You are the Reasoning Trajectory module (R_traj) in the CoEvolve framework, specializing in iterative plan refinement.

## Context
Problem: {problem}
Problem Understanding: {problem_understanding}
Current Plan (Iteration {iteration}): {plan}
Prior Plan (Iteration {prev_iteration}): {previous_plan}
Test Results: {test_results}
Error Analysis: {analysis}
Success Rate: {success_rate}%

## Task
Create a reasoning prompt for plan revision. Follow these steps:
1. Plan Error Localization
2. Historical Plan Analysis
3. Plan Adjustment Priorities, considering edge cases and special cases from the problem understanding
4. Reasoning Prompt for Plan Update

## Instructions
Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

## Plan Error Localization
[Your localization]

## Historical Plan Analysis
[Your analysis]

## Plan Adjustment Priorities
[Your priorities]

## Reasoning Prompt
[Your prompt]
"""
        return prompt

    def generate_prompt_for_code_update(
        self,
        iteration: int,
        error_analysis: Dict,
        problem: str,
        problem_understanding: str,
        plan: str,
        code: str,
        historical_logs: Optional[Dict] = None
    ) -> str:
        def sanitize(text: Optional[str]) -> str:
            if not isinstance(text, str):
                text = str(text)
            return text.replace('%', '%%')

        test_results = sanitize(error_analysis.get('test_results', ''))
        analysis = sanitize(error_analysis.get('analysis', ''))
        success_rate = sanitize(str(error_analysis.get('success_rate', '0')))
        problem = sanitize(problem)
        problem_understanding = sanitize(problem_understanding)
        plan = sanitize(plan)
        code = sanitize(code)

        if iteration == 1:
            prompt = f"""
You are the Reasoning Trajectory module (R_traj), specializing in iterative code refinement.

## Context
Problem: {problem}
Problem Understanding: {problem_understanding}
Plan: {plan}
Initial Code: {code}
Test Results: {test_results}
Error Analysis: {analysis}
Success Rate: {success_rate}%

## Task
Create the initial reasoning prompt for code revision in iteration 1. Focus on potential weaknesses, proven coding strategies, and priorities for robust code, considering edge cases and special cases from the problem understanding.

## Instructions
Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

## Predicted Code Vulnerabilities
[Your vulnerabilities]

## Domain Expertise Synthesis
[Your synthesis]

## Strategic Prioritization
[Your prioritization]

## Reasoning Prompt
[Your prompt]
"""
        else:
            prev = historical_logs or {}
            prev_iteration = prev.get('previous_iteration', iteration - 1)
            prev_success_rate = sanitize(str(prev.get('previous_success_rate', '0')))
            previous_code = sanitize(prev.get('previous_code', ''))

            prompt = f"""
You are the Reasoning Trajectory module (R_traj), specializing in iterative code refinement.

## Context
Problem: {problem}
Problem Understanding: {problem_understanding}
Plan: {plan}
Current Code (Iteration {iteration}): {code}
Prior Code (Iteration {prev_iteration}): {previous_code}
Test Results: {test_results}
Error Analysis: {analysis}
Success Rate: {success_rate}%

## Task
Create a reasoning prompt for code revision. Follow these steps:
1. Code Error Localization
2. Historical Code Analysis
3. Code Adjustment Priorities, considering edge cases and special cases from the problem understanding
4. Reasoning Prompt for Code Update

## Instructions
Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

## Code Error Localization
[Your localization]

## Historical Code Analysis
[Your analysis]

## Code Adjustment Priorities
[Your priorities]

## Reasoning Prompt
[Your prompt]
"""
        return prompt

    def update_historical_data(self, iteration: int, historical_logs: Dict):
        self.historical_data[iteration] = historical_logs

mapping = {
    1: "one (01)",
    2: "two (02)",
    3: "three (03)",
    4: "four (04)",
    5: "five (05)",
    6: "six (06)",
    7: "seven (07)",
    8: "eight (08)",
    9: "nine (09)",
}

class CoEvolvev2(BaseStrategy):
    def __init__(
        self,
        k: int = 3,
        t: int = 5,
        max_attempts: int = 3,
        include_mode: str = 'understanding_only',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.top_plan = 1
        self.t = t
        self.number_of_code_per_plan = 3
        self.trust_weights = {
            'plan': 0.4,
            'code': 0.3,
            'content': 0.3
        }
        self.analysis_meaning = {
            "plan": "The plan analysis identifies failures in the planning approach based on test logs and suggests specific modifications to the plan.",
            "code": "The code analysis identifies errors in the code implementation based on test logs and suggests specific fixes to the code.",
            "content": "The content analysis identifies mismatches between the problem, plan, and code, and suggests improvements for better alignment.",
        }
        self.history = []
        self.rt = ReasoningTrajectory(t=self.t)
        self.max_attempts = max_attempts
        if include_mode not in ['both', 'problem_only', 'understanding_only']:
            raise ValueError("include_mode must be 'both', 'problem_only', or 'understanding_only'")
        self.include_mode = include_mode

    def parse_structured_text(self, response: str) -> dict:
        response = response.strip()
        sections = {}
        current_key = None
        current_value = []
        
        for line in response.splitlines():
            if line.startswith('## '):
                if current_key:
                    sections[current_key] = '\n'.join(current_value).strip()
                current_key = line[3:].strip().lower().replace(' ', '_')
                current_value = []
            elif current_key:
                current_value.append(line)
        
        if current_key:
            sections[current_key] = '\n'.join(current_value).strip()
        
        if 'code' in sections:
            sections['code'] = self.parse_code(sections['code'])
        
        return sections

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response

        code_pattern = r'```((.|\n)*?)```'
        if "```Python" in response:
            code_pattern = r'```Python((.|\n)*?)```'
        if "```Python3" in response:
            code_pattern = r'```Python3((.|\n)*?)```'
        if "```python" in response:
            code_pattern = r'```python((.|\n)*?)```'
        if "```python3" in response:
            code_pattern = r'```python3((.|\n)*?)```'
        if "```C" in response:
            code_pattern = r'```C((.|\n)*?)```'
        if "```c" in response:
            code_pattern = r'```c((.|\n)*?)```'
        if "```C++" in response:
            code_pattern = r'```C\+\+((.|\n)*?)```'
        if "```c++" in response:
            code_pattern = r'```c\+\+((.|\n)*?)```'
        if "```Java" in response:
            code_pattern = r'```Java((.|\n)*?)```'
        if "```java" in response:
            code_pattern = r'```java((.|\n)*?)```'
        if "```Node" in response:
            code_pattern = r'```Node((.|\n)*?)```'
        if "```node" in response:
            code_pattern = r'```node((.|\n)*?)```'
        if "```Rust" in response:
            code_pattern = r'```Rust((.|\n)*?)```'
        if "```rust" in response:
            code_pattern = r'```rust((.|\n)*?)```'
        if "```PHP" in response:
            code_pattern = r'```PHP((.|\n)*?)```'
        if "```php" in response:
            code_pattern = r'```php((.|\n)*?)```'
        if "```Go" in response:
            code_pattern = r'```Go((.|\n)*?)```'
        if "```go" in response:
            code_pattern = r'```go((.|\n)*?)```'
        if "```Ruby" in response:
            code_pattern = r'```Ruby((.|\n)*?)```'
        if "```ruby" in response:
            code_pattern = r'```ruby((.|\n)*?)```'
        if "```C#" in response:
            code_pattern = r'```C#((.|\n)*?)```'
        if "```c#" in response:
            code_pattern = r'```c#((.|\n)*?)```'
        if "```csharp" in response:
            code_pattern = r'```csharp((.|\n)*?)```'

        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if isinstance(code_blocks[-1], (tuple, list)):
            code_str = "\n".join(code_blocks[-1])
        elif isinstance(code_blocks[-1], str):
            code_str = code_blocks[-1]
        else:
            code_str = response

        return code_str.strip()

    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, '').strip()

    def sanitize_input(self, text: Optional[str]) -> str:
        if not isinstance(text, str):
            text = str(text)
        return text.replace('%', '%%')

    def get_sample_io_str(self, item) -> str:
        if isinstance(self.data, XCodeDataset):
            return self.get_sample_io_xcode(item)
        else:
            sample_io = item['sample_io']
            if len(sample_io) > 0:
                if isinstance(sample_io[0], str):
                    return "\n".join(self.sanitize_input(io) for io in sample_io)
                if isinstance(sample_io[0], dict):
                    return "\n".join([f"Input:\n{self.sanitize_input(io['input'])}\nExpected output:\n{self.sanitize_input(io['output'][0])}" for io in sample_io])
            return ''

    def get_sample_io_xcode(self, item):
        return "\n".join([f"Input:\n{self.sanitize_input(item['sample_inputs'])}\nExpected output:\n{self.sanitize_input(item['sample_outputs'])}"])

    def get_problem_understanding(self, item) -> str:
        input_prompt = [
            {
                "role": "user",
                "content": f"""You are an expert in competitive programming. Your task is to analyze a problem description and provide a concise understanding of the problem, including its requirements, constraints, objectives, and potential edge cases or special cases. Identify edge cases (e.g., boundary conditions, invalid inputs, or extreme scenarios) and special cases (e.g., unique input patterns or problem-specific conditions) that need attention, and provide examples of these cases beyond the provided sample I/O.

    # Problem:
    {self.sanitize_input(self.data.get_prompt(item))}

    # Sample I/O:
    {self.get_sample_io_str(item)}

    ----------------
    IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

    ## Understanding
    [Your understanding]
"""
            }
        ]

        understanding = 'No understanding provided'
        pr_tok = 0
        com_tok = 0
        for attempt in range(self.max_attempts):
            try:
                response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_prompt)
                pr_tok += pr_tok_temp
                com_tok += com_tok_temp
                parsed = self.parse_structured_text(response)
                understanding = parsed.get('understanding', 'No understanding provided')
                break
            except Exception as e:
                print(f"Error in get_problem_understanding on attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Max attempts reached, using default understanding.")

        return understanding, pr_tok, com_tok

    def generate_plans(self, item):
        plannings = []
        pr_tok = 0
        com_tok = 0
        previous_approaches = ""

        problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
        pr_tok += pr_tok_u
        com_tok += com_tok_u

        problem_text = self.sanitize_input(self.data.get_prompt(item))
        context = ""
        if self.include_mode == 'both':
            context = f"# Problem:\n{problem_text}\n# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
        elif self.include_mode == 'problem_only':
            context = f"# Problem:\n{problem_text}"
        elif self.include_mode == 'understanding_only':
            context = f"# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

        for t in range(1, self.k + 1):
            diff_prompt = "" if t == 1 else f", different from the following previous approaches: {previous_approaches}"

            input_recall = [
                {
                    "role": "user",
                    "content": f"""Given a problem and its understanding, recall an approach that can solve it{diff_prompt}, provide a tutorial for the approach, then recall a relevant problem that uses this approach, and explain with plan and code.

{context}

# Approach:
Recall one approach (e.g., Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that can solve the problem.

# Tutorial:
Write a useful tutorial about the approach. Provide a high level generic tutorial for solving this type of problem. Do not generate code.

# Exemplar:
Recall one relevant and distinct problem (different from the problem mentioned above) that uses this approach. For the problem,
1. describe it
2. generate {self.language} code step by step to solve that problem using the approach
3. finally generate a planning to solve that problem using the approach

----------------
IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

## Approach Name
[Your approach name]

## Tutorial
[Your tutorial text]

## Problem Description
[Your problem description]

## Code
``` {self.language}
[Your code here]
```

## Planning
[Your planning text]
"""
                },
            ]

            parsed_response = None
            pr_tok_1 = 0
            com_tok_1 = 0
            for attempt in range(self.max_attempts):
                try:
                    response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_recall)
                    pr_tok_1 += pr_tok_temp
                    com_tok_1 += com_tok_temp
                    item['api_calls'] = item.get('api_calls', 0) + 1
                    parsed_response = self.parse_structured_text(response)
                    break
                except Exception as e:
                    print(f"Error in generate_plans (recall) on attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        print("Max attempts reached, skipping this approach.")

            pr_tok += pr_tok_1
            com_tok += com_tok_1
            if parsed_response is None:
                continue

            approach_name = parsed_response.get('approach_name', '')
            approach_tutorial = parsed_response.get('tutorial', '')
            algorithm_prompt = f"## Relevant Approach: {self.sanitize_input(approach_name)}\n{self.sanitize_input(approach_tutorial)}"
            example_description = parsed_response.get('problem_description', '')
            example_planning = parsed_response.get('planning', '')
            example_code = parsed_response.get('code', '')

            previous_approaches += f"\n- {self.sanitize_input(approach_name)}"

            sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item)}\n"

            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": f"""Given a competitive programming problem and its understanding, generate a concrete planning to solve the problem.

# Problem:
{example_description}

# Planning:
{example_planning}

{algorithm_prompt}

## Problem to be solved:
{problem_text}

## Problem Understanding:
{self.sanitize_input(problem_understanding) if self.include_mode in ['both', 'understanding_only'] else ''}

{sample_io_prompt}

## Planning:
[Your planning]
"""
                }
            ]

            try:
                planning, pr_tok_1, com_tok_1 = self.gpt_chat(
                    processed_input=input_for_problem_planning
                )
                pr_tok += pr_tok_1
                com_tok += com_tok_1
                item['api_calls'] += 1
            except Exception as e:
                print(f"Error in generate_plans (planning) for approach {t}: {e}")
                continue

            input_for_planning_verification = [
                {
                    "role": "user",
                    "content": f"""You are an expert evaluator for competitive programming plans. Assess a given plan based on problem-plan alignment and plan coherence. Provide explanations and scores, then compute an overall solvability score.

## Evaluation Criteria:
1. **Problem-Plan Alignment** (Score: 0.0 to 1.0):
   - Measures how well the plan addresses the problem's requirements, constraints, objectives, input/output formats, and edge cases.
   - 1.0: Perfect match—fully covers all aspects, including time/space complexity and pitfalls.
   - 0.5: Partial match—addresses core elements but misses some constraints or edge cases.
   - 0.0: No match—irrelevant or misunderstands the problem.

2. **Plan Coherence** (Score: 0.0 to 1.0):
   - Measures logical consistency, feasibility, and completeness of the plan.
   - 1.0: Fully coherent—logically sound, clear, feasible in {self.language}, no gaps.
   - 0.5: Moderately coherent—logical flow but minor inconsistencies or gaps.
   - 0.0: Incoherent—illogical, infeasible, or riddled with errors.

3. **Overall Solvability Score**:
   - Compute as (alignment score) * (coherence score) * 100, rounded to the nearest integer (0-100).

## Input:
Problem Description: {problem_text}
Problem Understanding: {self.sanitize_input(problem_understanding) if self.include_mode in ['both', 'understanding_only'] else ''}
Proposed Plan: {self.sanitize_input(planning)}
Sample I/O: {sample_io_prompt}

## Instructions:
Output in this exact structured text format with headings. Do not use JSON or extra text. Be objective and critical.

## Alignment Explanation
[Your explanation]

## Alignment Score
[Score as float]

## Coherence Explanation
[Your explanation]

## Coherence Score
[Score as float]

## Overall Solvability
[Integer score]
"""
                }
            ]

            verification_parsed = None
            pr_tok_1 = 0
            com_tok_1 = 0
            for attempt in range(self.max_attempts):
                try:
                    verification_res, pr_tok_temp, com_tok_temp = self.gpt_chat(
                        processed_input=input_for_planning_verification
                    )
                    pr_tok_1 += pr_tok_temp
                    com_tok_1 += com_tok_temp
                    item['api_calls'] += 1
                    verification_parsed = self.parse_structured_text(verification_res)
                    break
                except Exception as e:
                    print(f"Error in generate_plans (verification) on attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        print("Max attempts reached, skipping this plan.")

            pr_tok += pr_tok_1
            com_tok += com_tok_1
            if verification_parsed is None:
                continue

            try:
                alignment_score = float(verification_parsed.get('alignment_score', 0.0))
                coherence_score = float(verification_parsed.get('coherence_score', 0.0))
                confidence = int(alignment_score * coherence_score * 100)
            except (ValueError, TypeError) as e:
                print(f"Error calculating confidence in generate_plans: {e}")
                confidence = 0

            plannings.append((planning, confidence, {
                'description': example_description,
                'code': example_code,
                'planning': example_planning
            }))

        plannings.sort(key=lambda x: x[1], reverse=True)

        return plannings, pr_tok, com_tok

    def generate_codes_from_plan(self, item, plan, algorithm_prompt, sample_io_prompt):
        if isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset)):
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        codes = []
        pr_tok = 0
        com_tok = 0

        problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
        pr_tok += pr_tok_u
        com_tok += com_tok_u

        problem_text = self.sanitize_input(self.data.get_prompt(item))
        context = ""
        if self.include_mode == 'both':
            context = f"## Problem to be solved:\n{problem_text}\n## Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
        elif self.include_mode == 'problem_only':
            context = f"## Problem to be solved:\n{problem_text}"
        elif self.include_mode == 'understanding_only':
            context = f"## Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

        for i in range(self.number_of_code_per_plan):
            variation_prompt = "" #if i == 0 else "Generate a different code variation that still strictly follows the plan."

            input_for_code_generation = [
                {
                    "role": "user",
                    "content": f"""Given a competitive programming problem and its understanding, generate {self.language} code to solve the problem.

{self.sanitize_input(algorithm_prompt)}
{context}
## Planning:
{self.sanitize_input(plan)}
{sample_io_prompt}
## Let's think step by step.
{variation_prompt}

----------------
Important:
{std_input_prompt}
## Instructions:
Output only the {self.language} code to solve this problem. Do not add extra explanation or words.
"""
                }
            ]

            try:
                code_response, pr_tok_1, com_tok_1 = self.gpt_chat(
                    processed_input=input_for_code_generation
                )
                item['api_calls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1
                code = self.parse_code(code_response)
                codes.append(code)
            except Exception as e:
                print(f"Error in generate_codes_from_plan for code {i+1}: {e}")
                codes.append("")
                continue

        evaluated = []
        for idx, code in enumerate(codes):
            if not code:
                evaluated.append((code, 0.0, "No code generated", False))
                continue
            try:
                passed, test_log = self.data.evaluate_sample_io(item, code, self.language)
                num_passed = test_log.count("passed in test case")
                num_failed = test_log.count("failed in test case")
                total = num_passed + num_failed
                if total > 0:
                    passed_score = num_passed / total 
                else:
                    passed_score = 1.0 if passed else 0.0
                evaluated.append((code, passed_score, test_log, passed))
            except Exception as e:
                print(f"Error evaluating code {idx+1} in generate_codes_from_plan: {e}")
                evaluated.append((code, 0.0, f"Evaluation failed: {e}", False))

        evaluated.sort(key=lambda x: x[1], reverse=True)
        best_code = evaluated[0][0] if evaluated else ""
        flag = evaluated[0][3] if evaluated else False
        best_test_log = evaluated[0][2] if evaluated else ""

        return best_code, flag, best_test_log, evaluated[0][1] if evaluated else 0.0, pr_tok, com_tok

    def plan_analysis(self, plan: str, code: str, test_log: str, problem: str, problem_understanding: str) -> dict:
        context = ""
        if self.include_mode == 'both':
            context = f"# Problem:\n{self.sanitize_input(problem)}\n# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
        elif self.include_mode == 'problem_only':
            context = f"# Problem:\n{self.sanitize_input(problem)}"
        elif self.include_mode == 'understanding_only':
            context = f"# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

        input_prompt = [
            {
                "role": "user",
                "content": f"""Analyze a plan for solving a competitive programming problem, given the problem description, its understanding, and test log from code generated using the plan. Take a sample input from the test log and simulate the plan's execution step-by-step to pinpoint where the plan is failing based on the test log, and suggest specific improvements or modifications to fix those issues, considering edge cases and special cases from the problem understanding.

{context}

# Plan:
{self.sanitize_input(plan)}

# Current code implementation of the plan:
{self.sanitize_input(code)}

# Test Log:
{self.sanitize_input(test_log)}

----------------
IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

## Simulation
[Your simulation]

## Insights
[Your insights]
"""
            }
        ]

        simulation = 'No simulation provided'
        insights = 'No insights provided'
        pr_tok = 0
        com_tok = 0
        for attempt in range(self.max_attempts):
            try:
                response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_prompt)
                pr_tok += pr_tok_temp
                com_tok += com_tok_temp
                parsed = self.parse_structured_text(response)
                simulation = parsed.get('simulation', 'No simulation provided')
                insights = parsed.get('insights', 'No insights provided')
                break
            except Exception as e:
                print(f"Error in plan_analysis on attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Max attempts reached, using defaults.")

        return {
            'simulation': simulation,
            'insights': insights,
            'pr_tok': pr_tok,
            'com_tok': com_tok
        }

    def code_analysis(self, code: str, test_log: str, problem: str, problem_understanding: str) -> dict:
        context = ""
        if self.include_mode == 'both':
            context = f"# Problem:\n{self.sanitize_input(problem)}\n# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
        elif self.include_mode == 'problem_only':
            context = f"# Problem:\n{self.sanitize_input(problem)}"
        elif self.include_mode == 'understanding_only':
            context = f"# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

        input_prompt = [
            {
                "role": "user",
                "content": f"""Assess the generated code written in {self.language} programming language for a competitive programming problem, using the problem description, its understanding, and test log. Identify where the code is failing based on the test log, and suggest specific improvements or fixes to correct those issues, considering edge cases and special cases from the problem understanding.

{context}

# Code:
```{self.language}
{self.sanitize_input(code)}
```

# Test Log:
{self.sanitize_input(test_log)}

----------------
IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep insights concise (<150 words).

## Insights
[Your insights]
"""
            }
        ]

        insights = 'No insights provided'
        pr_tok = 0
        com_tok = 0
        for attempt in range(self.max_attempts):
            try:
                response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_prompt)
                pr_tok += pr_tok_temp
                com_tok += com_tok_temp
                parsed = self.parse_structured_text(response)
                insights = parsed.get('insights', 'No insights provided')
                break
            except Exception as e:
                print(f"Error in code_analysis on attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Max attempts reached, using default.")

        return {
            'insights': insights,
            'pr_tok': pr_tok,
            'com_tok': com_tok
        }

    def content_analysis(self, problem: str, problem_understanding: str, plan: str, code: str) -> dict:
        context = ""
        if self.include_mode == 'both':
            context = f"# Problem:\n{self.sanitize_input(problem)}\n# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
        elif self.include_mode == 'problem_only':
            context = f"# Problem:\n{self.sanitize_input(problem)}"
        elif self.include_mode == 'understanding_only':
            context = f"# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

        input_prompt = [
            {
                "role": "user",
                "content": f"""Evaluate how effectively a plan and generated code, written in {self.language} programming language, align with the requirements of a competitive programming problem, given the problem description and its understanding. Assess the alignment between the problem (and its understanding) and the plan, and between the plan and the code. Identify any mismatches or issues in these alignments, considering edge cases and special cases from the problem understanding. Provide separate confidence scores (0.0 to 1.0) for the problem-plan alignment and plan-code alignment (1.0: perfect match; 0.0: no alignment), and suggest specific improvements for each if the alignment is not strong.

{context}

# Plan:
{self.sanitize_input(plan)}

# Code:
```{self.language}
{self.sanitize_input(code)}
```

----------------
IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep insights concise (<150 words each).

## Problem Plan Confidence
[Score as float]

## Plan Code Confidence
[Score as float]

## Problem Plan Insights
[Your insights]

## Plan Code Insights
[Your insights]
"""
            }
        ]

        problem_plan_confidence = 0.0
        plan_code_confidence = 0.0
        overall_confidence = 0.0
        problem_plan_insights = 'No insights provided'
        plan_code_insights = 'No insights provided'
        pr_tok = 0
        com_tok = 0
        for attempt in range(self.max_attempts):
            try:
                response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_prompt)
                pr_tok += pr_tok_temp
                com_tok += com_tok_temp
                parsed = self.parse_structured_text(response)
                problem_plan_confidence = float(parsed.get('problem_plan_confidence', 0.0))
                plan_code_confidence = float(parsed.get('plan_code_confidence', 0.0))
                overall_confidence = problem_plan_confidence * plan_code_confidence
                problem_plan_insights = parsed.get('problem_plan_insights', 'No insights provided')
                plan_code_insights = parsed.get('plan_code_insights', 'No insights provided')
                break
            except Exception as e:
                print(f"Error in content_analysis on attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Max attempts reached, using defaults.")

        insights = f"## Problem-Plan Alignment Analysis: {problem_plan_insights}\n## Plan-Code Alignment Analysis: {plan_code_insights}"
        return {
            'problem_plan_confidence': max(0.0, min(problem_plan_confidence, 1.0)),
            'plan_code_confidence': max(0.0, min(plan_code_confidence, 1.0)),
            'confidence': max(0.0, min(overall_confidence, 1.0)),
            'problem_plan_insights': problem_plan_insights,
            'plan_code_insights': plan_code_insights,
            'insights': insights,
            'pr_tok': pr_tok,
            'com_tok': com_tok
        }

    def get_confidence(self, decision: str, analysis: dict, analysis_name: str) -> float:
        meaning = self.analysis_meaning.get(analysis_name, "")

        prompt = [
            {
                "role": "user",
                "content": f"""Given a {analysis_name} analysis. {meaning} Calculate the confidence score (0.0 to 1.0) for choosing to {decision}, where 1.0 means the analysis strongly supports the decision (e.g., insights indicate it's the best fix), and 0.0 means it does not support at all. Provide a brief explanation (50-100 words) of how the score was determined, referencing specific insights.

Insights:
{self.sanitize_input(analysis.get('insights', ''))}

=============
IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

## Confidence
[Score as float]

## Reasoning
[Your reasoning]
"""
            }
        ]

        score = 0.0
        for attempt in range(self.max_attempts):
            try:
                response, pr_tok, com_tok = self.gpt_chat(processed_input=prompt)
                parsed = self.parse_structured_text(response)
                score = float(parsed.get('confidence', 0.0))
                reasoning = parsed.get('reasoning', 'No reasoning provided')
                break
            except Exception as e:
                print(f"Error in get_confidence on attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Max attempts reached, using default score 0.0.")

        return max(0.0, min(score, 1.0))

    def get_consistency(
        self,
        decision: str,
        analysis1: dict, name1: str,
        analysis2: dict, name2: str
    ) -> float:
        ins1 = self.sanitize_input(analysis1.get('insights', '').strip())
        ins2 = self.sanitize_input(analysis2.get('insights', '').strip())

        name1_meaning = self.analysis_meaning.get(name1, "")
        name2_meaning = self.analysis_meaning.get(name2, "")

        prompt = [
            {
                "role": "user",
                "content": f"""Given insights from two analyses: {name1} and {name2}. 
{name1} meaning: {name1_meaning}
{name2} meaning: {name2_meaning}
Calculate the consistency score (0.0 to 1.0) for choosing to {decision}, where 1.0 means the insights from both analyses are highly consistent and support the decision (e.g., similar issues and fixes suggested), and 0.0 means they are inconsistent or contradictory. Provide a brief explanation (50-100 words) of how the score was determined, referencing specific insights from both analyses.

{name1} insights:
{ins1}

{name2} insights:
{ins2}

IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

## Consistency
[Score as float]

## Reasoning
[Your reasoning]
"""
            }
        ]

        score = 0.0
        for attempt in range(self.max_attempts):
            try:
                response, pr_tok, com_tok = self.gpt_chat(processed_input=prompt)
                parsed = self.parse_structured_text(response)
                score = float(parsed.get('consistency', 0.0))
                reasoning = parsed.get('reasoning', 'No reasoning provided')
                break
            except Exception as e:
                print(f"Error in get_consistency on attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Max attempts reached, using default score 0.0.")

        return max(0.0, min(score, 1.0))

    def collaborative_decision(self, plan: str, code: str, outcomes: str, item) -> str:
        try:
            problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
            problem_text = self.sanitize_input(self.data.get_prompt(item))

            A_plan = self.plan_analysis(plan, code, outcomes, problem_text, problem_understanding)
            A_code = self.code_analysis(code, outcomes, problem_text, problem_understanding)
            A_content = self.content_analysis(problem_text, problem_understanding, plan, code)

            decisions = ['update plan', 'update code only']
            scores = {}

            for d in decisions:
                total = 0.0
                for name, A_i in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
                    w = self.trust_weights[name]
                    conf = self.get_confidence(d, A_i, name)
                    cons_prod = 1.0
                    for oname, A_j in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
                        if oname != name:
                            cons_prod *= self.get_consistency(d, A_i, name, A_j, oname)
                    total += w * conf * cons_prod
                scores[d] = total

            final_decision = max(scores, key=scores.get)
            return final_decision
        except Exception as e:
            print(f"Error in collaborative_decision: {e}")
            return "update code only"

    def debug_plan(
        self,
        iteration: int,
        plan: str,
        error_analysis: Dict,
        problem: str,
        problem_understanding: str
    ):
        prev_logs = self.rt.historical_data.get(iteration - 1)
        rt_prompt = self.rt.generate_prompt_for_plan_update(
            iteration,
            error_analysis,
            problem,
            problem_understanding,
            plan,
            historical_logs=prev_logs
        )

        try:
            rt_response, pr_tok_rt, com_tok_rt = self.gpt_chat(
                processed_input=[{'role': 'user', 'content': rt_prompt}]
            )
            reasoning_trajectory = rt_response.strip()
        except Exception as e:
            print(f"Error in debug_plan (gpt_chat): {e}")
            reasoning_trajectory = "Error generating reasoning trajectory"

        context = ""
        if self.include_mode == 'both':
            context = f"# Problem: {self.sanitize_input(problem)}\n# Problem Understanding: {self.sanitize_input(problem_understanding)}"
        elif self.include_mode == 'problem_only':
            context = f"# Problem: {self.sanitize_input(problem)}"
        elif self.include_mode == 'understanding_only':
            context = f"# Problem Understanding: {self.sanitize_input(problem_understanding)}"

        update_prompt = [{
            'role': 'user',
            'content': (
                f"""Given a competitive programming problem, its understanding, and a plan to solve it, but this plan has some troubles that need to be updated with insights. Please modify the plan accordingly.

{context}

# Current Planning:
{self.sanitize_input(plan)}

# Insights:
{self.sanitize_input(error_analysis['insights'])}

# Reasoning Trajectory:
{self.sanitize_input(reasoning_trajectory)}

# Instructions:
Output only the revised plan text. Do not add extra explanation or words.
"""
            )
        }]

        try:
            updated, pr_tok_up, com_tok_up = self.gpt_chat(processed_input=update_prompt)
            revised_plan = updated.strip()
        except Exception as e:
            print(f"Error in debug_plan (update): {e}")
            revised_plan = plan

        self.rt.update_historical_data(iteration, {
            'previous_plan': plan,
            'previous_success_rate': error_analysis.get('success_rate'),
            'previous_iteration': iteration - 1
        })
        return revised_plan, reasoning_trajectory

    def debug_code(
        self,
        iteration: int,
        plan: str,
        code: str,
        error_analysis: Dict,
        problem: str,
        problem_understanding: str
    ):
        prev_logs = self.rt.historical_data.get(iteration - 1)
        rt_prompt = self.rt.generate_prompt_for_code_update(
            iteration,
            error_analysis,
            problem,
            problem_understanding,
            plan,
            code,
            historical_logs=prev_logs
        )

        try:
            rt_response, pr_tok_rt, com_tok_rt = self.gpt_chat(
                processed_input=[{'role': 'user', 'content': rt_prompt}]
            )
            reasoning_trajectory = rt_response.strip()
        except Exception as e:
            print(f"Error in debug_code (gpt_chat): {e}")
            reasoning_trajectory = "Error generating reasoning trajectory"

        context = ""
        if self.include_mode == 'both':
            context = f"# Problem: {self.sanitize_input(problem)}\n# Problem Understanding: {self.sanitize_input(problem_understanding)}"
        elif self.include_mode == 'problem_only':
            context = f"# Problem: {self.sanitize_input(problem)}"
        elif self.include_mode == 'understanding_only':
            context = f"# Problem Understanding: {self.sanitize_input(problem_understanding)}"

        code_prompt = [{
            'role': 'user',
            'content': (
                f"""Given a competitive programming problem, its understanding, and generated {self.language} code to solve the problem, but the generated code cannot pass sample test cases. Improve your code to solve the problem correctly.

{context}

## Planning:
{self.sanitize_input(plan)}

## Code:
```{self.language}
{self.sanitize_input(code)}
```

## Test Report:
{self.sanitize_input(error_analysis.get('test_results',''))}

## Insights:
{self.sanitize_input(error_analysis['insights'])}

## Reasoning Trajectory:
{self.sanitize_input(reasoning_trajectory)}

## Instructions:
Output only the revised {self.language} code. Do not add extra explanation or words.
"""
            )
        }]

        try:
            updated, pr_tok_up, com_tok_up = self.gpt_chat(processed_input=code_prompt)
            revised_code = self.parse_code(updated)
        except Exception as e:
            print(f"Error in debug_code (update): {e}")
            revised_code = code

        self.rt.update_historical_data(iteration, {
            'previous_code': code,
            'previous_success_rate': error_analysis.get('success_rate'),
            'previous_iteration': iteration - 1
        })
        return revised_code, reasoning_trajectory

    def _inner_run(self, item):
        pr_tok = 0
        com_tok = 0

        try:
            plannings, pr_tok_p, com_tok_p = self.generate_plans(item)
            pr_tok += pr_tok_p
            com_tok += com_tok_p
        except Exception as e:
            print(f"Error in _inner_run (generate_plans): {e}")
            plannings = []

        selected_plannings = plannings[:self.top_plan]

        best_code = ""
        flag = False
        test_log = ""

        for plan_idx, planning_with_ex in enumerate(selected_plannings, 1):
            plan, confidence, example = planning_with_ex
            approach_name = example.get('description', '') if 'description' in example else ''
            approach_tutorial = example.get('planning', '') if 'planning' in example else ''
            algorithm_prompt = f"## Relevant Approach: {self.sanitize_input(approach_name)}\n{self.sanitize_input(approach_tutorial)}"
            sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item)}\n"
            if isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset)):
                std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
            else:
                std_input_prompt = ""

            try:
                best_code, flag, test_log, score, pr_tok_c, com_tok_c = self.generate_codes_from_plan(item, plan, algorithm_prompt, sample_io_prompt)
                pr_tok += pr_tok_c
                com_tok += com_tok_c
            except Exception as e:
                print(f"Error in _inner_run (generate_codes_from_plan) for plan {plan_idx}: {e}")
                continue

            if flag:
                return best_code, pr_tok, com_tok

            for i in range(1, self.t + 1):
                try:
                    problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
                    pr_tok += pr_tok_u
                    com_tok += com_tok_u
                except Exception as e:
                    print(f"Error in _inner_run (get_problem_understanding) iteration {i}: {e}")
                    problem_understanding = ""

                try:
                    decision = self.collaborative_decision(plan, best_code, test_log, item)
                except Exception as e:
                    print(f"Error in _inner_run (collaborative_decision) iteration {i}: {e}")
                    decision = "update code only"

                if decision == 'update plan':
                    try:
                        A_plan = self.plan_analysis(plan, best_code, test_log, self.data.get_prompt(item), problem_understanding)
                        plan, reasoning_trajectory = self.debug_plan(i + 1, plan, {
                            'insights': A_plan['insights'],
                            'test_results': test_log,
                            'success_rate': score,
                        }, self.data.get_prompt(item), problem_understanding)

                        code, flag, test_log, score, p_c2, c_c2 = self.generate_codes_from_plan(
                            item, plan, algorithm_prompt="", sample_io_prompt=sample_io_prompt
                        )
                        pr_tok += p_c2
                        com_tok += c_c2
                        best_code = code
                    except Exception as e:
                        print(f"Error in _inner_run (update plan) iteration {i}: {e}")
                        continue
                else:
                    try:
                        A_code = self.code_analysis(best_code, test_log, self.data.get_prompt(item), problem_understanding)
                        best_code, reasoning_trajectory = self.debug_code(i + 1, plan, best_code, {
                            'insights': A_code['insights'],
                            'test_results': test_log,
                            'success_rate': score,
                        }, self.data.get_prompt(item), problem_understanding)
                    except Exception as e:
                        print(f"Error in _inner_run (update code) iteration {i}: {e}")
                        continue

                try:
                    flag, test_log = self.data.evaluate_sample_io(item, best_code, self.language)
                except Exception as e:
                    print(f"Error in _inner_run (evaluate_sample_io) iteration {i}: {e}")
                    flag, test_log = False, f"Evaluation failed: {e}"

                if flag:
                    return best_code, pr_tok, com_tok

        return best_code, pr_tok, com_tok

    def run_single_pass(self, item: dict):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                result = self._inner_run(item)
                return result
            except Exception as e:
                print(f"[run_single_pass] Attempt {attempt} caught exception: {e}")
                print(f"[run_single_pass] Item: {item}")
                if attempt == max_retries:
                    print(f"[run_single_pass] ERROR: All {max_retries} attempts failed. Returning fallback ('',0,0).")
                    return "", 0, 0



################################################################################################################################

# from typing import List, Optional, Dict
# import tiktoken
# import os
# import json
# import re
# import sys
# import time
# from datetime import datetime
# from copy import deepcopy
# import logging

# from .Base import BaseStrategy
# from models.Base import BaseModel

# from datasets.Dataset import Dataset
# from datasets.APPSDataset import APPSDataset
# from datasets.MBPPDataset import MBPPDataset
# from datasets.XCodeDataset import XCodeDataset
# from datasets.HumanEvalDataset import HumanDataset
# from datasets.CodeContestDataset import CodeContestDataset

# from results.Results import Results
# from evaluations.func_evaluate import evaluate_io
# import numpy as np
# from typing import Optional, Dict

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('coevolvev2.log', mode='a'),  # Append to log file
#         logging.StreamHandler()  # Also print to console
#     ]
# )
# logger = logging.getLogger(__name__)

# class ReasoningTrajectory:
#     def __init__(self, t: int):
#         self.t = t  
#         self.historical_data: Dict[int, Dict] = {}  

#     def generate_prompt_for_plan_update(
#         self,
#         iteration: int,
#         error_analysis: Dict,
#         problem: str,
#         problem_understanding: str,
#         plan: str,
#         code: Optional[str] = None,
#         historical_logs: Optional[Dict] = None
#     ) -> str:
#         def sanitize(text: Optional[str]) -> str:
#             if not isinstance(text, str):
#                 text = str(text)
#             return text.replace('%', '%%')

#         test_results = sanitize(error_analysis.get('test_results', ''))
#         analysis = sanitize(error_analysis.get('analysis', ''))
#         success_rate = sanitize(str(error_analysis.get('success_rate', '0')))
#         problem = sanitize(problem)
#         problem_understanding = sanitize(problem_understanding)
#         plan = sanitize(plan)

#         if iteration == 1:
#             prompt = f"""
# You are the Reasoning Trajectory module (R_traj) in the CoEvolve framework, specializing in iterative plan refinement for competitive programming.

# ## Context
# Problem: {problem}
# Problem Understanding: {problem_understanding}
# Initial Plan: {plan}
# Test Results: {test_results}
# Error Analysis: {analysis}
# Success Rate: {success_rate}%

# ## Task
# Create the initial reasoning prompt for plan revision in iteration 1. With no historical data, focus on:
# 1. Anticipating potential weaknesses in the plan based on the problem's details and understanding.
# 2. Drawing on planning strategies and common pitfalls, considering edge cases and special cases.
# 3. Setting forward-looking priorities for a thorough plan.

# ## Instructions
# Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

# ## Predicted Plan Vulnerabilities
# [Your vulnerabilities]

# ## Domain Expertise Synthesis
# [Your synthesis]

# ## Strategic Prioritization
# [Your prioritization]

# ## Reasoning Prompt
# [Your prompt]
# """
#         else:
#             prev = historical_logs or {}
#             prev_iteration = prev.get('previous_iteration', iteration - 1)
#             prev_success_rate = sanitize(str(prev.get('previous_success_rate', '0')))
#             previous_plan = sanitize(prev.get('previous_plan', ''))

#             prompt = f"""
# You are the Reasoning Trajectory module (R_traj) in the CoEvolve framework, specializing in iterative plan refinement.

# ## Context
# Problem: {problem}
# Problem Understanding: {problem_understanding}
# Current Plan (Iteration {iteration}): {plan}
# Prior Plan (Iteration {prev_iteration}): {previous_plan}
# Test Results: {test_results}
# Error Analysis: {analysis}
# Success Rate: {success_rate}%

# ## Task
# Create a reasoning prompt for plan revision. Follow these steps:
# 1. Plan Error Localization
# 2. Historical Plan Analysis
# 3. Plan Adjustment Priorities, considering edge cases and special cases from the problem understanding
# 4. Reasoning Prompt for Plan Update

# ## Instructions
# Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

# ## Plan Error Localization
# [Your localization]

# ## Historical Plan Analysis
# [Your analysis]

# ## Plan Adjustment Priorities
# [Your priorities]

# ## Reasoning Prompt
# [Your prompt]
# """
#         logger.info(f"Generated prompt for plan_update (iteration {iteration}):\n{prompt}")
#         return prompt

#     def generate_prompt_for_code_update(
#         self,
#         iteration: int,
#         error_analysis: Dict,
#         problem: str,
#         problem_understanding: str,
#         plan: str,
#         code: str,
#         historical_logs: Optional[Dict] = None
#     ) -> str:
#         def sanitize(text: Optional[str]) -> str:
#             if not isinstance(text, str):
#                 text = str(text)
#             return text.replace('%', '%%')

#         test_results = sanitize(error_analysis.get('test_results', ''))
#         analysis = sanitize(error_analysis.get('analysis', ''))
#         success_rate = sanitize(str(error_analysis.get('success_rate', '0')))
#         problem = sanitize(problem)
#         problem_understanding = sanitize(problem_understanding)
#         plan = sanitize(plan)
#         code = sanitize(code)

#         if iteration == 1:
#             prompt = f"""
# You are the Reasoning Trajectory module (R_traj), specializing in iterative code refinement.

# ## Context
# Problem: {problem}
# Problem Understanding: {problem_understanding}
# Plan: {plan}
# Initial Code: {code}
# Test Results: {test_results}
# Error Analysis: {analysis}
# Success Rate: {success_rate}%

# ## Task
# Create the initial reasoning prompt for code revision in iteration 1. Focus on potential weaknesses, proven coding strategies, and priorities for robust code, considering edge cases and special cases from the problem understanding.

# ## Instructions
# Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

# ## Predicted Code Vulnerabilities
# [Your vulnerabilities]

# ## Domain Expertise Synthesis
# [Your synthesis]

# ## Strategic Prioritization
# [Your prioritization]

# ## Reasoning Prompt
# [Your prompt]
# """
#         else:
#             prev = historical_logs or {}
#             prev_iteration = prev.get('previous_iteration', iteration - 1)
#             prev_success_rate = sanitize(str(prev.get('previous_success_rate', '0')))
#             previous_code = sanitize(prev.get('previous_code', ''))

#             prompt = f"""
# You are the Reasoning Trajectory module (R_traj), specializing in iterative code refinement.

# ## Context
# Problem: {problem}
# Problem Understanding: {problem_understanding}
# Plan: {plan}
# Current Code (Iteration {iteration}): {code}
# Prior Code (Iteration {prev_iteration}): {previous_code}
# Test Results: {test_results}
# Error Analysis: {analysis}
# Success Rate: {success_rate}%

# ## Task
# Create a reasoning prompt for code revision. Follow these steps:
# 1. Code Error Localization
# 2. Historical Code Analysis
# 3. Code Adjustment Priorities, considering edge cases and special cases from the problem understanding
# 4. Reasoning Prompt for Code Update

# ## Instructions
# Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

# ## Code Error Localization
# [Your localization]

# ## Historical Code Analysis
# [Your analysis]

# ## Code Adjustment Priorities
# [Your priorities]

# ## Reasoning Prompt
# [Your prompt]
# """
#         logger.info(f"Generated prompt for code_update (iteration {iteration}):\n{prompt}")
#         return prompt

#     def update_historical_data(self, iteration: int, historical_logs: Dict):
#         self.historical_data[iteration] = historical_logs
#         logger.info(f"Updated historical data for iteration {iteration}: {historical_logs}")

# mapping = {
#     1: "one (01)",
#     2: "two (02)",
#     3: "three (03)",
#     4: "four (04)",
#     5: "five (05)",
#     6: "six (06)",
#     7: "seven (07)",
#     8: "eight (08)",
#     9: "nine (09)",
# }

# class CoEvolvev2(BaseStrategy):
#     def __init__(
#         self,
#         k: int = 3,
#         t: int = 5,
#         max_attempts: int = 3,
#         include_mode: str = 'understanding_only',
#         *args,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.k = k
#         self.top_plan = 1
#         self.t = t
#         self.number_of_code_per_plan = 3
#         self.trust_weights = {
#             'plan': 0.4,
#             'code': 0.3,
#             'content': 0.3
#         }
#         self.analysis_meaning = {
#             "plan": "The plan analysis identifies failures in the planning approach based on test logs and suggests specific modifications to the plan.",
#             "code": "The code analysis identifies errors in the code implementation based on test logs and suggests specific fixes to the code.",
#             "content": "The content analysis identifies mismatches between the problem, plan, and code, and suggests improvements for better alignment.",
#         }
#         self.history = []
#         self.rt = ReasoningTrajectory(t=self.t)
#         self.max_attempts = max_attempts
#         if include_mode not in ['both', 'problem_only', 'understanding_only']:
#             raise ValueError("include_mode must be 'both', 'problem_only', or 'understanding_only'")
#         self.include_mode = include_mode
#         logger.info(f"Initialized CoEvolvev2 with k={k}, t={t}, max_attempts={max_attempts}, include_mode={include_mode}")

#     def parse_structured_text(self, response: str) -> dict:
#         response = response.strip()
#         sections = {}
#         current_key = None
#         current_value = []
        
#         for line in response.splitlines():
#             if line.startswith('## '):
#                 if current_key:
#                     sections[current_key] = '\n'.join(current_value).strip()
#                 current_key = line[3:].strip().lower().replace(' ', '_')
#                 current_value = []
#             elif current_key:
#                 current_value.append(line)
        
#         if current_key:
#             sections[current_key] = '\n'.join(current_value).strip()
        
#         if 'code' in sections:
#             sections['code'] = self.parse_code(sections['code'])
        
#         logger.info(f"Parsed structured text response into sections: {sections.keys()}")
#         return sections

#     def parse_code(self, response: str) -> str:
#         if "```" not in response:
#             return response

#         code_pattern = r'```((.|\n)*?)```'
#         if "```Python" in response:
#             code_pattern = r'```Python((.|\n)*?)```'
#         if "```Python3" in response:
#             code_pattern = r'```Python3((.|\n)*?)```'
#         if "```python" in response:
#             code_pattern = r'```python((.|\n)*?)```'
#         if "```python3" in response:
#             code_pattern = r'```python3((.|\n)*?)```'
#         if "```C" in response:
#             code_pattern = r'```C((.|\n)*?)```'
#         if "```c" in response:
#             code_pattern = r'```c((.|\n)*?)```'
#         if "```C++" in response:
#             code_pattern = r'```C\+\+((.|\n)*?)```'
#         if "```c++" in response:
#             code_pattern = r'```c\+\+((.|\n)*?)```'
#         if "```Java" in response:
#             code_pattern = r'```Java((.|\n)*?)```'
#         if "```java" in response:
#             code_pattern = r'```java((.|\n)*?)```'
#         if "```Node" in response:
#             code_pattern = r'```Node((.|\n)*?)```'
#         if "```node" in response:
#             code_pattern = r'```node((.|\n)*?)```'
#         if "```Rust" in response:
#             code_pattern = r'```Rust((.|\n)*?)```'
#         if "```rust" in response:
#             code_pattern = r'```rust((.|\n)*?)```'
#         if "```PHP" in response:
#             code_pattern = r'```PHP((.|\n)*?)```'
#         if "```php" in response:
#             code_pattern = r'```php((.|\n)*?)```'
#         if "```Go" in response:
#             code_pattern = r'```Go((.|\n)*?)```'
#         if "```go" in response:
#             code_pattern = r'```go((.|\n)*?)```'
#         if "```Ruby" in response:
#             code_pattern = r'```Ruby((.|\n)*?)```'
#         if "```ruby" in response:
#             code_pattern = r'```ruby((.|\n)*?)```'
#         if "```C#" in response:
#             code_pattern = r'```C#((.|\n)*?)```'
#         if "```c#" in response:
#             code_pattern = r'```c#((.|\n)*?)```'
#         if "```csharp" in response:
#             code_pattern = r'```csharp((.|\n)*?)```'

#         code_blocks = re.findall(code_pattern, response, re.DOTALL)

#         if isinstance(code_blocks[-1], (tuple, list)):
#             code_str = "\n".join(code_blocks[-1])
#         elif isinstance(code_blocks[-1], str):
#             code_str = code_blocks[-1]
#         else:
#             code_str = response

#         code_str = code_str.strip()
#         logger.info(f"Parsed code from response: {code_str[:100]}...")  # Log first 100 chars
#         return code_str

#     @staticmethod
#     def trim_text(text: str, trimmed_text: str):
#         result = text.replace(trimmed_text, '').strip()
#         logger.info(f"Trimmed text: {result[:100]}...")  # Log first 100 chars
#         return result

#     def sanitize_input(self, text: Optional[str]) -> str:
#         if not isinstance(text, str):
#             text = str(text)
#         result = text.replace('%', '%%')
#         logger.info(f"Sanitized input: {result[:100]}...")  # Log first 100 chars
#         return result

#     def get_sample_io_str(self, item) -> str:
#         if isinstance(self.data, XCodeDataset):
#             result = self.get_sample_io_xcode(item)
#         else:
#             sample_io = item['sample_io']
#             if len(sample_io) > 0:
#                 if isinstance(sample_io[0], str):
#                     result = "\n".join(self.sanitize_input(io) for io in sample_io)
#                 if isinstance(sample_io[0], dict):
#                     result = "\n".join([f"Input:\n{self.sanitize_input(io['input'])}\nExpected output:\n{self.sanitize_input(io['output'][0])}" for io in sample_io])
#             else:
#                 result = ''
#         logger.info(f"Generated sample I/O string: {result[:100]}...")  # Log first 100 chars
#         return result

#     def get_sample_io_xcode(self, item):
#         result = "\n".join([f"Input:\n{self.sanitize_input(item['sample_inputs'])}\nExpected output:\n{self.sanitize_input(item['sample_outputs'])}"])
#         logger.info(f"Generated Xcode sample I/O: {result[:100]}...")  # Log first 100 chars
#         return result

#     def get_problem_understanding(self, item) -> str:
#         input_prompt = [
#             {
#                 "role": "user",
#                 "content": f"""You are an expert in competitive programming. Your task is to analyze a problem description and provide a concise understanding of the problem, including its requirements, constraints, objectives, and potential edge cases or special cases. Identify edge cases (e.g., boundary conditions, invalid inputs, or extreme scenarios) and special cases (e.g., unique input patterns or problem-specific conditions) that need attention, and provide examples of these cases beyond the provided sample I/O.

#     # Problem:
#     {self.sanitize_input(self.data.get_prompt(item))}

#     # Sample I/O:
#     {self.get_sample_io_str(item)}

#     ----------------
#     IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

#     ## Understanding
#     [Your understanding]
# """
#             }
#         ]

#         logger.info(f"Prompt for get_problem_understanding:\n{input_prompt[0]['content']}")

#         understanding = 'No understanding provided'
#         pr_tok = 0
#         com_tok = 0
#         for attempt in range(self.max_attempts):
#             try:
#                 response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_prompt)
#                 pr_tok += pr_tok_temp
#                 com_tok += com_tok_temp
#                 logger.info(f"Response from get_problem_understanding (attempt {attempt + 1}):\n{response[:200]}...")  # Log first 200 chars
#                 parsed = self.parse_structured_text(response)
#                 understanding = parsed.get('understanding', 'No understanding provided')
#                 break
#             except Exception as e:
#                 logger.error(f"Error in get_problem_understanding on attempt {attempt + 1}: {e}", exc_info=True)
#                 if attempt == self.max_attempts - 1:
#                     logger.warning("Max attempts reached, using default understanding.")

#         logger.info(f"Parsed understanding: {understanding[:100]}...")  # Log first 100 chars
#         return understanding, pr_tok, com_tok

#     def generate_plans(self, item):
#         plannings = []
#         pr_tok = 0
#         com_tok = 0
#         previous_approaches = ""

#         problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
#         pr_tok += pr_tok_u
#         com_tok += com_tok_u
#         logger.info(f"Problem understanding: {problem_understanding[:100]}...")

#         problem_text = self.sanitize_input(self.data.get_prompt(item))
#         context = ""
#         if self.include_mode == 'both':
#             context = f"# Problem:\n{problem_text}\n# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
#         elif self.include_mode == 'problem_only':
#             context = f"# Problem:\n{problem_text}"
#         elif self.include_mode == 'understanding_only':
#             context = f"# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

#         for t in range(1, self.k + 1):
#             diff_prompt = "" if t == 1 else f", different from the following previous approaches: {previous_approaches}"

#             input_recall = [
#                 {
#                     "role": "user",
#                     "content": f"""Given a problem and its understanding, recall an approach that can solve it{diff_prompt}, provide a tutorial for the approach, then recall a relevant problem that uses this approach, and explain with plan and code.

# {context}

# # Approach:
# Recall one approach (e.g., Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that can solve the problem.

# # Tutorial:
# Write a useful tutorial about the approach. Provide a high level generic tutorial for solving this type of problem. Do not generate code.

# # Exemplar:
# Recall one relevant and distinct problem (different from the problem mentioned above) that uses this approach. For the problem,
# 1. describe it
# 2. generate {self.language} code step by step to solve that problem using the approach
# 3. finally generate a planning to solve that problem using the approach

# ----------------
# IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

# ## Approach Name
# [Your approach name]

# ## Tutorial
# [Your tutorial text]

# ## Problem Description
# [Your problem description]

# ## Code
# ``` {self.language}
# [Your code here]
# ```

# ## Planning
# [Your planning text]
# """
#                 },
#             ]

#             logger.info(f"Prompt for generate_plans (approach {t}):\n{input_recall[0]['content']}")

#             parsed_response = None
#             pr_tok_1 = 0
#             com_tok_1 = 0
#             for attempt in range(self.max_attempts):
#                 try:
#                     response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_recall)
#                     pr_tok_1 += pr_tok_temp
#                     com_tok_1 += com_tok_temp
#                     item['api_calls'] = item.get('api_calls', 0) + 1
#                     logger.info(f"Response from generate_plans (approach {t}, attempt {attempt + 1}):\n{response[:200]}...")
#                     parsed_response = self.parse_structured_text(response)
#                     break
#                 except Exception as e:
#                     logger.error(f"Error in generate_plans (recall) on attempt {attempt + 1}: {e}", exc_info=True)
#                     if attempt == self.max_attempts - 1:
#                         logger.warning("Max attempts reached, skipping this approach.")

#             pr_tok += pr_tok_1
#             com_tok += com_tok_1
#             if parsed_response is None:
#                 continue

#             approach_name = parsed_response.get('approach_name', '')
#             approach_tutorial = parsed_response.get('tutorial', '')
#             algorithm_prompt = f"## Relevant Approach: {self.sanitize_input(approach_name)}\n{self.sanitize_input(approach_tutorial)}"
#             example_description = parsed_response.get('problem_description', '')
#             example_planning = parsed_response.get('planning', '')
#             example_code = parsed_response.get('code', '')

#             previous_approaches += f"\n- {self.sanitize_input(approach_name)}"
#             logger.info(f"Parsed response for approach {t}: approach_name={approach_name}, tutorial={approach_tutorial[:50]}...")

#             sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item)}\n"

#             input_for_problem_planning = [
#                 {
#                     "role": "user",
#                     "content": f"""Given a competitive programming problem and its understanding, generate a concrete planning to solve the problem.

# # Problem:
# {example_description}

# # Planning:
# {example_planning}

# {algorithm_prompt}

# ## Problem to be solved:
# {problem_text}

# ## Problem Understanding:
# {self.sanitize_input(problem_understanding) if self.include_mode in ['both', 'understanding_only'] else ''}

# {sample_io_prompt}

# ## Planning:
# [Your planning]
# """
#                 }
#             ]

#             logger.info(f"Prompt for problem planning (approach {t}):\n{input_for_problem_planning[0]['content']}")

#             try:
#                 planning, pr_tok_1, com_tok_1 = self.gpt_chat(
#                     processed_input=input_for_problem_planning
#                 )
#                 pr_tok += pr_tok_1
#                 com_tok += com_tok_1
#                 item['api_calls'] += 1
#                 logger.info(f"Planning response for approach {t}:\n{planning[:200]}...")
#             except Exception as e:
#                 logger.error(f"Error in generate_plans (planning) for approach {t}: {e}", exc_info=True)
#                 continue

#             input_for_planning_verification = [
#                 {
#                     "role": "user",
#                     "content": f"""You are an expert evaluator for competitive programming plans. Assess a given plan based on problem-plan alignment and plan coherence. Provide explanations and scores, then compute an overall solvability score.

# ## Evaluation Criteria:
# 1. **Problem-Plan Alignment** (Score: 0.0 to 1.0):
#    - Measures how well the plan addresses the problem's requirements, constraints, objectives, input/output formats, and edge cases.
#    - 1.0: Perfect match—fully covers all aspects, including time/space complexity and pitfalls.
#    - 0.5: Partial match—addresses core elements but misses some constraints or edge cases.
#    - 0.0: No match—irrelevant or misunderstands the problem.

# 2. **Plan Coherence** (Score: 0.0 to 1.0):
#    - Measures logical consistency, feasibility, and completeness of the plan.
#    - 1.0: Fully coherent—logically sound, clear, feasible in {self.language}, no gaps.
#    - 0.5: Moderately coherent—logical flow but minor inconsistencies or gaps.
#    - 0.0: Incoherent—illogical, infeasible, or riddled with errors.

# 3. **Overall Solvability Score**:
#    - Compute as (alignment score) * (coherence score) * 100, rounded to the nearest integer (0-100).

# ## Input:
# Problem Description: {problem_text}
# Problem Understanding: {self.sanitize_input(problem_understanding) if self.include_mode in ['both', 'understanding_only'] else ''}
# Proposed Plan: {self.sanitize_input(planning)}
# Sample I/O: {sample_io_prompt}

# ## Instructions:
# Output in this exact structured text format with headings. Do not use JSON or extra text. Be objective and critical.

# ## Alignment Explanation
# [Your explanation]

# ## Alignment Score
# [Score as float]

# ## Coherence Explanation
# [Your explanation]

# ## Coherence Score
# [Score as float]

# ## Overall Solvability
# [Integer score]
# """
#                 }
#             ]

#             logger.info(f"Prompt for planning verification (approach {t}):\n{input_for_planning_verification[0]['content']}")

#             verification_parsed = None
#             pr_tok_1 = 0
#             com_tok_1 = 0
#             for attempt in range(self.max_attempts):
#                 try:
#                     verification_res, pr_tok_temp, com_tok_temp = self.gpt_chat(
#                         processed_input=input_for_planning_verification
#                     )
#                     pr_tok_1 += pr_tok_temp
#                     com_tok_1 += com_tok_temp
#                     item['api_calls'] += 1
#                     logger.info(f"Verification response for approach {t} (attempt {attempt + 1}):\n{verification_res[:200]}...")
#                     verification_parsed = self.parse_structured_text(verification_res)
#                     break
#                 except Exception as e:
#                     logger.error(f"Error in generate_plans (verification) on attempt {attempt + 1}: {e}", exc_info=True)
#                     if attempt == self.max_attempts - 1:
#                         logger.warning("Max attempts reached, skipping this plan.")

#             pr_tok += pr_tok_1
#             com_tok += com_tok_1
#             if verification_parsed is None:
#                 continue

#             try:
#                 alignment_score = float(verification_parsed.get('alignment_score', 0.0))
#                 coherence_score = float(verification_parsed.get('coherence_score', 0.0))
#                 confidence = int(alignment_score * coherence_score * 100)
#                 logger.info(f"Verification scores for approach {t}: alignment={alignment_score}, coherence={coherence_score}, confidence={confidence}")
#             except (ValueError, TypeError) as e:
#                 logger.error(f"Error calculating confidence in generate_plans: {e}", exc_info=True)
#                 confidence = 0

#             plannings.append((planning, confidence, {
#                 'description': example_description,
#                 'code': example_code,
#                 'planning': example_planning
#             }))
#             logger.info(f"Added planning for approach {t}: confidence={confidence}")

#         plannings.sort(key=lambda x: x[1], reverse=True)
#         logger.info(f"Sorted plannings: {[p[1] for p in plannings]}")

#         return plannings, pr_tok, com_tok

#     def generate_codes_from_plan(self, item, plan, algorithm_prompt, sample_io_prompt):
#         if isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset)):
#             std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
#         else:
#             std_input_prompt = ""

#         codes = []
#         pr_tok = 0
#         com_tok = 0

#         problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
#         pr_tok += pr_tok_u
#         com_tok += com_tok_u
#         logger.info(f"Problem understanding for code generation: {problem_understanding[:100]}...")

#         problem_text = self.sanitize_input(self.data.get_prompt(item))
#         context = ""
#         if self.include_mode == 'both':
#             context = f"## Problem to be solved:\n{problem_text}\n## Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
#         elif self.include_mode == 'problem_only':
#             context = f"## Problem to be solved:\n{problem_text}"
#         elif self.include_mode == 'understanding_only':
#             context = f"## Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

#         for i in range(self.number_of_code_per_plan):
#             variation_prompt = ""  #if i == 0 else "Generate a different code variation that still strictly follows the plan."

#             input_for_code_generation = [
#                 {
#                     "role": "user",
#                     "content": f"""Given a competitive programming problem and its understanding, generate {self.language} code to solve the problem.

# {self.sanitize_input(algorithm_prompt)}
# {context}
# ## Planning:
# {self.sanitize_input(plan)}
# {sample_io_prompt}
# ## Let's think step by step.
# {variation_prompt}

# ----------------
# Important:
# {std_input_prompt}
# ## Instructions:
# Output only the {self.language} code to solve this problem. Do not add extra explanation or words.
# """
#                 }
#             ]

#             logger.info(f"Prompt for generate_codes_from_plan (code {i+1}):\n{input_for_code_generation[0]['content']}")

#             try:
#                 code_response, pr_tok_1, com_tok_1 = self.gpt_chat(
#                     processed_input=input_for_code_generation
#                 )
#                 item['api_calls'] += 1
#                 pr_tok += pr_tok_1
#                 com_tok += com_tok_1
#                 logger.info(f"Response for code {i+1}:\n{code_response[:200]}...")
#                 code = self.parse_code(code_response)
#                 codes.append(code)
#             except Exception as e:
#                 logger.error(f"Error in generate_codes_from_plan for code {i+1}: {e}", exc_info=True)
#                 codes.append("")
#                 continue

#         evaluated = []
#         for idx, code in enumerate(codes):
#             if not code:
#                 evaluated.append((code, 0.0, "No code generated", False))
#                 logger.warning(f"No code generated for code {idx+1}")
#                 continue
#             try:
#                 passed, test_log = self.data.evaluate_sample_io(item, code, self.language)
#                 num_passed = test_log.count("passed in test case")
#                 num_failed = test_log.count("failed in test case")
#                 total = num_passed + num_failed
#                 if total > 0:
#                     passed_score = num_passed / total 
#                 else:
#                     passed_score = 1.0 if passed else 0.0
#                 evaluated.append((code, passed_score, test_log, passed))
#                 logger.info(f"Evaluated code {idx+1}: passed={passed}, score={passed_score}, test_log={test_log[:100]}...")
#             except Exception as e:
#                 logger.error(f"Error evaluating code {idx+1} in generate_codes_from_plan: {e}", exc_info=True)
#                 evaluated.append((code, 0.0, f"Evaluation failed: {e}", False))

#         evaluated.sort(key=lambda x: x[1], reverse=True)
#         best_code = evaluated[0][0] if evaluated else ""
#         flag = evaluated[0][3] if evaluated else False
#         best_test_log = evaluated[0][2] if evaluated else ""
#         logger.info(f"Best code selected: score={evaluated[0][1] if evaluated else 0.0}, passed={flag}")

#         return best_code, flag, best_test_log, evaluated[0][1] if evaluated else 0.0, pr_tok, com_tok

#     def plan_analysis(self, plan: str, code: str, test_log: str, problem: str, problem_understanding: str) -> dict:
#         context = ""
#         if self.include_mode == 'both':
#             context = f"# Problem:\n{self.sanitize_input(problem)}\n# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
#         elif self.include_mode == 'problem_only':
#             context = f"# Problem:\n{self.sanitize_input(problem)}"
#         elif self.include_mode == 'understanding_only':
#             context = f"# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

#         input_prompt = [
#             {
#                 "role": "user",
#                 "content": f"""Analyze a plan for solving a competitive programming problem, given the problem description, its understanding, and test log from code generated using the plan. Take a sample input from the test log and simulate the plan's execution step-by-step to pinpoint where the plan is failing based on the test log, and suggest specific improvements or modifications to fix those issues, considering edge cases and special cases from the problem understanding.

# {context}

# # Plan:
# {self.sanitize_input(plan)}

# # Current code implementation of the plan:
# {self.sanitize_input(code)}

# # Test Log:
# {self.sanitize_input(test_log)}

# ----------------
# IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

# ## Simulation
# [Your simulation]

# ## Insights
# [Your insights]
# """
#             }
#         ]

#         logger.info(f"Prompt for plan_analysis:\n{input_prompt[0]['content']}")

#         simulation = 'No simulation provided'
#         insights = 'No insights provided'
#         pr_tok = 0
#         com_tok = 0
#         for attempt in range(self.max_attempts):
#             try:
#                 response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_prompt)
#                 pr_tok += pr_tok_temp
#                 com_tok += com_tok_temp
#                 logger.info(f"Response from plan_analysis (attempt {attempt + 1}):\n{response[:200]}...")
#                 parsed = self.parse_structured_text(response)
#                 simulation = parsed.get('simulation', 'No simulation provided')
#                 insights = parsed.get('insights', 'No insights provided')
#                 break
#             except Exception as e:
#                 logger.error(f"Error in plan_analysis on attempt {attempt + 1}: {e}", exc_info=True)
#                 if attempt == self.max_attempts - 1:
#                     logger.warning("Max attempts reached, using defaults.")

#         logger.info(f"Plan analysis results: simulation={simulation[:50]}..., insights={insights[:50]}...")
#         return {
#             'simulation': simulation,
#             'insights': insights,
#             'pr_tok': pr_tok,
#             'com_tok': com_tok
#         }

#     def code_analysis(self, code: str, test_log: str, problem: str, problem_understanding: str) -> dict:
#         context = ""
#         if self.include_mode == 'both':
#             context = f"# Problem:\n{self.sanitize_input(problem)}\n# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
#         elif self.include_mode == 'problem_only':
#             context = f"# Problem:\n{self.sanitize_input(problem)}"
#         elif self.include_mode == 'understanding_only':
#             context = f"# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

#         input_prompt = [
#             {
#                 "role": "user",
#                 "content": f"""Assess the generated code written in {self.language} programming language for a competitive programming problem, using the problem description, its understanding, and test log. Identify where the code is failing based on the test log, and suggest specific improvements or fixes to correct those issues, considering edge cases and special cases from the problem understanding.

# {context}

# # Code:
# ```{self.language}
# {self.sanitize_input(code)}
# ```

# # Test Log:
# {self.sanitize_input(test_log)}

# ----------------
# IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep insights concise (<150 words).

# ## Insights
# [Your insights]
# """
#             }
#         ]

#         logger.info(f"Prompt for code_analysis:\n{input_prompt[0]['content']}")

#         insights = 'No insights provided'
#         pr_tok = 0
#         com_tok = 0
#         for attempt in range(self.max_attempts):
#             try:
#                 response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_prompt)
#                 pr_tok += pr_tok_temp
#                 com_tok += com_tok_temp
#                 logger.info(f"Response from code_analysis (attempt {attempt + 1}):\n{response[:200]}...")
#                 parsed = self.parse_structured_text(response)
#                 insights = parsed.get('insights', 'No insights provided')
#                 break
#             except Exception as e:
#                 logger.error(f"Error in code_analysis on attempt {attempt + 1}: {e}", exc_info=True)
#                 if attempt == self.max_attempts - 1:
#                     logger.warning("Max attempts reached, using default.")

#         logger.info(f"Code analysis insights: {insights[:50]}...")
#         return {
#             'insights': insights,
#             'pr_tok': pr_tok,
#             'com_tok': com_tok
#         }

#     def content_analysis(self, problem: str, problem_understanding: str, plan: str, code: str) -> dict:
#         context = ""
#         if self.include_mode == 'both':
#             context = f"# Problem:\n{self.sanitize_input(problem)}\n# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"
#         elif self.include_mode == 'problem_only':
#             context = f"# Problem:\n{self.sanitize_input(problem)}"
#         elif self.include_mode == 'understanding_only':
#             context = f"# Problem Understanding:\n{self.sanitize_input(problem_understanding)}"

#         input_prompt = [
#             {
#                 "role": "user",
#                 "content": f"""Evaluate how effectively a plan and generated code, written in {self.language} programming language, align with the requirements of a competitive programming problem, given the problem description and its understanding. Assess the alignment between the problem (and its understanding) and the plan, and between the plan and the code. Identify any mismatches or issues in these alignments, considering edge cases and special cases from the problem understanding. Provide separate confidence scores (0.0 to 1.0) for the problem-plan alignment and plan-code alignment (1.0: perfect match; 0.0: no alignment), and suggest specific improvements for each if the alignment is not strong.

# {context}

# # Plan:
# {self.sanitize_input(plan)}

# # Code:
# ```{self.language}
# {self.sanitize_input(code)}
# ```

# ----------------
# IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep insights concise (<150 words each).

# ## Problem Plan Confidence
# [Score as float]

# ## Plan Code Confidence
# [Score as float]

# ## Problem Plan Insights
# [Your insights]

# ## Plan Code Insights
# [Your insights]
# """
#             }
#         ]

#         logger.info(f"Prompt for content_analysis:\n{input_prompt[0]['content']}")

#         problem_plan_confidence = 0.0
#         plan_code_confidence = 0.0
#         overall_confidence = 0.0
#         problem_plan_insights = 'No insights provided'
#         plan_code_insights = 'No insights provided'
#         pr_tok = 0
#         com_tok = 0
#         for attempt in range(self.max_attempts):
#             try:
#                 response, pr_tok_temp, com_tok_temp = self.gpt_chat(processed_input=input_prompt)
#                 pr_tok += pr_tok_temp
#                 com_tok += com_tok_temp
#                 logger.info(f"Response from content_analysis (attempt {attempt + 1}):\n{response[:200]}...")
#                 parsed = self.parse_structured_text(response)
#                 problem_plan_confidence = float(parsed.get('problem_plan_confidence', 0.0))
#                 plan_code_confidence = float(parsed.get('plan_code_confidence', 0.0))
#                 overall_confidence = problem_plan_confidence * plan_code_confidence
#                 problem_plan_insights = parsed.get('problem_plan_insights', 'No insights provided')
#                 plan_code_insights = parsed.get('plan_code_insights', 'No insights provided')
#                 break
#             except Exception as e:
#                 logger.error(f"Error in content_analysis on attempt {attempt + 1}: {e}", exc_info=True)
#                 if attempt == self.max_attempts - 1:
#                     logger.warning("Max attempts reached, using defaults.")

#         insights = f"## Problem-Plan Alignment Analysis: {problem_plan_insights}\n## Plan-Code Alignment Analysis: {plan_code_insights}"
#         logger.info(f"Content analysis results: problem_plan_confidence={problem_plan_confidence}, plan_code_confidence={plan_code_confidence}")
#         return {
#             'problem_plan_confidence': max(0.0, min(problem_plan_confidence, 1.0)),
#             'plan_code_confidence': max(0.0, min(plan_code_confidence, 1.0)),
#             'confidence': max(0.0, min(overall_confidence, 1.0)),
#             'problem_plan_insights': problem_plan_insights,
#             'plan_code_insights': plan_code_insights,
#             'insights': insights,
#             'pr_tok': pr_tok,
#             'com_tok': com_tok
#         }

#     def get_confidence(self, decision: str, analysis: dict, analysis_name: str) -> float:
#         meaning = self.analysis_meaning.get(analysis_name, "")

#         prompt = [
#             {
#                 "role": "user",
#                 "content": f"""Given a {analysis_name} analysis. {meaning} Calculate the confidence score (0.0 to 1.0) for choosing to {decision}, where 1.0 means the analysis strongly supports the decision (e.g., insights indicate it's the best fix), and 0.0 means it does not support at all. Provide a brief explanation (50-100 words) of how the score was determined, referencing specific insights.

# Insights:
# {self.sanitize_input(analysis.get('insights', ''))}

# =============
# IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

# ## Confidence
# [Score as float]

# ## Reasoning
# [Your reasoning]
# """
#             }
#         ]

#         logger.info(f"Prompt for get_confidence (decision: {decision}, analysis: {analysis_name}):\n{prompt[0]['content']}")

#         score = 0.0
#         for attempt in range(self.max_attempts):
#             try:
#                 response, pr_tok, com_tok = self.gpt_chat(processed_input=prompt)
#                 logger.info(f"Response from get_confidence (attempt {attempt + 1}):\n{response[:200]}...")
#                 parsed = self.parse_structured_text(response)
#                 score = float(parsed.get('confidence', 0.0))
#                 reasoning = parsed.get('reasoning', 'No reasoning provided')
#                 break
#             except Exception as e:
#                 logger.error(f"Error in get_confidence on attempt {attempt + 1}: {e}", exc_info=True)
#                 if attempt == self.max_attempts - 1:
#                     logger.warning("Max attempts reached, using default score 0.0.")

#         logger.info(f"Confidence score for {decision}: {score}")
#         return max(0.0, min(score, 1.0))

#     def get_consistency(
#         self,
#         decision: str,
#         analysis1: dict, name1: str,
#         analysis2: dict, name2: str
#     ) -> float:
#         ins1 = self.sanitize_input(analysis1.get('insights', '').strip())
#         ins2 = self.sanitize_input(analysis2.get('insights', '').strip())

#         name1_meaning = self.analysis_meaning.get(name1, "")
#         name2_meaning = self.analysis_meaning.get(name2, "")

#         prompt = [
#             {
#                 "role": "user",
#                 "content": f"""Given insights from two analyses: {name1} and {name2}. 
# {name1} meaning: {name1_meaning}
# {name2} meaning: {name2_meaning}
# Calculate the consistency score (0.0 to 1.0) for choosing to {decision}, where 1.0 means the insights from both analyses are highly consistent and support the decision (e.g., similar issues and fixes suggested), and 0.0 means they are inconsistent or contradictory. Provide a brief explanation (50-100 words) of how the score was determined, referencing specific insights from both analyses.

# {name1} insights:
# {ins1}

# {name2} insights:
# {ins2}

# IMPORTANT: Output in this exact structured text format with headings. Do not use JSON or extra text. Keep concise.

# ## Consistency
# [Score as float]

# ## Reasoning
# [Your reasoning]
# """
#             }
#         ]

#         logger.info(f"Prompt for get_consistency (decision: {decision}, analyses: {name1}, {name2}):\n{prompt[0]['content']}")

#         score = 0.0
#         for attempt in range(self.max_attempts):
#             try:
#                 response, pr_tok, com_tok = self.gpt_chat(processed_input=prompt)
#                 logger.info(f"Response from get_consistency (attempt {attempt + 1}):\n{response[:200]}...")
#                 parsed = self.parse_structured_text(response)
#                 score = float(parsed.get('consistency', 0.0))
#                 reasoning = parsed.get('reasoning', 'No reasoning provided')
#                 break
#             except Exception as e:
#                 logger.error(f"Error in get_consistency on attempt {attempt + 1}: {e}", exc_info=True)
#                 if attempt == self.max_attempts - 1:
#                     logger.warning("Max attempts reached, using default score 0.0.")

#         logger.info(f"Consistency score for {decision}: {score}")
#         return max(0.0, min(score, 1.0))

#     def collaborative_decision(self, plan: str, code: str, outcomes: str, item) -> str:
#         try:
#             problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
#             problem_text = self.sanitize_input(self.data.get_prompt(item))

#             A_plan = self.plan_analysis(plan, code, outcomes, problem_text, problem_understanding)
#             A_code = self.code_analysis(code, outcomes, problem_text, problem_understanding)
#             A_content = self.content_analysis(problem_text, problem_understanding, plan, code)

#             decisions = ['update plan', 'update code only']
#             scores = {}

#             for d in decisions:
#                 total = 0.0
#                 for name, A_i in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
#                     w = self.trust_weights[name]
#                     conf = self.get_confidence(d, A_i, name)
#                     cons_prod = 1.0
#                     for oname, A_j in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
#                         if oname != name:
#                             cons_prod *= self.get_consistency(d, A_i, name, A_j, oname)
#                     total += w * conf * cons_prod
#                 scores[d] = total

#             final_decision = max(scores, key=scores.get)
#             logger.info(f"Collaborative decision: {final_decision}, scores={scores}")
#             return final_decision
#         except Exception as e:
#             logger.error(f"Error in collaborative_decision: {e}", exc_info=True)
#             return "update code only"

#     def debug_plan(
#         self,
#         iteration: int,
#         plan: str,
#         error_analysis: Dict,
#         problem: str,
#         problem_understanding: str
#     ):
#         prev_logs = self.rt.historical_data.get(iteration - 1)
#         rt_prompt = self.rt.generate_prompt_for_plan_update(
#             iteration,
#             error_analysis,
#             problem,
#             problem_understanding,
#             plan,
#             historical_logs=prev_logs
#         )

#         try:
#             rt_response, pr_tok_rt, com_tok_rt = self.gpt_chat(
#                 processed_input=[{'role': 'user', 'content': rt_prompt}]
#             )
#             logger.info(f"Response from debug_plan (iteration {iteration}):\n{rt_response[:200]}...")
#             reasoning_trajectory = rt_response.strip()
#         except Exception as e:
#             logger.error(f"Error in debug_plan (gpt_chat): {e}", exc_info=True)
#             reasoning_trajectory = "Error generating reasoning trajectory"

#         context = ""
#         if self.include_mode == 'both':
#             context = f"# Problem: {self.sanitize_input(problem)}\n# Problem Understanding: {self.sanitize_input(problem_understanding)}"
#         elif self.include_mode == 'problem_only':
#             context = f"# Problem: {self.sanitize_input(problem)}"
#         elif self.include_mode == 'understanding_only':
#             context = f"# Problem Understanding: {self.sanitize_input(problem_understanding)}"

#         update_prompt = [{
#             'role': 'user',
#             'content': (
#                 f"""Given a competitive programming problem, its understanding, and a plan to solve it, but this plan has some troubles that need to be updated with insights. Please modify the plan accordingly.

# {context}

# # Current Planning:
# {self.sanitize_input(plan)}

# # Insights:
# {self.sanitize_input(error_analysis['insights'])}

# # Reasoning Trajectory:
# {self.sanitize_input(reasoning_trajectory)}

# # Instructions:
# Output only the revised plan text. Do not add extra explanation or words.
# """
#             )
#         }]

#         logger.info(f"Prompt for debug_plan update (iteration {iteration}):\n{update_prompt[0]['content']}")

#         try:
#             updated, pr_tok_up, com_tok_up = self.gpt_chat(processed_input=update_prompt)
#             logger.info(f"Updated plan response (iteration {iteration}):\n{updated[:200]}...")
#             revised_plan = updated.strip()
#         except Exception as e:
#             logger.error(f"Error in debug_plan (update): {e}", exc_info=True)
#             revised_plan = plan

#         self.rt.update_historical_data(iteration, {
#             'previous_plan': plan,
#             'previous_success_rate': error_analysis.get('success_rate'),
#             'previous_iteration': iteration - 1
#         })
#         logger.info(f"Debug plan completed for iteration {iteration}: revised_plan={revised_plan[:100]}...")
#         return revised_plan, reasoning_trajectory

#     def debug_code(
#         self,
#         iteration: int,
#         plan: str,
#         code: str,
#         error_analysis: Dict,
#         problem: str,
#         problem_understanding: str
#     ):
#         prev_logs = self.rt.historical_data.get(iteration - 1)
#         rt_prompt = self.rt.generate_prompt_for_code_update(
#             iteration,
#             error_analysis,
#             problem,
#             problem_understanding,
#             plan,
#             code,
#             historical_logs=prev_logs
#         )

#         try:
#             rt_response, pr_tok_rt, com_tok_rt = self.gpt_chat(
#                 processed_input=[{'role': 'user', 'content': rt_prompt}]
#             )
#             logger.info(f"Response from debug_code (iteration {iteration}):\n{rt_response[:200]}...")
#             reasoning_trajectory = rt_response.strip()
#         except Exception as e:
#             logger.error(f"Error in debug_code (gpt_chat): {e}", exc_info=True)
#             reasoning_trajectory = "Error generating reasoning trajectory"

#         context = ""
#         if self.include_mode == 'both':
#             context = f"# Problem: {self.sanitize_input(problem)}\n# Problem Understanding: {self.sanitize_input(problem_understanding)}"
#         elif self.include_mode == 'problem_only':
#             context = f"# Problem: {self.sanitize_input(problem)}"
#         elif self.include_mode == 'understanding_only':
#             context = f"# Problem Understanding: {self.sanitize_input(problem_understanding)}"

#         code_prompt = [{
#             'role': 'user',
#             'content': (
#                 f"""Given a competitive programming problem, its understanding, and generated {self.language} code to solve the problem, but the generated code cannot pass sample test cases. Improve your code to solve the problem correctly.

# {context}

# ## Planning:
# {self.sanitize_input(plan)}

# ## Code:
# ```{self.language}
# {self.sanitize_input(code)}
# ```

# ## Test Report:
# {self.sanitize_input(error_analysis.get('test_results',''))}

# ## Insights:
# {self.sanitize_input(error_analysis['insights'])}

# ## Reasoning Trajectory:
# {self.sanitize_input(reasoning_trajectory)}

# ## Instructions:
# Output only the revised {self.language} code. Do not add extra explanation or words.
# """
#             )
#         }]

#         logger.info(f"Prompt for debug_code update (iteration {iteration}):\n{code_prompt[0]['content']}")

#         try:
#             updated, pr_tok_up, com_tok_up = self.gpt_chat(processed_input=code_prompt)
#             logger.info(f"Updated code response (iteration {iteration}):\n{updated[:200]}...")
#             revised_code = self.parse_code(updated)
#         except Exception as e:
#             logger.error(f"Error in debug_code (update): {e}", exc_info=True)
#             revised_code = code

#         self.rt.update_historical_data(iteration, {
#             'previous_code': code,
#             'previous_success_rate': error_analysis.get('success_rate'),
#             'previous_iteration': iteration - 1
#         })
#         logger.info(f"Debug code completed for iteration {iteration}: revised_code={revised_code[:100]}...")
#         return revised_code, reasoning_trajectory

#     def _inner_run(self, item):
#         pr_tok = 0
#         com_tok = 0

#         try:
#             plannings, pr_tok_p, com_tok_p = self.generate_plans(item)
#             pr_tok += pr_tok_p
#             com_tok += com_tok_p
#             logger.info(f"Generated plannings: {len(plannings)} plans")
#         except Exception as e:
#             logger.error(f"Error in _inner_run (generate_plans): {e}", exc_info=True)
#             plannings = []

#         selected_plannings = plannings[:self.top_plan]
#         logger.info(f"Selected top {self.top_plan} plannings")

#         best_code = ""
#         flag = False
#         test_log = ""

#         for plan_idx, planning_with_ex in enumerate(selected_plannings, 1):
#             plan, confidence, example = planning_with_ex
#             approach_name = example.get('description', '') if 'description' in example else ''
#             approach_tutorial = example.get('planning', '') if 'planning' in example else ''
#             algorithm_prompt = f"## Relevant Approach: {self.sanitize_input(approach_name)}\n{self.sanitize_input(approach_tutorial)}"
#             sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item)}\n"
#             if isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset)):
#                 std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
#             else:
#                 std_input_prompt = ""

#             try:
#                 best_code, flag, test_log, score, pr_tok_c, com_tok_c = self.generate_codes_from_plan(item, plan, algorithm_prompt, sample_io_prompt)
#                 pr_tok += pr_tok_c
#                 com_tok += com_tok_c
#                 logger.info(f"Generated code for plan {plan_idx}: score={score}, passed={flag}")
#             except Exception as e:
#                 logger.error(f"Error in _inner_run (generate_codes_from_plan) for plan {plan_idx}: {e}", exc_info=True)
#                 continue

#             if flag:
#                 logger.info(f"Plan {plan_idx} passed all tests, returning best code")
#                 return best_code, pr_tok, com_tok

#             for i in range(1, self.t + 1):
#                 try:
#                     problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
#                     pr_tok += pr_tok_u
#                     com_tok += com_tok_u
#                     logger.info(f"Problem understanding for iteration {i}: {problem_understanding[:100]}...")
#                 except Exception as e:
#                     logger.error(f"Error in _inner_run (get_problem_understanding) iteration {i}: {e}", exc_info=True)
#                     problem_understanding = ""

#                 try:
#                     decision = self.collaborative_decision(plan, best_code, test_log, item)
#                     logger.info(f"Decision for iteration {i}: {decision}")
#                 except Exception as e:
#                     logger.error(f"Error in _inner_run (collaborative_decision) iteration {i}: {e}", exc_info=True)
#                     decision = "update code only"

#                 if decision == 'update plan':
#                     try:
#                         A_plan = self.plan_analysis(plan, best_code, test_log, self.data.get_prompt(item), problem_understanding)
#                         plan, reasoning_trajectory = self.debug_plan(i + 1, plan, {
#                             'insights': A_plan['insights'],
#                             'test_results': test_log,
#                             'success_rate': score,
#                         }, self.data.get_prompt(item), problem_understanding)

#                         code, flag, test_log, score, p_c2, c_c2 = self.generate_codes_from_plan(
#                             item, plan, algorithm_prompt="", sample_io_prompt=sample_io_prompt
#                         )
#                         pr_tok += p_c2
#                         com_tok += c_c2
#                         best_code = code
#                         logger.info(f"Updated plan and code for iteration {i}: score={score}, passed={flag}")
#                     except Exception as e:
#                         logger.error(f"Error in _inner_run (update plan) iteration {i}: {e}", exc_info=True)
#                         continue
#                 else:
#                     try:
#                         A_code = self.code_analysis(best_code, test_log, self.data.get_prompt(item), problem_understanding)
#                         best_code, reasoning_trajectory = self.debug_code(i + 1, plan, best_code, {
#                             'insights': A_code['insights'],
#                             'test_results': test_log,
#                             'success_rate': score,
#                         }, self.data.get_prompt(item), problem_understanding)
#                         logger.info(f"Updated code for iteration {i}: {best_code[:100]}...")
#                     except Exception as e:
#                         logger.error(f"Error in _inner_run (update code) iteration {i}: {e}", exc_info=True)
#                         continue

#                 try:
#                     flag, test_log = self.data.evaluate_sample_io(item, best_code, self.language)
#                     logger.info(f"Evaluated code for iteration {i}: passed={flag}, test_log={test_log[:100]}...")
#                 except Exception as e:
#                     logger.error(f"Error in _inner_run (evaluate_sample_io) iteration {i}: {e}", exc_info=True)
#                     flag, test_log = False, f"Evaluation failed: {e}"

#                 if flag:
#                     logger.info(f"Code passed all tests in iteration {i}, returning best code")
#                     return best_code, pr_tok, com_tok

#         logger.info(f"No passing code found, returning best code: {best_code[:100]}...")
#         return best_code, pr_tok, com_tok

#     def run_single_pass(self, item: dict):
#         max_retries = 3
#         for attempt in range(1, max_retries + 1):
#             try:
#                 logger.info(f"Starting run_single_pass attempt {attempt} with item: {item}")
#                 result = self._inner_run(item)
#                 logger.info(f"run_single_pass completed successfully: result={result[0][:100]}...")
#                 return result
#             except Exception as e:
#                 logger.error(f"[run_single_pass] Attempt {attempt} caught exception: {e}", exc_info=True)
#                 logger.info(f"[run_single_pass] Item: {item}")
#                 if attempt == max_retries:
#                     logger.error(f"[run_single_pass] All {max_retries} attempts failed. Returning fallback ('',0,0).")
#                     return "", 0, 0