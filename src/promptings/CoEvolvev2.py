from typing import List, Optional, Dict
import tiktoken
import os
import json
import re
import sys
import time
from datetime import datetime
from copy import deepcopy
import xml.etree.ElementTree as ET

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
        test_results = error_analysis.get('test_results', '')
        analysis = error_analysis.get('analysis', '')
        success_rate = error_analysis.get('success_rate', '0')

        if iteration == 1:
            prompt = f"""
You are the Reasoning Trajectory module (R_traj) in the CoEvolve framework, an advanced AI specializing in iterative plan refinement for competitive programming. Your goal is to weave initial context into clear, strategic guidance that empowers the Planning agent to develop robust, error-resistant plans from the start.

### Context
- Problem: {problem}
- Problem Understanding: {problem_understanding}
- Initial Plan: {plan}
- Test Results: {test_results}
- Error Analysis: {analysis}
- Success Rate: {success_rate}%

### Task
Create the initial reasoning prompt for plan revision in iteration 1. With no historical data available, focus on:
1. Anticipating potential weaknesses in the plan based on the problem's details and understanding.
2. Drawing on planning strategies and common pitfalls, considering edge cases and special cases.
3. Setting forward-looking priorities for a thorough plan.

### Structured Output
## Predicted Plan Vulnerabilities
[...]

## Domain Expertise Synthesis
[...]

## Strategic Prioritization
[...]

## Reasoning Prompt for Plan Update
[Provide guidance here]

### Instructions
- Use the specified format.
- Keep it concise and precise.
"""
        else:
            prev = historical_logs or {}
            prev_iteration = prev.get('previous_iteration', iteration - 1)
            prev_success_rate = prev.get('previous_success_rate', '0')
            previous_plan = prev.get('previous_plan', '')

            prompt = f"""
You are the Reasoning Trajectory module (R_traj) in the CoEvolve framework, specializing in iterative plan refinement.

### Context
- Problem: {problem}
- Problem Understanding: {problem_understanding}
- Current Plan (Iteration {iteration}): {plan}
- Prior Plan (Iteration {prev_iteration}): {previous_plan}
- Test Results: {test_results}
- Error Analysis: {analysis}
- Success Rate: {success_rate}%

### Task
Create a reasoning prompt for plan revision. Follow these steps:
1. Plan Error Localization
2. Historical Plan Analysis
3. Plan Adjustment Priorities, considering edge cases and special cases from the problem understanding
4. Reasoning Prompt for Plan Update

### Instructions
- Structure output as specified.
- Focus on actionable guidance.
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
        test_results = error_analysis.get('test_results', '')
        analysis = error_analysis.get('analysis', '')
        success_rate = error_analysis.get('success_rate', '0')

        if iteration == 1:
            prompt = f"""
You are the Reasoning Trajectory module (R_traj), specializing in iterative code refinement.

### Context
- Problem: {problem}
- Problem Understanding: {problem_understanding}
- Plan: {plan}
- Initial Code: {code}
- Test Results: {test_results}
- Error Analysis: {analysis}
- Success Rate: {success_rate}%

### Task
Create the initial reasoning prompt for code revision in iteration 1. Focus on potential weaknesses, proven coding strategies, and priorities for robust code, considering edge cases and special cases from the problem understanding.

### Structured Output
## Predicted Code Vulnerabilities
[...]

## Domain Expertise Synthesis
[...]

## Strategic Prioritization
[...]

## Reasoning Prompt for Code Update
[Provide guidance here]

### Instructions
- Use the specified format.
- Keep it concise and precise.
"""
        else:
            prev = historical_logs or {}
            prev_iteration = prev.get('previous_iteration', iteration - 1)
            prev_success_rate = prev.get('previous_success_rate', '0')
            previous_code = prev.get('previous_code', '')

            prompt = f"""
You are the Reasoning Trajectory module (R_traj), specializing in iterative code refinement.

### Context
- Problem: {problem}
- Problem Understanding: {problem_understanding}
- Plan: {plan}
- Current Code (Iteration {iteration}): {code}
- Prior Code (Iteration {prev_iteration}): {previous_code}
- Test Results: {test_results}
- Error Analysis: {analysis}
- Success Rate: {success_rate}%

### Task
Create a reasoning prompt for code revision. Follow these steps:
1. Code Error Localization
2. Historical Code Analysis
3. Code Adjustment Priorities, considering edge cases and special cases from the problem understanding
4. Reasoning Prompt for Code Update

### Instructions
- Structure output as specified.
- Focus on actionable guidance.
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

    def xml_to_dict(self, element):
        result = {}
        for child in element:
            if child:
                child_data = self.xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            else:
                result[child.tag] = child.text
        return result

    def parse_xml(self, response: str) -> dict:
        if '```xml' in response:
            response = response.replace('```xml', '')
        if '```' in response:
            response = response.replace('```', '')

        try:
            root = ET.fromstring(response)
        except:
            try:
                root = ET.fromstring('<root>\n' + response + '\n</root>')
            except:
                root = ET.fromstring('<root>\n' + response)
        return self.xml_to_dict(root)

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

        if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
            code_str = "\n".join(code_blocks[-1])
        elif type(code_blocks[-1]) == str:
            code_str = code_blocks[-1]
        else:
            code_str = response

        return code_str

    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, '').strip()

    @staticmethod
    def replace_tag(text: str, tag: str):
        if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
            return text 
        else:
            return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()

    def get_sample_io_str(self, item) -> str:
        if type(self.data) == XCodeDataset:
            return self.get_sample_io_xcode(item)
        else:
            sample_io = item['sample_io']
            if len(sample_io) > 0:
                if type(sample_io[0]) == str:
                    return "\n".join(sample_io)
                if type(sample_io[0]) == dict:
                    return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
            return ''

    def get_sample_io_xcode(self, item):
        return "\n".join([f"Input:\n{item['sample_inputs']}\nExpected output:\n{item['sample_outputs']}"])

    def get_problem_understanding(self, item) -> str:
        input_prompt = [
            {
                "role": "user",
                "content": f"""You are an expert in competitive programming. Your task is to analyze a problem description and provide a concise understanding of the problem, including its requirements, constraints, objectives, and potential edge cases or special cases. Identify edge cases (e.g., boundary conditions, invalid inputs, or extreme scenarios) and special cases (e.g., unique input patterns or problem-specific conditions) that need attention, and provide examples of these cases beyond the provided sample I/O.

    # Problem:
    {self.data.get_prompt(item)}

    # Sample I/O:
    {self.get_sample_io_str(item)}

    ----------------
    Important: Respond in the following XML format. Keep the understanding concise, including a brief description of requirements, constraints, objectives, edge cases, special cases, and at least one example for each beyond the sample I/O.
    ```xml
    <root>
    <understanding>Text describing the problem understanding, including requirements, constraints, objectives, edge cases, special cases, and examples</understanding>
    </root>
    ```"""
            }
        ]

        response, pr_tok, com_tok = self.gpt_chat(processed_input=input_prompt)
        response = self.replace_tag(response, 'understanding')
        parsed = self.parse_xml(response)
        understanding = parsed.get('understanding', 'No understanding provided')

        return understanding, pr_tok, com_tok

    def generate_plans(self, item):
        plannings = []
        pr_tok = 0
        com_tok = 0
        previous_approaches = ""

        problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
        pr_tok += pr_tok_u
        com_tok += com_tok_u

        for t in range(1, self.k + 1):
            diff_prompt = "" if t == 1 else f", different from the following previous approaches: {previous_approaches}"

            input_recall = [
                {
                    "role": "user",
                    "content": f"""Given a problem and its understanding, recall an approach that can solve it{diff_prompt}, provide a tutorial for the approach, then recall a relevant problem that uses this approach, and explain with plan and code.

# Problem:
{self.data.get_prompt(item)}

# Problem Understanding:
{problem_understanding}

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
Important:
Your response must follow the following xml format-

<root>
<approach>
<name>The name of the approach</name>
<tutorial>The tutorial for the approach</tutorial>
</approach>
<problem>
<description>Describe the problem.</description>
<code>Let's think step by step to solve this problem in {self.language} programming language using the approach.</code>
<planning>Planning to solve this problem using the approach.</planning>
</problem>
</root>
"""
                },
            ]

            response, pr_tok_1, com_tok_1 = self.gpt_chat(processed_input=input_recall)
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            item['api_calls'] = item.get('api_calls', 0) + 1

            response = self.trim_text(response, "The name of the approach")
            response = self.trim_text(response, "The tutorial for the approach")
            response = self.trim_text(response, "Describe the problem.")
            response = self.trim_text(response, f"Let's think step by step to solve this problem in {self.language} programming language using the approach.")
            response = self.trim_text(response, "Planning to solve this problem using the approach.")
            response = self.replace_tag(response, 'name')
            response = self.replace_tag(response, 'tutorial')
            response = self.replace_tag(response, 'description')
            response = self.replace_tag(response, 'code')
            response = self.replace_tag(response, 'planning')
            try:
                parsed_response = self.parse_xml(response)
            except ET.ParseError as e:
                print(f"Error parsing XML when generating plans: {str(e)}")
                continue

            approach_name = parsed_response['approach']['name']
            approach_tutorial = parsed_response['approach']['tutorial']
            algorithm_prompt = f"## Relevant Approach: {approach_name}\n{approach_tutorial}"
            example = parsed_response['problem']

            previous_approaches += f"\n- {approach_name}"

            example_problem = example["description"]
            example_planning = example["planning"]

            sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item)}\n"

            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem and its understanding, generate a concrete planning to solve the problem.\n# Problem:\n{example_problem}\n# Planning:\n{example_planning}\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n## Problem Understanding:\n{problem_understanding}\n{sample_io_prompt}\n## Planning:\n\n----------------\nImportant: You should give only the planning to solve the problem. Do not add extra explanation or words."
                }
            ]

            planning, pr_tok_1, com_tok_1 = self.gpt_chat(
                processed_input=input_for_problem_planning
            )
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            item['api_calls'] += 1

            input_for_planning_verification = [
                {
                    "role": "user",
                    "content": f"""You are an expert evaluator for competitive programming plans. Your task is to assess a given plan for solving a programming problem based on two key criteria: problem-plan alignment and plan coherence. Provide reasoned explanations and scores for each, then compute an overall solvability score as the product of the two scores.

### Evaluation Criteria:
1. **Problem-Plan Alignment** (Score: 0.0 to 1.0):
   - Measures how well the plan addresses the problem's requirements, constraints, objectives, input/output formats, and edge cases.
   - 1.0: Perfect match—the plan fully covers all aspects of the problem, including time/space complexity, sample I/O (if provided), and potential pitfalls.
   - 0.5: Partial match—the plan addresses core elements but misses some constraints, edge cases, or optimizations.
   - 0.0: No match—the plan is irrelevant or fundamentally misunderstands the problem.
   - Consider: Does the plan handle all inputs/outputs correctly? Is it efficient for given constraints? Does it align with the problem's goals?

2. **Plan Coherence** (Score: 0.0 to 1.0):
   - Measures the internal logical consistency, feasibility, and completeness of the plan itself (independent of the problem).
   - 1.0: Fully coherent—the plan is logically sound, step-by-step clear, feasible to implement in {self.language}, and free of contradictions or gaps.
   - 0.5: Moderately coherent—the plan has logical flow but includes minor inconsistencies, ambiguities, or incomplete steps.
   - 0.0: Incoherent—the plan is illogical, infeasible, or riddled with errors/gaps.
   - Consider: Is the plan structured logically? Are steps executable? Does it use appropriate data structures/algorithms without contradictions?

3. **Overall Solvability Score**:
   - Compute as (problem-plan alignment score) * (plan coherence score), then multiply by 100 and round to the nearest integer (0-100).
   - This represents the estimated likelihood that the plan can solve the problem correctly.

### Input:
- Problem Description: {self.data.get_prompt(item)}
- Problem Understanding: {problem_understanding}
- Proposed Plan: {planning}
- Sample I/O (if available): {sample_io_prompt}

### Instructions:
- First, explain your reasoning for each criterion in concise, bullet-point form.
- Then, assign floating-point scores (0.0-1.0) based on the criteria.
- Finally, compute and output the overall solvability score as an integer (0-100).
- Be objective, evidence-based, and critical—reference specific parts of the problem, understanding, and plan.

### Output Format:
Respond ONLY in the following strict XML structure. Use CDATA for explanations to handle special characters. No additional text.
<root>
  <alignment_explanation><![CDATA[Your bullet-point explanation for alignment.]]></alignment_explanation>
  <alignment_score>Float between 0.0 and 1.0</alignment_score>
  <coherence_explanation><![CDATA[Your bullet-point explanation for coherence.]]></coherence_explanation>
  <coherence_score>Float between 0.0 and 1.0</coherence_score>
  <overall_solvability>Integer between 0 and 100</overall_solvability>
</root>
"""
                }
            ]

            verification_res, pr_tok_1, com_tok_1 = self.gpt_chat(
                processed_input=input_for_planning_verification
            )
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            item['api_calls'] += 1

            verification_res = self.replace_tag(verification_res, 'alignment_explanation')
            verification_res = self.replace_tag(verification_res, 'alignment_score')
            verification_res = self.replace_tag(verification_res, 'coherence_explanation')
            verification_res = self.replace_tag(verification_res, 'coherence_score')
            verification_res = self.replace_tag(verification_res, 'overall_solvability')
            try:
                verification_parsed = self.parse_xml(verification_res)
            except ET.ParseError as e:
                print(f"Error parsing XML response for verification of planning: {str(e)}")
                continue

            confidence = int(float(verification_parsed['alignment_score']) * 
                             float(verification_parsed['coherence_score']) * 100)

            plannings.append((planning, confidence, example))

        plannings.sort(key=lambda x: x[1], reverse=True)

        return plannings, pr_tok, com_tok

    def generate_codes_from_plan(self, item, plan, algorithm_prompt, sample_io_prompt):
        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        codes = []
        pr_tok = 0
        com_tok = 0

        problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
        pr_tok += pr_tok_u
        com_tok += com_tok_u

        for i in range(self.number_of_code_per_plan):
            variation_prompt = "" #if i == 0 else "Generate a different code variation that still strictly follows the plan."
    
            input_for_code_generation = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem and its understanding, generate {self.language} code to solve the problem.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n## Problem Understanding:\n{problem_understanding}\n## Planning:\n{plan}\n{sample_io_prompt}\n## Let's think step by step.\n{variation_prompt}\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
                }
            ]

            code_response, pr_tok_1, com_tok_1 = self.gpt_chat(
                processed_input=input_for_code_generation
            )
            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            code = self.parse_code(code_response)
            codes.append(code)

        evaluated = []
        for idx, code in enumerate(codes):
            passed, test_log = self.data.evaluate_sample_io(item, code, self.language)
            num_passed = test_log.count("passed in test case")
            num_failed = test_log.count("failed in test case")
            total = num_passed + num_failed
            if total > 0:
                passed_score = num_passed / total 
            else:
                passed_score = 1.0 if passed else 0.0
            evaluated.append((code, passed_score, test_log, passed))

        evaluated.sort(key=lambda x: x[1], reverse=True)
        best_code = evaluated[0][0]
        flag = evaluated[0][3]
        best_test_log = evaluated[0][2]

        return best_code, flag, best_test_log, evaluated[0][1], pr_tok, com_tok

    def plan_analysis(self, plan: str, code: str, test_log: str, problem: str, problem_understanding: str) -> dict:
        input_prompt = [
            {
                "role": "user",
                "content": f"""Analyze a plan for solving a competitive programming problem, given the problem description, its understanding, and test log from code generated using the plan. Take a sample input from the test log and simulate the plan's execution step-by-step to pinpoint where the plan is failing based on the test log, and suggest specific improvements or modifications to fix those issues, considering edge cases and special cases from the problem understanding.

    # Problem:
    {problem}

    # Problem Understanding:
    {problem_understanding}

    # Plan:
    {plan}

    # Current code implementation of the plan:
    {code}

    # Test Log:
    {test_log}

    ----------------
    Important: Respond in the following XML format. Keep simulation and insights concise, focusing on errors and suggestions.
    ```xml
    <root>
    <simulation>Text describing the step-by-step simulation of the plan with a sample input from the test log, identifying failures</simulation>
    <insights>Text identifying errors in the plan and suggesting specific fixes</insights>
    </root>
    ```
    """
            }
        ]

        response, pr_tok, com_tok = self.gpt_chat(processed_input=input_prompt)
        response = self.replace_tag(response, 'simulation')
        response = self.replace_tag(response, 'insights')
        parsed = self.parse_xml(response)

        try:
            simulation = parsed.get('simulation', 'No simulation provided')
            insights = parsed.get('insights', 'No insights provided')
        except (ValueError, TypeError):
            simulation = "Error parsing LLM response"
            insights = "Error parsing LLM response"

        return {
            'simulation': simulation,
            'insights': insights,
            'pr_tok': pr_tok,
            'com_tok': com_tok
        }

    def code_analysis(self, code: str, test_log: str, problem: str, problem_understanding: str) -> dict:
        input_prompt = [
            {
                "role": "user",
                "content": f"""Assess the generated code written in {self.language} programming language for a competitive programming problem, using the problem description, its understanding, and test log. Identify where the code is failing based on the test log, and suggest specific improvements or fixes to correct those issues, considering edge cases and special cases from the problem understanding.

    # Problem:
    {problem}

    # Problem Understanding:
    {problem_understanding}

    # Code:
    ```{self.language}
    {code}
    ```

    # Test Log:
    {test_log}

    ----------------
    Important: Respond in the following XML format. Keep insights concise (<150 words), focusing on errors and suggestions.
    ```xml
    <root>
    <insights>Text identifying errors in the code and suggesting specific fixes</insights>
    </root>
    ```
    """
            }
        ]

        response, pr_tok, com_tok = self.gpt_chat(processed_input=input_prompt)
        response = self.replace_tag(response, 'insights')
        parsed = self.parse_xml(response)

        insights = parsed.get('insights', 'No insights provided')

        return {
            'insights': insights,
            'pr_tok': pr_tok,
            'com_tok': com_tok
        }

    def content_analysis(self, problem: str, problem_understanding: str, plan: str, code: str) -> dict:
        input_prompt = [
            {
                "role": "user",
                "content": f"""Evaluate how effectively a plan and generated code, written in {self.language} programming language, align with the requirements of a competitive programming problem, given the problem description and its understanding. Specifically, assess the alignment between the problem (and its understanding) and the plan, and between the plan and the code. Identify any mismatches or issues in these alignments, considering edge cases and special cases from the problem understanding. Provide separate confidence scores (0.0 to 1.0) for the problem-plan alignment and plan-code alignment (1.0: perfect match; 0.0: no alignment), and suggest specific improvements for each if the alignment is not strong.

    # Problem:
    {problem}

    # Problem Understanding:
    {problem_understanding}

    # Plan:
    {plan}

    # Code:
    ```{self.language}
    {code}
    ```

    ----------------
    Important: Respond in the following XML format. Keep insights concise (<150 words each), focusing on mismatches and suggestions.
    ```xml
    <root>
    <problem_plan_confidence>Score between 0.0 and 1.0 for problem-plan alignment</problem_plan_confidence>
    <plan_code_confidence>Score between 0.0 and 1.0 for plan-code alignment</plan_code_confidence>
    <problem_plan_insights>Text identifying problem-plan alignment issues and suggesting specific fixes</problem_plan_insights>
    <plan_code_insights>Text identifying plan-code alignment issues and suggesting specific fixes</plan_code_insights>
    </root>
    ```
    """
            }
        ]

        response, pr_tok, com_tok = self.gpt_chat(processed_input=input_prompt)
        response = self.replace_tag(response, 'problem_plan_confidence')
        response = self.replace_tag(response, 'plan_code_confidence')
        response = self.replace_tag(response, 'problem_plan_insights')
        response = self.replace_tag(response, 'plan_code_insights')
        parsed = self.parse_xml(response)

        try:
            problem_plan_confidence = float(parsed.get('problem_plan_confidence', 0.0))
            plan_code_confidence = float(parsed.get('plan_code_confidence', 0.0))
            overall_confidence = problem_plan_confidence * plan_code_confidence
            problem_plan_insights = parsed.get('problem_plan_insights', 'No insights provided')
            plan_code_insights = parsed.get('plan_code_insights', 'No insights provided')
        except (ValueError, TypeError):
            problem_plan_confidence = 0.0
            plan_code_confidence = 0.0
            overall_confidence = 0.0
            problem_plan_insights = "Error parsing LLM response"
            plan_code_insights = "Error parsing LLM response"
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
                "content": f"""You are given a {analysis_name} analysis. {meaning} Calculate the confidence score (0.0 to 1.0) for choosing to {decision}, where 1.0 means the analysis strongly supports the decision (e.g., insights indicate it's the best fix), and 0.0 means it does not support at all. Provide a brief explanation (50-100 words) of how the score was determined, referencing specific insights.

    Insights:
    {analysis.get('insights', '')}

    ==============
    Return only XML in this format:
    <root>
    <confidence>A float between 0.0 and 1.0</confidence>
    <reasoning>Brief explanation of how the confidence score was determined</reasoning>
    </root>
    """
            }
        ]

        response, pr_tok, com_tok = self.gpt_chat(processed_input=prompt)
        response = self.replace_tag(response, 'confidence')
        response = self.replace_tag(response, 'reasoning')
        parsed = self.parse_xml(response)

        try:
            score = float(parsed.get('confidence', 0.0))
            reasoning = parsed.get('reasoning', 'No reasoning provided')
        except (TypeError, ValueError):
            score = 0.0
            reasoning = "Error parsing LLM response"

        return max(0.0, min(score, 1.0))

    def get_consistency(
        self,
        decision: str,
        analysis1: dict, name1: str,
        analysis2: dict, name2: str
    ) -> float:
        ins1 = analysis1.get('insights', '').strip()
        ins2 = analysis2.get('insights', '').strip()

        name1_meaning = self.analysis_meaning.get(name1, "")
        name2_meaning = self.analysis_meaning.get(name2, "")

        prompt = [
            {
                "role": "user",
                "content": f"""You are given insights from two analyses: {name1} and {name2}. 
    {name1} meaning: {name1_meaning}
    {name2} meaning: {name2_meaning}
    Calculate the consistency score (0.0 to 1.0) for choosing to {decision}, where 1.0 means the insights from both analyses are highly consistent and support the decision (e.g., similar issues and fixes suggested), and 0.0 means they are inconsistent or contradictory. Provide a brief explanation (50-100 words) of how the score was determined, referencing specific insights from both analyses.

    {name1} insights:
    {ins1}

    {name2} insights:
    {ins2}

    Important:
    Your response must follow the following XML format exactly:
    <root>
    <consistency>float between 0.0 and 1.0</consistency>
    <reasoning>Brief explanation of how the consistency score was determined</reasoning>
    </root>
    """
            }
        ]

        response, pr_tok, com_tok = self.gpt_chat(processed_input=prompt)
        response = self.replace_tag(response, 'consistency')
        response = self.replace_tag(response, 'reasoning')
        parsed = self.parse_xml(response)

        try:
            score = float(parsed.get('consistency', 0.0))
            reasoning = parsed.get('reasoning', 'No reasoning provided')
        except (TypeError, ValueError):
            score = 0.0
            reasoning = "Error parsing LLM response"

        return max(0.0, min(score, 1.0))

    def collaborative_decision(self, plan: str, code: str, outcomes: str, item) -> str:
        problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
        A_plan = self.plan_analysis(plan, code, outcomes, self.data.get_prompt(item), problem_understanding)
        A_code = self.code_analysis(code, outcomes, self.data.get_prompt(item), problem_understanding)
        A_content = self.content_analysis(self.data.get_prompt(item), problem_understanding, plan, code)

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

        rt_response, pr_tok_rt, com_tok_rt = self.gpt_chat(
            processed_input=[{'role': 'user', 'content': rt_prompt}]
        )
        reasoning_trajectory = rt_response.strip()

        update_prompt = [{
            'role': 'user',
            'content': (
                f"Given a competitive programming problem, its understanding, and a plan to solve it, but this plan has some troubles that need to be updated with insights. "
                f"Please modify the plan accordingly.\n"
                f"# Problem: {problem}\n"
                f"# Problem Understanding: {problem_understanding}\n"
                f"Current planning: {plan}\n"
                f"Insights: {error_analysis['insights']}\n"
                f"Reasoning Trajectory: {reasoning_trajectory}\n"
                "Important: return only the revised plan text. "
                "Important: You should give only the updated planning to solve the problem. "
                "Do not add extra explanation or words."
            )
        }]

        updated, pr_tok_up, com_tok_up = self.gpt_chat(processed_input=update_prompt)
        revised_plan = updated.strip()

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

        rt_response, pr_tok_rt, com_tok_rt = self.gpt_chat(
            processed_input=[{'role': 'user', 'content': rt_prompt}]
        )
        reasoning_trajectory = rt_response.strip()

        code_prompt = [{
            'role': 'user',
            'content': (
                f"Given a competitive programming problem, its understanding, and generated {self.language} code to solve the problem, "
                f"but the generated code cannot pass sample test cases. Improve your code to solve the problem correctly.\n"
                f"# Problem: {problem}\n"
                f"# Problem Understanding: {problem_understanding}\n"
                f"## Planning: {plan}\n"
                f"## Code:\n```{self.language}\n{code}\n```\n"
                f"## Test Report:\n{error_analysis.get('test_results','')}\n"
                f"Insights: {error_analysis['insights']}\n"
                f"Reasoning Trajectory: {reasoning_trajectory}\n"
                "Important: Respond only with the revised code. Do not add extra explanation or words."
            )
        }]

        updated, pr_tok_up, com_tok_up = self.gpt_chat(processed_input=code_prompt)
        revised_code = self.parse_code(updated)

        self.rt.update_historical_data(iteration, {
            'previous_code': code,
            'previous_success_rate': error_analysis.get('success_rate'),
            'previous_iteration': iteration - 1
        })
        return revised_code, reasoning_trajectory

    def _inner_run(self, item):
        pr_tok = 0
        com_tok = 0

        plannings, pr_tok_p, com_tok_p = self.generate_plans(item)
        pr_tok += pr_tok_p
        com_tok += com_tok_p

        selected_plannings = plannings[:self.top_plan]

        best_code = ""
        flag = False
        test_log = ""

        for plan_idx, planning_with_ex in enumerate(selected_plannings, 1):
            plan, confidence, example = planning_with_ex
            approach_name = example.get('name', '') if 'name' in example else ''
            approach_tutorial = example.get('tutorial', '') if 'tutorial' in example else ''
            algorithm_prompt = f"## Relevant Approach: {approach_name}\n{approach_tutorial}"
            sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item)}\n"
            if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
                std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
            else:
                std_input_prompt = ""

            best_code, flag, test_log, score, pr_tok_c, com_tok_c = self.generate_codes_from_plan(item, plan, algorithm_prompt, sample_io_prompt)
            pr_tok += pr_tok_c
            com_tok += com_tok_c

            if flag:
                return best_code, pr_tok, com_tok

            for i in range(1, self.t + 1):
                problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
                pr_tok += pr_tok_u
                com_tok += com_tok_u

                decision = self.collaborative_decision(plan, best_code, test_log, item)

                if decision == 'update plan':
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
                else:
                    A_code = self.code_analysis(best_code, test_log, self.data.get_prompt(item), problem_understanding)
                    best_code, reasoning_trajectory = self.debug_code(i + 1, plan, best_code, {
                        'insights': A_code['insights'],
                        'test_results': test_log,
                        'success_rate': score,
                    }, self.data.get_prompt(item), problem_understanding)

                flag, test_log = self.data.evaluate_sample_io(item, best_code, self.language)

                if flag:
                    return best_code, pr_tok, com_tok

        return best_code, pr_tok, com_tok

    def run_single_pass(self, item: dict):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                result = self._inner_run(item)
                return result
            except ET.ParseError as e:
                print(f"[run_single_pass] Attempt {attempt} caught ET.ParseError: {e}. Retrying...")
                if attempt == max_retries:
                    print(f"[run_single_pass] ERROR: All {max_retries} attempts failed due to XML parsing. Returning fallback ('',0,0).")
                    return "", 0, 0
            except Exception as e:
                print(f"[run_single_pass] Attempt {attempt} caught unexpected exception: {e}. Retrying...")
                if attempt == max_retries:
                    print(f"[run_single_pass] ERROR: All {max_retries} attempts failed due to unexpected errors. Returning fallback ('',0,0).")
                    return "", 0, 0