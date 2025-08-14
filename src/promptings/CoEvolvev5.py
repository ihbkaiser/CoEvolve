from typing import List, Optional, Dict, Tuple
import tiktoken
import os
import json
import re
import sys
import time
import random
from datetime import datetime
from copy import deepcopy
from .Base import BaseStrategy  # Assuming this is in your project structure
from models.Base import BaseModel  # Adjust paths as needed
from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset
from results.Results import Results
from evaluations.func_evaluate import evaluate_io
import numpy as np
from forms import *  # Assuming this imports your form models like PlanOutput, etc.
from multi_thread import multi_thread_task_dict

class AnalysisReflection:
    def __init__(self):
        self.historical_data = {}  # Dictionary to store iteration data

    def update_historical_data(self, iteration: int, data: Dict):
        """Store data for the given iteration."""
        self.historical_data[iteration] = data

    def generate_prompt_for_plan_reflection(self, iteration: int, error_analysis: Dict, problem: str, problem_understanding: str, plan: str, historical_logs: Dict) -> str:
        """Generate a conversational prompt for plan debugging, evolving from R(t-1)."""
        previous_reflection = historical_logs.get('analysis_reflection', 'No previous analysis reflection available')
        insights = error_analysis.get('insights', '')
        success_rate = error_analysis.get('success_rate', 0.0)
        test_log = error_analysis.get("test_results", "")
        success_rate_str = f"{success_rate:.2f}%"
      
    
        prompt = f"""You are a debugging assistant for a competitive programming problem. 
Your task is to write a single reflection that will guide the next update to the solution plan at iteration {iteration}.
You are provided with: the problem statement, the understanding of the problem, the current plan, the test log from the most recent run, the insights from the plan analysis, the previous analysis reflection (R(t-1)), and the success rate from the previous iteration.
In your reflection, identify the root causes of the current failures, integrate the new insights with R(t-1), review the findings from the previous iteration, and propose clear, actionable improvements to the plan.
Focus only on the essential reasoning and avoid irrelevant details.

Problem: {problem}
Problem Understanding: {problem_understanding}
Current Plan: {plan}
Test Log: {test_log}
Insights from Plan Analysis: {insights}
Previous Analysis Reflection (R(t-1)): {previous_reflection}
Success Rate from previous iteration t-1: {success_rate_str}

IMPORTANT: Output only the reflection as plain text, with no titles, no lists, no quotes, and no additional explanation.
"""
        return prompt

class CoEvolvev5(BaseStrategy):
    def __init__(
        self,
        k: int = 1,
        t: int = 5,
        max_attempts: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.top_plan = 1
        self.t = t
        self.number_of_code_per_plan = 1
        self.trust_weights = {
            'plan': 0.3,
            'code': 0.4,
            'content': 0.3
        }
        self.analysis_meaning = {
            "plan": "Identifies errors or problems in the planning approach.",
            "code": "Identifies errors or problems in the code implementation.",
            "content": "Identifies mismatches between problem, plan, and code."
        }
        self.history = []
        self.max_attempts = max_attempts
        self.verbose = True
        self.rt = AnalysisReflection()  # Initialize AnalysisReflection for debugging guidance
    def _extract_json_string(self, text: str) -> Optional[str]:
        m = re.search(r'```json\s*({[\s\S]*?})\s*```', text, re.DOTALL)
        if not m:
            m = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text, re.DOTALL)
        if not m:
            m = re.search(r'({[\s\S]*})', text, re.DOTALL)
        return m.group(1) if m else None
    def _fix_invalid_escapes(self, json_str: str) -> str:
        json_str = json_str.replace('\b', '\\b').replace('\f', '\\f').replace('\r', '\\r').replace('\t', '\\t')
        json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
        return json_str
    def parse_structured_output(self, response: str, model: BaseModel) -> BaseModel:
        if self.verbose:
            print("Step: Parsing structured output")
            print(f"Input response: {response}...")
        response = response.strip()
        schema = model.model_json_schema()
        wrapping_prompt = [
            {
                "role": "user",
                "content": f"""You are an expert in JSON structuring. Your task is to take a free-form response and convert it into structured JSON matching the provided schema. 
                Extract relevant information from the response and ensure the output is valid JSON. Please try to keep as much of the original content of the response as possible, even the layout.
Free-form response:
{response}
Target JSON schema:
{json.dumps(schema, indent=2)}
Instructions:
1. Analyze the response to identify content matching the schema fields.
2. If the response contains markdown (e.g., ```json ... ```), extract the relevant content.
3. If fields are missing, provide sensible defaults or leave them empty (e.g., "" for strings).
4. Output ONLY valid JSON matching the schema, with no extra text or markdown.
Example output:
{{
  "json": {{ ... }}
}}
"""
            }
        ]
        try:
            if self.verbose:
                print("Step: Making API call to wrap free-form response into JSON")
            wrapped_response, pr_tok, com_tok = self.gpt_chat(wrapping_prompt)
            if self.verbose:
                print(f"Step: Wrapped response: {wrapped_response}...")
        except Exception as e:
            print(f"Error wrapping response: {e}\nRaw response: {response}...")
            return model()
        json_str = self._extract_json_string(wrapped_response)
        if not json_str:
            print(f"Invalid output: No JSON structure found in wrapped response\nWrapped response: {wrapped_response}...\nRaw response: {response}...")
            return model()
        json_str = json_str.strip()
        json_str = re.sub(r'```+\s*$', '', json_str)
        data = None
        for attempt in range(3):
            try:
                data = json.loads(json_str)
                break
            except json.JSONDecodeError as e:
                if attempt == 2:
                    print(f"Invalid output: JSON decode error after retries: {e}\nFinal JSON candidate: {json_str}...\nWrapped response: {wrapped_response}...\nRaw response: {response}...")
                    return model()
                if self.verbose:
                    print(f"Step: JSON decode failed (attempt {attempt + 1}), applying escape fix: {e}")
                json_str = self._fix_invalid_escapes(json_str)
            except Exception as e:
                print(f"Unexpected error during JSON loading: {e}")
                return model()
        if data is None:
            return model()
        if isinstance(data, dict) and 'json' in data:
            data = data['json']
        try:
            parsed = model(**data)
            if self.verbose:
                print("Step: Parsing successful")
                print(f"Parsed data: {parsed}")
            return parsed
        except Exception as e:
            print(f"Invalid output: Model instantiation error: {e}\nParsed data dict: {data}")
            return model()
    def parse_code(self, response: str) -> str:
        if self.verbose:
            print("Step: Parsing code")
            print(f"Input response: {response}...")
        if "```" not in response:
            return response
        code_pattern = r'```((.|\n)*?)```'
        languages = ['Python', 'python', 'Python3', 'python3', 'C', 'c', 'C++', 'c++', 'Java', 'java', 'Node', 'node', 'Rust', 'rust', 'PHP', 'php', 'Go', 'go', 'Ruby', 'ruby', 'C#', 'c#', 'csharp']
        for lang in languages:
            if f"```{lang}" in response:
                code_pattern = r'```' + lang + r'((.|\n)*?)```'
                break
        code_blocks = re.findall(code_pattern, response, re.DOTALL)
        if code_blocks:
            code_str = code_blocks[-1][0] if isinstance(code_blocks[-1], tuple) else code_blocks[-1]
        else:
            code_str = response
        parsed_code = code_str.strip()
        if self.verbose:
            print("Step: Code parsing successful")
            print(f"Parsed code: {parsed_code}...")
        return parsed_code
    def get_sample_io_str(self, item) -> str:
        if self.verbose:
            print("Step: Getting sample I/O string")
        if isinstance(self.data, XCodeDataset):
            sample_io = f"Input:\n{item['sample_inputs']}\nExpected output:\n{item['sample_outputs']}"
        else:
            sample_io_list = item.get('sample_io', [])
            if sample_io_list:
                if isinstance(sample_io_list[0], str):
                    sample_io = "\n".join(io for io in sample_io_list)
                elif isinstance(sample_io_list[0], dict):
                    sample_io = "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io_list])
            else:
                sample_io = ''
        if self.verbose:
            print("Step: Sample I/O retrieved")
            print(f"Sample I/O: {sample_io}...")
        return sample_io
    def compute_score(self, test_log: str) -> float:
        print("Test Log:", test_log)
        if self.verbose:
            print("Step: Computing score")
            print(f"Test log: {test_log}...")
        num_passed = test_log.count("passed in test case")
        num_failed = test_log.count("failed in test case")
        total = num_passed + num_failed
        score = num_passed / total if total > 0 else 0.0
        if self.verbose:
            print("Step: Score computed")
            print(f"Score: {score} (passed: {num_passed}, failed: {num_failed})")
        return score
    def get_problem_understanding(self, item) -> Tuple[str, int, int]:
        return "", 0, 0
#         if self.verbose:
#             print("Step: Generating problem understanding")
#         problem_text = self.data.get_prompt(item)
#         input_for_understanding = [
#             {
#                 "role": "user",
#                 "content": f"""You are an expert in competitive programming. Your task is to:
# - Analyze a problem description and provide a concise understanding of the problem, including its requirements, constraints, objectives, and potential edge cases or special cases.
# - Identify edge cases (e.g., boundary conditions, invalid inputs, or extreme scenarios) and special cases (e.g., unique input patterns or problem-specific conditions) that need attention, and provide examples of these cases beyond the provided sample I/O.
# - Also, for each sample test case provided, explain step-by-step how the input is processed according to the problem to produce the expected output.
# # Problem:
# {problem_text}
# # Sample I/O:
# {self.get_sample_io_str(item)}
# """
#             }
#         ]
#         try:
#             if self.verbose:
#                 print("Step: Making API call for understanding")
#             understanding, pr_tok, com_tok = self.gpt_chat(processed_input=input_for_understanding)
#             item['api_calls'] += 1
#             if self.verbose:
#                 print("Step: Understanding parsed")
#                 print(f"Understanding: {understanding}...")
#             return understanding, pr_tok, com_tok
#         except Exception as e:
#             print(f"Error in get_problem_understanding: {e}")
#             return "", 0, 0
    def generate_code_from_plan(self, item, planning: str, problem_text: str, sample_io_prompt: str, previous_codes: str = "", understanding: str = "") -> Tuple[str, float, str, int, int]:
        if self.verbose:
            print("Step: Generating code from plan")
            print(f"Plan: {planning}...")
        codes = []
        pr_tok = 0
        com_tok = 0
        std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases." if isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset)) else ""
        context = f"Problem: {problem_text}\n"
        context += understanding if understanding else ""
        for c_idx in range(1, self.number_of_code_per_plan + 1):
            if self.verbose:
                print(f"Step: Generating code variant {c_idx}")
            diversity_prompt = "" if c_idx == 1 else f"""
Generate a distinct implementation from previous ones: {previous_codes}. Use a unique approach, such as alternative data structures (e.g., list vs. dictionary, array vs. set in {self.language}), varied coding patterns (e.g., functional vs. imperative style).
Ensure the implementation strictly follows the provided plan and solves the problem correctly.
"""
            input_for_code_generation = [
                {
                    "role": "user",
                    "content": f"""# Task:
Generate a {self.language} code solution for the given competitive programming problem, strictly adhering to the provided plan and input/output specifications.
# Problem Context:
{context}
# Planning:
{planning}
# Sample Test Cases:
{sample_io_prompt}
{diversity_prompt}
# Constraints:
{std_input_prompt}
# Instructions:
1. Follow the plan step-by-step to create a correct and efficient solution.
2. Provide the code in a clear, readable format, optionally within ```{self.language} ... ``` markdown.
IMPORTANT: Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words.
"""
                }
            ]
            try:
                if self.verbose:
                    print("Step: Making API call for code generation")
                code_response, pr_tok_1, com_tok_1 = self.gpt_chat(processed_input=input_for_code_generation)
                pr_tok += pr_tok_1
                com_tok += com_tok_1
                item['api_calls'] += 1
                code = self.parse_code(code_response)
                if self.verbose:
                    print(f"Generated code variant {c_idx}: {code}")
                codes.append(code)
                previous_codes += f"\n- {code}"
            except Exception as e:
                print(f"Error generating code {c_idx}: {e}")
                codes.append("")
        if self.verbose:
            print(f"Step: {len(codes)} code variants generated. Starting evaluation")
        evaluated_codes = []
        for code_idx, code in enumerate(codes, 1):
            if self.verbose:
                print(f"Step: Evaluating code variant {code_idx}")
                print(f"Code: {code}...")
            if not code:
                evaluated_codes.append((code, 0.0, ""))
                continue
            try:
                passed, test_log = self.data.evaluate_sample_io(item, code, self.language)
                score = self.compute_score(test_log)
                evaluated_codes.append((code, score, test_log))
                if self.verbose:
                    print("Step: Evaluation completed")
                    print(f"Score: {score}, Passed: {passed}")
            except Exception as e:
                print(f"Error evaluating code: {e}")
                evaluated_codes.append((code, 0.0, f"Evaluation failed: {e}"))
        max_score = max([score for _, score, _ in evaluated_codes], default=0.0)
        top_codes = [(code, test_log) for code, score, test_log in evaluated_codes if score == max_score]
        best_code, best_test_log = random.choice(top_codes) if top_codes else ("", "")
        if self.verbose:
            print("Step: Best code selected")
            print(f"Best code: {best_code}..., Score: {max_score}")
        return best_code, max_score, best_test_log, pr_tok, com_tok
    def generate_plans(self, item) -> Tuple[List[PlanOutput], int, int]:
        if self.verbose:
            print("Step: Starting plan generation")
        plans = []
        pr_tok = 0
        com_tok = 0
        previous_approaches = ""
        problem_text = self.data.get_prompt(item)
        sample_io_prompt = self.get_sample_io_str(item)
        problem_understanding, pr_u, com_u = self.get_problem_understanding(item)
        pr_tok += pr_u
        com_tok += com_u
        max_plans = self.k
        for t in range(1, max_plans + 1):
            if self.verbose:
                print(f"Step: Generating plan variant {t}")
            diff_prompt = "" if t == 1 else f", different from the following previous approaches: {previous_approaches}"
            input_recall = [
                {"role": "user", "content": f"""Given a problem and its understanding, recall an approach that can solve it{diff_prompt}, provide a tutorial for the approach, then recall a relevant problem that uses this approach, and explain with plan and code.
Problem: {problem_text}
Problem understanding: {problem_understanding}
# Approach:
Recall one approach (e.g., Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that can solve the problem.
# Tutorial:
Write a useful tutorial about the approach. Provide a high level generic tutorial for solving this type of problem. Do not generate code.
# Exemplar:
Recall one relevant and distinct problem (different from the problem mentioned above) that uses this approach. For the problem,
1. describe it
2. generate {self.language} code step by step to solve that problem using the approach
3. generate a planning to solve that problem using the approach
Provide the response in a clear, readable format, optionally using markdown for sections.
IMPORTANT: Structure your response with markdown sections for each field: ## approach_name, ## tutorial, ## problem_description, ## code, ## planning. Provide content under each section.
"""
                }
            ]
            parsed_response = None
            for attempt in range(self.max_attempts):
                if self.verbose:
                    print(f"Step: Approach recall attempt {attempt + 1} for variant {t}")
                try:
                    response, pr_tok_temp, com_tok_temp = self.gpt_chat(input_recall)
                    pr_tok += pr_tok_temp
                    com_tok += com_tok_temp
                    item['api_calls'] += 1
                    parsed_response = self.parse_structured_output(response, ApproachRecall)
                    if self.verbose:
                        print("Step: Approach recall successful")
                        print(f"Approach name: {parsed_response.approach_name}")
                    break
                except Exception as e:
                    print(f"Error in recall attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        continue
            if parsed_response is None:
                continue
            approach_name = parsed_response.approach_name
            approach_tutorial = parsed_response.tutorial
            algorithm_prompt = f"## Relevant Approach: {approach_name}\n{approach_tutorial}"
            example_description = parsed_response.problem_description
            example_planning = parsed_response.planning
            previous_approaches += f"\n- {approach_name}"
            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": (
                        "You are an expert competitive-programming coach. "
                        "Your task is to produce a precise, step-by-step plan to solve the Target Problem "
                        "using the specified algorithmic approach. "
                        "You are provided with a worked example. It is directly analogous to the Target Problem section below"
                        "Provide a concrete, step-by-step plan, can support well for writing code to solve the Target Problem"
                        f"# Example Problem Description:\n{example_description}\n\n"
                        f"# Approach Template:\n{algorithm_prompt}\n\n"
                        f"# Example Plan (for reference):\n{example_planning}\n\n"
                        f"# Target Problem:\n{problem_text}\n\n"
                        f"## Sample Test Cases:\n{sample_io_prompt}"
                        "IMPORTANT: You should give only the planning to solve the problem. Do not add extra explanation or words."
                    ),
                },
            ]
            for attempt in range(self.max_attempts):
                if self.verbose:
                    print(f"Step: Planning generation attempt {attempt + 1} for variant {t}")
                try:
                    planning, pr_tok_temp, com_tok_temp = self.gpt_chat(input_for_problem_planning)
                    pr_tok += pr_tok_temp
                    com_tok += com_tok_temp
                    item['api_calls'] += 1
                    break
                except Exception as e:
                    print(f"Error in planning attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        continue
            input_for_planning_verification = [
                {"role": "user", "content": f"""You are an expert evaluator for competitive programming plans.
                 Assess the plan for alignment with the problem and overall solvability (0-100). Provide explanations and scores.
Problem Description: {problem_text}
Proposed Plan: {planning}
Sample I/O: {sample_io_prompt}
Provide the response in a clear, readable format, optionally using markdown.
IMPORTANT: Structure your response with markdown sections for each field: ## alignment_explanation, ## alignment_score, ## coherence_explanation, ## coherence_score, ## overall_solvability. Provide content under each section.
"""
                }
            ]
            verification_parsed = None
            for attempt in range(self.max_attempts):
                if self.verbose:
                    print(f"Step: Planning verification attempt {attempt + 1} for variant {t}")
                try:
                    verification_res, pr_tok_temp, com_tok_temp = self.gpt_chat(input_for_planning_verification)
                    pr_tok += pr_tok_temp
                    com_tok += com_tok_temp
                    item['api_calls'] += 1
                    verification_parsed = self.parse_structured_output(verification_res, VerificationOutput)
                    if self.verbose:
                        print("Step: Verification successful")
                        print(f"Overall solvability: {verification_parsed.overall_solvability}")
                    break
                except Exception as e:
                    print(f"Error in verification attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        continue
            if verification_parsed is None:
                continue
            llm_score = verification_parsed.overall_solvability
            plans.append((planning, llm_score))
            if self.verbose:
                print(f"Step: Plan variant {t} completed")
                print(f"LLM score: {llm_score}")
        if len(plans) < self.k:
            print(f"Warning: Only {len(plans)}/{self.k} valid plans generated, attempting one more generation")
            additional_plans, pr_tok_add, com_tok_add = self.generate_plans(item)
            pr_tok += pr_tok_add
            com_tok += com_tok_add
            plans.extend(additional_plans)
        # Sort plans by llm_score and select the top one
        plans.sort(key=lambda x: x[1], reverse=True)
        if not plans:
            print("Warning: No valid plans generated. Returning empty result.")
            return [], pr_tok, com_tok
        best_plan, best_llm_score = plans[0]
        # Generate codes for the best plan only
        if self.verbose:
            print(f"Step: Generating codes for best plan with LLM score: {best_llm_score}")
        best_code, code_score, test_log, pr_tok_code, com_tok_code = self.generate_code_from_plan(
            item, best_plan, problem_text, sample_io_prompt, "", problem_understanding
        )
        pr_tok += pr_tok_code
        com_tok += com_tok_code
        # Return a single PlanOutput for the best plan
        result = [PlanOutput(
            planning=best_plan,
            llm_score=best_llm_score,
            code=best_code,
            code_score=code_score,
            test_log=test_log
        )]
        if self.verbose:
            print("Step: Best plan and code selected")
            print(f"Best plan LLM score: {best_llm_score}, Best code score: {code_score}")
        return result, pr_tok, com_tok
    def plan_analysis(self, plan: str, test_log: str, problem: str, problem_understanding: str) -> PlanAnalysisOutput:
        if self.verbose:
            print("Step: Performing plan analysis")
            print(f"Plan: {plan}...")
        schema = PlanAnalysisOutput.model_json_schema()
        input_prompt = [
            {
                "role": "user",
                "content": (
                    "You are an expert competitive-programming plan analyst with extensive experience in debugging and logical reasoning. "
                    "Given a proposed solution plan, the problem description, your understanding of the problem, and a test log showing failures, "
                    "Your primary task is to perform a detailed step-by-step simulation of the proposed solution plan using the specific input data from the failing test case(s) in the test log. "
                    "For each step of the plan, trace the logic using the test case input, identify where the simulation produces results that diverge from the expected output in the test log, and localize the specific logical flaws, incorrect assumptions, or mismatches in the plan. "
                    "Your mission is to pinpoint errors in the plan and provide actionable insights for refinement to align with the problem requirements."
                    "Structure your response with two sections: "
                    "1. Simulation: Provide a detailed step-by-step simulation of the plan on the failing test cases, highlighting divergences."
                    "2. Insight: Summarize the identified flaws and provide actionable insights for plan refinement."
                    f"Problem Description:\n{problem}\n\n"
                    f"Problem Understanding:\n{problem_understanding}\n\n"
                    f"Proposed Plan:\n{plan}\n\n"
                    f"Test Log (failing input/output):\n{test_log}"
                ),
            },
        ]
        pr_tok = 0
        com_tok = 0
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Plan analysis attempt {attempt + 1}")
            try:
                response, pr_tok_temp, com_tok_temp = self.gpt_chat(input_prompt)
                pr_tok += pr_tok_temp
                com_tok += com_tok_temp
                parsed = self.parse_structured_output(response, PlanAnalysisOutput)
                parsed.pr_tok = pr_tok
                parsed.com_tok = com_tok
                if self.verbose:
                    print("Step: Plan analysis successful")
                    print(f"Insights: {parsed.insights}...")
                return parsed
            except Exception as e:
                print(f"Error in plan_analysis attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    return PlanAnalysisOutput(pr_tok=pr_tok, com_tok=com_tok)
    def code_analysis(self, code: str, test_log: str, problem: str, problem_understanding: str) -> CodeAnalysisOutput:
        if self.verbose:
            print("Step: Performing code analysis")
            print(f"Code: {code}...")
        schema = CodeAnalysisOutput.model_json_schema()
        input_prompt = [
            {
                "role": "user",
                "content": (
                    "You are an expert competitive-programming code analyst. "
                    "Your task is to simulate the provided implementation on a failing test case, "
                    "executing it line-by-line to pinpoint and localize any specific errors, mismatches, "
                    "or logical flaws in the code. "
                    "Provide a detailed analysis in a clear, readable format, optionally using markdown."
                    "Structure your response with two sections: "
                    "1. Simulation: Provide a detailed line-by-line simulation of the code on the failing test cases, highlighting errors."
                    "2. Insight: Summarize the identified flaws and provide actionable insights for code refinement."
                    f"Problem Description:\n{problem}\n\n"
                    f"Problem Understanding:\n{problem_understanding}\n\n"
                    f"Code Implementation ({self.language}):\n"
                    f"```{self.language}\n{code}\n```\n\n"
                    f"Test Log (failing input/output):\n{test_log}"
                ),
            },
        ]
        pr_tok = 0
        com_tok = 0
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Code analysis attempt {attempt + 1}")
            try:
                response, pr_tok_temp, com_tok_temp = self.gpt_chat(input_prompt)
                pr_tok += pr_tok_temp
                com_tok += com_tok_temp
                parsed = self.parse_structured_output(response, CodeAnalysisOutput)
                parsed.pr_tok = pr_tok
                parsed.com_tok = com_tok
                if self.verbose:
                    print("Step: Code analysis successful")
                    print(f"Insights: {parsed.insights}...")
                return parsed
            except Exception as e:
                print(f"Error in code_analysis attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    return CodeAnalysisOutput(pr_tok=pr_tok, com_tok=com_tok)
    def content_analysis(self, problem: str, problem_understanding: str, plan: str, code: str) -> ContentAnalysisOutput:
        if self.verbose:
            print("Step: Performing content analysis")
        schema = ContentAnalysisOutput.model_json_schema()
        input_prompt = [
            {
                "role": "user",
                "content": (
                    "You are a senior competitive-programming evaluator. "
                    "Focus ONLY on how well the proposed solution plan and its implementation "
                    "align with the problem requirements and with each other. "
                    "Identify any mismatches, omissions, or unnecessary steps, "
                    "and provide a single concise insight summarizing the alignment issues and suggestions to improve alignment between problem, plan, and code. "
                    "Provide the analysis in a clear, readable format, optionally using markdown."
                    f"Problem Description:\n{problem}\n\n"
                    f"Problem Understanding:\n{problem_understanding}\n\n"
                    f"Proposed Plan:\n{plan}\n\n"
                    f"Code Implementation ({self.language}):\n"
                    f"```{self.language}\n{code}\n```\n\n"
                    "IMPORTANT: Structure your response with markdown sections for each field: ## plan_code_insights. Provide content under each section."
                ),
            },
        ]
        pr_tok = 0
        com_tok = 0
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Content analysis attempt {attempt + 1}")
            try:
                response, pr_tok_temp, com_tok_temp = self.gpt_chat(input_prompt)
                pr_tok += pr_tok_temp
                com_tok += com_tok_temp
                parsed = self.parse_structured_output(response, ContentAnalysisOutput)
                parsed.pr_tok = pr_tok
                parsed.com_tok = com_tok
                if self.verbose:
                    print("Step: Content analysis successful")
                    print(f"Insights: {parsed.plan_code_insights}...")
                return parsed
            except Exception as e:
                print(f"Error in content_analysis attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    return ContentAnalysisOutput(pr_tok=pr_tok, com_tok=com_tok)
    def get_confidence(self, decision: str, analysis: dict, analysis_name: str) -> ConfidenceOutput:
        if self.verbose:
            print(f"Step: Getting confidence for '{decision}' from {analysis_name}")
        agent_description = {
            "plan": "Plan Analyst tasked with identifying errors or logical flaws in the planning approach, such as incorrect algorithmic steps or missing edge cases.",
            "code": "Code Analyst tasked with identifying errors or problems in the code implementation, such as syntax errors, logical mistakes, or incorrect handling of inputs.",
            "content": "Content Evaluator tasked with identifying mismatches or misalignments between the problem requirements, the proposed plan, and the code implementation."
        }.get(analysis_name, "Unknown agent")
        insight = analysis.get('insights', '') or analysis.get('plan_code_insights', '')
        meaning = self.analysis_meaning.get(analysis_name, "")
        schema = ConfidenceOutput.model_json_schema()
        prompt = [
            {
                "role": "user",
                "content": f"""You are a senior competitive programmer and technical reviewer. Your task is to assign a confidence score (0.0 to 1.0) for the decision to '{decision}', based on the {analysis_name.replace('_', ' ').title()} provided by a specialized agent.
Agent Description: {agent_description}
{analysis_name.replace('_', ' ').title()}: {meaning}
Insight from {analysis_name.replace('_', ' ').title()}: {insight}
Instructions:
1. Analyze the insight to determine its relevance to the decision ('{decision}').
2. Assign a confidence score (0.0 to 1.0) based on how strongly the insight supports the decision.
3. Provide a brief reasoning for the confidence score.
4. Output a JSON object matching this schema:
{json.dumps(schema, indent=2)}
Provide the response as valid JSON, with no extra text or markdown.
"""
            }
        ]
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Confidence attempt {attempt + 1}")
            try:
                response, _, _ = self.gpt_chat(prompt)
                parsed = self.parse_structured_output(response, ConfidenceOutput)
                if self.verbose:
                    print("Step: Confidence calculated")
                    print(f"Confidence: {parsed.confidence}, Reasoning: {parsed.reasoning}...")
                return parsed
            except Exception as e:
                print(f"Error in get_confidence attempt {attempt + 1}: {e}")
        return ConfidenceOutput()

    def get_consistency(self, decision: str, analysis1: dict, name1: str, analysis2: dict, name2: str) -> ConsistencyOutput:
        if self.verbose:
            print(f"Step: Getting consistency for '{decision}' between {name1} and {name2}")
        agent_description1 = {
            "plan": "Plan Analyst tasked with identifying errors or logical flaws in the planning approach, such as incorrect algorithmic steps or missing edge cases.",
            "code": "Code Analyst tasked with identifying errors or problems in the code implementation, such as syntax errors, logical mistakes, or incorrect handling of inputs.",
            "content": "Content Evaluator tasked with identifying mismatches or misalignments between the problem requirements, the proposed plan, and the code implementation."
        }.get(name1, "Unknown agent")
        agent_description2 = {
            "plan": "Plan Analyst tasked with identifying errors or logical flaws in the planning approach, such as incorrect algorithmic steps or missing edge cases.",
            "code": "Code Analyst tasked with identifying errors or problems in the code implementation, such as syntax errors, logical mistakes, or incorrect handling of inputs.",
            "content": "Content Evaluator tasked with identifying mismatches or misalignments between the problem requirements, the proposed plan, and the code implementation."
        }.get(name2, "Unknown agent")
        ins1 = analysis1.get('insights', '') or analysis1.get('plan_code_insights', '')
        ins2 = analysis2.get('insights', '') or analysis2.get('plan_code_insights', '')
        name1_meaning = self.analysis_meaning.get(name1, "")
        name2_meaning = self.analysis_meaning.get(name2, "")
        schema = ConsistencyOutput.model_json_schema()
        prompt = [
            {
                "role": "user",
                "content": f"""You are a senior competitive programmer and technical reviewer. Your task is to compute a consistency score (0.0 to 1.0) for the decision to '{decision}', using insights from {name1.replace('_', ' ').title()} and {name2.replace('_', ' ').title()}.
{name1.replace('_', ' ').title()} Agent Description: {agent_description1}
{name1.replace('_', ' ').title()} Purpose: {name1_meaning}
Insight from {name1.replace('_', ' ').title()}: {ins1}
{name2.replace('_', ' ').title()} Agent Description: {agent_description2}
{name2.replace('_', ' ').title()} Purpose: {name2_meaning}
Insight from {name2.replace('_', ' ').title()}: {ins2}
Instructions:
1. Analyze both insights to determine how well they align in supporting or undermining the decision ('{decision}').
2. Compute a consistency score (0.0 to 1.0) based on the degree of agreement between the insights regarding the decision.
3. Provide a brief reasoning for the consistency score.
4. Output a JSON object matching this schema:
{json.dumps(schema, indent=2)}
Provide the response as valid JSON, with no extra text or markdown.
"""
            }
        ]
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Consistency attempt {attempt + 1}")
            try:
                response, _, _ = self.gpt_chat(prompt)
                parsed = self.parse_structured_output(response, ConsistencyOutput)
                if self.verbose:
                    print("Step: Consistency calculated")
                    print(f"Consistency: {parsed.consistency}, Reasoning: {parsed.reasoning}...")
                return parsed
            except Exception as e:
                print(f"Error in get_consistency attempt {attempt + 1}: {e}")
        return ConsistencyOutput()
    def get_all_confidence(self, decisions: List[str], analyses: Dict[str, Dict]) -> Dict[str, Dict[str, ConfidenceOutput]]:
        """
        Trả về: { decision: { analysis_type: ConfidenceOutput } }
        """
        if self.verbose:
            print("Step: Computing all confidence scores in a single API call")

        # Cố định thứ tự & chỉ giữ 3 loại hợp lệ
        ANALYSES_ORDER = ["plan", "code", "content"]
        analysis_names = [n for n in ANALYSES_ORDER if n in analyses]
        if not analysis_names:
            return {d: {n: ConfidenceOutput() for n in ANALYSES_ORDER} for d in decisions}

        agent_descriptions = {
            "plan": "Plan Analyst: finds logical flaws, missing steps, or edge cases in the plan.",
            "code": "Code Analyst: finds implementation bugs, logic mistakes, or I/O handling issues.",
            "content": "Content Evaluator: checks misalignment among problem, plan, and code."
        }
        analysis_meanings = {
            "plan": "Evaluates planning approach quality.",
            "code": "Evaluates code implementation quality.",
            "content": "Evaluates alignment between problem, plan, and code."
        }

        packed_analyses = [
            {
                "name": name,
                "role": agent_descriptions.get(name, ""),
                "purpose": analysis_meanings.get(name, ""),
                "insights": analyses.get(name, {}).get("insights", "")
            }
            for name in analysis_names
        ]

        user_content = (
            "You are a senior competitive programming reviewer. "
            "Evaluate how strongly each analysis type supports or refutes each decision.\n\n"
            f"Decisions:\n{json.dumps(decisions)}\n\n"
            f"Analysis Types and Insights:\n{json.dumps(packed_analyses, ensure_ascii=False, indent=2)}\n\n"
            "Scoring rules (confidence ∈ [0.0, 1.0]):\n"
            "- 1.0 = strongly supports with clear, direct evidence.\n"
            "- 0.7–0.9 = supports with mostly relevant reasoning, minor gaps.\n"
            "- 0.4–0.6 = weak/partial support; relevant but missing key links.\n"
            "- 0.1–0.3 = minimal/unclear relevance.\n"
            "- 0.0 = no relevance or contradicts the decision.\n\n"
            "Instructions:\n"
            "1) For each <decision> × <analysis_type>, read the insights and judge relevance & polarity.\n"
            "2) Assign confidence per rules above.\n"
            "3) Provide a brief reasoning (1–3 sentences) citing concrete insight points.\n"
            "4) If insights contradict the decision, confidence must be low (<0.3).\n\n"
            "Output JSON ONLY (no extra text, no markdown):\n"
            "{\n"
            '  "confidence_scores": {\n'
            '    "<decision>": {\n'
            '      "<analysis_type>": {\n'
            '        "confidence": float,\n'
            '        "reasoning": str\n'
            "      }\n"
            "    }\n"
            "  }\n"
            "}"
        )

        messages = [{"role": "user", "content": user_content}]

        result: Dict[str, Dict[str, ConfidenceOutput]] = {}
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Confidence API call attempt {attempt + 1}")
            try:
                # Nếu có, nên bật response_format='json' để khóa đầu ra JSON
                # response, _, _ = self.gpt_chat(messages, response_format='json')
                response, _, _ = self.gpt_chat(messages)
                json_str = self._extract_json_string(response)
                if not json_str:
                    if self.verbose:
                        print(f"Invalid output: No JSON found\nResponse head: {response}...")
                    continue

                json_str = self._fix_invalid_escapes(json_str)
                data = json.loads(json_str)
                confidence_scores = data.get("confidence_scores", {})

                for d in decisions:
                    result[d] = {}
                    for name in analysis_names:
                        item = confidence_scores.get(d, {}).get(name, {})
                        result[d][name] = ConfidenceOutput(
                            confidence=float(item.get("confidence", 0.0) or 0.0),
                            reasoning=str(item.get("reasoning", "") or "")
                        )

                if self.verbose:
                    print("Step: All confidence scores calculated")
                    for d in decisions:
                        for name in analysis_names:
                            print(f"[CONF] {d}/{name}: {result[d][name].confidence:.3f}")
                return result

            except Exception as e:
                print(f"Error in get_all_confidence attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Step: Max attempts reached, returning default confidence outputs")
                    return {
                        d: {name: ConfidenceOutput() for name in analysis_names}
                        for d in decisions
                    }


    def get_all_consistency(self, decisions: List[str], analyses: Dict[str, Dict]) -> Dict[str, Dict[str, ConsistencyOutput]]:
        """
        Trả về: { decision: { "name1-name2": ConsistencyOutput } }
        """
        if self.verbose:
            print("Step: Computing all consistency scores in a single API call")

        ANALYSES_ORDER = ["plan", "code", "content"]
        analysis_names = [n for n in ANALYSES_ORDER if n in analyses]
        if len(analysis_names) < 2:
            return {d: {} for d in decisions}

        # Sinh cặp ổn định
        pairs = []
        for i, n1 in enumerate(analysis_names):
            for n2 in analysis_names[i+1:]:
                pairs.append((n1, n2))

        agent_descriptions = {
            "plan": "Plan Analyst: finds logical flaws, missing steps, or edge cases in the plan.",
            "code": "Code Analyst: finds implementation bugs, logic mistakes, or I/O handling issues.",
            "content": "Content Evaluator: checks misalignment among problem, plan, and code."
        }
        analysis_meanings = {
            "plan": "Evaluates planning approach quality.",
            "code": "Evaluates code implementation quality.",
            "content": "Evaluates alignment between problem, plan, and code."
        }

        packed_analyses = [
            {
                "name": name,
                "role": agent_descriptions.get(name, ""),
                "purpose": analysis_meanings.get(name, ""),
                "insights": analyses.get(name, {}).get("insights", "")
            }
            for name in analysis_names
        ]

        user_content = (
            "You are a senior competitive programming reviewer. "
            "Evaluate how much two different analyses agree or disagree in supporting each decision.\n\n"
            f"Decisions:\n{json.dumps(decisions)}\n\n"
            f"Analysis Types and Insights:\n{json.dumps(packed_analyses, ensure_ascii=False, indent=2)}\n\n"
            f"Analysis Pairs:\n{json.dumps(pairs)}\n\n"
            "Scoring rules (consistency ∈ [0.0, 1.0]):\n"
            "- 1.0 = both clearly support or both clearly refute the decision with aligned reasoning.\n"
            "- 0.7–0.9 = generally agree with minor differences in focus.\n"
            "- 0.4–0.6 = mixed/partial agreement; some overlap but notable differences.\n"
            "- 0.1–0.3 = mostly disagree with conflicting reasoning.\n"
            "- 0.0 = fully contradictory or unrelated conclusions.\n\n"
            "Instructions:\n"
            "1) For each <decision> × <analysis1,analysis2>, compare their insights and judge agreement.\n"
            "2) Assign consistency per rules above.\n"
            "3) Provide a brief reasoning (1–3 sentences) that points out alignment/conflict specifics.\n"
            "4) If one analysis is irrelevant while the other is strongly relevant, consistency should be low (<0.4).\n\n"
            "Output JSON ONLY (no extra text, no markdown):\n"
            "{\n"
            '  "consistency_scores": {\n'
            '    "<decision>": {\n'
            '      "<analysis1>-<analysis2>": {\n'
            '        "consistency": float,\n'
            '        "reasoning": str\n'
            "      }\n"
            "    }\n"
            "  }\n"
            "}"
        )

        messages = [{"role": "user", "content": user_content}]

        result: Dict[str, Dict[str, ConsistencyOutput]] = {}
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Consistency API call attempt {attempt + 1}")
            try:
                # response, _, _ = self.gpt_chat(messages, response_format='json')
                response, _, _ = self.gpt_chat(messages)
                json_str = self._extract_json_string(response)
                if not json_str:
                    if self.verbose:
                        print(f"Invalid output: No JSON found\nResponse head: {response}...")
                    continue

                json_str = self._fix_invalid_escapes(json_str)
                data = json.loads(json_str)
                consistency_scores = data.get("consistency_scores", {})

                for d in decisions:
                    result[d] = {}
                    for n1, n2 in pairs:
                        key = f"{n1}-{n2}"
                        item = consistency_scores.get(d, {}).get(key, {})
                        result[d][key] = ConsistencyOutput(
                            consistency=float(item.get("consistency", 0.0) or 0.0),
                            reasoning=str(item.get("reasoning", "") or "")
                        )

                if self.verbose:
                    print("Step: All consistency scores calculated")
                    for d in decisions:
                        for key, obj in result[d].items():
                            print(f"[CONS] {d}/{key}: {obj.consistency:.3f}")
                return result

            except Exception as e:
                print(f"Error in get_all_consistency attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Step: Max attempts reached, returning default consistency outputs")
                    return {
                        d: {f"{n1}-{n2}": ConsistencyOutput() for n1, n2 in pairs}
                        for d in decisions
                    }

    def perform_analyses(self, plan: str, code: str, test_log: str, problem: str, problem_understanding: str) -> Dict:
        """
        Performs plan, code, and content analysis in parallel using multi-threading.
        Returns a dictionary containing all three analysis results.
        """
        if self.verbose:
            print("Step: Performing parallel analyses (plan + code + content)")
        task_dictionary = {
            'plan': lambda: self.plan_analysis(plan, test_log, problem, problem_understanding),
            'code': lambda: self.code_analysis(code, test_log, problem, problem_understanding),
            'content': lambda: self.content_analysis(problem, problem_understanding, plan, code)
        }
        results = multi_thread_task_dict(task_dictionary, num_workers=3, show_progress=self.verbose)
        pr_tok = sum(r.pr_tok for r in results.values())
        com_tok = sum(r.com_tok for r in results.values())
        analysis_result = {
            'plan_analysis': {'insights': results['plan'].insights},
            'code_analysis': {'insights': results['code'].insights},
            'content_analysis': {'insights': results['content'].plan_code_insights},
            'pr_tok': pr_tok,
            'com_tok': com_tok
        }
        if self.verbose:
            print("Step: Parallel analyses completed")
            print(f"Plan insights: {analysis_result['plan_analysis']['insights']}...")
            print(f"Code insights: {analysis_result['code_analysis']['insights']}...")
            print(f"Content insights: {analysis_result['content_analysis']['insights']}...")
        return analysis_result

    def collaborative_decision(self, plan: str, code: str, outcomes: str, item) -> str:
        if self.verbose:
            print("Step: Starting collaborative decision with merged analysis")
        merged_result = {
            'plan_analysis': {'insights': ''},
            'code_analysis': {'insights': ''},
            'content_analysis': {'insights': ''},
            'pr_tok': 0,
            'com_tok': 0
        }
        try:
            problem_understanding, _, _ = self.get_problem_understanding(item)
            problem_text = self.data.get_prompt(item)
            
            # Perform analyses in parallel
            merged_result = self.perform_analyses(plan, code, outcomes, problem_text, problem_understanding)
            
            # Extract analysis results
            A_plan = merged_result['plan_analysis']
            A_code = merged_result['code_analysis']
            A_content = merged_result['content_analysis']
            
            
            decisions = ['update plan', 'update code only']
            scores = {}
            for d in decisions:
                if self.verbose:
                    print(f"Step: Scoring decision '{d}'")
                total = 0.0
                for name, A_i in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
                    w = self.trust_weights[name]
                    conf_output = self.get_confidence(d, A_i, name)
                    conf = conf_output.confidence
                    cons_prod = 1.0
                    for oname, A_j in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
                        if oname != name:
                            cons_output = self.get_consistency(d, A_i, name, A_j, oname)
                            cons_prod *= cons_output.consistency
                    total += w * conf * cons_prod
                scores[d] = total
                if self.verbose:
                    print(f"Step: Score for '{d}': {total}")
            decision = max(scores, key=scores.get)
            if self.verbose:
                print("Step: Decision made")
                print(f"Decision: {decision}")
            return decision, merged_result
        except Exception as e:
            print(f"Error in collaborative_decision: {e}")
            return "update code only", merged_result

    def fast_collaborative_decision(self, plan: str, code: str, outcomes: str, item) -> str:
        """
        Updated collaborative_decision to use optimized confidence and consistency functions.
        """
        if self.verbose:
            print("Step: Starting collaborative decision with merged analysis")
        merged_result = {
            'plan_analysis': {'insights': ''},
            'code_analysis': {'insights': ''},
            'content_analysis': {'insights': ''},
            'pr_tok': 0,
            'com_tok': 0
        }
        try:
            problem_understanding, _, _ = self.get_problem_understanding(item)
            problem_text = self.data.get_prompt(item)
          
            # Perform analyses in parallel
            merged_result = self.perform_analyses(plan, code, outcomes, problem_text, problem_understanding)
          
            # Extract analysis results
            analyses = {
                'plan': merged_result['plan_analysis'],
                'code': merged_result['code_analysis'],
                'content': merged_result['content_analysis']
            }
          
            decisions = ['update plan', 'update code only']
          
            # Compute all confidence scores in one API call
            confidence_scores = self.get_all_confidence(decisions, analyses)
          
            # Compute all consistency scores in one API call
            consistency_scores = self.get_all_consistency(decisions, analyses)
          
            scores = {}
            for decision in decisions:
                if self.verbose:
                    print(f"Step: Scoring decision '{decision}'")
                total = 0.0
                for name in analyses.keys():
                    w = self.trust_weights[name]
                    conf = confidence_scores[decision][name].confidence
                    cons_prod = 1.0
                    for name2 in analyses.keys():
                        if name2 != name:
                            pair_key = f"{name}-{name2}" if name < name2 else f"{name2}-{name}"
                            cons = consistency_scores[decision].get(pair_key, ConsistencyOutput()).consistency
                            cons_prod *= cons
                    total += w * conf
                scores[decision] = total
                if self.verbose:
                    print(f"Step: Score for '{decision}': {total}")
          
            decision = max(scores, key=scores.get)
            if self.verbose:
                print("Step: Decision made")
                print(f"Decision: {decision}")
            return decision, merged_result
      
        except Exception as e:
            print(f"Error in collaborative_decision: {e}")
            return "update code only", merged_result

    def get_multiple_analyses(self, plans: List[str], codes: List[str], test_logs: List[str], problem: str, problem_understanding: str) -> List[Dict]:
        """
        Performs plan, code, and content analysis for multiple plan/code/test_log triples in parallel.
        Returns a list of dictionaries, each containing the three analysis results for one triple.
        """
        if self.verbose:
            print("Step: Performing multiple analyses")
        if len(plans) != len(codes) or len(plans) != len(test_logs):
            raise ValueError("Plans, codes, and test_logs must have the same length")
        
        task_dictionary = {}
        for idx in range(len(plans)):
            task_dictionary[idx] = lambda idx=idx: self.perform_analyses(
                plans[idx], codes[idx], test_logs[idx], problem, problem_understanding
            )
        
        results = multi_thread_task_dict(task_dictionary, num_workers=min(3, len(plans)), show_progress=self.verbose)
        analyses_list = [results[i] for i in sorted(results.keys())]
        
        if self.verbose:
            print("Step: Multiple analyses completed")
            for idx, analysis in enumerate(analyses_list):
                print(f"Analysis {idx}: Plan insights: {analysis['plan_analysis']['insights']}...")
        
        return analyses_list
    def debug_plan(self, iteration: int, plan: str, error_analysis: Dict, problem: str, problem_understanding: str, decision: str):
        if self.verbose:
            print(f"Step: Debugging plan at iteration {iteration}")
        prev_logs = self.rt.historical_data.get(iteration - 1, {})
        rt_prompt = self.rt.generate_prompt_for_plan_reflection(
            iteration, error_analysis, problem, problem_understanding, plan, historical_logs=prev_logs
        )
        try:
            if self.verbose:
                print("Step: Generating analysis reflection for plan")
                print(f"Prompt for analysis reflection: {rt_prompt}")
            analysis_reflection, _, _ = self.gpt_chat([{
                'role': 'user',
                'content': rt_prompt
            }])
            if self.verbose:
                print("Step: Analysis reflection generated")
                print(f"Reflection: {analysis_reflection}...")
        except Exception as e:
            print(f"Error generating analysis reflection for plan: {e}")
            analysis_reflection = "Error generating analysis reflection"
        update_prompt = [
            {
                'role': 'user',
                'content': f"""Given a competitive programming problem, its understanding, and a plan to solve it, but this plan has errors or problems that need to be addressed.
Problem: {problem}
Problem Understanding: {problem_understanding}
Current Planning: {plan}
Test Log: {error_analysis.get('test_results', '')}
Analysis Reflection: {analysis_reflection}
Using the provided analysis reflection and test log, refine the plan to correct the identified issues.
Provide the refined plan in a clear, readable format, optionally using markdown.
"""
            }
        ]
        try:
            if self.verbose:
                print("Step: Making API call for plan update")
            updated_response, _, _ = self.gpt_chat(update_prompt)
            updated_parsed = self.parse_structured_output(updated_response, Planning)
            revised_plan = updated_parsed.planning.strip()
            if self.verbose:
                print("Step: Plan updated")
                print(f"Revised plan: {revised_plan}...")
        except Exception as e:
            print(f"Error debugging plan: {e}")
            revised_plan = plan
        self.rt.update_historical_data(iteration, {
            'previous_plan': plan,
            'previous_success_rate': error_analysis.get('success_rate'),
            'previous_iteration': iteration - 1,
            'analysis_reflection': analysis_reflection
        })
        if self.verbose:
            print("Step: Historical data updated for plan")
        return revised_plan, analysis_reflection

    def debug_code(self, iteration: int, plan: str, code: str, error_analysis: Dict, problem: str, problem_understanding: str, decision: str):
        """
        Debug the code using only code analysis insights, without reflection analysis.
        """
        if self.verbose:
            print(f"Step: Debugging code at iteration {iteration} using code analysis only")
      
        # Extract insights and test results from error_analysis
        insights = error_analysis.get('insights', 'No insights provided')
        test_log = error_analysis.get('test_results', 'No test results provided')
      
        # Prompt for code refinement using code analysis insights directly
        code_prompt = [
            {
                "role": "user",
                "content": f"""Given a competitive programming problem, its understanding, and generated {self.language} code to solve the problem, but the generated code cannot pass sample test cases.
Problem: {problem}
Problem Understanding: {problem_understanding}
Current Plan: {plan}
Current Code:
```python
{code}
```
Test Log: {test_log}
Code Analysis Insights: {insights}
Using the provided code analysis insights and test log, refine the code to correct the identified issues.
Provide the revised code in a clear, readable format, optionally within ```{self.language} ... ``` markdown.
IMPORTANT: Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words.
"""
            }
        ]
      
        try:
            if self.verbose:
                print("Step: Making API call for code update")
            updated_response, _, _ = self.gpt_chat(code_prompt)
            revised_code = self.parse_code(updated_response)
            if self.verbose:
                print("Step: Code updated")
                print(f"Revised code: {revised_code}...")
        except Exception as e:
            print(f"Error debugging code: {e}")
            revised_code = code
      
        if self.verbose:
            print("Step: Historical data updated for code")
      
        return revised_code, insights
    def _inner_run(self, item):
        if self.verbose:
            print("Step: Starting inner run")
        pr_tok = 0
        com_tok = 0

        try:
            problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
            pr_tok += pr_tok_u
            com_tok += com_tok_u
        except Exception as e:
            print(f"Error getting problem understanding: {e}")
            problem_understanding = ""
        try:
            plans, pr_tok_p, com_tok_p = self.generate_plans(item)
            pr_tok += pr_tok_p
            com_tok += com_tok_p
            if self.verbose:
                print("Step: Plans generated")
                print(f"Number of plans: {len(plans)}")
        except Exception as e:
            print(f"Error generating plans: {e}")
            plans = []
        if not plans:
            print("Warning: No valid plans generated. Returning default code.")
            return "# No valid solution generated", pr_tok, com_tok
        selected_plannings = plans[:self.top_plan]
        best_code = ""
        test_log = ""
        code_score = 0.0
        for plan_idx, plan_output in enumerate(selected_plannings, 1):
            if self.verbose:
                print(f"Step: Processing plan {plan_idx}")
            planning = plan_output.planning
            best_code = plan_output.code
            code_score = plan_output.code_score
            test_log = plan_output.test_log
            try:
                passed, test_log = self.data.evaluate_sample_io(item, best_code, self.language)
                if passed:
                    if self.verbose:
                        print("Step: Code passed all sample test cases. Early stopping.")
                    return best_code, pr_tok, com_tok
                if self.verbose:
                    print(f"Step: Initial evaluation - Score: {code_score}, Passed: {passed}")
            except Exception as e:
                print(f"Error evaluating initial code: {e}")
                test_log = f"Evaluation failed: {e}"
            for i in range(1, self.t + 1):
                if self.verbose:
                    print(f"Step: Iteration {i} for plan {plan_idx}")
                try:
                    decision, merged_result = self.fast_collaborative_decision(planning, best_code, test_log, item)
                    if self.verbose:
                        print(f"Step: Decision made: {decision}")
                except Exception as e:
                    print(f"Error in decision: {e}")
                    decision = "update code only"
                A_content = merged_result['content_analysis']
                if decision == 'update plan':
                    try:
                        A_plan = merged_result['plan_analysis']
                        revised_plan, _ = self.debug_plan(i, planning, {
                            'insights': A_plan['insights'] + "\n" + A_content['insights'],
                            'test_results': test_log,
                            'success_rate': code_score * 100,
                        }, self.data.get_prompt(item), problem_understanding, decision)
                        best_code, code_score, test_log, pr_tok_code, com_tok_code = self.generate_code_from_plan(item, revised_plan, self.data.get_prompt(item), self.get_sample_io_str(item), "", problem_understanding)
                        planning = revised_plan
                        pr_tok += pr_tok_code
                        com_tok += com_tok_code
                        try:
                            passed, test_log = self.data.evaluate_sample_io(item, best_code, self.language)
                            if passed:
                                if self.verbose:
                                    print("Step: Updated code passed all sample test cases. Early stopping.")
                                return best_code, pr_tok, com_tok
                            if self.verbose:
                                print(f"Step: Plan updated - New code score: {code_score}, Passed: {passed}")
                        except Exception as e:
                            print(f"Error evaluating updated code: {e}")
                            test_log = f"Evaluation failed: {e}"
                    except Exception as e:
                        print(f"Error updating plan: {e}")
                        continue
                else:
                    try:
                        A_code = merged_result['code_analysis']
                        revised_code, _ = self.debug_code(i, planning, best_code, {
                            'insights': A_code['insights'] + "\n" + A_content['insights'],
                            'test_results': test_log,
                            'success_rate': code_score * 100,
                        }, self.data.get_prompt(item), problem_understanding, decision)
                        best_code = revised_code
                        try:
                            passed, test_log = self.data.evaluate_sample_io(item, best_code, self.language)
                            code_score = self.compute_score(test_log)
                            if passed:
                                if self.verbose:
                                    print("Step: Updated code passed all sample test cases. Early stopping.")
                                return best_code, pr_tok, com_tok
                            if self.verbose:
                                print(f"Step: Code updated - New code score: {code_score}, Passed: {passed}")
                        except Exception as e:
                            print(f"Error evaluating updated code: {e}")
                            test_log = f"Evaluation failed: {e}"
                            code_score = 0.0
                    except Exception as e:
                        print(f"Error updating code: {e}")
                        continue
        if self.verbose:
            print("Step: Process completed without passing all sample test cases")
        return best_code, pr_tok, com_tok

    def run_single_pass(self, item: dict):
        if self.verbose:
            print("Step: Starting single pass run")
        max_retries = 1
        for attempt in range(1, max_retries + 1):
            if self.verbose:
                print(f"Step: Run attempt {attempt}")
            try:
                item['api_calls'] = item.get('api_calls', 0)
                result = self._inner_run(item)
                if self.verbose:
                    print("Step: Run successful")
                return result
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    return "No_solution_found",0,0
    
