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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
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
from forms import *

def multi_thread_task_dict(task_dictionary, num_workers=1, show_progress=True):
    final_results = {}
    futures = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for id_, task in task_dictionary.items():
            futures.append(
                executor.submit(
                    lambda id_=id_, task=task: {"id": id_, "task_result": task()}
                )
            )

        if show_progress:
            with tqdm(total=len(futures)) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    final_results[result["id"]] = result["task_result"]
                    pbar.update(1)
        else:
            for future in as_completed(futures):
                result = future.result()
                final_results[result["id"]] = result["task_result"]

    return final_results

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
        
        # Base prompt structure
        prompt = f"""You are a debugging assistant for a competitive programming problem. Your task is to provide a conversational analysis reflection to guide plan refinement at iteration {iteration}. The reflection should evolve from the previous reflection (R(t-1)) and incorporate new insights from plan analysis.
Problem: {problem}
Problem Understanding: {problem_understanding}
Current Plan: {plan}
Test Log: {error_analysis.get('test_results', '')}
Insights from Plan Analysis: {insights}
Previous Analysis Reflection (R(t-1)): {previous_reflection}
Success Rate: {success_rate:.2f}%
Evolve the reflection from R(t-1) by addressing the new insights. Provide a reflection to guide the next plan update.
"""
        return prompt
    def generate_prompt_for_code_reflection(self, iteration: int, error_analysis: Dict, problem: str, problem_understanding: str, plan: str, code: str, historical_logs: Dict) -> str:
        """Generate a conversational prompt for code debugging, evolving from R(t-1)."""
        previous_reflection = historical_logs.get('analysis_reflection', 'No previous analysis reflection available')
        insights = error_analysis.get('insights', '')
        success_rate = error_analysis.get('success_rate', 0.0)
        
        # Base prompt structure
        prompt = f"""You are a debugging assistant for a competitive programming problem. Your task is to provide a conversational analysis reflection to guide code refinement at iteration {iteration}. The reflection should evolve from the previous reflection (R(t-1)) and incorporate new insights from code analysis.
Problem: {problem}
Problem Understanding: {problem_understanding}
Current Plan: {plan}
Code:
```python
{code}
```
Test Log: {error_analysis.get('test_results', '')}
Insights from Code Analysis: {insights}
Previous Analysis Reflection (R(t-1)): {previous_reflection}
Success Rate: {success_rate:.2f}%
Evolve the reflection from R(t-1) by addressing the new insights. Provide a reflection to guide the next code update.
"""
        return prompt

class CoEvolvev2(BaseStrategy):
    def __init__(
        self,
        k: int = 3,
        t: int = 5,
        max_attempts: int = 3,
        min_llm_score: int = 50,
        num_workers: int = 4,  # Added for parallelization
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.top_plan = 1
        self.t = t
        self.number_of_code_per_plan = 3
        self.min_llm_score = min_llm_score
        self.num_workers = num_workers  # Number of threads for parallel execution
        self.trust_weights = {
            'plan': 0.4,
            'code': 0.3,
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
        self.rt = AnalysisReflection()

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
                "content": f"""You are an expert in JSON structuring. Your task is to take a free-form response and convert it into structured JSON matching the provided schema. Extract relevant information from the response and ensure the output is valid JSON.
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
                print(f"Step: Wrapped response: {wrapped_response[:200]}...")
        except Exception as e:
            print(f"Error wrapping response: {e}\nRaw response: {response[:200]}...")
            return model()
        json_str = self._extract_json_string(wrapped_response)
        if not json_str:
            print(f"Invalid output: No JSON structure found in wrapped response\nWrapped response: {wrapped_response[:200]}...\nRaw response: {response[:200]}...")
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
                    print(f"Invalid output: JSON decode error after retries: {e}\nFinal JSON candidate: {json_str[:300]}...\nWrapped response: {wrapped_response[:300]}...\nRaw response: {response[:300]}...")
                    return model()
                if self.verbose:
                    print(f"Step: JSON decode failed (attempt {attempt + 1}), applying escape fix: {e}")
                json_str = self._fix_invalid_escapes(json_str)
            except Exception as e:
                print(f"Unexpected error during JSON loading: {e}")
                return model()
        if data is None:
            return model()
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
            print(f"Input response: {response[:100]}...")
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
            print(f"Parsed code: {parsed_code[:100]}...")
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
        if self.verbose:
            print("Step: Generating problem understanding")
        problem_text = self.data.get_prompt(item)
        input_for_understanding = [
            {
                "role": "user",
                "content": f"""You are an expert in competitive programming. Your task is to analyze a problem description and provide a concise understanding of the problem, including its requirements, constraints, objectives, and potential edge cases or special cases. Identify edge cases (e.g., boundary conditions, invalid inputs, or extreme scenarios) and special cases (e.g., unique input patterns or problem-specific conditions) that need attention, and provide examples of these cases beyond the provided sample I/O.
# Problem:
{problem_text}
# Sample I/O:
{self.get_sample_io_str(item)}
"""
            }
        ]
        try:
            if self.verbose:
                print("Step: Making API call for understanding")
            understanding, pr_tok, com_tok = self.gpt_chat(processed_input=input_for_understanding)
            item['api_calls'] += 1
            if self.verbose:
                print("Step: Understanding parsed")
                print(f"Understanding: {understanding}...")
            return understanding, pr_tok, com_tok
        except Exception as e:
            print(f"Error in get_problem_understanding: {e}")
            return "", 0, 0

    def generate_code_from_plan(self, item, planning: str, problem_text: str, sample_io_prompt: str, previous_codes: str = "", understanding: str = "") -> Tuple[str, float, str, int, int]:
        if self.verbose:
            print("Step: Generating code from plan")
            print(f"Plan: {planning}...")
        codes = []
        pr_tok = 0
        com_tok = 0
        api_calls = 0
        std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases." if isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset)) else ""
        context = f"Problem: {problem_text}\n"
        context += understanding if understanding else ""

        # Define task for generating a single code variant
        def generate_code_task(c_idx, previous_codes):
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
- Handle all edge cases (e.g., empty inputs, large inputs, boundary values) as specified in the problem and test cases.
- Optimize for correctness and clarity, ensuring alignment with the plan.
# Instructions:
1. Follow the plan step-by-step to create a correct and efficient solution.
2. Provide the code in a clear, readable format, optionally within ```{self.language} ... ``` markdown.
IMPORTANT: Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words.
"""
                }
            ]
            try:
                if self.verbose:
                    print(f"Step: Making API call for code generation (variant {c_idx})")
                code_response, pr_tok_1, com_tok_1 = self.gpt_chat(processed_input=input_for_code_generation)
                code = self.parse_code(code_response)
                if self.verbose:
                    print(f"Generated code variant {c_idx}: {code[:100]}...")
                return code, pr_tok_1, com_tok_1, 1
            except Exception as e:
                print(f"Error generating code {c_idx}: {e}")
                return "", 0, 0, 0

        # Create task dictionary for parallel code generation
        task_dict = {
            c_idx: lambda c_idx=c_idx, prev_codes=previous_codes: generate_code_task(c_idx, prev_codes)
            for c_idx in range(1, self.number_of_code_per_plan + 1)
        }
        previous_codes = ""
        code_results = multi_thread_task_dict(task_dict, num_workers=self.num_workers, show_progress=self.verbose)

        for c_idx in range(1, self.number_of_code_per_plan + 1):
            result = code_results.get(c_idx, ("", 0, 0, 0))
            code, pr_tok_1, com_tok_1, api_call = result
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            api_calls += api_call
            if code:
                codes.append(code)
                previous_codes += f"\n- {code}"

        if self.verbose:
            print(f"Step: {len(codes)} code variants generated. Starting evaluation")
        evaluated_codes = []
        for code_idx, code in enumerate(codes, 1):
            if self.verbose:
                print(f"Step: Evaluating code variant {code_idx}")
                print(f"Code: {code[:100]}...")
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
        
        item['api_calls'] = item.get('api_calls', 0) + api_calls
        max_score = max([score for _, score, _ in evaluated_codes], default=0.0)
        top_codes = [(code, test_log) for code, score, test_log in evaluated_codes if score == max_score]
        best_code, best_test_log = random.choice(top_codes) if top_codes else ("", "")
        if self.verbose:
            print("Step: Best code selected")
            print(f"Best code: {best_code[:100]}..., Score: {max_score}")
        return best_code, max_score, best_test_log, pr_tok, com_tok

    def generate_plans(self, item) -> Tuple[List[PlanOutput], int, int]:
        if self.verbose:
            print("Step: Starting plan generation")
        plans = []
        pr_tok = 0
        com_tok = 0
        api_calls = 0
        previous_approaches = ""
        problem_text = self.data.get_prompt(item)
        sample_io_prompt = self.get_sample_io_str(item)
        problem_understanding, pr_u, com_u = self.get_problem_understanding(item)
        pr_tok += pr_u
        com_tok += com_u
        api_calls += 1

        # Define task for generating a single plan
        def generate_plan_task(t, previous_approaches):
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
"""
                }
            ]
            parsed_response = None
            local_pr_tok = 0
            local_com_tok = 0
            local_api_calls = 0
            for attempt in range(self.max_attempts):
                if self.verbose:
                    print(f"Step: Approach recall attempt {attempt + 1} for variant {t}")
                try:
                    response, pr_tok_temp, com_tok_temp = self.gpt_chat(input_recall)
                    local_pr_tok += pr_tok_temp
                    local_com_tok += com_tok_temp
                    local_api_calls += 1
                    parsed_response = self.parse_structured_output(response, ApproachRecall)
                    if self.verbose:
                        print("Step: Approach recall successful")
                        print(f"Approach name: {parsed_response.approach_name}")
                    break
                except Exception as e:
                    print(f"Error in recall attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        return None, 0, 0, 0
            if parsed_response is None:
                return None, 0, 0, 0
            approach_name = parsed_response.approach_name
            approach_tutorial = parsed_response.tutorial
            algorithm_prompt = f"## Relevant Approach: {approach_name}\n{approach_tutorial}"
            example_description = parsed_response.problem_description
            example_planning = parsed_response.planning
            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": (
                        "You are an expert competitive-programming coach. "
                        "Your task is to produce a precise, step-by-step plan to solve the Target Problem "
                        "using the specified algorithmic approach. "
                        "You are provided with a worked example. It is directly analogous to the Target Problem section below"
                        "Provide the plan in a clear, readable format, optionally using markdown."
                        f"# Example Problem Description:\n{example_description}\n\n"
                        f"# Approach Template:\n{algorithm_prompt}\n\n"
                        f"# Example Plan (for reference):\n{example_planning}\n\n"
                        f"# Target Problem:\n{problem_text}\n\n"
                        f"## Sample Test Cases:\n{sample_io_prompt}"
                    ),
                },
            ]
            planning = None
            for attempt in range(self.max_attempts):
                if self.verbose:
                    print(f"Step: Planning generation attempt {attempt + 1} for variant {t}")
                try:
                    planning_response, pr_tok_temp, com_tok_temp = self.gpt_chat(input_for_problem_planning)
                    local_pr_tok += pr_tok_temp
                    local_com_tok += com_tok_temp
                    local_api_calls += 1
                    planning = planning_response
                    break
                except Exception as e:
                    print(f"Error in planning attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        return None, 0, 0, 0
            if planning is None:
                return None, 0, 0, 0
            input_for_planning_verification = [
                {"role": "user", "content": f"""You are an expert evaluator for competitive programming plans.
                 Assess the plan for alignment with the problem and overall solvability (0-100). Provide explanations and scores.
Problem Description: {problem_text}
Proposed Plan: {planning}
Sample I/O: {sample_io_prompt}
Provide the response in a clear, readable format, optionally using markdown.
"""
                }
            ]
            verification_parsed = None
            for attempt in range(self.max_attempts):
                if self.verbose:
                    print(f"Step: Planning verification attempt {attempt + 1} for variant {t}")
                try:
                    verification_res, pr_tok_temp, com_tok_temp = self.gpt_chat(input_for_planning_verification)
                    local_pr_tok += pr_tok_temp
                    local_com_tok += com_tok_temp
                    local_api_calls += 1
                    verification_parsed = self.parse_structured_output(verification_res, VerificationOutput)
                    if self.verbose:
                        print("Step: Verification successful")
                        print(f"Overall solvability: {verification_parsed.overall_solvability}")
                    break
                except Exception as e:
                    print(f"Error in verification attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        return None, 0, 0, 0
            if verification_parsed is None:
                return None, 0, 0, 0
            llm_score = verification_parsed.overall_solvability
            if llm_score < self.min_llm_score:
                return None, 0, 0, 0
            best_code, code_score, test_log, pr_tok_code, com_tok_code = self.generate_code_from_plan(item, planning, problem_text, sample_io_prompt)
            local_pr_tok += pr_tok_code
            local_com_tok += com_tok_code
            local_api_calls += item.get('api_calls', 0) - api_calls  # Account for API calls in generate_code_from_plan
            return PlanOutput(
                planning=planning,
                code=best_code,
                llm_score=llm_score,
                code_score=code_score,
                test_log=test_log
            ), local_pr_tok, local_com_tok, local_api_calls

        # Create task dictionary for parallel plan generation
        task_dict = {
            t: lambda t=t, prev_approaches=previous_approaches: generate_plan_task(t, prev_approaches)
            for t in range(1, self.k + 1)
        }
        plan_results = multi_thread_task_dict(task_dict, num_workers=self.num_workers, show_progress=self.verbose)

        for t in range(1, self.k + 1):
            result = plan_results.get(t, (None, 0, 0, 0))
            plan_output, local_pr_tok, local_com_tok, local_api_calls = result
            if plan_output:
                plans.append(plan_output)
                previous_approaches += f"\n- {plan_output.planning}"
            pr_tok += local_pr_tok
            com_tok += local_com_tok
            api_calls += local_api_calls

        item['api_calls'] = item.get('api_calls', 0) + api_calls

        if len(plans) < self.k:
            print(f"Warning: Only {len(plans)}/{self.k} valid plans generated, attempting one more generation")
            additional_plans, pr_tok_add, com_tok_add = self.generate_plans(item)
            pr_tok += pr_tok_add
            com_tok += com_tok_add
            plans.extend(additional_plans)

        # Sort plans by llm_score and select the top one
        plans.sort(key=lambda x: x.llm_score, reverse=True)
        if not plans:
            print("Warning: No valid plans generated. Returning empty result.")
            return [], pr_tok, com_tok
        
        best_plan = plans[0]
        if self.verbose:
            print(f"Step: Generating codes for best plan with LLM score: {best_plan.llm_score}")
        best_code, code_score, test_log, pr_tok_code, com_tok_code = self.generate_code_from_plan(
            item, best_plan.planning, problem_text, sample_io_prompt, "", problem_understanding
        )
        pr_tok += pr_tok_code
        com_tok += com_tok_code
        item['api_calls'] = item.get('api_calls', 0) + api_calls

        # Return a single PlanOutput for the best plan
        result = [PlanOutput(
            planning=best_plan.planning,
            llm_score=best_plan.llm_score,
            code=best_code,
            code_score=code_score,
            test_log=test_log
        )]
        if self.verbose:
            print("Step: Best plan and code selected")
            print(f"Best plan LLM score: {best_plan.llm_score}, Best code score: {code_score}")
        return result, pr_tok, com_tok

    def plan_analysis(self, plan: str, code: str, test_log: str, problem: str, problem_understanding: str) -> PlanAnalysisOutput:
        if self.verbose:
            print("Step: Performing plan analysis")
            print(f"Plan: {plan[:100]}...")
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
                    print(f"Insights: {parsed.insights[:100]}...")
                return parsed
            except Exception as e:
                print(f"Error in plan_analysis attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    return PlanAnalysisOutput(pr_tok=pr_tok, com_tok=com_tok)

    def code_analysis(self, code: str, test_log: str, problem: str, problem_understanding: str) -> CodeAnalysisOutput:
        if self.verbose:
            print("Step: Performing code analysis")
            print(f"Code: {code[:100]}...")
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
                    print(f"Insights: {parsed.insights[:100]}...")
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
                    "and provide concise insights on both the plan and the code to improve their alignment. "
                    "Provide the analysis in a clear, readable format, optionally using markdown."
                    f"Problem Description:\n{problem}\n\n"
                    f"Problem Understanding:\n{problem_understanding}\n\n"
                    f"Proposed Plan:\n{plan}\n\n"
                    f"Code Implementation ({self.language}):\n"
                    f"```{self.language}\n{code}\n```\n\n"
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
                    print(f"Insights: {parsed.plan_code_insights[:100]}...")
                return parsed
            except Exception as e:
                print(f"Error in content_analysis attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    return ContentAnalysisOutput(pr_tok=pr_tok, com_tok=com_tok)

    def get_confidence(self, decision: str, analysis: dict, analysis_name: str) -> ConfidenceOutput:
        if self.verbose:
            print(f"Step: Getting confidence for '{decision}' from {analysis_name}")
        meaning = self.analysis_meaning.get(analysis_name, "")
        schema = ConfidenceOutput.model_json_schema()
        prompt = [
            {
                "role": "user",
                "content": (
                    "You are a senior software engineer and technical reviewer. "
                    "Your task is to assign a confidence score (0.0 to 1.0) for the decision to "
                    f"'{decision}', based on the following analysis. "
                    "Provide the response in a clear, readable format, optionally using markdown."
                    f"Decision: {decision}\n"
                    f"Analysis Type: {analysis_name}\n"
                    f"Analysis Meaning: {meaning}\n"
                    f"Insights: {analysis.get('insights', '')}"
                ),
            },
        ]
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Confidence attempt {attempt + 1}")
            try:
                response, _, _ = self.gpt_chat(prompt)
                parsed = self.parse_structured_output(response, ConfidenceOutput)
                if self.verbose:
                    print("Step: Confidence calculated")
                    print(f"Confidence: {parsed.confidence}")
                return parsed
            except Exception as e:
                print(f"Error in get_confidence attempt {attempt + 1}: {e}")
        return ConfidenceOutput()

    def get_consistency(self, decision: str, analysis1: dict, name1: str, analysis2: dict, name2: str) -> ConsistencyOutput:
        if self.verbose:
            print(f"Step: Getting consistency for '{decision}' between {name1} and {name2}")
        ins1 = analysis1.get('insights', '')
        ins2 = analysis2.get('insights', '')
        name1_meaning = self.analysis_meaning.get(name1, "")
        name2_meaning = self.analysis_meaning.get(name2, "")
        schema = ConsistencyOutput.model_json_schema()
        prompt = [
            {
                "role": "user",
                "content": (
                    "You are a senior software engineer and technical reviewer. "
                    "Your task is to compute a consistency score (0.0 to 1.0) for the decision to "
                    f"'{decision}', using two separate analyses. "
                    "Compare both insights to determine how well they align. "
                    "Provide the response in a clear, readable format, optionally using markdown."
                    f"Decision: {decision}\n\n"
                    f"Analysis 1: {name1}\n"
                    f"Meaning: {name1_meaning}\n"
                    f"Insights: {ins1}\n\n"
                    f"Analysis 2: {name2}\n"
                    f"Meaning: {name2_meaning}\n"
                    f"Insights: {ins2}"
                ),
            },
        ]
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Consistency attempt {attempt + 1}")
            try:
                response, _, _ = self.gpt_chat(prompt)
                parsed = self.parse_structured_output(response, ConsistencyOutput)
                if self.verbose:
                    print("Step: Consistency calculated")
                    print(f"Consistency: {parsed.consistency}")
                return parsed
            except Exception as e:
                print(f"Error in get_consistency attempt {attempt + 1}: {e}")
        return ConsistencyOutput()

    def collaborative_decision(self, plan: str, code: str, outcomes: str, item) -> str:
        if self.verbose:
            print("Step: Starting collaborative decision")
        try:
            problem_understanding, _, _ = self.get_problem_understanding(item)
            problem_text = self.data.get_prompt(item)
            A_plan = self.plan_analysis(plan, code, outcomes, problem_text, problem_understanding)
            A_code = self.code_analysis(code, outcomes, problem_text, problem_understanding)
            A_content = self.content_analysis(problem_text, problem_understanding, plan, code)
            decisions = ['update plan', 'update code only']
            scores = {}
            for d in decisions:
                if self.verbose:
                    print(f"Step: Scoring decision '{d}'")
                total = 0.0
                for name, A_i in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
                    w = self.trust_weights[name]
                    conf_output = self.get_confidence(d, A_i.model_dump(), name)
                    conf = conf_output.confidence
                    cons_prod = 1.0
                    for oname, A_j in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
                        if oname != name:
                            cons_output = self.get_consistency(d, A_i.model_dump(), name, A_j.model_dump(), oname)
                            cons_prod *= cons_output.consistency
                    total += w * conf * cons_prod
                scores[d] = total
                if self.verbose:
                    print(f"Step: Score for '{d}': {total}")
            decision = max(scores, key=scores.get)
            if self.verbose:
                print("Step: Decision made")
                print(f"Decision: {decision}")
            return decision
        except Exception as e:
            print(f"Error in collaborative_decision: {e}")
            return "update code only"

    def debug_plan(self, iteration: int, plan: str, error_analysis: Dict, problem: str, problem_understanding: str, decision: str):
        if self.verbose:
            print(f"Step: Debugging plan at iteration {iteration}")
        prev_logs = self.rt.historical_data.get(iteration - 1, {})
        previous_reflection = prev_logs.get('analysis_reflection', 'No previous analysis reflection available')
        rt_prompt = self.rt.generate_prompt_for_plan_reflection(
            iteration, error_analysis, problem, problem_understanding, plan, historical_logs=prev_logs
        )
        try:
            if self.verbose:
                print("Step: Generating analysis reflection for plan")
            analysis_reflection, _, _ = self.gpt_chat([{
                'role': 'user',
                'content': rt_prompt
            }])
            if self.verbose:
                print("Step: Analysis reflection generated")
                print(f"Reflection: {analysis_reflection[:100]}...")
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
                print(f"Revised plan: {revised_plan[:100]}...")
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
        if self.verbose:
            print(f"Step: Debugging code at iteration {iteration}")
        prev_logs = self.rt.historical_data.get(iteration - 1, {})
        previous_reflection = prev_logs.get('analysis_reflection', 'No previous analysis reflection available')
        rt_prompt = self.rt.generate_prompt_for_code_reflection(
            iteration, error_analysis, problem, problem_understanding, plan, code, historical_logs=prev_logs
        )
        try:
            if self.verbose:
                print("Step: Generating analysis reflection for code")
            analysis_reflection, _, _ = self.gpt_chat([{
                'role': 'user',
                'content': rt_prompt
            }])
            if self.verbose:
                print("Step: Analysis reflection generated")
                print(f"Reflection: {analysis_reflection[:100]}...")
        except Exception as e:
            print(f"Error generating analysis reflection for code: {e}")
            analysis_reflection = "Error generating analysis reflection"
        code_prompt = [
            {
                'role': 'user',
                'content': f"""Given a competitive programming problem, its understanding, and generated {self.language} code to solve the problem, but the generated code cannot pass sample test cases.
Problem: {problem}
Problem Understanding: {problem_understanding}
Planning: {plan}
Code:
```python
{code}
```
Test Log (F(t)): {error_analysis.get('test_results','')}
Analysis Reflection: {analysis_reflection}
Using the provided analysis reflection and test log, refine the code to correct the identified issues.
Provide the revised code in a clear, readable format, optionally within ```{self.language} ... ``` markdown.
"""
            }
        ]
        try:
            if self.verbose:
                print("Step: Making API call for code update")
            updated_response, _, _ = self.gpt_chat(code_prompt)
            updated_parsed = self.parse_structured_output(updated_response, CodeOutput)
            revised_code = self.parse_code(updated_response)
            if self.verbose:
                print("Step: Code updated")
                print(f"Revised code: {revised_code[:100]}...")
        except Exception as e:
            print(f"Error debugging code: {e}")
            revised_code = code
        self.rt.update_historical_data(iteration, {
            'previous_code': code,
            'previous_success_rate': error_analysis.get('success_rate'),
            'previous_iteration': iteration - 1,
            'analysis_reflection': analysis_reflection
        })
        if self.verbose:
            print("Step: Historical data updated for code")
        return revised_code, analysis_reflection

    def _inner_run(self, item):
        if self.verbose:
            print("Step: Starting inner run")
        pr_tok = 0
        com_tok = 0
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
                    problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
                    pr_tok += pr_tok_u
                    com_tok += com_tok_u
                except Exception as e:
                    print(f"Error getting problem understanding: {e}")
                    problem_understanding = ""
                try:
                    decision = self.collaborative_decision(planning, best_code, test_log, item)
                    if self.verbose:
                        print(f"Step: Decision made: {decision}")
                except Exception as e:
                    print(f"Error in decision: {e}")
                    decision = "update code only"
                if decision == 'update plan':
                    try:
                        A_plan = self.plan_analysis(planning, best_code, test_log, self.data.get_prompt(item), problem_understanding)
                        revised_plan, _ = self.debug_plan(i, planning, {
                            'insights': A_plan.insights,
                            'test_results': test_log,
                            'success_rate': code_score * 100,
                        }, self.data.get_prompt(item), problem_understanding, decision)
                        best_code, code_score, test_log, pr_tok_code, com_tok_code = self.generate_code_from_plan(item, revised_plan, self.data.get_prompt(item), self.get_sample_io_str(item))
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
                        A_code = self.code_analysis(best_code, test_log, self.data.get_prompt(item), problem_understanding)
                        revised_code, _ = self.debug_code(i, planning, best_code, {
                            'insights': A_code.insights,
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
        max_retries = 3
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
                    return "# No valid solution generated", 0, 0