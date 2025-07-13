from typing import List
import tiktoken
import os
import json
import re
import sys
import time

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
from sentence_transformers import SentenceTransformer
import numpy as np

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

# KB + Exemplars + Example Planning + Problem Planning + Code Generation + Sample IO testing + Code Improvement


class CoEvolve(BaseStrategy):
    def __init__(
        self,
        k: int = 3,
        t: int = 5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.t = t
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
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free open-source embedding model

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
            return sample_io
    def get_sample_io_xcode(self, item):
        return "\n".join([f"Input:\n{item['sample_inputs']}\nExpected output:\n{item['sample_outputs']}"])
    
    def summarize_plan(self, plan: str) -> str:
        """
        Use LLM to summarize the plan concisely.
        """
        input_prompt = [
            {
                "role": "user",
                "content": f"""Summarize the following plan for solving a programming problem in a concise manner. Include the main approach used, data structures employed, a brief overview of the plan from input to output, and how edge cases are handled.

# Plan:
{plan}

----------------
Important: Return only the summary text. No extra words.
"""
            }
        ]
        response, _, _ = self.gpt_chat(processed_input=input_prompt)
        return response.strip()
    def get_embedding(self, text: str) -> np.array:
        """
        Get embedding using free sentence-transformers model.
        """
        return self.embedding_model.encode(text)
    def retrieval_rag(self, current_key: str, k: int = 2) -> List[dict]:
        """
        Retrieve top-k similar cases from history using cosine similarity on embeddings of summary || plan_analysis | test_log.
        """
        if not self.history:
            return []
        
        current_embedding = self.get_embedding(current_key)
        
        similarities = []
        for hist in self.history:
            hist_embedding = hist['embedding']
            cosine_sim = np.dot(current_embedding, hist_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(hist_embedding))
            similarities.append((cosine_sim, hist))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [sim[1] for sim in similarities[:k]]
    def plan_analysis(self, plan: str, test_log: str, problem: str) -> dict:
        """
        Return insights on the plan's effectiveness based on the test log and problem description.
        Returns {'insights': str, 'pr_tok': int, 'com_tok': int}
        """
        input_prompt = [
            {
                "role": "user",
                "content": f"""Analyze a plan for solving a competitive programming problem, given the problem description and test log from code generated using the plan. Take a sample input from the test log and simulate the plan's execution step-by-step to pinpoint where the plan is failing based on the test log, and suggest specific improvements or modifications to fix those issues.

    # Problem:
    {problem}

    # Plan:
    {plan}

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
    def code_analysis(self, code: str, test_log: str, problem: str) -> dict:
        """
        Analyzes code issues using LLM based on test log, focusing on errors and suggestions.
        Returns {'insights': str, 'pr_tok': int, 'com_tok': int}
        """
        input_prompt = [
            {
                "role": "user",
                "content": f"""Assess the generated code written in {self.language} programming language for a competitive programming problem, using the problem description and test log. Identify where the code is failing based on the test log, and suggest specific improvements or fixes to correct those issues.

    # Problem:
    {problem}

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
    def content_analysis(self, problem: str, plan: str, code: str) -> dict:
        """
        Analyzes problem-solution alignment using LLM based on problem, plan, and code.
        Returns {'problem_plan_confidence': float, 'plan_code_confidence': float, 'overall_confidence': float, 'problem_plan_insights': str, 'plan_code_insights': str, 'pr_tok': int, 'com_tok': int}
        """
        input_prompt = [
            {
                "role": "user",
                "content": f"""Evaluate how effectively a plan and generated code, written in {self.language} programming language, align with the requirements of a competitive programming problem, given the problem description. Specifically, assess the alignment between the problem and the plan, and between the plan and the code. Identify any mismatches or issues in these alignments. Provide separate confidence scores (0.0 to 1.0) for the problem-plan alignment and plan-code alignment (1.0: perfect match; 0.0: no alignment), and suggest specific improvements for each if the alignment is not strong.

    # Problem:
    {problem}

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
        """
        Compute a confidence score (0.0–1.0) for a given decision based on a single analysis.
        - If analysis_name is "plan" or "code": evaluates the reliability of that analysis.
        - If analysis_name is "content": evaluates how well the plan and code align.
        """
        meaning = self.analysis_meaning.get(analysis_name, "")

        prompt = [
            {
                "role": "user",
                "content": f"""You are given a {analysis_name} analysis. {meaning} Please calculate the confidence score (0.0 to 1.0) for choosing to {decision}, where 1.0 means the analysis strongly supports the decision (e.g., insights indicate it's the best fix), and 0.0 means it does not support at all.

    Insights:
    {analysis.get('insights', '')}

    ==============
    Return only XML in this format:
    <root>
    <confidence>A float between 0.0 and 1.0</confidence>
    </root>
    """
            }
        ]
        response, pr_tok, com_tok = self.gpt_chat(processed_input=prompt)
        # extract and parse the <confidence> tag
        response = self.replace_tag(response, 'confidence')
        parsed = self.parse_xml(response)
        try:
            score = float(parsed.get('confidence', 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return max(0.0, min(score, 1.0))

    def get_consistency(
        self,
        decision: str,
        analysis1: dict, name1: str,
        analysis2: dict, name2: str
    ) -> float:
        """
        Compute a consistency score (0.0–1.0) for choosing `decision`
        given two analyses: name1 and name2.
        """
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
    Calculate the consistency score (0.0 to 1.0) for choosing to {decision}, where 1.0 means the insights from both analyses are highly consistent and support the decision (e.g., similar issues and fixes suggested), and 0.0 means they are inconsistent or contradictory.

    {name1} insights:
    {ins1}

    {name2} insights:
    {ins2}

    Important:
    Your response must follow the following XML format exactly:
    <root>
    <consistency>float between 0.0 and 1.0</consistency>
    </root>
    """
            }
        ]

        response, pr_tok, com_tok = self.gpt_chat(processed_input=prompt)
        response = self.replace_tag(response, 'consistency')
        parsed = self.parse_xml(response)
        try:
            score = float(parsed.get('consistency', 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return max(0.0, min(score, 1.0))
    def collaborative_decision(self, plan: str, code: str, outcomes: str, item) -> str:
        """
        Compute D_final = consensus over plan, code, and content analyses.
        Returns either 'update plan' or 'update code only'.
        """
        A_plan = self.plan_analysis(plan, outcomes, self.data.get_prompt(item))
        A_code = self.code_analysis(code, outcomes, self.data.get_prompt(item))
        A_content = self.content_analysis(self.data.get_prompt(item), plan, code)

        decisions = ['update plan', 'update code only']
        scores = {}

        for d in decisions:
            total = 0.0
            for name, A_i in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
                w = self.trust_weights[name]
                conf = self.get_confidence(d, A_i, name)
                # consistency with other agents
                cons_prod = 1.0
                for oname, A_j in [('plan', A_plan), ('code', A_code), ('content', A_content)]:
                    if oname != name:
                        cons_prod *= self.get_consistency(d, A_i, name, A_j, oname)
                total += w * conf * cons_prod
            scores[d] = total

        # choose decision with max consensus score
        return max(scores, key=scores.get)

 

    
    def _inner_run(self, item):
            print("", flush=True)

            input_kb_exemplars = [
                {
                    "role": "user",
                    "content": f"""Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.
    # Problem:
    {self.data.get_prompt(item)}

    # Exemplars:
    Recall {mapping[self.k]} relevant and distinct problems (different from problem mentioned above). For each problem,
    1. describe it
    2. generate {self.language} code step by step to solve that problem
    3. finally generate a planning to solve that problem

    # Algorithm:

    ----------------
    Important:
    Your response must follow the following xml format-

    <root>
    <problem>
    # Recall {mapping[self.k]} relevant and distinct problems (different from problem mentioned above). Write each problem in the following format.
    <description>
    # Describe the problem.
    </description>
    <code>
    # Let's think step by step to solve this problem in {self.language} programming language.
    </code>
    <planning>
    # Planning to solve this problem.
    </planning>
    </problem>

    # similarly add more problems here...

    <algorithm>
    # Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.
    # Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.
    </algorithm>
    </root>
    """,
                },
            ]

            print("\n\n________________________")
            print("Input for knowledge base and exemplars: ")
            print(input_kb_exemplars[0]['content'], flush=True)

            response, pr_tok, com_tok = self.gpt_chat(
                processed_input=input_kb_exemplars
            )
            item['api_calls'] = item.get('api_calls', 0) + 1

            # Post processing
            response = self.trim_text(
                response, "# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.")
            response = self.trim_text(
                response, "# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.")
            response = self.trim_text(
                response, "# Planning to solve this problem:")
            response = self.trim_text(
                response, f"# Let's think step by step to solve this problem in {self.language} programming language.")
            response = self.replace_tag(response, 'algorithm')
            response = self.replace_tag(response, 'description')
            response = self.replace_tag(response, 'code')
            response = self.replace_tag(response, 'planning')

            print("\n\n________________________")
            print("Response from knowledge base and exemplars: ")
            print(response, flush=True)

            response = self.parse_xml(response)

            algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{ response['algorithm']}"
            sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item)}\n"
            # sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_xcode(item)}\n"
            # if type(self.data) != MBPPDataset and type(self.data) != XCodeDataset else ""

            plannings = []
            for example_no, example in enumerate(response["problem"], start=1):
                example_problem = example["description"]
                example_planning = example["planning"]

                input_for_problem_planning = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem generate a concrete planning to solve the problem.\n# Problem:\n{example_problem}\n# Planning:\n{example_planning}\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{sample_io_prompt}\n## Planning:\n\n----------------\nImportant: You should give only the planning to solve the problem. Do not add extra explanation or words."
                    }
                ]

                print("\n\n________________________")
                print(
                    f"Input for our problem planning using example: {example_no}: ")
                print(input_for_problem_planning[0]['content'], flush=True)

                planning, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_problem_planning
                )
                item['api_calls'] += 1
                # time.sleep(1)
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                # planning = self.parse_xml(planning)
                # planning['confidence'] = int(str(planning['confidence']).strip())

                print("\n\n________________________")
                print("Response from our problem planning: ")
                print(planning, flush=True)

                input_for_planning_verification = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem and a plan to solve the problem in {self.language}, tell whether the plan is correct to solve this problem.\n\n# Problem:\n{self.data.get_prompt(item)}\n# Planning:\n{planning}\n\n----------------\nImportant: Your response must follow the following xml format-```\n<root>\n<explanation> Discuss whether the given competitive programming problem is solvable by using the above mentioned planning.</explanation>\n<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>\n</root>\n```"
                    }
                ]

                print("Input for planning verification: ")
                print(input_for_planning_verification[0]['content'], flush=True)

                verification_res, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_planning_verification
                )
                item['api_calls'] += 1
                # time.sleep(1)
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                verification_res = self.replace_tag(
                    verification_res, 'explanation')
                verification_res = self.replace_tag(verification_res, 'confidence')

                verification_res = self.parse_xml(verification_res)

                verification_res['confidence'] = int(
                    str(verification_res['confidence']).strip())

                print("Response from planning verification: ")
                print(verification_res, flush=True)

                plannings.append((
                    planning,
                    verification_res['confidence'],
                    example
                ))

                # if type(self.data) == MBPPDataset and verification_res['confidence'] == 100:
                #     break

            plannings.sort(key=lambda x: x[1], reverse=True)
            # time.sleep(1)

            if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
                std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
            else:
                std_input_prompt = ""

            for planning_with_ex in plannings:
                planning, confidence, example = planning_with_ex

                input_for_final_code_generation = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem generate {self.language} code to solve the problem.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n## Planning:\n{planning}\n{sample_io_prompt}\n## Let's think step by step.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
                    }
                ]

                print("\n\n________________________")
                print("Input for final code generation: ")
                print(input_for_final_code_generation[0]['content'], flush=True)

                code, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_final_code_generation
                )
                item['api_calls'] += 1
                # time.sleep(1)

                code = self.parse_code(code)
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                print("\n\n________________________")
                print("Response from final code generation: ")
                print(code, flush=True)

                response = f"## Planning: {planning}\n## Code:\n```\n{code}\n```"
                passed = False

                for i in range(1, self.t + 1):
                    
                    passed, test_log = self.data.evaluate_sample_io(
                        item,
                        code,
                        self.language
                    )

                    # if passed:
                    #     break
                    # if not break then run decision making  
                    decision = self.collaborative_decision(
                        planning,
                        code,
                        test_log,
                        item
                    )
                    print("Decision made: ", decision)
                    summary = self.summarize_plan(planning)
                    A_plan = self.plan_analysis(
                        planning, test_log, self.data.get_prompt(item)
                    )
                    A_code = self.code_analysis(
                        code, test_log, self.data.get_prompt(item)
                    )
                    entry_key = summary + A_plan['insights'] + test_log
                    embedding = self.get_embedding(entry_key)
                    self.history.append({
                            'summary': summary,
                            'plan': planning,
                            'plan_analysis': A_plan['insights'],
                            'code': code,
                            'code_analysis': A_code['insights'],
                            'test_log': test_log,
                            'embedding': embedding
                        })
                    summary = self.summarize_plan(planning)
                    current_key = summary + A_plan['insights'] + test_log
                    if decision == 'update plan':
                        similar_cases = self.retrieval_rag(current_key, k=2)
                        similar_str = "\n".join([f"Past failed plan {j+1}: {case['plan']}\nPlan Analysis: {case['plan_analysis']}" for j, case in enumerate(similar_cases)])
                        similar_codes_str = "\n".join([f"Past failed code {j+1}:\n```{self.language}\n{case['code']}\n```\nCode Analysis: {case['code_analysis']}" for j, case in enumerate(similar_cases)])
                        prompt_update = [{"role": "user", "content": (
                            f"Given a competitive programming problem and a plan to solve it, but this plan has some troubles that need to be updated with insights. Please modify the plan accordingly."
                            f"# Problem: {self.data.get_prompt(item)}"
                            f"Current planning: {planning}"
                            f"Insights: {A_plan['insights']}"
                            f"Similar past failed plans and analyses:\n{similar_str}"
                            f"Important: Generate a revised plan that is different from the past failed plans."
                            "Important: return only the revised plan text. Important: You should give only the updated planning to solve the problem. Do not add extra explanation or words."
                        )}]

                        revised_plan, p_up, c_up = self.gpt_chat(processed_input=prompt_update)
                        print("Revised plan: ", revised_plan, flush=True)
                        item['api_calls'] += 1
                        planning = revised_plan.strip()
                        input_for_new_code_generation = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem generate {self.language} code to solve the problem.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n## Planning:\n{planning}\n{sample_io_prompt}\n## Let's think step by step.\nSimilar past failed codes and analyses:\n{similar_codes_str}\nImportant: Generate code that is different from the past failed codes and follow the Planning given.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
                    }
                ]
                        new_code_response, pr_tok_1, com_tok_1 = self.gpt_chat(
                            input_for_new_code_generation
                        )
                        item['api_calls'] += 1
                        code = self.parse_code(new_code_response)
                    else:
                        current_key = summary + A_plan['insights'] + test_log
                        similar_cases = self.retrieval_rag(current_key, k=2)
                        similar_codes_str = "\n".join([f"Past failed code {j+1}:\n```{self.language}\n{case['code']}\n```\nCode Analysis: {case['code_analysis']}" for j, case in enumerate(similar_cases)])
                        
                        print(f"Input for improving code generation: {i}")

                        input_for_improving_code = [
                            {
                                "role": "user",
                                "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{response}\n## Test Report:\n{test_log}\n## Insights:{A_code['insights']}\n## Let's think step by step to modify {self.language} Code for solving this problem.\nSimilar past failed codes and analyses:\n{similar_codes_str}\nImportant: Generate modified code that is different from the past failed codes.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."
                            }
                        ]

                        print("\n\n________________________")
                        print("Input for improving code generation: ")
                        print(input_for_improving_code[0]['content'], flush=True)

                        response, pr_tok_1, com_tok_1 = self.gpt_chat(
                            input_for_improving_code
                        )
                        item['api_calls'] += 1
                        # time.sleep(1)

                        code = self.parse_code(response)
                        pr_tok += pr_tok_1
                        com_tok += com_tok_1

                        print("\n\n________________________")
                        print("Response from improving code generation: ")
                        print(response, flush=True)

                # got a code that passed all sample test cases
                
                if passed:
                    break

            print("________________________\n\n", flush=True)
            return code, pr_tok, com_tok

    def run_single_pass(self, item: dict):
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                return self._inner_run(item)
            except ET.ParseError as e:
                print(f"[run_single_pass] Attempt {attempt} caught ET.ParseError: {e}. Retrying...")
                if attempt == max_retries:
                    print(f"[run_single_pass] ERROR: All {max_retries} attempts failed due to XML parsing. Returning fallback ('',0,0).")
                    return "", 0, 0
            except Exception as e:
                # raise e
                # Nếu có lỗi khác không phải ParseError, cũng bẫy lại để retry
                print(f"[run_single_pass] Attempt {attempt} caught unexpected exception: {e}. Retrying...")
                if attempt == max_retries:
                    print(f"[run_single_pass] ERROR: All {max_retries} attempts failed due to unexpected errors. Returning fallback ('',0,0).")
                    return "", 0, 0
