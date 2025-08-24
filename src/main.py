import sys
from datetime import datetime
from constants.paths import *
from models.Gemini import Gemini
from models.OpenAI import OpenAIModel
from results.Results import Results
from promptings.PromptingFactory import PromptingFactory
from datasets.DatasetFactory import DatasetFactory
from models.ModelFactory import ModelFactory
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="HumanEval",
    choices=[
        "HumanEval",
        "HumanEvalET",
        "MBPP",
        "APPS",
        "XCode",
        "CC",
        "LCB"
    ]
)
parser.add_argument(
    "--strategy",
    type=str,
    default="MapCoder",
    choices=[
        "Direct",
        "CoT",
        "SelfPlanning",
        "Analogical",
        "MapCoder",
        "CoEvolve",
        "CoEvolvev2",
        "CoEvolvev4",
        "CoEvolvev5"
    ]
)
parser.add_argument(
    "--model",
    type=str,
    default="ChatGPT",
    choices=[
        "ChatGPT",
        "GPT4",
        "Gemini",
        "Together"
    ]
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0
)
parser.add_argument(
    "--pass_at_k",
    type=int,
    default=1
)
parser.add_argument(
    "--language",
    type=str,
    default="Python3",
    choices=[
        "C",
        "C#",
        "C++",
        "Go",
        "PHP",
        "Python3",
        "Ruby",
        "Rust",
    ]
)
parser.add_argument(
    "--problem_index",
    type=str,
    default=None,
    help="Comma-separated list of problem indices to run (e.g., '20' or '20,21'). If None, run all problems."
)
args = parser.parse_args()

DATASET = args.dataset
STRATEGY = args.strategy
MODEL_NAME = args.model
TEMPERATURE = args.temperature
PASS_AT_K = args.pass_at_k
LANGUAGE = args.language
RUN_NAME = f"{MODEL_NAME}-{STRATEGY}-{DATASET}-{LANGUAGE}-{TEMPERATURE}-{PASS_AT_K}"
RESULTS_PATH = f"./outputs/{RUN_NAME}.jsonl"
print(f"#########################\nRunning start {RUN_NAME}, Time: {datetime.now()}\n##########################\n")

strategy = PromptingFactory.get_prompting_class(STRATEGY)(
    model=ModelFactory.get_model_class(MODEL_NAME)(temperature=TEMPERATURE),
    data=DatasetFactory.get_dataset_class(DATASET)(),
    language=LANGUAGE,
    pass_at_k=PASS_AT_K,
    results=Results(RESULTS_PATH),
)

if args.problem_index is None:
    strategy.run()
else:
    indices = [int(idx.strip()) for idx in args.problem_index.split(',')]
    for idx in indices:
        strategy.run_problem(idx)

print(f"#########################\nRunning end {RUN_NAME}, Time: {datetime.now()}\n##########################\n")
