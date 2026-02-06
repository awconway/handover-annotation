import argparse

import dspy

from config.dspy_settings import configure_dspy
from config.settings import DATA_FILE, EVAL_RESULTS_FILE, OUTPUT_MODEL_FILE
from data.dataset import prepare_dataset
from eval.evaluator import evaluate
from uncertain_span_task.signatures import build_predictor

lm = dspy.LM(model="openai/gpt-5-nano")
configure_dspy(lm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotator-id",
        default=None,
        help="Filter examples by _annotator_id (e.g. handover_db-user1).",
    )
    return parser.parse_args()


args = parse_args()
_, testset = prepare_dataset(DATA_FILE, annotator_id=args.annotator_id)

predictor = build_predictor()
case = testset[17]
print(f"text: {case.text}")
print(predictor(text=case.text))
