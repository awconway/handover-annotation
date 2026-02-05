import argparse

import dspy

from config.settings import DATA_FILE, EVAL_RESULTS_FILE, OUTPUT_MODEL_FILE
from data.dataset import prepare_dataset
from eval.evaluator import evaluate
from sbar_span_task.signatures import build_predictor

lm = dspy.LM(model="openai/gpt-5-nano")
dspy.settings.configure(lm=lm)


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

# score = evaluate(predictor, testset, EVAL_RESULTS_FILE)
print(predictor(text=testset[0].text))
# print("Evaluation complete. Score:", score)
