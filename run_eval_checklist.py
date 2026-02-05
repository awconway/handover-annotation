import argparse
from tkinter.constants import TRUE

import dspy

from checklist_task.signatures import build_predictor
from config.settings import (
    DATA_FILE,
    EVAL_RESULTS_FILE,
    MODEL_NAME,
    OUTPUT_MODEL_FILE,
    with_annotator_suffix,
)
from data.dataset import prepare_dataset
from eval.evaluator import evaluate_checklist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotator-id",
        default=None,
        help="Filter examples by _annotator_id (e.g. handover_db-user1).",
    )
    parser.add_argument(
        "--suffix-eval-only",
        action="store_true",
        help="Suffix only eval output files (load shared model).",
    )
    return parser.parse_args()


args = parse_args()
_, testset = prepare_dataset(DATA_FILE, annotator_id=args.annotator_id)
if args.annotator_id and args.suffix_eval_only:
    output_model_file = OUTPUT_MODEL_FILE
    eval_results_file = with_annotator_suffix(EVAL_RESULTS_FILE, args.annotator_id)
else:
    output_model_file = with_annotator_suffix(OUTPUT_MODEL_FILE, args.annotator_id)
    eval_results_file = with_annotator_suffix(EVAL_RESULTS_FILE, args.annotator_id)

predictor = build_predictor()

lm = dspy.LM(model="openai/gpt-5.1")
dspy.settings.configure(lm=lm)
predictor.load(output_model_file)
for name, pred in predictor.named_predictors():
    print("================================")
    print(f"Predictor: {name}")
    print("================================")
    print("Prompt:")
    print(pred.signature.instructions)
    print("*********************************")
score = evaluate_checklist(predictor, testset, eval_results_file)
print("Evaluation complete. Score:", score)
