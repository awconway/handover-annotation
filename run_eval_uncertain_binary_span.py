import argparse

from config.dspy_settings import configure_dspy
from config.model_registry import load_model
from data.dataset import (
    prepare_dataset_uncertainty_binary_span,
    prepare_dataset_uncertainty_binary_span_all,
)
from eval.evaluator import evaluate_sbar
from uncertain_binary_span_task.signatures import build_predictor

DATA_FILE = "./annotated_data/db_20260129_tokenised.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-model-file",
        required=True,
        help="Path to the trained program to load.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model registry key (see src/config/model_registry.py).",
    )
    parser.add_argument(
        "--eval-results-file",
        required=True,
        help="Path to write eval JSONL results.",
    )
    parser.add_argument(
        "--annotator-id",
        default=None,
        help="Filter examples by _annotator_id (e.g. handover_db-user1).",
    )
    parser.add_argument(
        "--use-all",
        action="store_true",
        help="Evaluate on all matching examples (no train/test split).",
    )
    return parser.parse_args()


args = parse_args()
if args.use_all:
    testset = prepare_dataset_uncertainty_binary_span_all(
        DATA_FILE, annotator_id=args.annotator_id
    )
else:
    _, testset = prepare_dataset_uncertainty_binary_span(
        DATA_FILE, annotator_id=args.annotator_id
    )
output_model_file = args.output_model_file
eval_results_file = args.eval_results_file

predictor = build_predictor()
predictor.load(output_model_file)

lm = load_model(args.model_name)
configure_dspy(lm)

score = evaluate_sbar(predictor, testset, eval_results_file)
print(predictor.inspect_history(-1))
print("Evaluation complete. Score:", score)
