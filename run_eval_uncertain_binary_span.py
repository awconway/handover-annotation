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
        "--data-file",
        default=DATA_FILE,
        help="Path to tokenised Prodigy JSONL data.",
    )
    parser.add_argument(
        "--output-model-file",
        default=None,
        help="Path to the trained program to load (omit when using --baseline).",
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
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate using the initial DSPy signature (no trained program load).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads for evaluation. Defaults to DSPy settings.num_threads.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh by overwriting existing eval JSONL instead of resuming.",
    )
    args = parser.parse_args()

    if args.baseline and args.output_model_file:
        parser.error("Use either --baseline or --output-model-file, not both.")
    if not args.baseline and not args.output_model_file:
        parser.error("--output-model-file is required unless --baseline is set.")

    return args


args = parse_args()
if args.use_all:
    testset = prepare_dataset_uncertainty_binary_span_all(
        args.data_file, annotator_id=args.annotator_id
    )
else:
    _, testset = prepare_dataset_uncertainty_binary_span(
        args.data_file, annotator_id=args.annotator_id
    )
eval_results_file = args.eval_results_file

predictor = build_predictor()

lm = load_model(args.model_name)
configure_dspy(lm)

if not args.baseline:
    predictor.load(args.output_model_file)

score = evaluate_sbar(
    predictor,
    testset,
    eval_results_file,
    resume=not args.no_resume,
    num_threads=args.num_threads,
)
print(predictor.inspect_history(-1))
print("Evaluation complete. Score:", score)
