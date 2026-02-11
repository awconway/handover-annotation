import argparse
import os

from uncertain_span_task.langextract_experiment import (
    run_langextract_uncertainty_experiment,
)

DATA_FILE = "./annotated_data/db_20260129_tokenised.jsonl"
OUTPUT_FILE = "./evals/eval_uncertain_langextract.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        default=DATA_FILE,
        help="Path to tokenised Prodigy JSONL data.",
    )
    parser.add_argument(
        "--output-file",
        default=OUTPUT_FILE,
        help="Path to write experiment predictions/metrics JSONL.",
    )
    parser.add_argument(
        "--model-id",
        default="gpt-5.2",
        help="LangExtract model_id (e.g. gpt-5.2, gemini-2.5-flash).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override. If omitted, read from --api-key-env.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable to read API key from when --api-key is omitted.",
    )
    parser.add_argument(
        "--annotator-id",
        default=None,
        help="Optional _annotator_id filter (e.g. handover_db-user1).",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=24,
        help="Number of few-shot examples for LangExtract.",
    )
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=20,
        help="Number of held-out records to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=339,
        help="Seed for deterministic train/eval split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LangExtract API calls and write empty predictions for smoke tests.",
    )
    parser.add_argument(
        "--prompt-validation-level",
        choices=["off", "warning", "error"],
        default="warning",
        help=(
            "Prompt alignment validation mode for few-shot examples. "
            "Validation runs once per experiment, not per eval record."
        ),
    )
    parser.add_argument(
        "--prompt-validation-strict",
        action="store_true",
        help=(
            "When validation level is 'error', also fail on non-exact "
            "alignment (fuzzy/lesser)."
        ),
    )
    parser.set_defaults(show_progress=True)
    parser.add_argument(
        "--show-progress",
        dest="show_progress",
        action="store_true",
        help="Show LangExtract progress during each extraction call.",
    )
    parser.add_argument(
        "--no-show-progress",
        dest="show_progress",
        action="store_false",
        help="Hide LangExtract progress output.",
    )
    parser.add_argument(
        "--use-dataset-test-split",
        action="store_true",
        help=(
            "Use the same deterministic split as data.dataset.prepare_dataset_uncertainty_span "
            "and evaluate on its test set."
        ),
    )
    parser.set_defaults(fence_output=True, use_schema_constraints=False)
    parser.add_argument(
        "--fence-output",
        dest="fence_output",
        action="store_true",
        help="Enable fenced output in LangExtract.",
    )
    parser.add_argument(
        "--no-fence-output",
        dest="fence_output",
        action="store_false",
        help="Disable fenced output in LangExtract.",
    )
    parser.add_argument(
        "--use-schema-constraints",
        dest="use_schema_constraints",
        action="store_true",
        help="Enable schema constraints in LangExtract.",
    )
    parser.add_argument(
        "--no-use-schema-constraints",
        dest="use_schema_constraints",
        action="store_false",
        help="Disable schema constraints in LangExtract.",
    )
    return parser.parse_args()


args = parse_args()
api_key = args.api_key or os.environ.get(args.api_key_env)

summary = run_langextract_uncertainty_experiment(
    data_file=args.data_file,
    output_file=args.output_file,
    model_id=args.model_id,
    train_examples=args.train_examples,
    eval_examples=args.eval_examples,
    annotator_id=args.annotator_id,
    seed=args.seed,
    api_key=api_key,
    fence_output=args.fence_output,
    use_schema_constraints=args.use_schema_constraints,
    prompt_validation_level=args.prompt_validation_level,
    prompt_validation_strict=args.prompt_validation_strict,
    show_progress=args.show_progress,
    use_dataset_test_split=args.use_dataset_test_split,
    dry_run=args.dry_run,
)

print(
    "LangExtract uncertainty experiment complete.",
    f"train={int(summary['num_train_examples'])}",
    f"eval={int(summary['num_eval_examples'])}",
    f"avg_f1={summary['average_f1']:.4f}",
)
print("Wrote:", args.output_file)
