import argparse

from config.dspy_settings import configure_dspy
from config.model_registry import load_model
from config.optimiser_registry import OPTIM_REGISTRY, load_optimiser
from data.dataset import (
    UNKNOWN_FACT_LABEL,
    prepare_dataset_unknown_fact_binary_span,
)
from training.run_logging import enable_local_training_file_logging
from unknown_fact_binary_span_task.signatures import build_predictor

DATA_FILE = "./annotated_data/db_20260129_tokenised.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        default=DATA_FILE,
        help="Path to tokenised Prodigy JSONL data.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model registry key (see src/config/model_registry.py).",
    )
    parser.add_argument(
        "--optimiser-name",
        required=True,
        help="Optimiser registry key (see src/config/optimiser_registry.py).",
    )
    parser.add_argument(
        "--output-model-file",
        required=True,
        help="Path to save the trained program.",
    )
    parser.add_argument(
        "--annotator-id",
        default=None,
        help="Filter examples by _annotator_id (e.g. handover_db-user1).",
    )
    parser.add_argument(
        "--gepa-log-dir",
        default=None,
        help="Optional GEPA run directory used for checkpoint/resume.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default=None,
        help=(
            "Optional reasoning effort for OpenAI GPT models. "
            "When omitted, model defaults are used."
        ),
    )
    return parser.parse_args()


def _summarise_unknown_fact_split(name: str, examples: list) -> None:
    total = len(examples)
    positive_examples = 0
    gold_spans = 0

    for ex in examples:
        spans = ex["gold_spans"]
        unknown_spans = [
            span for span in spans if span.get("label") == UNKNOWN_FACT_LABEL
        ]
        if unknown_spans:
            positive_examples += 1
        gold_spans += len(unknown_spans)

    print(
        f"{name}: examples={total}, positive_examples={positive_examples}, "
        f"gold_{UNKNOWN_FACT_LABEL.replace(' ', '_').lower()}_spans={gold_spans}"
    )


args = parse_args()
enable_local_training_file_logging(__file__)
allowed_span_optimisers = {
    name for name in OPTIM_REGISTRY if name == "none" or name.endswith("_span")
}
if args.optimiser_name not in allowed_span_optimisers:
    raise ValueError(
        "Unsupported optimiser for unknown-fact binary span. "
        "Use a span optimiser like 'gepa_light_span' or 'gepa_heavy_span'."
    )

trainset, valset = prepare_dataset_unknown_fact_binary_span(
    args.data_file, annotator_id=args.annotator_id
)
_summarise_unknown_fact_split("train", trainset)
_summarise_unknown_fact_split("val", valset)
output_model_file = args.output_model_file

if args.reasoning_effort is not None:
    print(f"Reasoning effort override enabled: {args.reasoning_effort}.")
lm = load_model(args.model_name, reasoning_effort=args.reasoning_effort)
configure_dspy(lm)

predictor = build_predictor()
optimiser_fn = load_optimiser(args.optimiser_name)
predictor = optimiser_fn(
    predictor,
    trainset,
    valset,
    gepa_log_dir=args.gepa_log_dir,
)

predictor.save(output_model_file)
print("Training complete. Saved to", output_model_file)
