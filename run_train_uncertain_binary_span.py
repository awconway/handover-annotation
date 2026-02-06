import argparse

from config.dspy_settings import configure_dspy
from config.model_registry import load_model
from config.optimiser_registry import OPTIM_REGISTRY, load_optimiser
from data.dataset import prepare_dataset_uncertainty_binary_span
from uncertain_binary_span_task.signatures import build_predictor

DATA_FILE = "./annotated_data/db_20260129_tokenised.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


args = parse_args()
allowed_span_optimisers = {
    name for name in OPTIM_REGISTRY if name == "none" or name.endswith("_span")
}
if args.optimiser_name not in allowed_span_optimisers:
    raise ValueError(
        "Unsupported optimiser for uncertainty binary span. "
        "Use a span optimiser like 'gepa_light_span' or 'gepa_heavy_span'."
    )

trainset, valset = prepare_dataset_uncertainty_binary_span(
    DATA_FILE, annotator_id=args.annotator_id
)
output_model_file = args.output_model_file

lm = load_model(args.model_name)
configure_dspy(lm)

predictor = build_predictor()
optimiser_fn = load_optimiser(args.optimiser_name)
predictor = optimiser_fn(predictor, trainset, valset)

predictor.save(output_model_file)
print("Training complete. Saved to", output_model_file)
