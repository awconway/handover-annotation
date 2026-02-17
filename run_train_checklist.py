import argparse

from data.dataset import prepare_dataset
from training.trainer import train

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
    return parser.parse_args()


args = parse_args()
trainset, valset = prepare_dataset(args.data_file, annotator_id=args.annotator_id)
output_model_file = args.output_model_file

predictor = train(
    args.model_name,
    args.optimiser_name,
    trainset,
    valset,
    gepa_log_dir=args.gepa_log_dir,
)
predictor.save(output_model_file)
for name, pred in predictor.named_predictors():
    print("================================")
    print(f"Predictor: {name}")
    print("================================")
    print("Prompt:")
    print(pred.signature.instructions)
    print("*********************************")

print("Training complete. Saved to", output_model_file)
