import argparse

from config.settings import (
    DATA_FILE,
    MODEL_NAME,
    OPTIMISER_NAME,
    OUTPUT_MODEL_FILE,
    with_annotator_suffix,
)
from data.dataset import prepare_dataset
from training.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotator-id",
        default=None,
        help="Filter examples by _annotator_id (e.g. handover_db-user1).",
    )
    return parser.parse_args()


args = parse_args()
trainset, valset = prepare_dataset(DATA_FILE, annotator_id=args.annotator_id)
output_model_file = with_annotator_suffix(OUTPUT_MODEL_FILE, args.annotator_id)

predictor = train(MODEL_NAME, OPTIMISER_NAME, trainset, valset)
predictor.save(output_model_file)
for name, pred in predictor.named_predictors():
    print("================================")
    print(f"Predictor: {name}")
    print("================================")
    print("Prompt:")
    print(pred.signature.instructions)
    print("*********************************")

print("Training complete. Saved to", output_model_file)
