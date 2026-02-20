import argparse
import json

from config.dspy_settings import configure_dspy
from config.model_registry import load_model
from data.dataset import prepare_dataset_sbar_span
from sbar_span_task.signatures import build_predictor

DEFAULT_DATA_FILE = "./annotated_data/db_20260129_tokenised_consensus.jsonl"
DEFAULT_MODEL_FILE = "./compiled_programs/sbar_span_gpt5-2_consensus.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a compiled SBAR span program, run it on the first example in the "
            "deterministic test split, and print the prompt/messages sent to the model."
        )
    )
    parser.add_argument(
        "--data-file",
        default=DEFAULT_DATA_FILE,
        help="Path to consensus JSONL data.",
    )
    parser.add_argument(
        "--output-model-file",
        default=DEFAULT_MODEL_FILE,
        help="Path to the compiled DSPy program JSON.",
    )
    parser.add_argument(
        "--annotator-id",
        default="consensus",
        help="Annotator filter used when constructing the deterministic split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _, testset = prepare_dataset_sbar_span(args.data_file, annotator_id=args.annotator_id)
    if not testset:
        raise ValueError("No examples in test split for the requested data/annotator.")

    example = testset[0]

    lm = load_model("gpt_5.2")
    configure_dspy(lm)

    predictor = build_predictor()
    predictor.load(args.output_model_file)

    prediction = predictor(text=example.text)
    print("Prediction:")
    print(prediction)

    if not lm.history:
        print("\nNo LM history captured, cannot print prompt.")
        return

    last_request = lm.history[-1]
    messages = last_request.get("messages")
    prompt = last_request.get("prompt")

    print("\nPrompt used (raw request messages):")
    if messages:
        print(json.dumps(messages, indent=2, ensure_ascii=False))
    else:
        print(prompt if isinstance(prompt, str) else "<empty prompt>")


if __name__ == "__main__":
    main()
