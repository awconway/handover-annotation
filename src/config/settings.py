import os

# Select exactly one model and one optimiser per run

MODEL_NAME = "gpt_5.1"  # options: see model_registry.py
OPTIMISER_NAME = "gepa_heavy_checklist"  # options: see optimiser_registry.py

DATA_FILE = "./annotated_data/db_20260129_tokenised.jsonl"

OUTPUT_MODEL_FILE = f"./compiled_programs/trained_{MODEL_NAME}_{OPTIMISER_NAME}.json"

def with_annotator_suffix(path: str, annotator_id: str | None) -> str:
    if not annotator_id:
        return path
    safe_id = annotator_id.replace("/", "_").replace("\\", "_")
    root, ext = os.path.splitext(path)
    return f"{root}_{safe_id}{ext}"


EVAL_RESULTS_FILE = f"./evals/eval_{MODEL_NAME}_{OPTIMISER_NAME}.jsonl"
