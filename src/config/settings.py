# Select exactly one model and one optimiser per run

MODEL_NAME = "gpt_5.1"  # options: see model_registry.py
OPTIMISER_NAME = "gepa_heavy_checklist"  # options: see optimiser_registry.py

DATA_FILE = "./annotated_data/db_20251127_tokenised.jsonl"

OUTPUT_MODEL_FILE = f"./compiled_programs/trained_{MODEL_NAME}_{OPTIMISER_NAME}.json"


EVAL_RESULTS_FILE = f"./evals/eval_{MODEL_NAME}_{OPTIMISER_NAME}.jsonl"
