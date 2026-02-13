# DSpy handover

This project trains and evaluates DSPy pipelines for data extraction from clinical handover transcripts.  

The project uses:

- uv for environment and runtime management
- src/ layout for clean imports
- modular components for models, optimisers, data loading, training and eval

============================================================
1. Project Structure
============================================================

project/
  pyproject.toml
  run_train_checklist.py
  run_eval.py
  run_eval_checklist.py
  run_eval_sbar_span.py
  src/
    checklist_task/
      labels.py
      metrics.py
      signatures.py
    config/
      settings.py
      model_registry.py
      optimiser_registry.py
    data/
      dataset.py
    training/
      trainer.py
    eval/
      evaluator.py

============================================================
2. Installation
============================================================

2.1. Install uv (if not already installed)

curl -LsSf https://astral.sh/uv/install.sh | sh

or macOS:

brew install uv

2.2. Create / sync the environment

uv sync

2.3. Install the project in editable mode (recommended)

uv pip install -e .

This makes the src/ packages importable.

============================================================
3. Configuration
============================================================

Training and eval scripts take required CLI flags instead of settings.py defaults. Paths are used exactly as provided (no auto suffixing).

Fixed data file:

./annotated_data/db_20260129_tokenised.jsonl

Training flags (required):
- `--model-name`: key from `src/config/model_registry.py`
- `--optimiser-name`: key from `src/config/optimiser_registry.py`
- `--output-model-file`: path to save the trained program

Evaluation flags (required):
- `--output-model-file`: path to the trained program to load
- `--eval-results-file`: path to write JSONL results

There is no automatic looping — exactly one model and one optimiser run per training.

============================================================
4. Running Training
============================================================

From the project root:

uv run run_train_checklist.py --model-name gpt_nano --optimiser-name gepa_heavy_checklist --output-model-file ./compiled_programs/trained_gpt_nano_gepa_heavy_checklist.json

Optional:

uv run run_train_checklist.py --model-name gpt_nano --optimiser-name gepa_heavy_checklist --output-model-file ./compiled_programs/trained_gpt_nano_gepa_heavy_checklist.json --annotator-id handover_db-user1


This:

- loads the model from the registry  
- loads the optimiser from the registry  
- trains on the train split  
- saves the program to `--output-model-file`  

Output example:

Training complete. Saved to trained_model.json

============================================================
5. Running Evaluation
============================================================

After training (checklist):

uv run run_eval.py --output-model-file ./compiled_programs/trained_gpt_nano_gepa_heavy_checklist.json --eval-results-file ./evals/eval_gpt_nano_gepa_heavy_checklist.jsonl

Optional:

uv run run_eval.py --output-model-file ./compiled_programs/trained_gpt_nano_gepa_heavy_checklist.json --eval-results-file ./evals/eval_gpt_nano_gepa_heavy_checklist.jsonl --annotator-id handover_db-user1
uv run run_eval.py --output-model-file ./compiled_programs/trained_gpt_nano_gepa_heavy_checklist.json --eval-results-file ./evals/eval_gpt_nano_gepa_heavy_checklist.jsonl --annotator-id handover_db-user1 --use-all

Flag meanings:
- `--output-model-file`: model file to load.
- `--eval-results-file`: JSONL output path.
- `--annotator-id`: filter examples to a single annotator (matches `_annotator_id` in the JSONL).
- `--use-all`: evaluate on all matching examples (no 75/25 train-test split).

If you want per-annotator outputs, pass a distinct `--eval-results-file` (and/or `--output-model-file` for training).

Checklist eval with checklist-specific metrics:

uv run run_eval_checklist.py --output-model-file ./compiled_programs/trained_gpt_nano_gepa_heavy_checklist.json --eval-results-file ./evals/eval_gpt_nano_gepa_heavy_checklist.jsonl
uv run run_eval_checklist.py --output-model-file ./compiled_programs/trained_gpt_nano_gepa_heavy_checklist.json --eval-results-file ./evals/eval_gpt_nano_gepa_heavy_checklist.jsonl --annotator-id handover_db-user1 --use-all

SBAR span eval:

uv run run_eval_sbar_span.py

Note: run_eval_sbar_span.py currently uses the SBAR span dataset and the configured LM without loading a trained program.

SBAR LangExtract experiment (few-shot extraction from SBAR spans):

uv run run_experiment_sbar_langextract.py --model-id gemini-2.5-flash --annotator-id handover_db-user1 --train-examples 24 --eval-examples 20

The experiment uses the same IoU span metric as `src/span_metric/gepa_span_metric.py`.

Use the same test split as `prepare_dataset_sbar_span`:

uv run run_experiment_sbar_langextract.py --model-id gpt-5.2 --use-dataset-test-split

OpenAI-style settings (matching LangExtract docs):

OPENAI_API_KEY=... uv run run_experiment_sbar_langextract.py --model-id gpt-5.2 --fence-output --no-use-schema-constraints

If prompt alignment warnings are noisy, turn them off:

uv run run_experiment_sbar_langextract.py --prompt-validation-level off

If you want to hide LangExtract progress lines:

uv run run_experiment_sbar_langextract.py --no-show-progress

If you only want to validate data loading and output formatting without API calls:

uv run run_experiment_sbar_langextract.py --dry-run

run_eval.py and run_eval_checklist.py:

- loads `--output-model-file`  
- prepares the test split  
- runs evaluation  
- writes `--eval-results-file`  

Output example:

Evaluation complete. Score: 0.72

Each JSONL row contains:

{
  "example": { ... },
  "prediction": { ... },
  "score": 0.85
}

============================================================
6. Adding Models or Optimisers
============================================================

To add a model:
  edit src/config/model_registry.py

To add an optimiser:
  edit src/config/optimiser_registry.py

Then pass it via `--model-name` or `--optimiser-name`.

============================================================
7. Notes
============================================================

- Always run scripts via uv run for correct environment activation.
- src/ layout ensures clean imports.
- Trained DSPy programs are saved/loaded via .save() and .load().

============================================================
8. Reproducible Checklist Eval Analysis
============================================================

To generate reproducible error-analysis artifacts (summary JSON + label CSV + bucket CSV + per-example error CSV):

uv run python analysis/analyze_checklist_eval.py ./evals/eval_checklist_consensus_gpt_5_2_test.jsonl

By default, outputs are written to:

./evals/eval_checklist_consensus_gpt_5_2_test_analysis/

Optional:

- `--out-dir <path>` to choose a different output directory.
- `--top-k <int>` to change how many top FN/FP labels and FN->FP pairs are reported.

============================================================
