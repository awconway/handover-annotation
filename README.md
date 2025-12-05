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
  run_train.py
  run_eval.py
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

Edit:

src/config/settings.py

Example:

MODEL_NAME = "gpt_nano"           # gpt_nano, gpt_mini, gpt_o3
OPTIMISER_NAME = "mipro_light"    # none, mipro_light, mipro_heavy
DATA_FILE = "./annotated_data/db_20251120_tokenised.jsonl"
OUTPUT_MODEL_FILE = "trained_model.json"

Change the model or optimiser by modifying those two lines.

There is no automatic looping — exactly one model and one optimiser run per training.

============================================================
4. Running Training
============================================================

From the project root:

uv run run_train.py

This:

- loads the model from settings.py  
- loads the optimiser  
- trains on the train split  
- saves the program to OUTPUT_MODEL_FILE  

Output example:

Training complete. Saved to trained_model.json

============================================================
5. Running Evaluation
============================================================

After training:

uv run run_eval.py

This:

- loads OUTPUT_MODEL_FILE  
- prepares the test split  
- runs evaluation  
- writes eval_results.jsonl  

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

Then set it in settings.py.

============================================================
7. Notes
============================================================

- Always run scripts via uv run for correct environment activation.
- src/ layout ensures clean imports.
- Trained DSPy programs are saved/loaded via .save() and .load().

============================================================
