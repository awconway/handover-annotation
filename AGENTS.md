# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds all package code (src layout). Key areas: `src/config/` for settings and registries, `src/data/` for dataset loading, `src/training/` and `src/eval/` for training/evaluation, plus task modules like `src/checklist_task/`, `src/sbar_span_task/`, and `src/uncertain_span_task/`.
- `tests/` contains pytest files (e.g., `tests/test_tokeniser.py`).
- Data lives in JSONL files under `annotated_data/` and `unlabelled_data/`. Evaluation outputs are often in `evals/` and model artifacts may be stored in `compiled_programs/`.
- Top-level scripts (e.g., `run_eval.py`, `run_train_checklist.py`, `run_test_span.py`) are entry points for common workflows.

## Build, Test, and Development Commands
- `uv sync`: create or update the virtual environment from `pyproject.toml`.
- `uv pip install -e .`: install in editable mode so `src/` imports resolve cleanly.
- `uv run run_train_checklist.py`: run checklist training using `src/config/settings.py`.
- `uv run run_eval.py`: evaluate the last trained program and write JSONL results.
- `uv run pytest`: run the test suite (uses `pytest.ini` to set `pythonpath=src`).

## Coding Style & Naming Conventions
- Python code follows PEP 8 with 4-space indentation.
- Keep module names lowercase with underscores (e.g., `span_metric`), and test files as `test_*.py`.
- Prefer explicit, typed configuration in `src/config/settings.py` and registry updates in `src/config/model_registry.py` or `src/config/optimiser_registry.py`.

## Testing Guidelines
- Framework: `pytest`.
- Place new tests in `tests/` and follow existing patterns such as `test_soft_span_eval.py`.
- Run `uv run pytest` before submitting changes; add tests for new metrics, data transforms, or task logic.

## Commit & Pull Request Guidelines
- Commit history is minimal and not standardized (e.g., `init`, `inc`). Use short, imperative messages that describe the change.
- PRs should include: a concise description, relevant command output (e.g., `uv run pytest`), and links to related issues or data files when applicable.

## Configuration Tips
- Edit `src/config/settings.py` to switch models, optimisers, and data files. Training and evaluation scripts read from this configuration.
- Always run scripts via `uv run` to ensure the correct environment and spaCy model are used.
