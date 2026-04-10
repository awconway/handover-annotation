from __future__ import annotations

import ast
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = REPO_ROOT / "prod"
PACKAGE_ROOT = PROD_ROOT / "handover_prod"


def test_prod_package_does_not_import_repo_src_modules():
    forbidden_roots = {
        "checklist_task",
        "config",
        "data",
        "eval",
        "sbar_span_task",
        "span_metric",
        "training",
        "uncertain_binary_span_task",
        "uncertain_span_task",
        "unknown_fact_binary_span_task",
    }

    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in forbidden_roots, (
                        f"{path} imports forbidden module {alias.name!r}"
                    )
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden_roots, (
                    f"{path} imports forbidden module {node.module!r}"
                )


def test_prod_registry_loads_compiled_programs_without_repo_src():
    sys.path.insert(0, str(PROD_ROOT))
    try:
        from handover_prod.config import Settings
        from handover_prod.predictors import PredictorRegistry

        settings = Settings.from_env()
        registry = PredictorRegistry.from_settings(settings, configure_lm=False)

        assert registry.available_tasks() == ["checklist", "sbar", "unknown_fact"]

        metadata = registry.task_metadata()
        assert Path(metadata["checklist"]["program_path"]).name == (
            "checklist_gpt_5-2_consensus.json"
        )
        assert Path(metadata["sbar"]["program_path"]).name == (
            "sbar_span_gpt5-2_consensus.json"
        )
        assert Path(metadata["unknown_fact"]["program_path"]).name == (
            "unknown_fact_binary_gpt5-2_user2_v2.json"
        )
    finally:
        sys.path.pop(0)
