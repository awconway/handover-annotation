import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from inter_rater_agreement import (  # noqa: E402
    compute_agreement_rows,
    format_markdown_table,
)


OPTIONS = [{"id": "item_a"}, {"id": "item_b"}]
TOKENS = [
    {"text": "one", "start": 0, "end": 3, "id": 0, "ws": True},
    {"text": "two", "start": 4, "end": 7, "id": 1, "ws": True},
    {"text": "three", "start": 8, "end": 13, "id": 2, "ws": False},
]


def _row(annotator, task_hash, *, accepts, spans, tokens=None):
    return {
        "_annotator_id": annotator,
        "_task_hash": task_hash,
        "text": "one two three",
        "tokens": TOKENS if tokens is None else tokens,
        "options": OPTIONS,
        "accept": accepts,
        "spans": spans,
    }


def _span(start, end, token_start, token_end, label):
    return {
        "start": start,
        "end": end,
        "token_start": token_start,
        "token_end": token_end,
        "label": label,
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_compute_agreement_rows_supports_overlapping_token_labels(tmp_path):
    rows = [
        _row(
            "rn-a",
            1,
            accepts=["item_a"],
            spans=[
                _span(0, 7, 0, 1, "SITUATION"),
                _span(4, 13, 1, 2, "ASSESSMENT"),
                _span(8, 13, 2, 2, "Unknown fact"),
            ],
        ),
        _row(
            "rn-b",
            1,
            accepts=["item_a"],
            spans=[
                _span(0, 7, 0, 1, "SITUATION"),
                _span(4, 13, 1, 2, "ASSESSMENT"),
                _span(8, 13, 2, 2, "Unknown fact"),
            ],
        ),
        _row(
            "rn-a",
            2,
            accepts=["item_b"],
            spans=[
                _span(0, 3, 0, 0, "BACKGROUND"),
                _span(4, 7, 1, 1, "Vagueness"),
            ],
        ),
        _row(
            "rn-b",
            2,
            accepts=[],
            spans=[
                _span(0, 3, 0, 0, "BACKGROUND"),
                _span(4, 13, 1, 2, "Vagueness"),
            ],
        ),
    ]
    source = tmp_path / "paired.jsonl"
    _write_jsonl(source, rows)

    results = compute_agreement_rows(
        source,
        bootstrap_samples=50,
        seed=7,
    )

    assert [row["task"] for row in results] == [
        "Checklist content detection",
        "SBAR span labelling",
        "Uncertainty span labelling",
        "Unknown-fact span labelling",
    ]
    assert results[0]["n_transcripts"] == 2
    assert results[0]["n_decisions"] == 4
    assert results[0]["both_positive"] == 1
    assert results[1]["n_decisions"] == 24  # 2 documents × 3 tokens × 4 labels
    assert results[1]["both_positive"] == 5  # includes both overlapping labels
    assert results[1]["span_f1"] == pytest.approx(1.0)
    assert results[2]["span_f1"] == pytest.approx(1.0)
    assert results[2]["mean_matched_iou"] == pytest.approx(2 / 3)
    assert results[3]["span_f1"] == pytest.approx(1.0)

    table = format_markdown_table(results)
    assert "| Checklist content detection |" in table
    assert "| Unknown-fact span labelling |" in table
    assert "Cohen κ" in table


def test_compute_agreement_rows_rejects_token_mismatch(tmp_path):
    altered_tokens = [dict(token) for token in TOKENS]
    altered_tokens[0]["text"] = "changed"
    rows = [
        _row("rn-a", 1, accepts=[], spans=[]),
        _row("rn-b", 1, accepts=[], spans=[], tokens=altered_tokens),
    ]
    source = tmp_path / "mismatch.jsonl"
    _write_jsonl(source, rows)

    with pytest.raises(ValueError, match="Token mismatch"):
        compute_agreement_rows(source, bootstrap_samples=10)
