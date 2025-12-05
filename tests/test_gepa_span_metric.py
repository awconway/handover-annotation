# test_gepa_span_metric.py
from types import SimpleNamespace

import dspy
import pytest

from gepa_span_metric import gepa_span_metric


def test_gepa_span_metric_exact_match():
    """Prediction matches gold spans exactly → F1 = 1."""
    example = {
        "text": "I’ve escalated to the medical registrar—awaiting review.",
        "gold_spans": [
            {"start": 0, "end": 56, "label": "Unknown fact"},
        ],
    }

    pred = SimpleNamespace(
        spans=[
            {
                "label": "Unknown fact",
                "quote": "I’ve escalated to the medical registrar—awaiting review.",
            }
        ]
    )

    result = gepa_span_metric(example, pred, pred_name="pred")
    assert result.score == pytest.approx(1.0)


def test_gepa_span_metric_partial_overlap():
    """Partial overlap span should produce a fractional F1 (0 < F1 < 1)."""
    example = {
        "text": "I’ve escalated to the medical registrar—awaiting review.",
        "gold_spans": [
            {"start": 0, "end": 56, "label": "Unknown fact"},
        ],
    }

    # Overlaps partially but not exact
    pred = SimpleNamespace(
        spans=[
            {
                "label": "Unknown fact",
                "quote": "escalated to the medical registrar",
            }
        ]
    )

    result = gepa_span_metric(example, pred, pred_name="pred")
    assert 0 < result.score < 1


def test_gepa_span_metric_label_mismatch():
    """
    GEPA should award F1=0 when labels don't match, even if spans overlap.
    """
    example = {
        "text": "I’ve escalated to the medical registrar—awaiting review.",
        "gold_spans": [
            {"start": 0, "end": 56, "label": "Unknown fact"},
        ],
    }

    pred = SimpleNamespace(
        spans=[
            {
                "label": "Procedural uncertainty",  # WRONG LABEL
                "quote": "I’ve escalated to the medical registrar—awaiting review.",
            }
        ]
    )

    result = gepa_span_metric(example, pred, pred_name="pred")
    assert result.score == 0.0


def test_gepa_span_metric_label_mismatch_feedback_present():
    """
    GEPA mode (with pred_name): returns a Prediction with score and feedback,
    even when score == 0.
    """
    example = {
        "text": "I’ve escalated to the medical registrar—awaiting review.",
        "gold_spans": [
            {"start": 0, "end": 56, "label": "Unknown fact"},
        ],
    }

    pred = SimpleNamespace(
        spans=[
            {
                "label": "Procedural uncertainty",  # WRONG LABEL
                "quote": "I’ve escalated to the medical registrar—awaiting review.",
            }
        ]
    )

    result = gepa_span_metric(example, pred, pred_name="span_module")

    # 1) We got a Prediction object
    assert isinstance(result, dspy.Prediction)

    # 2) The score is still 0.0
    assert result.score == pytest.approx(0.0)

    # 3) Feedback exists and is non-empty
    assert isinstance(result.feedback, str)
    assert result.feedback.strip() != ""

    # 4) (Optional) Check that the feedback mentions the relevant spans
    assert "Unknown fact" in result.feedback
    assert "Procedural uncertainty" in result.feedback
    assert "No matched spans" in result.feedback


def test_gepa_span_metric_multiple_preds_golds():
    """
    Ensure greedy matching behaves correctly with multiple spans.
    """
    text = (
        "I’ve escalated to the medical registrar—awaiting review. "
        "I’ve requested an IDC insertion—awaiting medical clearance."
    )

    example = {
        "text": text,
        "gold_spans": [
            {"start": 0, "end": 56, "label": "Unknown fact"},
            {"start": 55, "end": 125, "label": "Procedural uncertainty"},
        ],
    }

    pred = SimpleNamespace(
        spans=[
            {
                "label": "Unknown fact",
                "quote": "I’ve escalated to the medical registrar—awaiting review.",
            },
            {
                "label": "Procedural uncertainty",
                "quote": "requested an IDC insertion—awaiting",
            },
        ]
    )

    result = gepa_span_metric(example, pred, pred_name="pred")
    assert 0 < result.score <= 1.0  # soft F1 or exact


def test_gepa_span_metric_feedback_returned():
    """
    GEPA should return a Prediction object when pred_name is provided.
    """
    example = {
        "text": "Some text",
        "gold_spans": [],
    }

    pred = SimpleNamespace(spans=[])

    out = gepa_span_metric(example, pred, pred_name="span_module.predict")
    assert hasattr(out, "score")
    assert hasattr(out, "feedback")


def test_gepa_span_metric_no_spans_is_perfect():
    example = {
        "text": "Some text",
        "gold_spans": [],
    }
    pred = SimpleNamespace(spans=[])

    # Plain metric mode
    score = gepa_span_metric(example, pred)
    assert score == pytest.approx(1.0)

    # GEPA mode
    result = gepa_span_metric(example, pred, pred_name="span_module")
    assert isinstance(result, dspy.Prediction)
    assert result.score == pytest.approx(1.0)
    assert "desired behavior" in result.feedback
