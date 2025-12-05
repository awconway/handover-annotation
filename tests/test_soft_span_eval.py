# test_span_eval.py
import pytest

from sbar_span_task.metric_gepa import label_aware_soft_f1

# sample paragraph (use the same text used in your earlier examples)
PARAGRAPH = (
    "Patient presents following a pacing procedure. "
    "I’ve escalated to the medical registrar—awaiting review. "
    "I’ve requested an IDC insertion—awaiting medical clearance due to the recent pacing procedure."
)

GOLD = [
    # single gold span covering the first 'escalated...' sentence (example start/end indexes)
    {"start": 31, "end": 85, "label": "SITUATION"},
]


def test_label_mismatch_counts_as_nonmatch():
    """If labels differ, there should be no TP even if quotes overlap."""
    preds = [
        {
            "label": "Procedural uncertainty",
            "quote": "I’ve escalated to the medical registrar—awaiting review.",
        },
    ]
    res = label_aware_soft_f1(
        PARAGRAPH, GOLD, preds, iou_threshold=None, require_label_match=True
    )
    # Since label mismatches, no true positives should be counted
    assert res["tp"] == 0 or res["f1"] == 0.0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0


def test_label_match_partial_overlap_has_positive_f1():
    """If labels match and there is partial overlap, IoU should give positive TP -> f1 > 0"""
    preds = [
        # shorter quoted substring (partial) but with matching label
        {
            "label": "SITUATION",
            "quote": "I’ve escalated to the medical registrar—awaiting",
        },
    ]
    res = label_aware_soft_f1(
        PARAGRAPH, GOLD, preds, iou_threshold=None, require_label_match=True
    )
    assert res["tp"] > 0  # fractional IoU credit
    assert res["f1"] > 0


def test_exact_match_full_credit():
    """Exact quote + same label should yield full credit -> f1 == 1"""
    preds = [
        {
            "label": "SITUATION",
            "quote": "I’ve escalated to the medical registrar—awaiting review.",
        },
    ]
    res = label_aware_soft_f1(
        PARAGRAPH, GOLD, preds, iou_threshold=0.5, require_label_match=True
    )
    # With iou_threshold=0.5, exact or near-exact match should count as credit=1
    assert res["tp"] >= 1.0
    assert res["precision"] == pytest.approx(1.0, rel=1e-6)
    assert res["recall"] == pytest.approx(1.0, rel=1e-6)
    assert res["f1"] == pytest.approx(1.0, rel=1e-6)
