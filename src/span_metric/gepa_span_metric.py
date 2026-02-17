import json
import logging
import os

import dspy

# --- import your metric ---
from .soft_f1 import label_aware_soft_f1


logger = logging.getLogger(__name__)
DEBUG_LOGS_ENABLED = os.getenv("GEPA_SPAN_METRIC_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _debug_log(message: str) -> None:
    if DEBUG_LOGS_ENABLED:
        logger.info(message)


def _extract_pred_items(pred):
    items = getattr(pred, "pred_spans", None)
    if items is None and isinstance(pred, dict):
        items = pred.get("pred_spans")

    if not isinstance(items, list):
        return []

    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        quote = item.get("quote")
        if not isinstance(label, str) or not isinstance(quote, str):
            continue
        cleaned.append({"label": label, "quote": quote})
    return cleaned


def format_span(span):
    if span is None:
        return "None"
    return f"[{span[0]}, {span[1]}]"


def _extract_span_quote(text, span):
    """
    Safely extract quote text for a [start, end] char span.
    Returns None when span is missing/invalid.
    """
    if span is None or not isinstance(span, (list, tuple)) or len(span) != 2:
        return None

    start, end = span
    if not isinstance(start, int) or not isinstance(end, int):
        return None

    text_len = len(text)
    start = max(0, min(start, text_len))
    end = max(0, min(end, text_len))
    if end <= start:
        return ""
    return text[start:end]


def _format_quote(quote):
    if quote is None:
        return "None"
    return json.dumps(quote, ensure_ascii=False)


def build_feedback(text, golds, preds, detailed):
    """
    Build natural-language feedback for GEPA based on the detailed output
    from label_aware_soft_f1().
    """
    fb = []

    fb.append("Span Evaluation Feedback:\n")

    # --- GOLD SPANS ---
    fb.append("Gold spans:\n")
    for g in detailed["golds"]:
        gold_quote = _extract_span_quote(text, g["span"])
        fb.append(
            f"  - Gold #{g['idx']} — label='{g['label']}', "
            f"quote={_format_quote(gold_quote)}, span={format_span(g['span'])}"
        )

    # --- PRED SPANS ---
    fb.append("\nPredicted spans:\n")
    for p in detailed["preds"]:
        fb.append(
            f"  - Pred #{p['idx']} — label='{p['label']}', "
            f"quote={_format_quote(p['quote'])}, mapped_span={format_span(p['span'])}"
        )

    # --- MATCHES ---
    fb.append("\nMatches (IoU-based):\n")
    matches = detailed.get("matches", [])
    golds = detailed.get("golds", [])
    preds = detailed.get("preds", [])

    if not matches:
        if golds and not preds:
            fb.append(
                "  - No matched spans because the model did not predict any spans, "
                f"while there are {len(golds)} gold spans."
            )
        elif preds and not golds:
            fb.append(
                "  - No matched spans because there are no gold spans, "
                f"but the model predicted {len(preds)} spans."
            )
        elif golds and preds:
            fb.append(
                "  - No matched spans: predicted spans did not overlap sufficiently "
                "with any gold spans (IoU below threshold and/or label mismatch)."
            )
        else:
            fb.append("  - No spans to compare (no gold spans and no predicted spans).")
    else:
        for m in matches:
            fb.append(
                f"  - Pred #{m['pred_idx']} ↔ Gold #{m['gold_idx']} "
                f"(IoU={m['iou']:.3f})"
            )

    # --- UNMATCHED GOLD ---
    matched_golds = {m["gold_idx"] for m in detailed["matches"]}
    unmatched_gold = [g for g in detailed["golds"] if g["idx"] not in matched_golds]

    if unmatched_gold:
        fb.append("\nUnmatched gold spans:")
        for g in unmatched_gold:
            gold_quote = _extract_span_quote(text, g["span"])
            fb.append(
                f"  - Gold #{g['idx']} (label='{g['label']}', "
                f"quote={_format_quote(gold_quote)}, span={format_span(g['span'])}) was not predicted."
            )

    # --- UNMATCHED PREDS ---
    matched_preds = {m["pred_idx"] for m in detailed["matches"]}
    unmatched_preds = [p for p in detailed["preds"] if p["idx"] not in matched_preds]

    if unmatched_preds:
        fb.append("\nUnmatched predicted spans:")
        for p in unmatched_preds:
            fb.append(
                f"  - Pred #{p['idx']} (label='{p['label']}', "
                f"quote={_format_quote(p['quote'])}) did not match any gold span."
            )

    # Final summary / guidance
    if detailed["golds"] or detailed["preds"]:
        fb.append(
            "\nThink about improving extraction boundaries or label selection to improve IoU-based matching."
        )
    else:
        fb.append(
            "\nNo spans were required and none were predicted; this is the desired behavior."
        )

    feedback_str = "\n".join(fb)
    _debug_log(
        "GEPA span metric feedback generated "
        f"(chars={len(feedback_str)}, golds={len(detailed.get('golds', []))}, preds={len(detailed.get('preds', []))})"
    )

    return feedback_str


def gepa_span_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA-compatible metric wrapper around label_aware_soft_f1.

    example: dict containing:
        - "text": full input string
        - "gold_spans": list of {"start":..., "end":..., "label":...}

    pred: LM output containing:
        - pred.spans = [{"label":..., "quote":...}, ...]

    Returns:
        - If pred_name is None: return float score only
        - Else: return dspy.Prediction(score=..., feedback=...)
    """

    text = example["text"]
    gold_spans = example["gold_spans"]
    # Ensure prediction has the right structure
    pred_items = _extract_pred_items(pred)

    if DEBUG_LOGS_ENABLED:
        _debug_log(
            "GEPA span metric input: "
            f"text_chars={len(text)}, gold_count={len(gold_spans)}, pred_count={len(pred_items)}"
        )

    # Run span F1 evaluator
    out = label_aware_soft_f1(
        text=text,
        gold_spans=gold_spans,
        pred_items=pred_items,
        fuzzy_threshold=0.6,
        iou_threshold=None,
        require_label_match=True,
    )

    score = out["f1"]

    # Expose all computed metrics on the prediction so eval JSONL can include them.
    # (The metric function must still return a numeric score for dspy.Evaluate.)
    try:
        pred["span_metrics"] = out
    except Exception:
        # If `pred` isn't dictlike for some reason, skip attaching metrics.
        pass

    # --- special-case: no golds and no preds => perfect behavior ---
    if not out["detailed"]["golds"] and not out["detailed"]["preds"]:
        score = 1.0

    if DEBUG_LOGS_ENABLED:
        detailed = out["detailed"]
        mapped_preds = sum(1 for p in detailed["preds"] if p["span"] is not None)
        _debug_log(
            "GEPA span metric output: "
            f"precision={out['precision']:.4f}, recall={out['recall']:.4f}, f1={score:.4f}, tp={out['tp']:.4f}, "
            f"matches={len(detailed['matches'])}, mapped_preds={mapped_preds}/{len(detailed['preds'])}, "
            f"gold_count={out['gold_count']}, pred_count={out['pred_count']}"
        )

    if pred_name is None:
        # GEPA global scoring
        return score

    # GEPA wants textual feedback targeted at the module
    feedback = build_feedback(
        text=text,
        golds=gold_spans,
        preds=pred_items,
        detailed=out["detailed"],
    )

    return dspy.Prediction(score=score, feedback=feedback, metrics=out)
