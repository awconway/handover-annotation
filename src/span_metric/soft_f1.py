from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

Span = Tuple[int, int]  # (start_char, end_char) exclusive
GoldSpan = Dict[str, object]  # {"start": int, "end": int, "label": str}
PredItem = Dict[str, str]  # {"label": str, "quote": str}


def find_best_char_span(
    text: str, quote: str, fuzzy_threshold: float = 0.6
) -> Optional[Span]:
    """
    Try to find the best span in `text` corresponding to `quote`.
    1. Try exact match
    2. If not found, do a sliding-window fuzzy match using SequenceMatcher ratio.
    Returns (start, end) or None if no decent match found.
    """
    text_n = text
    quote_n = quote
    # text_n = normalize_text(text)
    # quote_n = normalize_text(quote)

    # exact
    idx = text_n.find(quote_n)
    if idx != -1:
        return (idx, idx + len(quote_n))

    # fuzzy sliding window
    best_ratio = 0.0
    best_span = None
    qlen = len(quote_n)
    if qlen == 0:
        return None

    # choose window sizes around quote length for robustness
    min_w = max(10, qlen - 30)
    max_w = min(len(text_n), qlen + 30)

    # We'll slide across text using a modest step to avoid heavy cost
    step = max(1, qlen // 4)
    for w in range(min_w, max_w + 1, 10):
        for start in range(0, max(1, len(text_n) - w + 1), step):
            window = text_n[start : start + w]
            ratio = SequenceMatcher(None, quote_n, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (start, start + w)

    if best_ratio >= fuzzy_threshold:
        return best_span
    return None


def span_iou(a: Span, b: Span) -> float:
    """IoU for two char spans."""
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def label_aware_soft_f1(
    text: str,
    gold_spans: List[GoldSpan],
    pred_items: List[PredItem],
    fuzzy_threshold: float = 0.6,
    iou_threshold: Optional[float] = None,
    require_label_match: bool = True,
) -> Dict:
    """
    Evaluate predicted quote items against gold spans.
    - text: full paragraph string
    - gold_spans: list of {"start": int, "end": int, "label": str}
    - pred_items: list of {"label": str, "quote": str}
    - fuzzy_threshold: used when mapping quotes -> char spans
    - iou_threshold: if set, a match is binary (1 if iou>=threshold), otherwise fractional IoU credit
    - require_label_match: if True, only consider pred/gold with same label
    Returns dictionary with precision, recall, f1, tp, counts and detailed matches.
    """
    text_n = text
    # text_n = normalize_text(text)

    # convert gold to tuple form + label
    golds = [((g["start"], g["end"]), g.get("label")) for g in gold_spans]

    # map preds to spans
    preds_with_spans: List[Tuple[Span, str, str]] = []  # (span, label, quote)
    for p in pred_items:
        span = find_best_char_span(text_n, p["quote"], fuzzy_threshold=fuzzy_threshold)
        preds_with_spans.append((span, p.get("label"), p.get("quote")))

    # Build score matrix (pred_idx x gold_idx) only if labels match (when required)
    scores = []  # rows per pred: list of (gold_idx, iou)
    for pi, (pspan, plabel, pquote) in enumerate(preds_with_spans):
        row = []
        if pspan is None:
            scores.append(row)
            continue
        for gi, (gspan, glabel) in enumerate(golds):
            if require_label_match and (plabel != glabel):
                # label mismatch -> zero score (but keep entry for completeness)
                row.append((gi, 0.0))
                continue
            s = span_iou(pspan, gspan)
            row.append((gi, s))
        scores.append(row)

    # Greedy matching by highest score, enforce 1-to-1
    flat = []
    for pi, row in enumerate(scores):
        for gi, s in row:
            if s > 0:
                flat.append((s, pi, gi))
    flat.sort(reverse=True, key=lambda x: x[0])

    used_preds = set()
    used_golds = set()
    matches = []  # (pred_idx, gold_idx, iou, credit)
    tp_total = 0.0

    for s, pi, gi in flat:
        if pi in used_preds or gi in used_golds:
            continue
        credit = 1.0 if (iou_threshold is not None and s >= iou_threshold) else s
        if credit > 0:
            matches.append({"pred_idx": pi, "gold_idx": gi, "iou": s, "credit": credit})
            tp_total += credit
            used_preds.add(pi)
            used_golds.add(gi)

    pred_count = len(pred_items)
    gold_count = len(gold_spans)
    precision = tp_total / pred_count if pred_count > 0 else 0.0
    recall = tp_total / gold_count if gold_count > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # Build debug-friendly structures
    detailed = {
        "preds": [
            {"idx": i, "label": p[1], "quote": p[2], "span": p[0]}
            for i, p in enumerate(preds_with_spans)
        ],
        "golds": [{"idx": i, "label": g[1], "span": g[0]} for i, g in enumerate(golds)],
        "matches": matches,
    }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_total,
        "pred_count": pred_count,
        "gold_count": gold_count,
        "detailed": detailed,
    }
