"""Pre-consensus inter-rater agreement for the handover annotations.

The analysis pairs the two RN annotations by ``_task_hash`` and treats each
transcript as the independent sampling unit. Checklist agreement is pooled over
transcript-by-item binary decisions. Span-task agreement is pooled over
token-by-label binary decisions, which preserves overlapping labels.

For span tasks, token-level agreement is complemented by label-aware span
matching. Two spans match when they have the same label and any character
overlap; matches are assigned greedily in descending intersection-over-union
(IoU) order. Confidence intervals use a transcript-cluster bootstrap.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path("annotated_data/db_20260129_tokenised.jsonl")
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_SEED = 339

SBAR_LABELS = (
    "SITUATION",
    "BACKGROUND",
    "ASSESSMENT",
    "RECOMMENDATION",
)
UNCERTAINTY_LABELS = (
    "Vagueness",
    "Hedging",
    "Unknown fact",
    "Indefinite timing",
    "Source uncertainty",
    "Procedural uncertainty",
    "Responsibility uncertainty",
)
UNKNOWN_FACT_LABELS = ("Unknown fact",)

BinaryCounts = np.ndarray
SpanCounts = np.ndarray
AgreementRow = dict[str, Any]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _pair_annotations(
    rows: Sequence[dict[str, Any]],
) -> tuple[str, str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    by_annotator: dict[str, dict[Any, dict[str, Any]]] = defaultdict(dict)

    for row in rows:
        annotator = row.get("_annotator_id")
        task_hash = row.get("_task_hash")
        if not isinstance(annotator, str) or not annotator:
            raise ValueError("Every row must have a non-empty _annotator_id")
        if task_hash is None:
            raise ValueError("Every row must have a _task_hash")
        if task_hash in by_annotator[annotator]:
            raise ValueError(
                f"Duplicate row for annotator={annotator!r}, "
                f"_task_hash={task_hash!r}"
            )
        by_annotator[annotator][task_hash] = row

    annotators = sorted(by_annotator)
    if len(annotators) != 2:
        raise ValueError(
            f"Expected exactly two annotators, found {len(annotators)}: {annotators}"
        )

    annotator_a, annotator_b = annotators
    hashes_a = set(by_annotator[annotator_a])
    hashes_b = set(by_annotator[annotator_b])
    if hashes_a != hashes_b:
        missing_for_a = sorted(hashes_b - hashes_a, key=str)
        missing_for_b = sorted(hashes_a - hashes_b, key=str)
        raise ValueError(
            "Annotators do not have identical task coverage; "
            f"missing for {annotator_a}: {missing_for_a[:5]}, "
            f"missing for {annotator_b}: {missing_for_b[:5]}"
        )

    pairs = [
        (by_annotator[annotator_a][task_hash], by_annotator[annotator_b][task_hash])
        for task_hash in sorted(hashes_a, key=str)
    ]
    for row_a, row_b in pairs:
        if row_a.get("text") != row_b.get("text"):
            raise ValueError(
                f"Text mismatch for _task_hash={row_a.get('_task_hash')!r}"
            )
        if row_a.get("tokens") != row_b.get("tokens"):
            raise ValueError(
                f"Token mismatch for _task_hash={row_a.get('_task_hash')!r}"
            )

    return annotator_a, annotator_b, pairs


def _checklist_labels(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[str, ...]:
    if not pairs:
        raise ValueError("No paired annotations were found")

    first_options = pairs[0][0].get("options") or []
    labels = tuple(option.get("id") for option in first_options)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Checklist options must contain non-empty string ids")
    if len(labels) != len(set(labels)):
        raise ValueError("Checklist option ids must be unique")

    expected = set(labels)
    for row_a, row_b in pairs:
        for row in (row_a, row_b):
            row_labels = {
                option.get("id")
                for option in (row.get("options") or [])
                if isinstance(option, dict)
            }
            if row_labels != expected:
                raise ValueError(
                    "Checklist option ids differ across annotation rows for "
                    f"_task_hash={row.get('_task_hash')!r}"
                )
            unknown_accepts = set(row.get("accept") or []) - expected
            if unknown_accepts:
                raise ValueError(
                    f"Unknown checklist selections {sorted(unknown_accepts)} for "
                    f"_task_hash={row.get('_task_hash')!r}"
                )
    return labels


def _token_label_sets(row: dict[str, Any]) -> list[set[str]]:
    tokens = row.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError(
            f"tokens must be a list for _task_hash={row.get('_task_hash')!r}"
        )
    labels_by_token: list[set[str]] = [set() for _ in tokens]

    for span in row.get("spans") or []:
        if not isinstance(span, dict):
            raise ValueError("Every span must be a JSON object")
        token_start = span.get("token_start")
        token_end = span.get("token_end")
        label = span.get("label")
        if (
            not isinstance(token_start, int)
            or not isinstance(token_end, int)
            or not isinstance(label, str)
            or not (0 <= token_start <= token_end < len(tokens))
        ):
            raise ValueError(
                "Invalid token span for "
                f"_task_hash={row.get('_task_hash')!r}: {span!r}"
            )
        for token_index in range(token_start, token_end + 1):
            labels_by_token[token_index].add(label)

    return labels_by_token


def _binary_counts(
    ratings_a: Iterable[bool], ratings_b: Iterable[bool]
) -> BinaryCounts:
    values_a = np.asarray(list(ratings_a), dtype=bool)
    values_b = np.asarray(list(ratings_b), dtype=bool)
    if values_a.shape != values_b.shape:
        raise ValueError("Rater decision arrays must have the same shape")

    return np.array(
        (
            np.sum(values_a & values_b),
            np.sum(values_a & ~values_b),
            np.sum(~values_a & values_b),
            np.sum(~values_a & ~values_b),
        ),
        dtype=float,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _binary_metrics(counts: BinaryCounts) -> dict[str, float]:
    both_positive, a_only, b_only, both_negative = map(float, counts)
    total = both_positive + a_only + b_only + both_negative
    observed_agreement = _safe_ratio(both_positive + both_negative, total)
    prevalence_a = _safe_ratio(both_positive + a_only, total)
    prevalence_b = _safe_ratio(both_positive + b_only, total)
    expected_agreement = (
        prevalence_a * prevalence_b
        + (1.0 - prevalence_a) * (1.0 - prevalence_b)
    )

    return {
        "observed_agreement": observed_agreement,
        "cohen_kappa": _safe_ratio(
            observed_agreement - expected_agreement,
            1.0 - expected_agreement,
        ),
        "positive_agreement": _safe_ratio(
            2.0 * both_positive,
            2.0 * both_positive + a_only + b_only,
        ),
        "negative_agreement": _safe_ratio(
            2.0 * both_negative,
            2.0 * both_negative + a_only + b_only,
        ),
    }


def _span_iou(span_a: tuple[int, int, str], span_b: tuple[int, int, str]) -> float:
    intersection = max(
        0,
        min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]),
    )
    union = max(span_a[1], span_b[1]) - min(span_a[0], span_b[0])
    return intersection / union if union > 0 else 0.0


def _selected_spans(
    row: dict[str, Any], allowed_labels: frozenset[str]
) -> list[tuple[int, int, str]]:
    selected: list[tuple[int, int, str]] = []
    for span in row.get("spans") or []:
        label = span.get("label")
        if label not in allowed_labels:
            continue
        start = span.get("start")
        end = span.get("end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(label, str)
            or start < 0
            or end <= start
        ):
            raise ValueError(
                "Invalid character span for "
                f"_task_hash={row.get('_task_hash')!r}: {span!r}"
            )
        selected.append((start, end, label))
    return selected


def _match_spans(
    spans_a: Sequence[tuple[int, int, str]],
    spans_b: Sequence[tuple[int, int, str]],
) -> SpanCounts:
    candidates: list[tuple[float, int, int]] = []
    for index_a, span_a in enumerate(spans_a):
        for index_b, span_b in enumerate(spans_b):
            if span_a[2] != span_b[2]:
                continue
            overlap = _span_iou(span_a, span_b)
            if overlap > 0:
                candidates.append((overlap, index_a, index_b))

    candidates.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched_ious: list[float] = []
    for overlap, index_a, index_b in candidates:
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matched_ious.append(overlap)

    return np.array(
        (
            len(spans_a),
            len(spans_b),
            len(matched_ious),
            sum(matched_ious),
        ),
        dtype=float,
    )


def _span_metrics(counts: SpanCounts) -> dict[str, float]:
    count_a, count_b, matched_count, iou_sum = map(float, counts)
    return {
        "span_f1": _safe_ratio(
            2.0 * matched_count,
            count_a + count_b,
        ),
        "mean_matched_iou": _safe_ratio(iou_sum, matched_count),
    }


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        return math.nan, math.nan
    low, high = np.percentile(finite_values, (2.5, 97.5))
    return float(low), float(high)


def _bootstrap_binary_metrics(
    document_counts: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, tuple[float, float]]:
    sampled_counts = document_counts[bootstrap_indices].sum(axis=1)
    both_positive = sampled_counts[:, 0]
    a_only = sampled_counts[:, 1]
    b_only = sampled_counts[:, 2]
    both_negative = sampled_counts[:, 3]
    total = sampled_counts.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        observed = (both_positive + both_negative) / total
        prevalence_a = (both_positive + a_only) / total
        prevalence_b = (both_positive + b_only) / total
        expected = (
            prevalence_a * prevalence_b
            + (1.0 - prevalence_a) * (1.0 - prevalence_b)
        )
        kappa = (observed - expected) / (1.0 - expected)
        positive = (
            2.0
            * both_positive
            / (2.0 * both_positive + a_only + b_only)
        )
        negative = (
            2.0
            * both_negative
            / (2.0 * both_negative + a_only + b_only)
        )

    return {
        "observed_agreement": _percentile_interval(observed),
        "cohen_kappa": _percentile_interval(kappa),
        "positive_agreement": _percentile_interval(positive),
        "negative_agreement": _percentile_interval(negative),
    }


def _bootstrap_span_metrics(
    document_counts: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, tuple[float, float]]:
    sampled_counts = document_counts[bootstrap_indices].sum(axis=1)
    count_a = sampled_counts[:, 0]
    count_b = sampled_counts[:, 1]
    matched_count = sampled_counts[:, 2]
    iou_sum = sampled_counts[:, 3]

    with np.errstate(divide="ignore", invalid="ignore"):
        span_f1 = 2.0 * matched_count / (count_a + count_b)
        mean_matched_iou = iou_sum / matched_count

    return {
        "span_f1": _percentile_interval(span_f1),
        "mean_matched_iou": _percentile_interval(mean_matched_iou),
    }


def _make_agreement_row(
    *,
    task: str,
    unit: str,
    labels: Sequence[str],
    document_binary_counts: Sequence[BinaryCounts],
    bootstrap_indices: np.ndarray,
    annotator_a: str,
    annotator_b: str,
    document_span_counts: Sequence[SpanCounts] | None = None,
) -> AgreementRow:
    binary_array = np.asarray(document_binary_counts, dtype=float)
    binary_totals = binary_array.sum(axis=0)
    point_metrics = _binary_metrics(binary_totals)
    binary_intervals = _bootstrap_binary_metrics(binary_array, bootstrap_indices)

    row: AgreementRow = {
        "task": task,
        "unit": unit,
        "labels": list(labels),
        "annotator_a": annotator_a,
        "annotator_b": annotator_b,
        "n_transcripts": int(binary_array.shape[0]),
        "n_decisions": int(binary_totals.sum()),
        "both_positive": int(binary_totals[0]),
        "annotator_a_only": int(binary_totals[1]),
        "annotator_b_only": int(binary_totals[2]),
        "both_negative": int(binary_totals[3]),
    }
    for metric, estimate in point_metrics.items():
        interval = binary_intervals[metric]
        row[metric] = float(estimate)
        row[f"{metric}_ci_low"] = interval[0]
        row[f"{metric}_ci_high"] = interval[1]

    if document_span_counts is None:
        row.update(
            {
                "span_count_annotator_a": None,
                "span_count_annotator_b": None,
                "matched_span_count": None,
                "span_f1": None,
                "span_f1_ci_low": None,
                "span_f1_ci_high": None,
                "mean_matched_iou": None,
                "mean_matched_iou_ci_low": None,
                "mean_matched_iou_ci_high": None,
            }
        )
        return row

    span_array = np.asarray(document_span_counts, dtype=float)
    span_totals = span_array.sum(axis=0)
    span_point_metrics = _span_metrics(span_totals)
    span_intervals = _bootstrap_span_metrics(span_array, bootstrap_indices)
    row.update(
        {
            "span_count_annotator_a": int(span_totals[0]),
            "span_count_annotator_b": int(span_totals[1]),
            "matched_span_count": int(span_totals[2]),
        }
    )
    for metric, estimate in span_point_metrics.items():
        interval = span_intervals[metric]
        row[metric] = float(estimate)
        row[f"{metric}_ci_low"] = interval[0]
        row[f"{metric}_ci_high"] = interval[1]
    return row


def compute_agreement_rows(
    path: str | Path = DEFAULT_INPUT,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[AgreementRow]:
    """Return four aggregate pre-consensus agreement rows.

    The returned rows are, in order: checklist content detection, SBAR span
    labelling, seven-category uncertainty span labelling, and unknown-fact span
    labelling. All confidence intervals are percentile intervals from a
    transcript-cluster bootstrap.
    """

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")

    rows = _load_rows(Path(path))
    annotator_a, annotator_b, pairs = _pair_annotations(rows)
    checklist_labels = _checklist_labels(pairs)
    token_pairs = [
        (_token_label_sets(row_a), _token_label_sets(row_b))
        for row_a, row_b in pairs
    ]

    random = np.random.default_rng(seed)
    bootstrap_indices = random.integers(
        0,
        len(pairs),
        size=(bootstrap_samples, len(pairs)),
    )

    checklist_counts: list[BinaryCounts] = []
    for row_a, row_b in pairs:
        accepted_a = set(row_a.get("accept") or [])
        accepted_b = set(row_b.get("accept") or [])
        checklist_counts.append(
            _binary_counts(
                (label in accepted_a for label in checklist_labels),
                (label in accepted_b for label in checklist_labels),
            )
        )

    def span_task_counts(
        labels: Sequence[str],
    ) -> tuple[list[BinaryCounts], list[SpanCounts]]:
        allowed = frozenset(labels)
        binary_counts: list[BinaryCounts] = []
        span_counts: list[SpanCounts] = []
        for (row_a, row_b), (tokens_a, tokens_b) in zip(
            pairs, token_pairs, strict=True
        ):
            decisions_a: list[bool] = []
            decisions_b: list[bool] = []
            for token_labels_a, token_labels_b in zip(
                tokens_a, tokens_b, strict=True
            ):
                decisions_a.extend(
                    label in token_labels_a for label in labels
                )
                decisions_b.extend(
                    label in token_labels_b for label in labels
                )
            binary_counts.append(_binary_counts(decisions_a, decisions_b))
            span_counts.append(
                _match_spans(
                    _selected_spans(row_a, allowed),
                    _selected_spans(row_b, allowed),
                )
            )
        return binary_counts, span_counts

    sbar_binary, sbar_spans = span_task_counts(SBAR_LABELS)
    uncertainty_binary, uncertainty_spans = span_task_counts(UNCERTAINTY_LABELS)
    unknown_binary, unknown_spans = span_task_counts(UNKNOWN_FACT_LABELS)

    return [
        _make_agreement_row(
            task="Checklist content detection",
            unit="transcript × checklist item",
            labels=checklist_labels,
            document_binary_counts=checklist_counts,
            bootstrap_indices=bootstrap_indices,
            annotator_a=annotator_a,
            annotator_b=annotator_b,
        ),
        _make_agreement_row(
            task="SBAR span labelling",
            unit="token × SBAR label",
            labels=SBAR_LABELS,
            document_binary_counts=sbar_binary,
            document_span_counts=sbar_spans,
            bootstrap_indices=bootstrap_indices,
            annotator_a=annotator_a,
            annotator_b=annotator_b,
        ),
        _make_agreement_row(
            task="Uncertainty span labelling",
            unit="token × uncertainty label",
            labels=UNCERTAINTY_LABELS,
            document_binary_counts=uncertainty_binary,
            document_span_counts=uncertainty_spans,
            bootstrap_indices=bootstrap_indices,
            annotator_a=annotator_a,
            annotator_b=annotator_b,
        ),
        _make_agreement_row(
            task="Unknown-fact span labelling",
            unit="token × unknown-fact label",
            labels=UNKNOWN_FACT_LABELS,
            document_binary_counts=unknown_binary,
            document_span_counts=unknown_spans,
            bootstrap_indices=bootstrap_indices,
            annotator_a=annotator_a,
            annotator_b=annotator_b,
        ),
    ]


def _format_estimate_interval(row: AgreementRow, metric: str) -> str:
    estimate = row.get(metric)
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if (
        estimate is None
        or low is None
        or high is None
        or not all(math.isfinite(float(value)) for value in (estimate, low, high))
    ):
        return "—"
    return f"{float(estimate):.3f} ({float(low):.3f}–{float(high):.3f})"


def format_markdown_table(rows: Sequence[AgreementRow]) -> str:
    """Format aggregate agreement rows as a manuscript-ready Markdown table."""

    header = (
        "| Task | Analysis unit (decisions) | Observed agreement (95% CI) | "
        "Cohen κ (95% CI) | Positive agreement (95% CI) | "
        "Label-aware span F1 (95% CI) | Mean matched IoU (95% CI) |"
    )
    separator = (
        "|---|---|---:|---:|---:|---:|---:|"
    )
    body = []
    for row in rows:
        unit = f"{row['unit']} ({row['n_decisions']:,})"
        body.append(
            "| "
            + " | ".join(
                (
                    str(row["task"]),
                    unit,
                    _format_estimate_interval(row, "observed_agreement"),
                    _format_estimate_interval(row, "cohen_kappa"),
                    _format_estimate_interval(row, "positive_agreement"),
                    _format_estimate_interval(row, "span_f1"),
                    _format_estimate_interval(row, "mean_matched_iou"),
                )
            )
            + " |"
        )
    return "\n".join((header, separator, *body))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate pre-consensus RN inter-rater agreement with "
            "transcript-cluster bootstrap confidence intervals."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Paired annotation JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=(
            "Number of transcript-cluster bootstrap samples "
            f"(default: {DEFAULT_BOOTSTRAP_SAMPLES})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Bootstrap random seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = compute_agreement_rows(
        args.input,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(format_markdown_table(rows))


if __name__ == "__main__":
    main()
