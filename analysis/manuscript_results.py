from __future__ import annotations

import html
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.render_checklist_md_table import (
    BUCKET_DISPLAY,
    BUCKET_ORDER,
    LABEL_TEXT,
    LABEL_TO_BUCKET,
)


TASK_LABELS = {
    "checklist": "Checklist item prediction",
    "sbar": "SBAR span extraction",
    "uncertain": "Uncertainty span extraction",
    "unknown-fact": "Unknown-fact span extraction",
}

TASK_ORDER = [
    TASK_LABELS["checklist"],
    TASK_LABELS["sbar"],
    TASK_LABELS["uncertain"],
    TASK_LABELS["unknown-fact"],
]

APPROACH_ORDER = ["Baseline", "DSPy/GEPA", "LangExtract"]
MODEL_ORDER = ["GPT-5.2", "GPT-5-nano", "MedGemma 27B"]

MODEL_LABELS = {
    "gpt-5.2": "GPT-5.2",
    "gpt-nano": "GPT-5-nano",
    "medgemma": "MedGemma 27B",
    "gemma3-1b": "Gemma 3 1B",
}

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260611


def fmt3(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def fmt_signed3(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):+.2f}"


def fmt_estimate_ci(value: float | int | None, low: float | None, high: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return fmt3(value)
    return f"{fmt3(value)} ({fmt3(low)}-{fmt3(high)})"


def fmt_estimate_ci_unless_full_range(
    value: float | int | None,
    low: float | None,
    high: float | None,
) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return fmt3(value)
    value_text = fmt3(value)
    low_text = fmt3(low)
    high_text = fmt3(high)
    if (low_text == "0.00" and high_text == "1.00") or (
        value_text == "1.00" and low_text == "1.00" and high_text == "1.00"
    ):
        return fmt3(value)
    return fmt_estimate_ci(value, low, high)


def fmt_estimate_ci_text(value: float | int | None, low: float | None, high: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return fmt3(value)
    return f"{fmt3(value)} (95% CI {fmt3(low)}-{fmt3(high)})"


def fmt_metric_ci(row: Any, metric: str) -> str:
    return fmt_estimate_ci(
        _row_value(row, metric),
        _row_value(row, f"{metric}_ci_low"),
        _row_value(row, f"{metric}_ci_high"),
    )


def fmt_metric_ci_text(row: Any, metric: str) -> str:
    return fmt_estimate_ci_text(
        _row_value(row, metric),
        _row_value(row, f"{metric}_ci_low"),
        _row_value(row, f"{metric}_ci_high"),
    )


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key)
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def markdown_table(headers: list[str], aligns: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        escaped = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def _safe_div(numerator: float | np.ndarray, denominator: float | np.ndarray) -> float | np.ndarray:
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    result = np.divide(
        numerator_arr,
        denominator_arr,
        out=np.zeros_like(numerator_arr, dtype=float),
        where=denominator_arr != 0,
    )
    if result.ndim == 0:
        return float(result)
    return result


def _f1_from_counts(
    tp: float | np.ndarray,
    fp: float | np.ndarray,
    fn: float | np.ndarray,
) -> float | np.ndarray:
    precision_value = _safe_div(tp, np.asarray(tp) + np.asarray(fp))
    recall_value = _safe_div(tp, np.asarray(tp) + np.asarray(fn))
    return _safe_div(2 * precision_value * recall_value, precision_value + recall_value)


def _ci(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()
    return (BOOTSTRAP_SEED + int(digest[:8], 16)) % (2**32)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _metric_payload(point: float, samples: np.ndarray) -> dict[str, float | None]:
    low, high = _ci(samples)
    return {"point": float(point), "low": low, "high": high}


def _checklist_bootstrap_metrics(
    path: Path,
    *,
    label: str | None = None,
) -> dict[str, dict[str, float | None]]:
    rows = _read_jsonl(path)
    if not rows:
        return {}

    labels = [label] if label is not None else list(LABEL_TEXT)
    components = np.zeros((len(rows), len(labels), 4), dtype=float)
    exact = np.zeros(len(rows), dtype=float)

    for row_index, obj in enumerate(rows):
        gold = set(obj.get("example", {}).get("labels", []) or [])
        pred = set(obj.get("prediction", {}).get("labels", []) or [])
        exact[row_index] = float(gold == pred) if label is None else float((label in gold) == (label in pred))

        for label_index, label_id in enumerate(labels):
            gold_positive = label_id in gold
            pred_positive = label_id in pred
            if gold_positive and pred_positive:
                components[row_index, label_index, 0] = 1
            elif (not gold_positive) and pred_positive:
                components[row_index, label_index, 1] = 1
            elif gold_positive and (not pred_positive):
                components[row_index, label_index, 2] = 1
            else:
                components[row_index, label_index, 3] = 1

    rng = np.random.default_rng(_stable_seed(path, "checklist", label or "all"))
    indices = rng.integers(0, len(rows), size=(BOOTSTRAP_RESAMPLES, len(rows)))
    resampled = components[indices].sum(axis=1)

    point_counts_by_label = components.sum(axis=0)
    point_counts = point_counts_by_label.sum(axis=0)
    sample_counts = resampled.sum(axis=1)

    tp, fp, fn, tn = point_counts
    sample_tp = sample_counts[:, 0]
    sample_fp = sample_counts[:, 1]
    sample_fn = sample_counts[:, 2]
    sample_tn = sample_counts[:, 3]

    metrics = {
        "accuracy": _metric_payload(
            _safe_div(tp + tn, tp + fp + fn + tn),
            _safe_div(sample_tp + sample_tn, sample_tp + sample_fp + sample_fn + sample_tn),
        ),
        "exact_accuracy": _metric_payload(
            float(exact.mean()),
            exact[indices].mean(axis=1),
        ),
        "micro_precision": _metric_payload(
            _safe_div(tp, tp + fp),
            _safe_div(sample_tp, sample_tp + sample_fp),
        ),
        "micro_recall": _metric_payload(
            _safe_div(tp, tp + fn),
            _safe_div(sample_tp, sample_tp + sample_fn),
        ),
        "micro_f1": _metric_payload(
            _f1_from_counts(tp, fp, fn),
            _f1_from_counts(sample_tp, sample_fp, sample_fn),
        ),
    }

    if label is None:
        label_tp = point_counts_by_label[:, 0]
        label_fp = point_counts_by_label[:, 1]
        label_fn = point_counts_by_label[:, 2]
        label_f1 = _f1_from_counts(label_tp, label_fp, label_fn)
        support = label_tp + label_fn
        point_macro_f1 = float(np.mean(label_f1)) if label_f1.size else 0.0
        point_weighted_f1 = float(_safe_div(np.sum(label_f1 * support), np.sum(support)))

        sample_label_tp = resampled[:, :, 0]
        sample_label_fp = resampled[:, :, 1]
        sample_label_fn = resampled[:, :, 2]
        sample_label_f1 = _f1_from_counts(sample_label_tp, sample_label_fp, sample_label_fn)
        sample_support = sample_label_tp + sample_label_fn
        sample_macro_f1 = np.mean(sample_label_f1, axis=1)
        sample_weighted_f1 = _safe_div(
            np.sum(sample_label_f1 * sample_support, axis=1),
            np.sum(sample_support, axis=1),
        )
        metrics["macro_f1"] = _metric_payload(point_macro_f1, sample_macro_f1)
        metrics["support_weighted_f1"] = _metric_payload(point_weighted_f1, sample_weighted_f1)

    return metrics


def _span_bootstrap_metrics(
    path: Path,
    *,
    label: str | None = None,
) -> dict[str, dict[str, float | None]]:
    rows = _read_jsonl(path)
    if not rows:
        return {}

    labels: list[str]
    if label is None:
        labels_set: set[str] = set()
        for obj in rows:
            detailed = obj.get("prediction", {}).get("span_metrics", {}).get("detailed", {})
            for span in detailed.get("golds", []) or []:
                if isinstance(span, dict) and isinstance(span.get("label"), str):
                    labels_set.add(span["label"])
            for span in detailed.get("preds", []) or []:
                if isinstance(span, dict) and isinstance(span.get("label"), str):
                    labels_set.add(span["label"])
        labels = sorted(labels_set)
    else:
        labels = [label]

    label_index = {label_name: idx for idx, label_name in enumerate(labels)}
    components = np.zeros((len(rows), len(labels), 3), dtype=float)
    iou_sum = np.zeros((len(rows), len(labels)), dtype=float)
    iou_count = np.zeros((len(rows), len(labels)), dtype=float)
    exact = np.zeros(len(rows), dtype=float)

    for row_index, obj in enumerate(rows):
        detailed = obj.get("prediction", {}).get("span_metrics", {}).get("detailed", {})
        golds = detailed.get("golds", []) or []
        preds = detailed.get("preds", []) or []
        matches = detailed.get("matches", []) or []
        exact[row_index] = float(float(obj.get("score", 0.0) or 0.0) >= 1.0 - 1e-12)

        gold_by_idx = {
            span.get("idx"): span
            for span in golds
            if isinstance(span, dict) and span.get("idx") is not None
        }
        pred_by_idx = {
            span.get("idx"): span
            for span in preds
            if isinstance(span, dict) and span.get("idx") is not None
        }
        matched_gold_by_label = np.zeros(len(labels), dtype=float)
        matched_pred_by_label = np.zeros(len(labels), dtype=float)

        for span in golds:
            if not isinstance(span, dict):
                continue
            idx = label_index.get(span.get("label"))
            if idx is not None:
                components[row_index, idx, 2] += 1

        for span in preds:
            if not isinstance(span, dict):
                continue
            idx = label_index.get(span.get("label"))
            if idx is not None:
                components[row_index, idx, 1] += 1

        for match in matches:
            if not isinstance(match, dict):
                continue
            gold_span = gold_by_idx.get(match.get("gold_idx"))
            pred_span = pred_by_idx.get(match.get("pred_idx"))
            if gold_span is None or pred_span is None:
                continue
            gold_label = gold_span.get("label")
            pred_label = pred_span.get("label")
            if gold_label != pred_label:
                continue
            idx = label_index.get(gold_label)
            if idx is None:
                continue
            matched_gold_by_label[idx] += 1
            matched_pred_by_label[idx] += 1
            if isinstance(match.get("iou"), (int, float)):
                iou_sum[row_index, idx] += float(match["iou"])
                iou_count[row_index, idx] += 1

        components[row_index, :, 0] = matched_pred_by_label
        components[row_index, :, 1] = np.maximum(0, components[row_index, :, 1] - matched_pred_by_label)
        components[row_index, :, 2] = np.maximum(0, components[row_index, :, 2] - matched_gold_by_label)

    rng = np.random.default_rng(_stable_seed(path, "span", label or "all"))
    indices = rng.integers(0, len(rows), size=(BOOTSTRAP_RESAMPLES, len(rows)))
    resampled = components[indices].sum(axis=1)
    resampled_iou_sum = iou_sum[indices].sum(axis=1)
    resampled_iou_count = iou_count[indices].sum(axis=1)

    point_counts_by_label = components.sum(axis=0)
    point_counts = point_counts_by_label.sum(axis=0)
    sample_counts = resampled.sum(axis=1)

    tp, fp, fn = point_counts
    sample_tp = sample_counts[:, 0]
    sample_fp = sample_counts[:, 1]
    sample_fn = sample_counts[:, 2]

    point_iou_sum = iou_sum.sum()
    point_iou_count = iou_count.sum()
    sample_iou = _safe_div(resampled_iou_sum.sum(axis=1), resampled_iou_count.sum(axis=1))

    metrics = {
        "exact_accuracy": _metric_payload(
            float(exact.mean()),
            exact[indices].mean(axis=1),
        ),
        "micro_precision": _metric_payload(
            _safe_div(tp, tp + fp),
            _safe_div(sample_tp, sample_tp + sample_fp),
        ),
        "micro_recall": _metric_payload(
            _safe_div(tp, tp + fn),
            _safe_div(sample_tp, sample_tp + sample_fn),
        ),
        "micro_f1": _metric_payload(
            _f1_from_counts(tp, fp, fn),
            _f1_from_counts(sample_tp, sample_fp, sample_fn),
        ),
        "mean_iou": _metric_payload(
            _safe_div(point_iou_sum, point_iou_count),
            sample_iou,
        ),
    }

    if label is None and labels:
        label_tp = point_counts_by_label[:, 0]
        label_fp = point_counts_by_label[:, 1]
        label_fn = point_counts_by_label[:, 2]
        label_precision = _safe_div(label_tp, label_tp + label_fp)
        label_recall = _safe_div(label_tp, label_tp + label_fn)
        label_f1 = _f1_from_counts(label_tp, label_fp, label_fn)
        sample_label_tp = resampled[:, :, 0]
        sample_label_fp = resampled[:, :, 1]
        sample_label_fn = resampled[:, :, 2]
        sample_label_precision = _safe_div(sample_label_tp, sample_label_tp + sample_label_fp)
        sample_label_recall = _safe_div(sample_label_tp, sample_label_tp + sample_label_fn)
        sample_label_f1 = _f1_from_counts(sample_label_tp, sample_label_fp, sample_label_fn)

        metrics["macro_precision"] = _metric_payload(
            float(np.mean(label_precision)),
            np.mean(sample_label_precision, axis=1),
        )
        metrics["macro_recall"] = _metric_payload(
            float(np.mean(label_recall)),
            np.mean(sample_label_recall, axis=1),
        )
        metrics["macro_f1"] = _metric_payload(
            float(np.mean(label_f1)),
            np.mean(sample_label_f1, axis=1),
        )

    return metrics


def bootstrap_metrics(
    eval_path: str | Path,
    task: str,
    *,
    label: str | None = None,
) -> dict[str, dict[str, float | None]]:
    path = Path(eval_path)
    if task == TASK_LABELS["checklist"]:
        return _checklist_bootstrap_metrics(path, label=label)
    return _span_bootstrap_metrics(path, label=label)


def _add_metric_ci_columns(
    row: dict[str, Any],
    metrics: dict[str, dict[str, float | None]],
    *,
    mapping: dict[str, str],
) -> None:
    for metric_name, column_name in mapping.items():
        payload = metrics.get(metric_name)
        if not payload:
            continue
        row[column_name] = payload["point"]
        row[f"{column_name}_ci_low"] = payload["low"]
        row[f"{column_name}_ci_high"] = payload["high"]


def bar_chart_html(title: str, rows: list[tuple[str, float]]) -> str:
    ordered = sorted(rows, key=lambda item: item[1], reverse=True)
    bars = []
    for label, value in ordered:
        width = max(0.0, min(100.0, float(value) * 100))
        safe_label = html.escape(label)
        bars.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{safe_label}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width: {width:.1f}%;"></div></div>'
            f'<div class="bar-value">{float(value):.2f}</div>'
            f'</div>'
        )
    return f'<section class="chart-card"><h4>{html.escape(title)}</h4>{"".join(bars)}</section>'


def comparison_plot_svg(overview_df: pd.DataFrame, *, width: int = 980) -> str:
    if overview_df.empty:
        return "<p>No comparison data available.</p>"

    colors = {
        "baseline": "#8b95a7",
        "dspy": "#1f6feb",
        "langextract": "#d97706",
        "grid": "#d8dcef",
        "axis": "#4d5770",
        "text": "#192038",
        "panel_bg": "#ffffff",
        "panel_border": "#d8dcef",
        "value": "#2b3345",
    }

    tasks = [task for task in TASK_ORDER if task in set(overview_df["Task"])]
    columns = 2
    panel_w = 450
    panel_h = 220
    outer_pad = 22
    col_gap = 28
    row_gap = 28
    legend_h = 60
    rows_n = (len(tasks) + columns - 1) // columns
    height = outer_pad * 2 + legend_h + rows_n * panel_h + max(0, rows_n - 1) * row_gap
    view_w = outer_pad * 2 + columns * panel_w + (columns - 1) * col_gap

    if width != view_w:
        scale = width / view_w
        height = int(height * scale)
    else:
        scale = 1.0

    plot_left = 150
    plot_right = 62
    plot_top = 44
    plot_bottom = 24
    plot_w = panel_w - plot_left - plot_right
    plot_h = panel_h - plot_top - plot_bottom

    def x_pos(value: float) -> float:
        return plot_left + plot_w * value

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {view_w} {height / scale:.1f}" role="img" aria-labelledby="comparison-title comparison-desc" '
        f'style="max-width:100%;height:auto;">',
        "<title id=\"comparison-title\">Within-model comparison of baseline, DSPy/GEPA, and LangExtract micro-F1 performance</title>",
        "<desc id=\"comparison-desc\">Four-panel grouped horizontal bar chart across checklist, SBAR, uncertainty, and unknown-fact tasks.</desc>",
        "<style>",
        ".plot-title{font:700 14px system-ui, sans-serif; fill:#192038;}",
        ".panel-title{font:700 12px system-ui, sans-serif; fill:#192038;}",
        ".axis-label{font:10px system-ui, sans-serif; fill:#4d5770;}",
        ".tick-label{font:10px system-ui, sans-serif; fill:#4d5770;}",
        ".model-label{font:11px system-ui, sans-serif; fill:#192038;}",
        ".legend-label{font:11px system-ui, sans-serif; fill:#192038;}",
        ".value-label{font:10px system-ui, sans-serif; fill:#2b3345;}",
        "</style>",
    ]

    legend_y = outer_pad + 16
    parts.append(
        f'<text x="{outer_pad}" y="{legend_y}" class="plot-title">Within-model micro-F1 comparisons across tasks</text>'
    )
    legend_items = [
        ("Baseline", "baseline"),
        ("DSPy/GEPA", "dspy"),
        ("LangExtract", "langextract"),
    ]
    legend_x = outer_pad
    legend_row_y = outer_pad + 38
    for label, key in legend_items:
        parts.append(
            f'<rect x="{legend_x}" y="{legend_row_y - 10}" width="16" height="10" '
            f'fill="{colors[key]}" rx="2" ry="2" />'
        )
        parts.append(
            f'<text x="{legend_x + 24}" y="{legend_row_y - 1}" class="legend-label">{html.escape(label)}</text>'
        )
        legend_x += 132

    tick_values = [0.0, 0.5, 1.0]
    group_height = 44
    bar_height = 10
    bar_gap = 4
    approach_offsets = {
        "Baseline": 0,
        "DSPy/GEPA": bar_height + bar_gap,
        "LangExtract": 2 * (bar_height + bar_gap),
    }

    for idx, task in enumerate(tasks):
        col = idx % columns
        row = idx // columns
        panel_x = outer_pad + col * (panel_w + col_gap)
        panel_y = outer_pad + legend_h + row * (panel_h + row_gap)
        subset = overview_df[overview_df["Task"] == task].copy()
        subset = subset.set_index("Model")

        parts.append(
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="10" ry="10" '
            f'fill="{colors["panel_bg"]}" stroke="{colors["panel_border"]}" stroke-width="1.2" />'
        )
        parts.append(
            f'<text x="{panel_x + 16}" y="{panel_y + 20}" class="panel-title">{html.escape(task)}</text>'
        )

        for tick in tick_values:
            x = panel_x + x_pos(tick)
            parts.append(
                f'<line x1="{x:.1f}" y1="{panel_y + plot_top - 8:.1f}" x2="{x:.1f}" y2="{panel_y + panel_h - plot_bottom:.1f}" '
                f'stroke="{colors["grid"]}" stroke-width="1" />'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{panel_y + panel_h - 8:.1f}" text-anchor="middle" class="tick-label">{tick:.2f}</text>'
            )

        parts.append(
            f'<text x="{panel_x + plot_left + plot_w / 2:.1f}" y="{panel_y + panel_h - 18:.1f}" '
            f'text-anchor="middle" class="axis-label">Micro-F1</text>'
        )

        for model_idx, model in enumerate(MODEL_ORDER):
            if model not in subset.index:
                continue
            row_data = subset.loc[model]
            group_top = panel_y + plot_top + model_idx * group_height
            model_label_y = group_top + bar_height + 8
            parts.append(
                f'<text x="{panel_x + plot_left - 12:.1f}" y="{model_label_y:.1f}" text-anchor="end" class="model-label">{html.escape(model)}</text>'
            )

            for approach, color_key in [
                ("Baseline", "baseline"),
                ("DSPy/GEPA", "dspy"),
                ("LangExtract", "langextract"),
            ]:
                value = row_data[approach]
                if pd.isna(value):
                    continue
                bar_y = group_top + approach_offsets[approach]
                bar_x = panel_x + plot_left
                bar_width = max(1.0, plot_w * float(value))
                label_x = min(bar_x + bar_width + 6, panel_x + panel_w - 34)
                parts.append(
                    f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                    f'fill="{colors[color_key]}" rx="3" ry="3" />'
                )
                parts.append(
                    f'<text x="{label_x:.1f}" y="{bar_y + bar_height - 1:.1f}" class="value-label">{float(value):.2f}</text>'
                )

    parts.append("</svg>")
    return "".join(parts)


def comparison_plot_png(
    overview_df: pd.DataFrame,
    output_path: str | Path,
    *,
    width: int = 1800,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "baseline": "#8b95a7",
        "dspy": "#1f6feb",
        "langextract": "#d97706",
        "grid": "#d8dcef",
        "axis": "#4d5770",
        "text": "#192038",
        "panel_bg": "#ffffff",
        "panel_border": "#d8dcef",
        "value": "#2b3345",
    }

    font_paths = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    regular_path = next((path for path in font_paths if path.exists()), None)
    bold_path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    bold_path = bold_path if bold_path.exists() else regular_path

    def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        path = bold_path if bold else regular_path
        if path is None:
            return ImageFont.load_default()
        return ImageFont.truetype(str(path), size=size)

    title_font = font(34, bold=True)
    panel_font = font(25, bold=True)
    label_font = font(23)
    axis_font = font(20)
    value_font = font(20)

    tasks = [task for task in TASK_ORDER if task in set(overview_df["Task"])]
    columns = 2
    panel_w = 830
    panel_h = 330
    outer_pad = 42
    col_gap = 44
    row_gap = 42
    legend_h = 98
    rows_n = (len(tasks) + columns - 1) // columns
    height = outer_pad * 2 + legend_h + rows_n * panel_h + max(0, rows_n - 1) * row_gap

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def text_width(text: str, text_font: ImageFont.ImageFont) -> int:
        bbox = draw.textbbox((0, 0), text, font=text_font)
        return bbox[2] - bbox[0]

    draw.text(
        (outer_pad, outer_pad - 8),
        "Within-model micro-F1 comparisons across tasks",
        fill=colors["text"],
        font=title_font,
    )

    legend_items = [
        ("Baseline", "baseline"),
        ("DSPy/GEPA", "dspy"),
        ("LangExtract", "langextract"),
    ]
    legend_x = outer_pad
    legend_y = outer_pad + 54
    for label, key in legend_items:
        draw.rounded_rectangle(
            (legend_x, legend_y - 18, legend_x + 34, legend_y + 2),
            radius=5,
            fill=colors[key],
        )
        draw.text((legend_x + 46, legend_y - 21), label, fill=colors["text"], font=label_font)
        legend_x += 235

    plot_left = 235
    plot_right = 88
    plot_top = 66
    plot_bottom = 52
    plot_w = panel_w - plot_left - plot_right
    group_height = 66
    bar_height = 15
    bar_gap = 7
    approach_offsets = {
        "Baseline": 0,
        "DSPy/GEPA": bar_height + bar_gap,
        "LangExtract": 2 * (bar_height + bar_gap),
    }

    tick_values = [0.0, 0.5, 1.0]

    for idx, task in enumerate(tasks):
        col = idx % columns
        row = idx // columns
        panel_x = outer_pad + col * (panel_w + col_gap)
        panel_y = outer_pad + legend_h + row * (panel_h + row_gap)
        subset = overview_df[overview_df["Task"] == task].copy().set_index("Model")

        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
            radius=12,
            fill=colors["panel_bg"],
            outline=colors["panel_border"],
            width=2,
        )
        draw.text((panel_x + 24, panel_y + 20), task, fill=colors["text"], font=panel_font)

        for tick in tick_values:
            x = panel_x + plot_left + plot_w * tick
            draw.line(
                (x, panel_y + plot_top - 14, x, panel_y + panel_h - plot_bottom),
                fill=colors["grid"],
                width=2,
            )
            tick_label = f"{tick:.2f}"
            draw.text(
                (x - text_width(tick_label, axis_font) / 2, panel_y + panel_h - 34),
                tick_label,
                fill=colors["axis"],
                font=axis_font,
            )

        axis_label = "Micro-F1"
        draw.text(
            (
                panel_x + plot_left + plot_w / 2 - text_width(axis_label, axis_font) / 2,
                panel_y + panel_h - 58,
            ),
            axis_label,
            fill=colors["axis"],
            font=axis_font,
        )

        for model_idx, model in enumerate(MODEL_ORDER):
            if model not in subset.index:
                continue
            row_data = subset.loc[model]
            group_top = panel_y + plot_top + model_idx * group_height
            model_y = group_top + bar_height + 12
            draw.text(
                (
                    panel_x + plot_left - 18 - text_width(model, label_font),
                    model_y - 18,
                ),
                model,
                fill=colors["text"],
                font=label_font,
            )

            for approach, color_key in [
                ("Baseline", "baseline"),
                ("DSPy/GEPA", "dspy"),
                ("LangExtract", "langextract"),
            ]:
                value = row_data[approach]
                if pd.isna(value):
                    continue
                value = float(value)
                bar_x = panel_x + plot_left
                bar_y = group_top + approach_offsets[approach]
                bar_width = max(3, plot_w * value)
                draw.rounded_rectangle(
                    (bar_x, bar_y, bar_x + bar_width, bar_y + bar_height),
                    radius=5,
                    fill=colors[color_key],
                )
                value_label = f"{value:.2f}"
                value_x = min(bar_x + bar_width + 10, panel_x + panel_w - 58)
                draw.text(
                    (value_x, bar_y - 5),
                    value_label,
                    fill=colors["value"],
                    font=value_font,
                )

    image.save(output_path, dpi=(220, 220))
    return output_path


def _load_collate_module(root: Path):
    module_path = root / "analysis" / "collate_eval_results_webpage.py"
    spec = importlib.util.spec_from_file_location("collate_eval_results_webpage", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detect_task(short_name: str) -> str:
    if short_name.startswith("checklist"):
        return TASK_LABELS["checklist"]
    if short_name.startswith("sbar"):
        return TASK_LABELS["sbar"]
    if short_name.startswith("uncertain"):
        return TASK_LABELS["uncertain"]
    if short_name.startswith("unknown-fact"):
        return TASK_LABELS["unknown-fact"]
    return short_name


def _detect_approach(name: str, short_name: str) -> str:
    raw_name = name.lower()
    if "_baseline_" in raw_name or raw_name.startswith("eval_baseline_"):
        return "Baseline"
    if "langextract" in short_name:
        return "LangExtract"
    if "baseline" in short_name:
        return "Baseline"
    return "DSPy/GEPA"


def _detect_model(short_name: str) -> str:
    for key, label in MODEL_LABELS.items():
        if key in short_name:
            return label
    return "Other"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_all_results(root: str | Path = ".") -> pd.DataFrame:
    root = Path(root)
    eval_dir = root / "evals"
    collate = _load_collate_module(root)

    checklist_rows, _ = collate._collect_checklist_results(eval_dir=eval_dir, skip_missing=False)
    sbar_rows, _ = collate._collect_span_results(eval_dir=eval_dir, task_prefix="sbar", top_labels=20)
    uncertain_rows, _ = collate._collect_span_results(
        eval_dir=eval_dir,
        task_prefix="uncertain",
        top_labels=20,
        include_unknown_fact=False,
    )
    unknown_rows, _ = collate._collect_span_results(
        eval_dir=eval_dir,
        task_prefix="unknown_fact",
        top_labels=20,
        include_unknown_fact=True,
    )

    entries = checklist_rows + sbar_rows + uncertain_rows + unknown_rows
    rows: list[dict[str, Any]] = []
    for item in entries:
        short_name = collate._short_name(item["name"])
        summary = item["summary"]
        micro = summary.get("micro", {})
        macro = summary.get("macro", {})
        task = _detect_task(short_name)
        row = {
            "name": item["name"],
            "short_name": short_name,
            "task": task,
            "approach": _detect_approach(item["name"], short_name),
            "model": _detect_model(short_name),
            "n_examples": int(summary.get("n_examples", 0)),
            "avg_score": float(summary.get("avg_score", 0.0)),
            "micro_precision": float(micro.get("precision", 0.0)),
            "micro_recall": float(micro.get("recall", 0.0)),
            "micro_f1": float(micro.get("f1", 0.0)),
            "mean_iou": float(micro.get("mean_iou", 0.0)) if "mean_iou" in micro else None,
            "macro_f1": float(macro.get("f1", 0.0)) if "f1" in macro else None,
            "support_weighted_f1": (
                float(macro.get("weighted_f1", 0.0))
                if "weighted_f1" in macro
                else None
            ),
            "eval_path": str(item["eval_file"]),
        }
        metrics = bootstrap_metrics(item["eval_file"], task)
        _add_metric_ci_columns(
            row,
            metrics,
            mapping={
                "exact_accuracy": "accuracy",
                "micro_precision": "micro_precision",
                "micro_recall": "micro_recall",
                "micro_f1": "micro_f1",
                "mean_iou": "mean_iou",
                "macro_f1": "macro_f1",
                "support_weighted_f1": "support_weighted_f1",
            },
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["task_order"] = df["task"].map({name: idx for idx, name in enumerate(TASK_ORDER)})
    df["approach_order"] = df["approach"].map({name: idx for idx, name in enumerate(APPROACH_ORDER)})
    df["model_order"] = df["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)}).fillna(99)
    return df.sort_values(["task_order", "approach_order", "model_order", "micro_f1"], ascending=[True, True, True, False]).reset_index(drop=True)


def curated_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    curated = manuscript_scope_results(df)
    idx = curated.groupby(["task", "approach", "model"])["micro_f1"].idxmax()
    curated = curated.loc[idx].copy()
    return curated.sort_values(["task_order", "approach_order", "model_order"]).reset_index(drop=True)


def manuscript_scope_results(df: pd.DataFrame) -> pd.DataFrame:
    scoped = df[df["model"].isin(MODEL_ORDER)].copy()
    scoped = scoped[~((scoped["task"] == TASK_LABELS["uncertain"]) & (scoped["n_examples"] < 10))]
    return scoped.sort_values(["task_order", "approach_order", "model_order", "micro_f1"], ascending=[True, True, True, False]).reset_index(drop=True)


def main_overview_table(curated_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        for model in MODEL_ORDER:
            subset = curated_df[
                (curated_df["task"] == task) & (curated_df["model"] == model)
            ].copy()
            if subset.empty:
                continue

            def row_for(approach: str) -> pd.Series | None:
                rows_for_approach = subset[subset["approach"] == approach]
                if rows_for_approach.empty:
                    return None
                return rows_for_approach.iloc[0]

            baseline = row_for("Baseline")
            dspy = row_for("DSPy/GEPA")
            langextract = row_for("LangExtract")
            first_available = subset.iloc[0]

            rows.append(
                {
                    "Task": task,
                    "Model": model,
                    "n": int(first_available["n_examples"]),
                    "Baseline": None if baseline is None else float(baseline["micro_f1"]),
                    "DSPy/GEPA": None if dspy is None else float(dspy["micro_f1"]),
                    "Delta DSPy/GEPA vs baseline": (
                        float(dspy["micro_f1"] - baseline["micro_f1"])
                        if baseline is not None and dspy is not None
                        else None
                    ),
                    "LangExtract": None if langextract is None else float(langextract["micro_f1"]),
                    "Baseline label": None if baseline is None else baseline["short_name"],
                    "DSPy/GEPA label": None if dspy is None else dspy["short_name"],
                    "LangExtract label": None if langextract is None else langextract["short_name"],
                }
            )
    return pd.DataFrame(rows)


def main_overview_lookup(overview_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["Task"], row["Model"]): row
        for row in overview_df.to_dict("records")
    }


def legacy_best_overview_table(curated_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        task_df = curated_df[curated_df["task"] == task].copy()
        if task_df.empty:
            continue

        def best_row(approach: str) -> pd.Series | None:
            subset = task_df[task_df["approach"] == approach]
            if subset.empty:
                return None
            return subset.sort_values("micro_f1", ascending=False).iloc[0]

        baseline = best_row("Baseline")
        dspy = best_row("DSPy/GEPA")
        langextract = best_row("LangExtract")
        best_overall = task_df.sort_values("micro_f1", ascending=False).iloc[0]

        rows.append(
            {
                "Task": task,
                "n": int(best_overall["n_examples"]),
                "Best baseline": (
                    f'{baseline["model"]} ({fmt3(baseline["micro_f1"])})' if baseline is not None else "NA"
                ),
                "Best DSPy/GEPA": (
                    f'{dspy["model"]} ({fmt3(dspy["micro_f1"])})' if dspy is not None else "NA"
                ),
                "Best LangExtract": (
                    f'{langextract["model"]} ({fmt3(langextract["micro_f1"])})'
                    if langextract is not None
                    else "NA"
                ),
                "Delta DSPy/GEPA vs baseline": (
                    float(dspy["micro_f1"] - baseline["micro_f1"])
                    if baseline is not None and dspy is not None
                    else None
                ),
                "Best overall configuration": f'{best_overall["approach"]}, {best_overall["model"]}',
                "_baseline_row_name": None if baseline is None else baseline["name"],
                "_dspy_row_name": None if dspy is None else dspy["name"],
                "_langextract_row_name": None if langextract is None else langextract["name"],
            }
        )
    return pd.DataFrame(rows)


def main_overview_markdown_table(overview_df: pd.DataFrame) -> str:
    rows = [
        [
            row["Task"],
            row["Model"],
            int(row["n"]),
            fmt3(row["Baseline"]),
            fmt3(row["DSPy/GEPA"]),
            fmt_signed3(row["Delta DSPy/GEPA vs baseline"]),
            fmt3(row["LangExtract"]),
        ]
        for _, row in overview_df.iterrows()
    ]
    return markdown_table(
        headers=[
            "Task",
            "Model",
            "n",
            "Baseline",
            "DSPy/GEPA",
            "Delta DSPy/GEPA vs baseline",
            "LangExtract",
        ],
        aligns=["---", "---", "---:", "---:", "---:", "---:", "---:"],
        rows=rows,
    )


def full_overview_table(curated_df: pd.DataFrame) -> pd.DataFrame:
    table = curated_df.copy()
    table["Micro precision"] = table["micro_precision"].map(fmt3)
    table["Micro recall"] = table["micro_recall"].map(fmt3)
    table["Micro F1"] = table["micro_f1"].map(fmt3)
    table["Mean IoU"] = table["mean_iou"].map(fmt3)
    table["Average score"] = table["avg_score"].map(fmt3)
    return table[
        [
            "task",
            "approach",
            "model",
            "n_examples",
            "Average score",
            "Micro precision",
            "Micro recall",
            "Micro F1",
            "Mean IoU",
            "short_name",
        ]
    ].rename(
        columns={
            "task": "Task",
            "approach": "Approach",
            "model": "Model",
            "n_examples": "n",
            "short_name": "Saved evaluation label",
        }
    )


def full_overview_markdown_table(table_df: pd.DataFrame) -> str:
    rows = [list(row) for row in table_df.itertuples(index=False, name=None)]
    return markdown_table(
        headers=list(table_df.columns),
        aligns=["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
        rows=rows,
    )


def uncertainty_results_markdown_table(curated_df: pd.DataFrame) -> str:
    table_df = curated_df[
        curated_df["task"].isin(
            [TASK_LABELS["uncertain"], TASK_LABELS["unknown-fact"]]
        )
    ].copy()
    table_df["Task"] = table_df["task"].replace(
        {
            TASK_LABELS["uncertain"]: "Uncertainty",
            TASK_LABELS["unknown-fact"]: "Unknown fact",
        }
    )
    table_df = table_df.sort_values(["task_order", "model_order", "approach_order"])

    rows: list[list[object]] = []
    previous_task: str | None = None
    previous_model: str | None = None
    for row in table_df.itertuples(index=False):
        task_label = row.Task if row.Task != previous_task else ""
        model_label = row.model if row.Task != previous_task or row.model != previous_model else ""
        rows.append(
            [
                task_label,
                model_label,
                row.approach,
                fmt_estimate_ci(
                    row.micro_precision,
                    row.micro_precision_ci_low,
                    row.micro_precision_ci_high,
                ),
                fmt_estimate_ci(
                    row.micro_recall,
                    row.micro_recall_ci_low,
                    row.micro_recall_ci_high,
                ),
                fmt_estimate_ci(row.micro_f1, row.micro_f1_ci_low, row.micro_f1_ci_high),
                fmt_estimate_ci(row.mean_iou, row.mean_iou_ci_low, row.mean_iou_ci_high),
            ]
        )
        previous_task = row.Task
        previous_model = row.model

    return markdown_table(
        headers=["Task", "Model", "Approach", "Precision", "Recall", "F1", "Mean IoU"],
        aligns=["---", "---", "---", "---:", "---:", "---:", "---:"],
        rows=rows,
    )


def chart_rows_by_task(curated_df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    charts: dict[str, list[tuple[str, float]]] = {}
    for task in TASK_ORDER:
        subset = curated_df[curated_df["task"] == task].copy()
        if subset.empty:
            continue
        charts[task] = [
            (f'{row["approach"]}, {row["model"]}', float(row["micro_f1"]))
            for _, row in subset.sort_values(["micro_f1", "approach_order", "model_order"], ascending=[False, True, True]).iterrows()
        ]
    return charts


def load_sbar_main_tables(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    base_csv = root / "evals" / "eval_sbar_baseline_gpt_5.2_consensus_per_label_analysis.csv"
    opt_csv = root / "evals" / "eval_sbar_span_gpt_5-2_consensus_reasoning_none_per_label_analysis.csv"
    base_jsonl = root / "evals" / "eval_sbar_baseline_gpt_5.2_consensus.jsonl"
    opt_jsonl = root / "evals" / "eval_sbar_span_gpt_5-2_consensus_reasoning_none.jsonl"

    base_df = pd.read_csv(base_csv)
    opt_df = pd.read_csv(opt_csv)
    comp_df = opt_df.merge(
        base_df[["label", "f1"]].rename(columns={"f1": "baseline_f1"}),
        on="label",
        how="left",
    )
    comp_df["delta_f1"] = comp_df["f1"] - comp_df["baseline_f1"]

    label_metrics = {}
    for label in comp_df["label"]:
        label_metrics[label] = bootstrap_metrics(opt_jsonl, TASK_LABELS["sbar"], label=label)
    for metric_name, source_name in [
        ("recall", "micro_recall"),
        ("precision", "micro_precision"),
        ("f1", "micro_f1"),
        ("mean_iou", "mean_iou"),
    ]:
        comp_df[f"{metric_name}_ci_low"] = comp_df["label"].map(
            lambda label: label_metrics[label].get(source_name, {}).get("low")
        )
        comp_df[f"{metric_name}_ci_high"] = comp_df["label"].map(
            lambda label: label_metrics[label].get(source_name, {}).get("high")
        )

    base_metrics = bootstrap_metrics(base_jsonl, TASK_LABELS["sbar"])
    opt_metrics = bootstrap_metrics(opt_jsonl, TASK_LABELS["sbar"])

    def count_jsonl(path: Path) -> int:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    return {
        "base_df": base_df,
        "opt_df": opt_df,
        "comp_df": comp_df,
        "base_n": count_jsonl(base_jsonl),
        "opt_n": count_jsonl(opt_jsonl),
        "base_macro_precision": float(base_df["precision"].mean()),
        "base_macro_recall": float(base_df["recall"].mean()),
        "base_macro_f1": float(base_df["f1"].mean()),
        "opt_macro_precision": float(opt_df["precision"].mean()),
        "opt_macro_recall": float(opt_df["recall"].mean()),
        "opt_macro_f1": float(opt_df["f1"].mean()),
        "base_macro_precision_ci": (
            base_metrics.get("macro_precision", {}).get("low"),
            base_metrics.get("macro_precision", {}).get("high"),
        ),
        "base_macro_recall_ci": (
            base_metrics.get("macro_recall", {}).get("low"),
            base_metrics.get("macro_recall", {}).get("high"),
        ),
        "base_macro_f1_ci": (
            base_metrics.get("macro_f1", {}).get("low"),
            base_metrics.get("macro_f1", {}).get("high"),
        ),
        "opt_macro_precision_ci": (
            opt_metrics.get("macro_precision", {}).get("low"),
            opt_metrics.get("macro_precision", {}).get("high"),
        ),
        "opt_macro_recall_ci": (
            opt_metrics.get("macro_recall", {}).get("low"),
            opt_metrics.get("macro_recall", {}).get("high"),
        ),
        "opt_macro_f1_ci": (
            opt_metrics.get("macro_f1", {}).get("low"),
            opt_metrics.get("macro_f1", {}).get("high"),
        ),
        "mean_iou_min": float(opt_df["mean_iou"].min()),
        "mean_iou_max": float(opt_df["mean_iou"].max()),
    }


def sbar_main_markdown_table(comp_df: pd.DataFrame) -> str:
    rows = [
        [
            row.label,
            int(row.gold),
            int(row.total_pred_spans),
            fmt_estimate_ci(row.recall, row.recall_ci_low, row.recall_ci_high),
            fmt_estimate_ci(row.precision, row.precision_ci_low, row.precision_ci_high),
            fmt_estimate_ci(row.mean_iou, row.mean_iou_ci_low, row.mean_iou_ci_high),
            fmt_estimate_ci(row.f1, row.f1_ci_low, row.f1_ci_high),
        ]
        for row in comp_df.itertuples(index=False)
    ]
    return markdown_table(
        headers=[
            "Label",
            "Gold",
            "Predicted spans",
            "Recall",
            "Precision",
            "Mean IoU",
            "F1",
        ],
        aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:"],
        rows=rows,
    )


def _checklist_grouped_display_df(
    per_label_df: pd.DataFrame, n_examples: int
) -> pd.DataFrame:
    table_df = per_label_df.copy()
    bucket_rank = {bucket: idx for idx, bucket in enumerate(BUCKET_ORDER)}
    table_df["category_key"] = table_df["label"].map(LABEL_TO_BUCKET).fillna("other")
    table_df["bucket_order"] = table_df["category_key"].map(bucket_rank).fillna(
        len(BUCKET_ORDER)
    )
    table_df["Category"] = table_df["category_key"].map(BUCKET_DISPLAY).fillna(
        table_df["category_key"]
    )
    table_df["Label"] = table_df["label"].map(LABEL_TEXT).fillna(table_df["label"])
    table_df["TN"] = n_examples - table_df["tp"] - table_df["fp"] - table_df["fn"]
    table_df = table_df.rename(
        columns={
            "support": "Support",
            "tp": "TP",
            "fp": "FP",
            "fn": "FN",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        }
    ).sort_values(
        ["bucket_order", "Support", "label"],
        ascending=[True, False, True],
    )
    display_df = table_df[
        [
            "category_key",
            "Category",
            "Label",
            "Accuracy",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "Precision",
            "precision_ci_low",
            "precision_ci_high",
            "Recall",
            "recall_ci_low",
            "recall_ci_high",
            "F1",
            "f1_ci_low",
            "f1_ci_high",
        ]
    ]
    return display_df


def checklist_grouped_markdown_table(
    per_label_df: pd.DataFrame,
    n_examples: int,
) -> str:
    display_df = _checklist_grouped_display_df(per_label_df, n_examples)
    rows: list[list[object]] = []

    for bucket in BUCKET_ORDER:
        bucket_rows = display_df[display_df["category_key"] == bucket]
        if bucket_rows.empty:
            continue

        category = BUCKET_DISPLAY.get(bucket, bucket.title())
        rows.append([f"**{category}**", "", "", "", ""])

        for row in bucket_rows.itertuples(index=False):
            rows.append(
                [
                    row.Label,
                    fmt_estimate_ci_unless_full_range(
                        row.Accuracy,
                        row.accuracy_ci_low,
                        row.accuracy_ci_high,
                    ),
                    fmt_estimate_ci_unless_full_range(
                        row.Precision,
                        row.precision_ci_low,
                        row.precision_ci_high,
                    ),
                    fmt_estimate_ci_unless_full_range(
                        row.Recall,
                        row.recall_ci_low,
                        row.recall_ci_high,
                    ),
                    fmt_estimate_ci_unless_full_range(
                        row.F1,
                        row.f1_ci_low,
                        row.f1_ci_high,
                    ),
                ]
            )

    return markdown_table(
        headers=["Checklist item", "Accuracy", "Precision", "Recall", "F1"],
        aligns=[
            "--------------------------------------------",
            "-------------:",
            "-------------:",
            "-------------:",
            "-------------:",
        ],
        rows=rows,
    )


def load_checklist_main_tables(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    cur_summary = _read_json(root / "evals" / "eval_checklist_consensus_gpt_5_2_test_analysis" / "summary.json")["summary"]
    base_summary = _read_json(root / "evals" / "eval_baseline_checklist_consensus_gpt_5_2_test_analysis" / "summary.json")["summary"]
    cur_bucket = pd.read_csv(root / "evals" / "eval_checklist_consensus_gpt_5_2_test_analysis" / "per_bucket.csv")
    base_bucket = pd.read_csv(root / "evals" / "eval_baseline_checklist_consensus_gpt_5_2_test_analysis" / "per_bucket.csv")
    per_label = pd.read_csv(root / "evals" / "eval_checklist_consensus_gpt_5_2_test_analysis" / "per_label.csv")
    baseline_per_label = pd.read_csv(root / "evals" / "eval_baseline_checklist_consensus_gpt_5_2_test_analysis" / "per_label.csv")

    label_metrics = {
        label: bootstrap_metrics(
            root / "evals" / "eval_checklist_consensus_gpt_5_2_test.jsonl",
            TASK_LABELS["checklist"],
            label=label,
        )
        for label in per_label["label"]
    }
    for metric_name, source_name in [
        ("accuracy", "accuracy"),
        ("precision", "micro_precision"),
        ("recall", "micro_recall"),
        ("f1", "micro_f1"),
    ]:
        per_label[metric_name] = per_label["label"].map(
            lambda label: label_metrics[label].get(source_name, {}).get("point")
        )
        per_label[f"{metric_name}_ci_low"] = per_label["label"].map(
            lambda label: label_metrics[label].get(source_name, {}).get("low")
        )
        per_label[f"{metric_name}_ci_high"] = per_label["label"].map(
            lambda label: label_metrics[label].get(source_name, {}).get("high")
        )

    full_grouped_md = checklist_grouped_markdown_table(
        per_label,
        int(cur_summary["n_examples"]),
    )

    merged = cur_bucket.merge(
        base_bucket[["bucket", "f1"]].rename(columns={"f1": "baseline_f1"}),
        on="bucket",
        how="left",
    )
    merged["delta_f1"] = merged["f1"] - merged["baseline_f1"]
    merged["category"] = merged["bucket"].map(
        {
            "assessment": "Assessment",
            "background": "Background",
            "id": "Identification",
            "patient_involvement": "Patient involvement",
            "recommendation": "Recommendation",
            "situation": "Situation",
        }
    )

    total_support = float(per_label["support"].sum())
    baseline_total_support = float(baseline_per_label["support"].sum())
    weighted_precision = float((per_label["precision"] * per_label["support"]).sum() / total_support) if total_support else 0.0
    weighted_recall = float((per_label["recall"] * per_label["support"]).sum() / total_support) if total_support else 0.0
    baseline_weighted_precision = (
        float((baseline_per_label["precision"] * baseline_per_label["support"]).sum() / baseline_total_support)
        if baseline_total_support
        else 0.0
    )
    baseline_weighted_recall = (
        float((baseline_per_label["recall"] * baseline_per_label["support"]).sum() / baseline_total_support)
        if baseline_total_support
        else 0.0
    )

    overall_rows = pd.DataFrame(
        [
            {
                "category": "Overall (micro)",
                "support": int(cur_summary["micro"]["tp"] + cur_summary["micro"]["fn"]),
                "precision": float(cur_summary["micro"]["precision"]),
                "recall": float(cur_summary["micro"]["recall"]),
                "f1": float(cur_summary["micro"]["f1"]),
                "delta_f1": float(cur_summary["micro"]["f1"] - base_summary["micro"]["f1"]),
            },
            {
                "category": "Overall (macro)",
                "support": "NA",
                "precision": float(cur_summary["macro"]["precision"]),
                "recall": float(cur_summary["macro"]["recall"]),
                "f1": float(cur_summary["macro"]["f1"]),
                "delta_f1": float(cur_summary["macro"]["f1"] - base_summary["macro"]["f1"]),
            },
            {
                "category": "Overall (support-weighted)",
                "support": int(cur_summary["micro"]["tp"] + cur_summary["micro"]["fn"]),
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1": float(cur_summary["macro"]["weighted_f1"]),
                "delta_f1": float(cur_summary["macro"]["weighted_f1"] - base_summary["macro"]["weighted_f1"]),
                "delta_precision": float(weighted_precision - baseline_weighted_precision),
                "delta_recall": float(weighted_recall - baseline_weighted_recall),
            },
        ]
    )

    return {
        "summary": cur_summary,
        "baseline_summary": base_summary,
        "per_bucket_df": merged,
        "overall_df": overall_rows,
        "per_label_df": per_label,
        "full_grouped_md": full_grouped_md,
        "zero_f1_count": int((per_label["f1"] == 0).sum()),
    }


def checklist_category_markdown_table(bucket_df: pd.DataFrame, overall_df: pd.DataFrame) -> str:
    ordered = bucket_df.copy()
    ordered["bucket_order"] = ordered["bucket"].map(
        {
            "situation": 1,
            "background": 2,
            "assessment": 3,
            "recommendation": 4,
            "patient_involvement": 5,
            "id": 6,
        }
    )
    ordered = ordered.sort_values("bucket_order")
    rows = [
        [
            row.category,
            int(row.support),
            fmt3(row.precision),
            fmt3(row.recall),
            fmt3(row.f1),
            fmt_signed3(row.delta_f1),
        ]
        for row in ordered.itertuples(index=False)
    ]
    for row in overall_df.itertuples(index=False):
        rows.append(
            [
                row.category,
                row.support,
                fmt3(row.precision),
                fmt3(row.recall),
                fmt3(row.f1),
                fmt_signed3(row.delta_f1),
            ]
        )
    return markdown_table(
        headers=["Category", "Support", "Precision", "Recall", "F1", "Delta F1 (vs baseline)"],
        aligns=["---", "---:", "---:", "---:", "---:", "---:"],
        rows=rows,
    )
