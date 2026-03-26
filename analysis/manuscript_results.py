from __future__ import annotations

import html
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


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


def fmt3(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def fmt_signed3(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):+.3f}"


def markdown_table(headers: list[str], aligns: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        escaped = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


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
            f'<div class="bar-value">{float(value):.3f}</div>'
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
        rows.append(
            {
                "name": item["name"],
                "short_name": short_name,
                "task": _detect_task(short_name),
                "approach": _detect_approach(item["name"], short_name),
                "model": _detect_model(short_name),
                "n_examples": int(summary.get("n_examples", 0)),
                "avg_score": float(summary.get("avg_score", 0.0)),
                "micro_precision": float(micro.get("precision", 0.0)),
                "micro_recall": float(micro.get("recall", 0.0)),
                "micro_f1": float(micro.get("f1", 0.0)),
                "mean_iou": float(micro.get("mean_iou", 0.0)) if "mean_iou" in micro else None,
                "macro_f1": float(macro.get("f1", 0.0)) if "f1" in macro else None,
                "eval_path": str(item["eval_file"]),
            }
        )

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
        "mean_iou_min": float(opt_df["mean_iou"].min()),
        "mean_iou_max": float(opt_df["mean_iou"].max()),
    }


def sbar_main_markdown_table(comp_df: pd.DataFrame) -> str:
    rows = [
        [
            row.label,
            int(row.gold),
            int(row.total_pred_spans),
            fmt3(row.recall),
            fmt3(row.precision),
            fmt3(row.mean_iou),
            fmt3(row.f1),
            fmt_signed3(row.delta_f1),
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
            "Delta F1 (vs baseline)",
        ],
        aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
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
    full_grouped_md = (root / "evals" / "eval_checklist_consensus_gpt_5_2_test_analysis" / "table_per_label_grouped_with_baseline_delta.md").read_text(encoding="utf-8")

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
