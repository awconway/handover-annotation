"""Build a self contained HTML dashboard from all eval outputs.

This script collates:
1) checklist evals using analysis/analyze_checklist_eval.py outputs
2) SBAR, uncertain, and unknown-fact span evals using per-label span metrics from JSONL
   rows (prediction.span_metrics.detailed).

It writes one HTML report with summary tables, per-label tables, and simple
embedded plots suitable for quick slide review.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collate checklist + span eval outputs into one HTML report."
    )
    parser.add_argument(
        "--eval-dir",
        default="evals",
        help="Directory containing eval JSONL and *_analysis outputs (default: evals).",
    )
    parser.add_argument(
        "--out",
        default="evals/eval_results_dashboard.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=20,
        help="Max per-label rows displayed per eval section.",
    )
    parser.add_argument(
        "--skip-missing-checklist-analyses",
        action="store_true",
        help=(
            "Skip checklist evals when *_analysis cannot be found. "
            "By default, missing analyses are generated from evaluate files."
        ),
    )
    return parser.parse_args()


def safe_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _tokenize(name: str) -> list[str]:
    return [t for t in name.lower().replace(".jsonl", "").split("_") if t]


def precision(tp: float, fp: float) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(tp: float, fn: float) -> float:
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1(precision_value: float, recall_value: float) -> float:
    return (
        2 * precision_value * recall_value / (precision_value + recall_value)
        if (precision_value + recall_value)
        else 0.0
    )


def _run_analyze_checklist_eval(eval_jsonl: Path, out_dir: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(ROOT / "analysis" / "analyze_checklist_eval.py"),
        str(eval_jsonl),
        "--out-dir",
        str(out_dir),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if proc.stdout:
            return True, proc.stdout.strip()
        return True, proc.stderr.strip()
    except subprocess.CalledProcessError as exc:
        return False, (exc.stdout or "") + "\n" + (exc.stderr or "")


def _load_csv_rows(path: Path, numeric_fields: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric_fields = numeric_fields or set()
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in numeric_fields:
                    try:
                        parsed[key] = int(value) if value.isdigit() else float(value)
                    except ValueError:
                        parsed[key] = 0
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _load_checklist_analysis(
    analysis_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = analysis_dir / "summary.json"
    per_label_path = analysis_dir / "per_label.csv"
    per_bucket_path = analysis_dir / "per_bucket.csv"

    if not (summary_path.exists() and per_label_path.exists() and per_bucket_path.exists()):
        raise FileNotFoundError(
            f"Missing expected files in {analysis_dir}: "
            "summary.json, per_label.csv, per_bucket.csv"
        )

    summary = json.loads(summary_path.read_text())
    per_label = _load_csv_rows(
        per_label_path,
        numeric_fields={"support", "pred", "tp", "fp", "fn"},
    )
    # keep precision/recall/f1 numeric too
    for row in per_label:
        for key in ("precision", "recall", "f1"):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                row[key] = 0.0

    per_bucket = _load_csv_rows(
        per_bucket_path,
        numeric_fields={"support", "pred", "tp", "fp", "fn"},
    )
    for row in per_bucket:
        for key in ("precision", "recall", "f1"):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                row[key] = 0.0

    return summary, per_label, per_bucket


def _find_checklist_analysis_dir(
    eval_jsonl: Path,
    eval_dir: Path,
) -> Path | None:
    stem = eval_jsonl.stem
    direct = eval_jsonl.with_name(f"{stem}_analysis")
    if direct.exists() and direct.is_dir():
        return direct

    candidates = [
        stem.replace("eval_checklist_", "eval_", 1) + "_analysis",
    ]
    if stem.startswith("eval_checklist_baseline_"):
        candidates.append("eval_baseline_" + stem[len("eval_checklist_"):])

    for candidate in candidates:
        candidate_path = eval_dir / f"{candidate}_analysis"
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_path

    target_tokens = set(_tokenize(stem))
    best_score = -1
    best_dir: Path | None = None
    for p in eval_dir.glob("*_analysis"):
        if not p.is_dir():
            continue
        dir_tokens = set(_tokenize(p.stem.replace("_analysis", "")))
        if "checklist" not in dir_tokens or "eval" not in dir_tokens:
            continue
        overlap = len(target_tokens.intersection(dir_tokens))
        if overlap > best_score and overlap >= 4:
            best_score = overlap
            best_dir = p
    if best_dir is not None:
        return best_dir

    return None


def _analyze_span_eval(jsonl_path: Path, out_csv: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels: set[str] = set()
    gold_count: defaultdict[str, int] = defaultdict(int)
    pred_count: defaultdict[str, int] = defaultdict(int)
    matched_gold: defaultdict[str, int] = defaultdict(int)
    matched_pred: defaultdict[str, int] = defaultdict(int)
    match_ious: defaultdict[str, list[float]] = defaultdict(list)

    score_values: list[float] = []
    parsed_rows = 0
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            parsed_rows += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            pred = obj.get("prediction", {}) if isinstance(obj, dict) else {}
            if not isinstance(pred, dict):
                continue
            span_metrics = pred.get("span_metrics")
            if not isinstance(span_metrics, dict):
                continue
            detailed = span_metrics.get("detailed", {})
            if not isinstance(detailed, dict):
                continue

            score = obj.get("score", 0.0)
            if isinstance(score, (int, float)):
                score_values.append(float(score))

            golds = detailed.get("golds", [])
            preds = detailed.get("preds", [])
            matches = detailed.get("matches", [])
            if not isinstance(golds, list) or not isinstance(preds, list):
                continue

            gold_by_idx = {
                g.get("idx"): g
                for g in golds
                if isinstance(g, dict) and g.get("idx") is not None
            }
            pred_by_idx = {
                p.get("idx"): p
                for p in preds
                if isinstance(p, dict) and p.get("idx") is not None
            }

            for g in golds:
                if not isinstance(g, dict):
                    continue
                lab = g.get("label")
                if not isinstance(lab, str):
                    continue
                labels.add(lab)
                gold_count[lab] += 1

            for p in preds:
                if not isinstance(p, dict):
                    continue
                lab = p.get("label")
                if not isinstance(lab, str):
                    continue
                labels.add(lab)
                pred_count[lab] += 1

            if isinstance(matches, list):
                for m in matches:
                    if not isinstance(m, dict):
                        continue
                    gi = m.get("gold_idx")
                    pi = m.get("pred_idx")
                    iou = m.get("iou")
                    g = gold_by_idx.get(gi)
                    p = pred_by_idx.get(pi)
                    if g is None or p is None:
                        continue
                    glab = g.get("label")
                    plab = p.get("label")
                    if not isinstance(glab, str) or not isinstance(plab, str):
                        continue
                    matched_gold[glab] += 1
                    matched_pred[plab] += 1
                    if isinstance(iou, (int, float)):
                        match_ious[glab].append(float(iou))

    rows: list[dict[str, Any]] = []
    for lab in sorted(labels):
        g = gold_count[lab]
        p = pred_count[lab]
        mg = matched_gold[lab]
        mp = matched_pred[lab]
        recall_value = (mg / g) if g else 0.0
        precision_value = (mp / p) if p else 0.0
        f1_value = f1(precision_value, recall_value)
        ious = match_ious[lab]
        mean_iou = (sum(ious) / len(ious)) if ious else 0.0
        rows.append(
            {
                "label": lab,
                "gold": g,
                "total_pred_spans": p,
                "matched_gold": mg,
                "matched_pred": mp,
                "recall": recall_value,
                "precision": precision_value,
                "f1": f1_value,
                "mean_iou": mean_iou,
            }
        )

    total_gold = sum(gold_count.values())
    total_pred = sum(pred_count.values())
    total_match_gold = sum(matched_gold.values())
    total_match_pred = sum(matched_pred.values())

    micro_precision_value = precision(total_match_pred, max(0, total_pred - total_match_pred))
    micro_recall_value = recall(total_match_gold, max(0, total_gold - total_match_gold))
    micro_f1 = f1(micro_precision_value, micro_recall_value)

    all_ious: list[float] = []
    for vals in match_ious.values():
        all_ious.extend(vals)

    macro_f1 = sum(row["f1"] for row in rows) / len(rows) if rows else 0.0
    avg_score = sum(score_values) / len(score_values) if score_values else 0.0

    summary = {
        "n_examples": parsed_rows,
        "avg_score": avg_score,
        "micro": {
            "precision": micro_precision_value,
            "recall": micro_recall_value,
            "f1": micro_f1,
            "tp": total_match_pred,
            "fp": max(0, total_pred - total_match_pred),
            "fn": max(0, total_gold - total_match_gold),
            "mean_iou": (sum(all_ious) / len(all_ious)) if all_ious else 0.0,
        },
        "macro": {
            "precision": sum(row["precision"] for row in rows) / len(rows) if rows else 0.0,
            "recall": sum(row["recall"] for row in rows) / len(rows) if rows else 0.0,
            "f1": macro_f1,
        },
    }

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "gold",
                "total_pred_spans",
                "matched_gold",
                "matched_pred",
                "recall",
                "precision",
                "f1",
                "mean_iou",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (f"{v:.6f}" if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            )

    return summary, rows


def _collect_checklist_results(eval_dir: Path, skip_missing: bool) -> tuple[
    list[dict[str, Any]], list[str]
]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for eval_jsonl in sorted(eval_dir.glob("eval_checklist_*.jsonl")):
        analysis_dir = _find_checklist_analysis_dir(eval_jsonl, eval_dir)

        if analysis_dir is None:
            if skip_missing:
                notes.append(f"Skipped {eval_jsonl.name}: no checklist analysis directory")
                continue
            target_dir = eval_jsonl.with_name(f"{eval_jsonl.stem}_analysis")
            ok, msg = _run_analyze_checklist_eval(eval_jsonl, target_dir)
            if not ok:
                notes.append(f"Failed to generate analysis for {eval_jsonl.name}: {msg}")
                continue
            analysis_dir = target_dir

        try:
            summary, per_label, per_bucket = _load_checklist_analysis(analysis_dir)
        except Exception as exc:
            notes.append(f"Failed to load checklist analysis for {eval_jsonl.name}: {exc}")
            continue

        rows.append(
            {
                "name": eval_jsonl.stem,
                "eval_file": eval_jsonl,
                "analysis_dir": analysis_dir,
                "summary": summary,
                "per_label": sorted(
                    per_label,
                    key=lambda r: (
                        safe_float(r.get("support")),
                        str(r.get("label", "")),
                    ),
                    reverse=True,
                ),
                "per_bucket": sorted(
                    per_bucket,
                    key=lambda r: str(r.get("bucket", "")),
                ),
            }
        )

    return rows, notes


def _collect_span_results(
    eval_dir: Path,
    task_prefix: str,
    top_labels: int,
    *,
    include_unknown_fact: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []

    for eval_jsonl in sorted(eval_dir.glob(f"eval_{task_prefix}_*.jsonl")):
        if not include_unknown_fact and "unknown_fact_binary" in eval_jsonl.stem:
            continue

        out_csv = eval_jsonl.with_name(f"{eval_jsonl.stem}_per_label_analysis.csv")
        try:
            summary, label_rows = _analyze_span_eval(eval_jsonl, out_csv)
        except Exception as exc:
            notes.append(f"Failed span analysis for {eval_jsonl.name}: {exc}")
            continue

        rows.append(
            {
                "name": eval_jsonl.stem,
                "eval_file": eval_jsonl,
                "summary": summary,
                "labels": sorted(
                    label_rows,
                    key=lambda r: (r["f1"], r["gold"]),
                    reverse=True,
                ),
                "label_csv": out_csv,
                "top_labels": top_labels,
            }
        )

    return rows, notes


def _format_num(value: float, *, percent=False, digits: int = 4) -> str:
    if percent:
        return f"{value * 100:.{digits}f}%"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _short_name(name: str) -> str:
    if name.startswith("eval_"):
        return name[len("eval_"):]
    return name


def _rel(path: Path, out_path: Path) -> str:
    try:
        return path.relative_to(out_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _table_html(headers: list[str], rows: list[dict[str, Any]], formatters: dict[str, Any] | None = None) -> str:
    formatters = formatters or {}
    header_cells = "".join([f"<th>{html.escape(h)}</th>" for h in headers])
    body_rows = []
    for row in rows:
        cells = []
        for header in headers:
            if header in row:
                value = row[header]
            else:
                value = ""
            if formatter := formatters.get(header):
                value = formatter(value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        "<table>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _bar_chart_html(title: str, rows: list[tuple[str, float]], value_fmt: str = "f1") -> str:
    entries = sorted(rows, key=lambda r: r[1], reverse=True)
    bars = []
    for label, value in entries:
        safe_label = html.escape(label)
        width = max(0.0, min(100.0, float(value) * 100))
        bars.append(
            f"""
            <div class="bar-row">
              <div class="bar-label" title="{safe_label}">{safe_label}</div>
              <div class="bar-track">
                <div class="bar-fill" style="width: {width:.1f}%;"></div>
              </div>
              <div class="bar-value">{_format_num(value, percent=False, digits=3)}</div>
            </div>
            """
        )

    return (
        f"<section class=\"panel chart\"><h3>{html.escape(title)}</h3>"
        f"{''.join(bars)}</section>"
    )


def _join_top_pairs(items: list[Any]) -> str:
    rendered = []
    for pair in items[:10]:
        if isinstance(pair, dict):
            label = pair.get("fn_label", "") if pair.get("fn_label") is not None else ""
            if "fp_label" in pair:
                rendered.append(f"{label}→{pair.get('fp_label')} ({pair.get('count', 0)})")
            else:
                rendered.append(f"{label} ({pair.get('count', 0)})")
        elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
            rendered.append(f"{pair[0]} ({pair[1]})")
        else:
            rendered.append(str(pair))
    return ", ".join(rendered)


def _render_checklist_sections(results: list[dict[str, Any]], top_labels: int) -> str:
    blocks = []
    for item in results:
        summary = item["summary"]
        micro = summary.get("micro", {})
        macro = summary.get("macro", {})
        top_pairs_fn = summary.get("top_fn_labels", [])
        top_pairs_fp = summary.get("top_fp_labels", [])
        bucket_rows = item["per_bucket"]
        label_rows = item["labels"] if False else item["per_label"]

        label_headers = [
            "label",
            "support",
            "pred",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ]

        bucket_headers = ["bucket", "support", "pred", "tp", "fp", "fn", "precision", "recall", "f1"]

        per_label_rows = []
        for row in label_rows[:top_labels]:
            per_label_rows.append(
                {
                    **row,
                    "precision": row.get("precision", 0.0),
                    "recall": row.get("recall", 0.0),
                    "f1": row.get("f1", 0.0),
                }
            )

        blocks.append(
            f"""
            <details class="eval-block">
              <summary>{html.escape(_short_name(item['name']))}</summary>
              <div class="stats-grid">
                <div class="stat">N examples: <strong>{summary.get('n_examples', 0)}</strong></div>
                <div class="stat">Avg score: <strong>{_format_num(summary.get('avg_score', 0.0), digits=4)}</strong></div>
                <div class="stat">Micro F1: <strong>{_format_num(micro.get('f1', 0.0), digits=4)}</strong></div>
                <div class="stat">Macro F1: <strong>{_format_num(macro.get('f1', 0.0), digits=4)}</strong></div>
              </div>
              <p>Top FN labels: {_join_top_pairs(top_pairs_fn)}<br/>
              Top FP labels: {_join_top_pairs(top_pairs_fp)}</p>
              <p>
                Sources:
                <a href="{_rel(item['analysis_dir'] / 'summary.json', Path(_LAST_OUT_PATH))}">summary.json</a>,
                <a href="{_rel(item['analysis_dir'] / 'per_label.csv', Path(_LAST_OUT_PATH))}">per_label.csv</a>,
                <a href="{_rel(item['analysis_dir'] / 'per_bucket.csv', Path(_LAST_OUT_PATH))}">per_bucket.csv</a>
              </p>
              <h4>Per-bucket</h4>
              {_table_html(
                  bucket_headers,
                  bucket_rows,
                  {
                      "precision": lambda v: _format_num(v, digits=4),
                      "recall": lambda v: _format_num(v, digits=4),
                      "f1": lambda v: _format_num(v, digits=4),
                  },
              )}
              <h4>Top per-label (support-ordered)</h4>
              {_table_html(
                  label_headers,
                  per_label_rows,
                  {
                      "precision": lambda v: _format_num(v, digits=4),
                      "recall": lambda v: _format_num(v, digits=4),
                      "f1": lambda v: _format_num(v, digits=4),
                  },
              )}
            </details>
            """
        )
    return "".join(blocks)


def _render_span_sections(
    results: list[dict[str, Any]],
    top_labels: int,
) -> str:
    blocks = []
    label_headers = [
        "label",
        "gold",
        "total_pred_spans",
        "matched_gold",
        "matched_pred",
        "precision",
        "recall",
        "f1",
        "mean_iou",
    ]
    for item in results:
        summary = item["summary"]
        micro = summary.get("micro", {})
        blocks.append(
            f"""
            <details class="eval-block">
              <summary>{html.escape(_short_name(item['name']))}</summary>
              <div class="stats-grid">
                <div class="stat">N examples: <strong>{summary.get('n_examples', 0)}</strong></div>
                <div class="stat">Avg score: <strong>{_format_num(summary.get('avg_score', 0.0), digits=4)}</strong></div>
                <div class="stat">Micro F1: <strong>{_format_num(micro.get('f1', 0.0), digits=4)}</strong></div>
                <div class="stat">Mean IoU (matched): <strong>{_format_num(micro.get('mean_iou', 0.0), digits=4)}</strong></div>
              </div>
              <p>
                Per-label CSV: <a href="{_rel(item['label_csv'], Path(_LAST_OUT_PATH))}">download</a>
              </p>
              <h4>Top per-label</h4>
              {_table_html(
                  label_headers,
                  item["labels"][:top_labels],
                  {
                      "precision": lambda v: _format_num(v, digits=4),
                      "recall": lambda v: _format_num(v, digits=4),
                      "f1": lambda v: _format_num(v, digits=4),
                      "mean_iou": lambda v: _format_num(v, digits=4),
                  },
              )}
            </details>
            """
        )
    return "".join(blocks)


def _summary_rows_for_table(entries: list[dict[str, Any]], task_label: str) -> tuple[list[dict[str, Any]], list[tuple[str, float]], list[tuple[str, float]]]:
    rows: list[dict[str, Any]] = []
    chart_micro: list[tuple[str, float]] = []
    chart_macro: list[tuple[str, float]] = []
    for item in entries:
        summary = item["summary"]
        micro = summary.get("micro", {})
        macro = summary.get("macro", {})
        rows.append(
            {
                "task": task_label,
                "eval": _short_name(item["name"]),
                "n_examples": summary.get("n_examples", 0),
                "avg_score": _format_num(summary.get("avg_score", 0.0), digits=4),
                "micro_p": _format_num(micro.get("precision", 0.0), digits=4),
                "micro_r": _format_num(micro.get("recall", 0.0), digits=4),
                "micro_f1": _format_num(micro.get("f1", 0.0), digits=4),
                "macro_f1": _format_num(macro.get("f1", 0.0), digits=4),
                "weighted_f1": _format_num(macro.get("weighted_f1", macro.get("f1", 0.0)), digits=4),
                "exact_match": _format_num(
                    summary.get("exact_match_count", 0)
                    / summary.get("n_examples", 1),
                    digits=4,
                ),
            }
        )
        try:
            chart_micro.append((item["name"], safe_float(micro.get("f1", 0.0))))
            chart_macro.append((item["name"], safe_float(macro.get("f1", 0.0))))
        except Exception:
            continue

    return rows, chart_micro, chart_macro


def _span_summary_rows(entries: list[dict[str, Any]], task_label: str) -> tuple[list[dict[str, Any]], list[tuple[str, float]]]:
    rows: list[dict[str, Any]] = []
    chart_micro: list[tuple[str, float]] = []
    for item in entries:
        summary = item["summary"]
        micro = summary.get("micro", {})
        rows.append(
            {
                "task": task_label,
                "eval": _short_name(item["name"]),
                "n_examples": summary.get("n_examples", 0),
                "avg_score": _format_num(summary.get("avg_score", 0.0), digits=4),
                "micro_p": _format_num(micro.get("precision", 0.0), digits=4),
                "micro_r": _format_num(micro.get("recall", 0.0), digits=4),
                "micro_f1": _format_num(micro.get("f1", 0.0), digits=4),
                "mean_iou": _format_num(micro.get("mean_iou", 0.0), digits=4),
            }
        )
        chart_micro.append((item["name"], safe_float(micro.get("f1", 0.0))))

    return rows, chart_micro


def _build_page(
    checklist_rows: list[dict[str, Any]],
    sbar_rows: list[dict[str, Any]],
    uncertain_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    notes: list[str],
    top_labels: int,
) -> str:
    checklist_summary_rows, checklist_chart_micro, checklist_chart_macro = _summary_rows_for_table(
        checklist_rows,
        "checklist",
    )
    sbar_summary_rows, sbar_chart = _span_summary_rows(sbar_rows, "sbar")
    uncertain_summary_rows, uncertain_chart = _span_summary_rows(uncertain_rows, "uncertain")
    unknown_summary_rows, unknown_chart = _span_summary_rows(unknown_rows, "unknown_fact")

    all_rows = (
        checklist_summary_rows
        + sbar_summary_rows
        + uncertain_summary_rows
        + unknown_summary_rows
    )

    checklist_summary_table = _table_html(
        [
            "task",
            "eval",
            "n_examples",
            "avg_score",
            "micro_p",
            "micro_r",
            "micro_f1",
            "macro_f1",
            "weighted_f1",
            "exact_match",
        ],
        all_rows,
    )

    sbar_section = _render_span_sections(sbar_rows, top_labels)
    uncertain_section = _render_span_sections(uncertain_rows, top_labels)
    unknown_section = _render_span_sections(unknown_rows, top_labels)
    checklist_section = _render_checklist_sections(checklist_rows, top_labels)

    notes_html = ""
    if notes:
        notes_html = (
            '<section class="panel"><h2>Warnings</h2><ul>'
            + "".join([f"<li>{html.escape(note)}</li>" for note in notes])
            + "</ul></section>"
        )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Evaluation Results Dashboard</title>
    <style>
      :root {{
        --bg: #f4f5f8;
        --panel: #ffffff;
        --ink: #192038;
        --muted: #4d5770;
        --line: #d8dcef;
        --accent: #1f6feb;
      }}
      body {{
        margin: 0;
        padding: 1.2rem;
        font-family: Inter, "Segoe UI", Arial, Helvetica, sans-serif;
        background: var(--bg);
        color: var(--ink);
      }}
      h1, h2, h3, h4 {{ margin: 0 0 .5rem 0; }}
      .panel {{
        border: 1px solid var(--line);
        background: var(--panel);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
      }}
      .chart-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 1rem;
      }}
      .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 0.5rem;
        margin: .5rem 0 1rem;
      }}
      .stat {{
        padding: .6rem;
        border-radius: 6px;
        background: #f9faff;
        border: 1px solid var(--line);
      }}
      .bar-row {{
        display: grid;
        grid-template-columns: minmax(180px, 1fr) 1fr auto;
        gap: 0.5rem;
        align-items: center;
        margin: 0.45rem 0;
      }}
      .bar-label {{
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 0.9rem;
      }}
      .bar-track {{
        height: 1rem;
        border-radius: 999px;
        border: 1px solid #ccc;
        background: #ebedf3;
        overflow: hidden;
      }}
      .bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #2563eb);
      }}
      .bar-value {{
        font-size: 0.85rem;
        color: var(--muted);
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin: .6rem 0 1rem;
        font-size: 0.85rem;
      }}
      th, td {{
        text-align: left;
        padding: 0.45rem 0.5rem;
        border-bottom: 1px solid var(--line);
      }}
      th {{
        background: #eef1fb;
        position: sticky;
        top: 0;
      }}
      .eval-block {{
        padding: 0.6rem 0;
        border-top: 1px solid #dee3f2;
      }}
      details > summary {{
        cursor: pointer;
        font-weight: 600;
        margin: 0.5rem 0;
      }}
      a {{ color: var(--accent); }}
    </style>
  </head>
  <body>
    <h1>Evaluation Results Dashboard</h1>
    <p>Generated from <code>{ROOT / 'evals'}</code>. All figures are derived from saved eval outputs.</p>

    <div class="panel">
      <h2>Overview</h2>
      {checklist_summary_table}
    </div>

    <div class="chart-grid">
      {_bar_chart_html("Checklist micro F1", checklist_chart_micro)}
      {_bar_chart_html("Checklist macro F1", checklist_chart_macro)}
      {_bar_chart_html("SBAR micro F1", sbar_chart)}
      {_bar_chart_html("Uncertain micro F1", uncertain_chart)}
      {_bar_chart_html("Unknown-fact micro F1", unknown_chart)}
    </div>

    {notes_html}

    <section class="panel">
      <h2>Checklist task evaluations</h2>
      {checklist_section}
    </section>

    <section class="panel">
      <h2>SBAR span task evaluations</h2>
      {sbar_section}
    </section>

    <section class="panel">
      <h2>Uncertain span task evaluations</h2>
      {uncertain_section}
    </section>

    <section class="panel">
      <h2>Unknown-fact span task evaluations</h2>
      {unknown_section}
    </section>
  </body>
</html>
"""


_LAST_OUT_PATH: str | None = None


def main() -> None:
    global _LAST_OUT_PATH
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"Missing eval dir: {eval_dir}")

    out_path = Path(args.out)
    _LAST_OUT_PATH = out_path.as_posix()

    checklist_rows, notes = _collect_checklist_results(
        eval_dir=eval_dir,
        skip_missing=args.skip_missing_checklist_analyses,
    )
    sbar_rows, sbar_notes = _collect_span_results(
        eval_dir=eval_dir,
        task_prefix="sbar",
        top_labels=args.top_labels,
    )
    uncertain_rows, uncertain_notes = _collect_span_results(
        eval_dir=eval_dir,
        task_prefix="uncertain",
        top_labels=args.top_labels,
        include_unknown_fact=False,
    )
    unknown_rows, unknown_notes = _collect_span_results(
        eval_dir=eval_dir,
        task_prefix="unknown_fact_binary",
        top_labels=args.top_labels,
        include_unknown_fact=True,
    )

    all_notes = notes + sbar_notes + uncertain_notes + unknown_notes

    html = _build_page(
        checklist_rows=checklist_rows,
        sbar_rows=sbar_rows,
        uncertain_rows=uncertain_rows,
        unknown_rows=unknown_rows,
        notes=all_notes,
        top_labels=args.top_labels,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    print(f"Wrote report: {out_path}")
    print(f"Checklist evals: {len(checklist_rows)}")
    print(f"SBAR span evals: {len(sbar_rows)}")
    print(f"Uncertain span evals: {len(uncertain_rows)}")
    print(f"Unknown-fact span evals: {len(unknown_rows)}")


if __name__ == "__main__":
    main()
