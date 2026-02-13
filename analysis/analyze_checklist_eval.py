import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze checklist eval JSONL outputs and write reproducible summary "
            "artifacts (JSON + CSV)."
        )
    )
    parser.add_argument(
        "eval_jsonl",
        help="Path to eval JSONL file (rows with example.labels, prediction.labels, score).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory to write outputs. "
            "Default: eval path stem with suffix '_analysis' next to the input file."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many top FN/FP labels and error pairs to include in summary JSON.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Row {line_num}: expected object, got {type(obj)}")

            ex = obj.get("example", {})
            pred = obj.get("prediction", {})
            if not isinstance(ex, dict) or not isinstance(pred, dict):
                raise ValueError(
                    f"Row {line_num}: expected dict for example/prediction fields."
                )

            ex_labels = ex.get("labels", [])
            pred_labels = pred.get("labels", [])
            if not isinstance(ex_labels, list) or not isinstance(pred_labels, list):
                raise ValueError(f"Row {line_num}: labels must be lists.")
            if not all(isinstance(x, str) for x in ex_labels + pred_labels):
                raise ValueError(f"Row {line_num}: labels must be strings.")

            rows.append(obj)
    return rows


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def truncated_text(text: str, max_chars: int = 240) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3] + "..."


def build_analysis(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    label_set: set[str] = set()
    for row in rows:
        label_set.update(row["example"]["labels"])
        label_set.update(row["prediction"]["labels"])
    labels = sorted(label_set)

    stats = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "pred": 0}
        for label in labels
    }
    counters = Counter()
    fn_counter: Counter[str] = Counter()
    fp_counter: Counter[str] = Counter()
    pair_counter: Counter[tuple[str, str]] = Counter()
    error_count_dist: Counter[int] = Counter()
    score_band_dist: Counter[str] = Counter()
    example_rows: list[dict[str, Any]] = []

    label_to_bucket = {
        "introduction_clinicians": "patient_involvement",
        "introduction_patients": "patient_involvement",
        "invitation": "patient_involvement",
        "id_check": "id",
        "sit_diagnosis": "situation",
        "sit_events": "situation",
        "sit_status": "situation",
        "bg_history": "background",
        "bg_fall_risk": "background",
        "bg_pi_risk": "background",
        "bg_allergies": "background",
        "bg_acp": "background",
        "assess_obs": "assessment",
        "assess_pain": "assessment",
        "assess_devices": "assessment",
        "assess_monitoring": "assessment",
        "assess_nutrition": "assessment",
        "assess_fluid_balance": "assessment",
        "assess_infusions": "assessment",
        "assess_medications": "assessment",
        "assess_pathology": "assessment",
        "assess_mobility": "assessment",
        "assess_skin_integrity": "assessment",
        "rec_discharge_plan": "recommendation",
        "rec_actions": "recommendation",
        "rec_plan": "recommendation",
        "rec_patient_goals": "recommendation",
    }
    bucket_stats: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "pred": 0}
    )

    for idx, row in enumerate(rows):
        gold = set(row["example"]["labels"])
        pred = set(row["prediction"]["labels"])
        score = float(row.get("score", 0.0))

        missing = sorted(gold - pred)
        extra = sorted(pred - gold)

        counters["exact_match"] += int(gold == pred)
        counters["all_gold_covered"] += int(gold.issubset(pred))
        counters["no_fp"] += int(pred.issubset(gold))
        counters["missing_any"] += int(bool(missing))
        counters["extra_any"] += int(bool(extra))

        if score >= 0.95:
            score_band_dist[">=0.95"] += 1
        elif score >= 0.90:
            score_band_dist["0.90-0.949"] += 1
        elif score >= 0.80:
            score_band_dist["0.80-0.899"] += 1
        elif score >= 0.70:
            score_band_dist["0.70-0.799"] += 1
        else:
            score_band_dist["<0.70"] += 1

        error_count_dist[len(missing) + len(extra)] += 1
        fn_counter.update(missing)
        fp_counter.update(extra)
        for miss_label in missing:
            for extra_label in extra:
                pair_counter[(miss_label, extra_label)] += 1

        text = row["example"].get("text", "")
        if not isinstance(text, str):
            text = ""
        example_rows.append(
            {
                "idx": idx,
                "score": score,
                "gold_count": len(gold),
                "pred_count": len(pred),
                "fn_count": len(missing),
                "fp_count": len(extra),
                "error_count": len(missing) + len(extra),
                "gold_labels": "|".join(sorted(gold)),
                "pred_labels": "|".join(sorted(pred)),
                "missing_labels": "|".join(missing),
                "extra_labels": "|".join(extra),
                "text_excerpt": truncated_text(text),
            }
        )

        for label in labels:
            in_gold = label in gold
            in_pred = label in pred
            if in_gold:
                stats[label]["support"] += 1
            if in_pred:
                stats[label]["pred"] += 1
            if in_gold and in_pred:
                stats[label]["tp"] += 1
            elif in_pred and not in_gold:
                stats[label]["fp"] += 1
            elif in_gold and not in_pred:
                stats[label]["fn"] += 1

    tp_total = sum(s["tp"] for s in stats.values())
    fp_total = sum(s["fp"] for s in stats.values())
    fn_total = sum(s["fn"] for s in stats.values())
    p_micro = precision(tp_total, fp_total)
    r_micro = recall(tp_total, fn_total)
    f1_micro = f1(p_micro, r_micro)

    label_rows: list[dict[str, Any]] = []
    for label in labels:
        s = stats[label]
        p = precision(s["tp"], s["fp"])
        r = recall(s["tp"], s["fn"])
        label_rows.append(
            {
                "label": label,
                "support": s["support"],
                "pred": s["pred"],
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "precision": p,
                "recall": r,
                "f1": f1(p, r),
            }
        )

    macro_p = sum(r["precision"] for r in label_rows) / len(label_rows) if label_rows else 0.0
    macro_r = sum(r["recall"] for r in label_rows) / len(label_rows) if label_rows else 0.0
    macro_f = sum(r["f1"] for r in label_rows) / len(label_rows) if label_rows else 0.0
    weighted_f = (
        sum(r["f1"] * r["support"] for r in label_rows)
        / max(1, sum(r["support"] for r in label_rows))
    )

    for label in labels:
        bucket = label_to_bucket.get(label, "other")
        s = stats[label]
        for key in ("tp", "fp", "fn", "support", "pred"):
            bucket_stats[bucket][key] += s[key]

    bucket_rows: list[dict[str, Any]] = []
    for bucket_name in sorted(bucket_stats):
        s = bucket_stats[bucket_name]
        p = precision(s["tp"], s["fp"])
        r = recall(s["tp"], s["fn"])
        bucket_rows.append(
            {
                "bucket": bucket_name,
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "support": s["support"],
                "pred": s["pred"],
                "precision": p,
                "recall": r,
                "f1": f1(p, r),
            }
        )

    never_in_gold = [r["label"] for r in label_rows if r["support"] == 0]
    never_pred = [r["label"] for r in label_rows if r["pred"] == 0]

    scores = [float(r.get("score", 0.0)) for r in rows]
    gold_sizes = [len(set(r["example"]["labels"])) for r in rows]
    pred_sizes = [len(set(r["prediction"]["labels"])) for r in rows]

    summary = {
        "n_examples": len(rows),
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "exact_match_count": counters["exact_match"],
        "all_gold_covered_count": counters["all_gold_covered"],
        "no_fp_count": counters["no_fp"],
        "missing_any_count": counters["missing_any"],
        "extra_any_count": counters["extra_any"],
        "micro": {
            "precision": p_micro,
            "recall": r_micro,
            "f1": f1_micro,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
        },
        "macro": {
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f,
            "weighted_f1": weighted_f,
        },
        "cardinality": {
            "gold_avg": (sum(gold_sizes) / len(gold_sizes)) if gold_sizes else 0.0,
            "pred_avg": (sum(pred_sizes) / len(pred_sizes)) if pred_sizes else 0.0,
            "gold_min": min(gold_sizes) if gold_sizes else 0,
            "gold_max": max(gold_sizes) if gold_sizes else 0,
            "pred_min": min(pred_sizes) if pred_sizes else 0,
            "pred_max": max(pred_sizes) if pred_sizes else 0,
        },
        "error_count_distribution": {
            str(k): v for k, v in sorted(error_count_dist.items(), key=lambda x: x[0])
        },
        "score_band_distribution": {
            k: score_band_dist[k]
            for k in [">=0.95", "0.90-0.949", "0.80-0.899", "0.70-0.799", "<0.70"]
            if k in score_band_dist
        },
        "top_fn_labels": fn_counter.most_common(top_k),
        "top_fp_labels": fp_counter.most_common(top_k),
        "top_fn_fp_pairs": [
            {"fn_label": fn_label, "fp_label": fp_label, "count": count}
            for (fn_label, fp_label), count in pair_counter.most_common(top_k)
        ],
        "never_in_gold": never_in_gold,
        "never_pred": never_pred,
    }

    label_rows_sorted = sorted(
        label_rows, key=lambda r: (-r["support"], r["label"])
    )
    example_rows_sorted = sorted(
        example_rows, key=lambda r: (-r["error_count"], r["score"], r["idx"])
    )

    return {
        "summary": summary,
        "per_label": label_rows_sorted,
        "per_bucket": bucket_rows,
        "per_example": example_rows_sorted,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_jsonl)
    if not eval_path.exists():
        raise SystemExit(f"Missing eval JSONL: {eval_path}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else eval_path.parent / f"{eval_path.stem}_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(eval_path)
    analysis = build_analysis(rows, top_k=args.top_k)

    summary_path = out_dir / "summary.json"
    label_csv_path = out_dir / "per_label.csv"
    bucket_csv_path = out_dir / "per_bucket.csv"
    example_csv_path = out_dir / "per_example_errors.csv"

    write_json(summary_path, analysis)
    write_csv(
        label_csv_path,
        analysis["per_label"],
        fieldnames=[
            "label",
            "support",
            "pred",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_csv(
        bucket_csv_path,
        analysis["per_bucket"],
        fieldnames=[
            "bucket",
            "support",
            "pred",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_csv(
        example_csv_path,
        analysis["per_example"],
        fieldnames=[
            "idx",
            "score",
            "error_count",
            "fn_count",
            "fp_count",
            "gold_count",
            "pred_count",
            "missing_labels",
            "extra_labels",
            "gold_labels",
            "pred_labels",
            "text_excerpt",
        ],
    )

    summary = analysis["summary"]
    micro = summary["micro"]
    print(f"Wrote analysis to {out_dir}")
    print(f"Examples: {summary['n_examples']}")
    print(f"Average score: {summary['avg_score']:.4f}")
    print(
        "Micro: "
        f"P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f} "
        f"(TP={micro['tp']} FP={micro['fp']} FN={micro['fn']})"
    )
    print(f"Top FN labels: {summary['top_fn_labels'][:5]}")
    print(f"Top FP labels: {summary['top_fp_labels'][:5]}")


if __name__ == "__main__":
    main()
