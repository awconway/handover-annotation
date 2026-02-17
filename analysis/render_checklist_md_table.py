from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BUCKET_ORDER = [
    "id",
    "situation",
    "background",
    "assessment",
    "recommendation",
    "patient_involvement",
    "other",
]

BUCKET_DISPLAY = {
    "id": "Identification",
    "situation": "Situation",
    "background": "Background",
    "assessment": "Assessment",
    "recommendation": "Recommendation",
    "patient_involvement": "Patient Involvement",
    "other": "Other",
}

LABEL_TO_BUCKET = {
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

LABEL_TEXT = {
    "introduction_clinicians": "Introduction of clinicians involved in handover",
    "introduction_patients": "Introduction of clinicians involved in handover to patient/carer",
    "invitation": "Invitation for patient/carer to participate in handover",
    "id_check": "ID check of 3 patient identifiers",
    "sit_diagnosis": "Primary diagnosis | reason for admission",
    "sit_events": "Significant events or complications",
    "sit_status": "Current status (awaiting tests/procedures, on interim orders/plan)",
    "bg_history": "Relevant clinical and social history | comorbidities",
    "bg_fall_risk": "Alerts - falls risk",
    "bg_pi_risk": "Alerts - pressure injury risk",
    "bg_allergies": "Alerts - allergies",
    "bg_acp": "Advanced care planning",
    "assess_obs": "Observations | Q-ADDS | recent escalations",
    "assess_pain": "Pain management",
    "assess_devices": "Devices | lines | vascular access",
    "assess_monitoring": "Critical monitoring | alarms",
    "assess_nutrition": "Nutrition | restrictions",
    "assess_fluid_balance": "Fluid balance | restrictions",
    "assess_infusions": "Infusions",
    "assess_medications": "Medication chart | flag high risk meds",
    "assess_pathology": "Pathology",
    "assess_mobility": "Mobility | aids",
    "assess_skin_integrity": "Skin integrity | interventions",
    "rec_discharge_plan": "Discharge plan",
    "rec_actions": "Critical actions required",
    "rec_plan": "Care plan/pathway actions to follow up",
    "rec_patient_goals": "Asked patient/carer about goals and preferences",
}


@dataclass
class LabelRow:
    label_id: str
    category: str
    label_text: str
    support: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render per-label checklist metrics as a grouped Markdown table with "
            "category subtotals and overall summary rows."
        )
    )
    parser.add_argument(
        "per_label_csv",
        help="Path to per_label.csv from analyze_checklist_eval.py",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help=(
            "Path to summary.json from analyze_checklist_eval.py. "
            "Default: sibling summary.json next to per_label.csv."
        ),
    )
    parser.add_argument(
        "--out-file",
        default=None,
        help="Output Markdown file path. Default: print to stdout.",
    )
    parser.add_argument(
        "--include-id-subtotal",
        action="store_true",
        help="Include a subtotal row for Identification (off by default).",
    )
    parser.add_argument(
        "--baseline-per-label-csv",
        default=None,
        help=(
            "Optional baseline per_label.csv. If provided, add a far-right "
            "delta column: ΔF1 = (current F1 - baseline F1)."
        ),
    )
    return parser.parse_args()


def load_n_examples(summary_json: Path) -> int:
    data = json.loads(summary_json.read_text())
    try:
        return int(data["summary"]["n_examples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not read n_examples from {summary_json}. "
            "Expected key summary.n_examples."
        ) from exc


def load_rows(per_label_csv: Path, n_examples: int | None) -> list[LabelRow]:
    rows: list[LabelRow] = []
    with per_label_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            label_id = r["label"]
            tp = int(r["tp"])
            fp = int(r["fp"])
            fn = int(r["fn"])
            tn = (n_examples - tp - fp - fn) if n_examples is not None else 0
            rows.append(
                LabelRow(
                    label_id=label_id,
                    category=LABEL_TO_BUCKET.get(label_id, "other"),
                    label_text=LABEL_TEXT.get(label_id, label_id),
                    support=int(r["support"]),
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    tn=tn,
                    precision=float(r["precision"]),
                    recall=float(r["recall"]),
                    f1=float(r["f1"]),
                )
            )
    return rows


def fmt_f(x: float) -> str:
    return f"{x:.3f}"


def fmt_delta(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:+.3f}"


def md_row(cols: Iterable[str]) -> str:
    safe_cols = [c.replace("|", "\\|") for c in cols]
    return "| " + " | ".join(safe_cols) + " |"


def subtotal_prf(rows: list[LabelRow]) -> tuple[float, float, float]:
    tp = sum(r.tp for r in rows)
    fp = sum(r.fp for r in rows)
    fn = sum(r.fn for r in rows)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def subtotal_row(
    category: str, rows: list[LabelRow], baseline_rows: list[LabelRow] | None = None
) -> str:
    tp = sum(r.tp for r in rows)
    fp = sum(r.fp for r in rows)
    fn = sum(r.fn for r in rows)
    tn = sum(r.tn for r in rows)
    support = sum(r.support for r in rows)
    p, r, f1 = subtotal_prf(rows)

    delta_p = None
    delta_r = None
    delta_f1 = None
    if baseline_rows is not None:
        b_p, b_r, b_f1 = subtotal_prf(baseline_rows)
        delta_p = p - b_p
        delta_r = r - b_r
        delta_f1 = f1 - b_f1

    return md_row(
        [
            category,
            f"**Subtotal ({category})**",
            f"**{support}**",
            f"**{tp}**",
            f"**{fp}**",
            f"**{fn}**",
            f"**{tn}**",
            f"**{fmt_f(p)}**",
            f"**{fmt_delta(delta_p)}**",
            f"**{fmt_f(r)}**",
            f"**{fmt_delta(delta_r)}**",
            f"**{fmt_f(f1)}**",
            f"**{fmt_delta(delta_f1)}**",
        ]
    )


def build_table(
    rows: list[LabelRow],
    include_id_subtotal: bool,
    baseline_rows: list[LabelRow] | None = None,
) -> list[str]:
    baseline_by_label = (
        {r.label_id: r for r in baseline_rows} if baseline_rows is not None else {}
    )

    table: list[str] = []
    table.append(
        md_row(
            [
                "Category",
                "Label",
                "Support",
                "TP",
                "FP",
                "FN",
                "TN",
                "Precision",
                "ΔPrecision vs Baseline",
                "Recall",
                "ΔRecall vs Baseline",
                "F1",
                "ΔF1 vs Baseline",
            ]
        )
    )
    table.append(
        md_row(
            [
                "---",
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
            ]
        )
    )

    bucket_rank = {k: i for i, k in enumerate(BUCKET_ORDER)}
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            bucket_rank.get(r.category, len(BUCKET_ORDER)),
            -r.support,
            r.label_id,
        ),
    )

    for bucket in BUCKET_ORDER:
        bucket_rows = [r for r in rows_sorted if r.category == bucket]
        if not bucket_rows:
            continue
        cat_name = BUCKET_DISPLAY.get(bucket, bucket.title())
        for r in bucket_rows:
            baseline_row = baseline_by_label.get(r.label_id)
            delta_p = (
                r.precision - baseline_row.precision
                if baseline_row is not None
                else None
            )
            delta_r = (
                r.recall - baseline_row.recall
                if baseline_row is not None
                else None
            )
            delta_f1 = (r.f1 - baseline_row.f1) if baseline_row is not None else None
            table.append(
                md_row(
                    [
                        cat_name,
                        r.label_text,
                        str(r.support),
                        str(r.tp),
                        str(r.fp),
                        str(r.fn),
                        str(r.tn),
                        fmt_f(r.precision),
                        fmt_delta(delta_p),
                        fmt_f(r.recall),
                        fmt_delta(delta_r),
                        fmt_f(r.f1),
                        fmt_delta(delta_f1),
                    ]
                )
            )

        if bucket != "id" or include_id_subtotal:
            baseline_bucket_rows = (
                [r for r in baseline_rows if r.category == bucket]
                if baseline_rows is not None
                else None
            )
            table.append(subtotal_row(cat_name, bucket_rows, baseline_bucket_rows))

    tp = sum(r.tp for r in rows)
    fp = sum(r.fp for r in rows)
    fn = sum(r.fn for r in rows)
    tn = sum(r.tn for r in rows)
    support = sum(r.support for r in rows)

    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    macro_p = sum(r.precision for r in rows) / len(rows) if rows else 0.0
    macro_r = sum(r.recall for r in rows) / len(rows) if rows else 0.0
    macro_f1 = sum(r.f1 for r in rows) / len(rows) if rows else 0.0

    weighted_p = (
        sum(r.precision * r.support for r in rows) / support if support else 0.0
    )
    weighted_r = (
        sum(r.recall * r.support for r in rows) / support if support else 0.0
    )
    weighted_f1 = sum(r.f1 * r.support for r in rows) / support if support else 0.0

    baseline_micro_p = None
    baseline_micro_r = None
    baseline_micro_f1 = None
    baseline_macro_p = None
    baseline_macro_r = None
    baseline_macro_f1 = None
    baseline_weighted_p = None
    baseline_weighted_r = None
    baseline_weighted_f1 = None
    if baseline_rows is not None and baseline_rows:
        b_tp = sum(r.tp for r in baseline_rows)
        b_fp = sum(r.fp for r in baseline_rows)
        b_fn = sum(r.fn for r in baseline_rows)
        b_support = sum(r.support for r in baseline_rows)
        baseline_micro_p = b_tp / (b_tp + b_fp) if (b_tp + b_fp) else 0.0
        baseline_micro_r = b_tp / (b_tp + b_fn) if (b_tp + b_fn) else 0.0
        baseline_micro_f1 = (
            (2 * baseline_micro_p * baseline_micro_r / (baseline_micro_p + baseline_micro_r))
            if (baseline_micro_p + baseline_micro_r)
            else 0.0
        )
        baseline_macro_p = sum(r.precision for r in baseline_rows) / len(baseline_rows)
        baseline_macro_r = sum(r.recall for r in baseline_rows) / len(baseline_rows)
        baseline_macro_f1 = sum(r.f1 for r in baseline_rows) / len(baseline_rows)
        baseline_weighted_p = (
            sum(r.precision * r.support for r in baseline_rows) / b_support
            if b_support
            else 0.0
        )
        baseline_weighted_r = (
            sum(r.recall * r.support for r in baseline_rows) / b_support
            if b_support
            else 0.0
        )
        baseline_weighted_f1 = (
            sum(r.f1 * r.support for r in baseline_rows) / b_support if b_support else 0.0
        )

    table.append(
        md_row(
            [
                "**Overall (Micro)**",
                "**All labels pooled**",
                f"**{support}**",
                f"**{tp}**",
                f"**{fp}**",
                f"**{fn}**",
                f"**{tn}**",
                f"**{fmt_f(micro_p)}**",
                f"**{fmt_delta((micro_p - baseline_micro_p) if baseline_micro_p is not None else None)}**",
                f"**{fmt_f(micro_r)}**",
                f"**{fmt_delta((micro_r - baseline_micro_r) if baseline_micro_r is not None else None)}**",
                f"**{fmt_f(micro_f1)}**",
                f"**{fmt_delta((micro_f1 - baseline_micro_f1) if baseline_micro_f1 is not None else None)}**",
            ]
        )
    )
    table.append(
        md_row(
            [
                "**Overall (Macro)**",
                "**Unweighted label mean**",
                "**-**",
                "**-**",
                "**-**",
                "**-**",
                "**-**",
                f"**{fmt_f(macro_p)}**",
                f"**{fmt_delta((macro_p - baseline_macro_p) if baseline_macro_p is not None else None)}**",
                f"**{fmt_f(macro_r)}**",
                f"**{fmt_delta((macro_r - baseline_macro_r) if baseline_macro_r is not None else None)}**",
                f"**{fmt_f(macro_f1)}**",
                f"**{fmt_delta((macro_f1 - baseline_macro_f1) if baseline_macro_f1 is not None else None)}**",
            ]
        )
    )
    table.append(
        md_row(
            [
                "**Overall (Support-Weighted)**",
                "**Weighted by label support**",
                f"**{support}**",
                "**-**",
                "**-**",
                "**-**",
                "**-**",
                f"**{fmt_f(weighted_p)}**",
                f"**{fmt_delta((weighted_p - baseline_weighted_p) if baseline_weighted_p is not None else None)}**",
                f"**{fmt_f(weighted_r)}**",
                f"**{fmt_delta((weighted_r - baseline_weighted_r) if baseline_weighted_r is not None else None)}**",
                f"**{fmt_f(weighted_f1)}**",
                f"**{fmt_delta((weighted_f1 - baseline_weighted_f1) if baseline_weighted_f1 is not None else None)}**",
            ]
        )
    )
    return table


def main() -> None:
    args = parse_args()
    per_label_csv = Path(args.per_label_csv)
    if not per_label_csv.exists():
        raise SystemExit(f"Missing per_label CSV: {per_label_csv}")

    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else per_label_csv.parent / "summary.json"
    )
    if not summary_json.exists():
        raise SystemExit(
            f"Missing summary JSON: {summary_json}\n"
            "Pass --summary-json with a valid path."
        )

    n_examples = load_n_examples(summary_json)
    rows = load_rows(per_label_csv, n_examples)

    baseline_rows = None
    if args.baseline_per_label_csv:
        baseline_csv = Path(args.baseline_per_label_csv)
        if not baseline_csv.exists():
            raise SystemExit(f"Missing baseline per_label CSV: {baseline_csv}")
        baseline_rows = load_rows(baseline_csv, n_examples=None)

    table_lines = build_table(
        rows,
        include_id_subtotal=args.include_id_subtotal,
        baseline_rows=baseline_rows,
    )
    output = "\n".join(table_lines) + "\n"

    if args.out_file:
        out_path = Path(args.out_file)
        out_path.write_text(output)
        print(f"Wrote: {out_path}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
