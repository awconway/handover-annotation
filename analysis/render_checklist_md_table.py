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


def load_rows(per_label_csv: Path, n_examples: int) -> list[LabelRow]:
    rows: list[LabelRow] = []
    with per_label_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            label_id = r["label"]
            tp = int(r["tp"])
            fp = int(r["fp"])
            fn = int(r["fn"])
            tn = n_examples - tp - fp - fn
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


def md_row(cols: Iterable[str]) -> str:
    safe_cols = [c.replace("|", "\\|") for c in cols]
    return "| " + " | ".join(safe_cols) + " |"


def subtotal_row(category: str, rows: list[LabelRow]) -> str:
    tp = sum(r.tp for r in rows)
    fp = sum(r.fp for r in rows)
    fn = sum(r.fn for r in rows)
    tn = sum(r.tn for r in rows)
    support = sum(r.support for r in rows)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
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
            f"**{fmt_f(r)}**",
            f"**{fmt_f(f1)}**",
        ]
    )


def build_table(
    rows: list[LabelRow], include_id_subtotal: bool
) -> list[str]:
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
                "Recall",
                "F1",
            ]
        )
    )
    table.append(md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))

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
                        fmt_f(r.recall),
                        fmt_f(r.f1),
                    ]
                )
            )

        if bucket != "id" or include_id_subtotal:
            table.append(subtotal_row(cat_name, bucket_rows))

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
                f"**{fmt_f(micro_r)}**",
                f"**{fmt_f(micro_f1)}**",
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
                f"**{fmt_f(macro_r)}**",
                f"**{fmt_f(macro_f1)}**",
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
                f"**{fmt_f(weighted_r)}**",
                f"**{fmt_f(weighted_f1)}**",
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
    table_lines = build_table(rows, include_id_subtotal=args.include_id_subtotal)
    output = "\n".join(table_lines) + "\n"

    if args.out_file:
        out_path = Path(args.out_file)
        out_path.write_text(output)
        print(f"Wrote: {out_path}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
