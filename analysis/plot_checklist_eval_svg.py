import argparse
import csv
import math
from pathlib import Path


BUCKET_COLORS = {
    "situation": "#1f77b4",
    "background": "#2ca02c",
    "assessment": "#ff7f0e",
    "recommendation": "#d62728",
    "patient_involvement": "#9467bd",
    "id": "#8c564b",
    "other": "#7f7f7f",
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready SVG figures (lollipop and dumbbell) from "
            "per_label.csv output."
        )
    )
    parser.add_argument(
        "per_label_csv",
        help="Path to per_label.csv from analyze_checklist_eval.py",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for SVG files. "
            "Default: same directory as per_label.csv."
        ),
    )
    parser.add_argument(
        "--min-support-dumbbell",
        type=int,
        default=5,
        help="Minimum support for labels included in dumbbell plot.",
    )
    return parser.parse_args()


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "label": r["label"],
                    "support": int(r["support"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "f1": float(r["f1"]),
                    "fp": int(r["fp"]),
                    "fn": int(r["fn"]),
                }
            )
    return rows


def draw_axes(svg: list[str], x0: int, y0: int, x1: int, y1: int) -> None:
    svg.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222" stroke-width="1.5"/>'
    )
    svg.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" stroke-width="1.5"/>'
    )


def draw_x_ticks(svg: list[str], x0: int, x1: int, y0: int) -> None:
    for i in range(11):
        v = i / 10
        x = x0 + (x1 - x0) * v
        svg.append(
            f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0+6}" stroke="#333" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{y0+24}" font-size="11" text-anchor="middle" fill="#333">{v:.1f}</text>'
        )
        if 0 < i < 10:
            svg.append(
                f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0-9999}" stroke="#ececec" stroke-width="1"/>'
            )


def bucket(label: str) -> str:
    return LABEL_TO_BUCKET.get(label, "other")


def label_text(label: str) -> str:
    return LABEL_TEXT.get(label, label)


def render_lollipop(rows: list[dict], out_path: Path) -> None:
    bucket_rank = {name: i for i, name in enumerate(BUCKET_ORDER)}
    rows = sorted(
        rows,
        key=lambda r: (
            bucket_rank.get(bucket(r["label"]), len(BUCKET_ORDER)),
            -r["support"],
            r["label"],
        ),
    )
    n = len(rows)

    width = 1400
    left = 330
    right = 120
    top = 120
    row_h = 28
    bottom = 80
    height = top + bottom + max(1, n) * row_h

    x0 = left
    x1 = width - right
    y_axis = height - bottom

    svg: list[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(
        '<text x="40" y="42" font-size="26" font-weight="700" fill="#111">Per-label F1 (Lollipop)</text>'
    )
    svg.append(
        '<text x="40" y="68" font-size="14" fill="#444">Sorted by bucket then support. Dot color denotes bucket and dot size denotes support.</text>'
    )

    draw_axes(svg, x0, y_axis, x1, top - 20)
    draw_x_ticks(svg, x0, x1, y_axis)

    max_support = max((r["support"] for r in rows), default=1)

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        f1 = row["f1"]
        x = x0 + (x1 - x0) * f1
        b = bucket(row["label"])
        color = BUCKET_COLORS[b]
        radius = 3.0 + 7.0 * math.sqrt(row["support"] / max_support)

        svg.append(
            f'<line x1="{x0}" y1="{y}" x2="{x:.2f}" y2="{y}" stroke="#c9c9c9" stroke-width="2"/>'
        )
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y}" r="{radius:.2f}" fill="{color}" fill-opacity="0.9" stroke="#fff" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{x0-12}" y="{y+4}" font-size="12" text-anchor="end" fill="#222">{esc(label_text(row["label"]))}</text>'
        )
        svg.append(
            f'<text x="{x1+8}" y="{y+4}" font-size="11" text-anchor="start" fill="#444">F1={f1:.3f} | n={row["support"]}</text>'
        )

    legend_x = 40
    legend_y = 92
    dx = 210
    for i, b in enumerate(
        ["id", "situation", "background", "assessment", "recommendation", "patient_involvement"]
    ):
        x = legend_x + i * dx
        svg.append(
            f'<circle cx="{x}" cy="{legend_y}" r="6" fill="{BUCKET_COLORS[b]}"/>'
        )
        svg.append(
            f'<text x="{x+12}" y="{legend_y+4}" font-size="12" fill="#333">{esc(BUCKET_DISPLAY.get(b, b.title()))}</text>'
        )

    size_legend_x = 40
    size_legend_y = legend_y + 26
    for j, frac in enumerate([0.2, 0.6, 1.0]):
        r = 3.0 + 7.0 * math.sqrt(frac)
        x = size_legend_x + j * 90
        svg.append(
            f'<circle cx="{x}" cy="{size_legend_y}" r="{r:.2f}" fill="#888" fill-opacity="0.35" stroke="#666" stroke-width="0.8"/>'
        )
        support_val = max(1, round(max_support * frac))
        svg.append(
            f'<text x="{x+16}" y="{size_legend_y+4}" font-size="11" fill="#444">n~{support_val}</text>'
        )

    svg.append("</svg>")
    out_path.write_text("\n".join(svg))


def render_dumbbell(rows: list[dict], min_support: int, out_path: Path) -> None:
    rows = [r for r in rows if r["support"] >= min_support]
    rows = sorted(rows, key=lambda r: abs(r["recall"] - r["precision"]), reverse=True)
    n = len(rows)

    width = 1400
    left = 350
    right = 160
    top = 120
    row_h = 30
    bottom = 80
    height = top + bottom + max(1, n) * row_h

    x0 = left
    x1 = width - right
    y_axis = height - bottom

    p_color = "#1f77b4"
    r_color = "#ff7f0e"

    svg: list[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(
        '<text x="40" y="42" font-size="26" font-weight="700" fill="#111">Per-label Precision vs Recall (Dumbbell)</text>'
    )
    svg.append(
        f'<text x="40" y="68" font-size="14" fill="#444">Sorted by |Recall-Precision|, support >= {min_support}. Marker size scales with support.</text>'
    )

    draw_axes(svg, x0, y_axis, x1, top - 20)
    draw_x_ticks(svg, x0, x1, y_axis)

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        p = row["precision"]
        r = row["recall"]
        xp = x0 + (x1 - x0) * p
        xr = x0 + (x1 - x0) * r
        rad = 3.5 + 0.45 * math.sqrt(row["support"])

        svg.append(
            f'<line x1="{xp:.2f}" y1="{y}" x2="{xr:.2f}" y2="{y}" stroke="#bdbdbd" stroke-width="2.4"/>'
        )
        svg.append(
            f'<circle cx="{xp:.2f}" cy="{y}" r="{rad:.2f}" fill="{p_color}" fill-opacity="0.9" stroke="white" stroke-width="1"/>'
        )
        svg.append(
            f'<circle cx="{xr:.2f}" cy="{y}" r="{rad:.2f}" fill="{r_color}" fill-opacity="0.9" stroke="white" stroke-width="1"/>'
        )
        delta = r - p
        svg.append(
            f'<text x="{x0-12}" y="{y+4}" font-size="12" text-anchor="end" fill="#222">{esc(label_text(row["label"]))}</text>'
        )
        svg.append(
            f'<text x="{x1+8}" y="{y+4}" font-size="11" text-anchor="start" fill="#444">Δ={delta:+.3f} | n={row["support"]}</text>'
        )

    legend_x = 40
    legend_y = 95
    svg.append(
        f'<circle cx="{legend_x}" cy="{legend_y}" r="6" fill="{p_color}"/><text x="{legend_x+12}" y="{legend_y+4}" font-size="12" fill="#333">Precision</text>'
    )
    svg.append(
        f'<circle cx="{legend_x+120}" cy="{legend_y}" r="6" fill="{r_color}"/><text x="{legend_x+132}" y="{legend_y+4}" font-size="12" fill="#333">Recall</text>'
    )

    svg.append("</svg>")
    out_path.write_text("\n".join(svg))


def main() -> None:
    args = parse_args()
    csv_path = Path(args.per_label_csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing file: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    lollipop_path = out_dir / "fig_option1_lollipop_f1.svg"
    dumbbell_path = out_dir / "fig_option5_dumbbell_pr.svg"

    render_lollipop(rows, lollipop_path)
    render_dumbbell(rows, args.min_support_dumbbell, dumbbell_path)

    print(f"Wrote: {lollipop_path}")
    print(f"Wrote: {dumbbell_path}")


if __name__ == "__main__":
    main()
