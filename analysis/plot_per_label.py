import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jsonl_path",
        help="Path to eval JSONL (expects prediction.span_metrics.detailed).",
    )
    parser.add_argument(
        "output_path",
        help="Path to write plot image (e.g., .png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.jsonl_path)
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    labels = set()
    gold_count = defaultdict(int)
    pred_count = defaultdict(int)
    matched_gold = defaultdict(int)
    matched_pred = defaultdict(int)
    match_ious = defaultdict(list)

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred = obj.get("prediction", {})
            if not isinstance(pred, dict):
                continue
            sm = pred.get("span_metrics")
            if not isinstance(sm, dict):
                continue
            detailed = sm.get("detailed", {})
            if not isinstance(detailed, dict):
                continue

            golds = detailed.get("golds", []) or []
            preds = detailed.get("preds", []) or []
            matches = detailed.get("matches", []) or []

            gold_by_idx = {g.get("idx"): g for g in golds if isinstance(g, dict)}
            pred_by_idx = {p.get("idx"): p for p in preds if isinstance(p, dict)}

            for g in golds:
                if not isinstance(g, dict):
                    continue
                lab = g.get("label")
                if lab is None:
                    continue
                labels.add(lab)
                gold_count[lab] += 1

            for p in preds:
                if not isinstance(p, dict):
                    continue
                lab = p.get("label")
                if lab is None:
                    continue
                labels.add(lab)
                pred_count[lab] += 1

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
                if glab is None or plab is None:
                    continue
                matched_gold[glab] += 1
                matched_pred[plab] += 1
                if isinstance(iou, (int, float)):
                    match_ious[glab].append(float(iou))

    rows = []
    for lab in sorted(labels):
        g = gold_count[lab]
        p = pred_count[lab]
        mg = matched_gold[lab]
        mp = matched_pred[lab]
        recall = (mg / g) if g else 0.0
        precision = (mp / p) if p else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        ious = match_ious[lab]
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        rows.append((lab, precision, recall, f1, mean_iou))

    labels_sorted = [r[0] for r in rows]
    precision_vals = [r[1] for r in rows]
    recall_vals = [r[2] for r in rows]
    f1_vals = [r[3] for r in rows]
    iou_vals = [r[4] for r in rows]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".svg":
        _write_svg(
            output_path,
            labels_sorted,
            precision_vals,
            recall_vals,
            f1_vals,
            iou_vals,
        )
        return

    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:
        fallback = output_path.with_suffix(".svg")
        _write_svg(
            fallback,
            labels_sorted,
            precision_vals,
            recall_vals,
            f1_vals,
            iou_vals,
        )
        print(f"Matplotlib not available; wrote SVG to {fallback}")
        return

    x = np.arange(len(labels_sorted))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 7), dpi=160, sharex=True, height_ratios=[2, 1]
    )

    ax1.bar(x - width, precision_vals, width, label="Precision")
    ax1.bar(x, recall_vals, width, label="Recall")
    ax1.bar(x + width, f1_vals, width, label="F1")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score")
    ax1.set_title("Per-label Precision / Recall / F1")
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.legend(loc="lower right")

    ax2.bar(x, iou_vals, width=0.5, color="tab:gray")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Mean IoU")
    ax2.set_title("Per-label Mean IoU (Matched)")
    ax2.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_sorted)

    fig.tight_layout()
    fig.savefig(output_path)


def _write_svg(
    output_path: Path,
    labels: list[str],
    precision_vals: list[float],
    recall_vals: list[float],
    f1_vals: list[float],
    iou_vals: list[float],
) -> None:
    width = 900
    height = 620
    margin_left = 80
    margin_right = 40
    chart_width = width - margin_left - margin_right

    chart1_top = 50
    chart1_height = 260
    chart2_top = 370
    chart2_height = 170

    n = max(len(labels), 1)
    group_width = chart_width / n
    bar_width = group_width * 0.18

    def y_for(value: float, top: float, h: float) -> float:
        return top + (1 - value) * h

    def rect(x: float, y: float, w: float, h: float, fill: str) -> str:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"/>'

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#111}</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    # Axes and gridlines for chart 1 and 2
    for top, h in ((chart1_top, chart1_height), (chart2_top, chart2_height)):
        lines.append(
            f'<line x1="{margin_left}" y1="{top}" x2="{margin_left}" y2="{top + h}" stroke="#111" stroke-width="1"/>'
        )
        lines.append(
            f'<line x1="{margin_left}" y1="{top + h}" x2="{margin_left + chart_width}" y2="{top + h}" stroke="#111" stroke-width="1"/>'
        )
        for i in range(1, 5):
            y = top + h * (i / 5)
            lines.append(
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + chart_width}" y2="{y:.1f}" stroke="#e0e0e0" stroke-width="1"/>'
            )

    lines.append(
        '<text x="80" y="28" style="font-size:14px;font-weight:bold">Per-label Precision / Recall / F1</text>'
    )
    lines.append(
        '<text x="80" y="348" style="font-size:14px;font-weight:bold">Per-label Mean IoU (Matched)</text>'
    )

    colors = {
        "precision": "#1f77b4",
        "recall": "#2ca02c",
        "f1": "#ff7f0e",
        "iou": "#7f7f7f",
    }

    # Bars for chart 1
    for i, lab in enumerate(labels):
        group_x = margin_left + i * group_width
        px = group_x + group_width * 0.2
        rx = group_x + group_width * 0.42
        fx = group_x + group_width * 0.64

        for val, x, color in (
            (precision_vals[i], px, colors["precision"]),
            (recall_vals[i], rx, colors["recall"]),
            (f1_vals[i], fx, colors["f1"]),
        ):
            y = y_for(val, chart1_top, chart1_height)
            h = chart1_top + chart1_height - y
            lines.append(rect(x, y, bar_width, h, color))

    # Bars for chart 2
    for i, val in enumerate(iou_vals):
        group_x = margin_left + i * group_width
        bx = group_x + group_width * 0.4
        y = y_for(val, chart2_top, chart2_height)
        h = chart2_top + chart2_height - y
        lines.append(rect(bx, y, bar_width * 1.5, h, colors["iou"]))

    # Legend
    legend_x = margin_left + chart_width - 220
    legend_y = chart1_top + 10
    legend_items = [
        ("Precision", colors["precision"]),
        ("Recall", colors["recall"]),
        ("F1", colors["f1"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        y = legend_y + i * 18
        lines.append(rect(legend_x, y - 10, 12, 12, color))
        lines.append(f'<text x="{legend_x + 18}" y="{y}">{label}</text>')

    # X labels
    for i, lab in enumerate(labels):
        x = margin_left + i * group_width + group_width * 0.4
        lines.append(
            f'<text x="{x:.1f}" y="{chart2_top + chart2_height + 24}" text-anchor="middle">{lab}</text>'
        )

    # Y-axis labels
    lines.append(
        f'<text x="16" y="{chart1_top + 12}" transform="rotate(-90 16,{chart1_top + 12})">Score</text>'
    )
    lines.append(
        f'<text x="16" y="{chart2_top + 12}" transform="rotate(-90 16,{chart2_top + 12})">Mean IoU</text>'
    )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
