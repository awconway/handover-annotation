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
        rows.append((lab, g, p, mg, mp, recall, precision, f1, mean_iou))

    print(f"Per-label analysis for {path.name}")
    print(
        "label\tgold\tpred\tmatched_gold\tmatched_pred\trecall\tprecision\tf1\tmean_iou"
    )
    for lab, g, p, mg, mp, recall, precision, f1, mean_iou in rows:
        print(
            f"{lab}	{g}	{p}	{mg}	{mp}	{recall:.3f}	{precision:.3f}	{f1:.3f}	{mean_iou:.3f}"
        )


if __name__ == "__main__":
    main()
