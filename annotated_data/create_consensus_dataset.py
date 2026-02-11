from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_INPUT = "annotated_data/db_20260129_tokenised.jsonl"
DEFAULT_OUTPUT = "annotated_data/db_20260129_tokenised_consensus.jsonl"
CONSENSUS_METHOD = "token_level_total_agreement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a derived JSONL dataset with consensus spans for two annotators. "
            "Consensus spans are tokens where both annotators have exactly the same "
            "non-empty label set."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--annotator-a",
        default=None,
        help="First annotator id. If omitted, auto-detect exactly two annotators.",
    )
    parser.add_argument(
        "--annotator-b",
        default=None,
        help="Second annotator id. If omitted, auto-detect exactly two annotators.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop records with no consensus spans.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on malformed pairs/inconsistent rows. Use --no-strict to skip.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for raw in file:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def detect_annotators(rows: list[dict[str, Any]]) -> tuple[str, str]:
    annotators = sorted(
        {
            row.get("_annotator_id")
            for row in rows
            if isinstance(row.get("_annotator_id"), str) and row.get("_annotator_id")
        }
    )
    if len(annotators) != 2:
        raise ValueError(
            "Expected exactly two annotators for auto-detection, "
            f"found {len(annotators)}: {annotators}"
        )
    return annotators[0], annotators[1]


def build_pairs(
    rows: list[dict[str, Any]],
    annotator_a: str,
    annotator_b: str,
    *,
    strict: bool,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    target_annotators = {annotator_a, annotator_b}
    grouped: dict[Any, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in rows:
        annotator_id = row.get("_annotator_id")
        if annotator_id not in target_annotators:
            continue

        task_hash = row.get("_task_hash")
        if task_hash is None:
            if strict:
                raise ValueError("Found row without _task_hash for target annotators.")
            continue

        if annotator_id in grouped[task_hash]:
            if strict:
                raise ValueError(
                    f"Duplicate row for task_hash={task_hash}, annotator={annotator_id}."
                )
            continue

        grouped[task_hash][annotator_id] = row

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_hash, pair in grouped.items():
        if set(pair.keys()) != target_annotators:
            if strict:
                raise ValueError(
                    "Expected exactly one row per annotator for "
                    f"task_hash={task_hash}, found {sorted(pair.keys())}."
                )
            continue
        pairs.append((pair[annotator_a], pair[annotator_b]))

    return pairs


def token_label_sets(row: dict[str, Any], *, strict: bool) -> list[set[str]]:
    tokens = row.get("tokens") or []
    labels_by_token: list[set[str]] = [set() for _ in tokens]

    for span in row.get("spans") or []:
        token_start = span.get("token_start")
        token_end = span.get("token_end")
        label = span.get("label")

        valid = (
            isinstance(token_start, int)
            and isinstance(token_end, int)
            and isinstance(label, str)
        )
        if not valid:
            if strict:
                raise ValueError(f"Invalid span format: {span}")
            continue

        in_range = 0 <= token_start <= token_end < len(tokens)
        if not in_range:
            if strict:
                raise ValueError(
                    "Span token bounds out of range. "
                    f"token_start={token_start}, token_end={token_end}, "
                    f"n_tokens={len(tokens)}"
                )
            continue

        for token_idx in range(token_start, token_end + 1):
            labels_by_token[token_idx].add(label)

    return labels_by_token


def consensus_token_labels(
    labels_a: list[set[str]], labels_b: list[set[str]]
) -> list[set[str]]:
    if len(labels_a) != len(labels_b):
        raise ValueError(
            "Token length mismatch between annotators: "
            f"{len(labels_a)} vs {len(labels_b)}"
        )

    agreed: list[set[str]] = []
    for left, right in zip(labels_a, labels_b):
        if left == right and left:
            agreed.append(set(left))
        else:
            agreed.append(set())
    return agreed


def labels_to_spans(
    labels_by_token: list[set[str]], tokens: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    all_labels = sorted({label for labels in labels_by_token for label in labels})
    spans: list[dict[str, Any]] = []

    for label in all_labels:
        current_start: int | None = None

        for idx, labels in enumerate(labels_by_token):
            has_label = label in labels
            if has_label and current_start is None:
                current_start = idx
            elif not has_label and current_start is not None:
                spans.append(make_span(tokens, current_start, idx - 1, label))
                current_start = None

        if current_start is not None:
            spans.append(make_span(tokens, current_start, len(labels_by_token) - 1, label))

    spans.sort(key=lambda span: (span["token_start"], span["token_end"], span["label"]))
    return spans


def make_span(
    tokens: list[dict[str, Any]], token_start: int, token_end: int, label: str
) -> dict[str, Any]:
    return {
        "start": tokens[token_start]["start"],
        "end": tokens[token_end]["end"],
        "token_start": token_start,
        "token_end": token_end,
        "label": label,
    }


def consensus_accept(row_a: dict[str, Any], row_b: dict[str, Any]) -> list[str]:
    overlap = set(row_a.get("accept") or []).intersection(row_b.get("accept") or [])
    if not overlap:
        return []

    ordered: list[str] = []
    for option in row_a.get("options") or []:
        option_id = option.get("id")
        if option_id in overlap and option_id not in ordered:
            ordered.append(option_id)

    for option_id in sorted(overlap):
        if option_id not in ordered:
            ordered.append(option_id)

    return ordered


def build_consensus_row(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    *,
    annotator_a: str,
    annotator_b: str,
    strict: bool,
) -> dict[str, Any]:
    text_a = row_a.get("text")
    text_b = row_b.get("text")
    tokens_a = row_a.get("tokens") or []
    tokens_b = row_b.get("tokens") or []

    if strict and text_a != text_b:
        raise ValueError("Annotator text mismatch for same _task_hash.")
    if strict and len(tokens_a) != len(tokens_b):
        raise ValueError("Annotator token count mismatch for same _task_hash.")

    labels_a = token_label_sets(row_a, strict=strict)
    labels_b = token_label_sets(row_b, strict=strict)
    agreed_labels = consensus_token_labels(labels_a, labels_b)
    spans = labels_to_spans(agreed_labels, tokens_a)

    consensus_row = copy.deepcopy(row_a)
    consensus_row["spans"] = spans
    consensus_row["accept"] = consensus_accept(row_a, row_b)
    consensus_row["_annotator_id"] = "consensus"
    consensus_row["_session_id"] = "consensus"
    consensus_row["consensus_info"] = {
        "annotators": [annotator_a, annotator_b],
        "method": CONSENSUS_METHOD,
    }
    consensus_row.pop("_timestamp", None)
    return consensus_row


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)

    if args.annotator_a is None and args.annotator_b is None:
        annotator_a, annotator_b = detect_annotators(rows)
    elif args.annotator_a and args.annotator_b:
        annotator_a, annotator_b = args.annotator_a, args.annotator_b
    else:
        raise ValueError("Provide both --annotator-a and --annotator-b, or neither.")

    pairs = build_pairs(rows, annotator_a, annotator_b, strict=args.strict)
    consensus_rows = [
        build_consensus_row(
            row_a,
            row_b,
            annotator_a=annotator_a,
            annotator_b=annotator_b,
            strict=args.strict,
        )
        for row_a, row_b in pairs
    ]
    if args.drop_empty:
        consensus_rows = [row for row in consensus_rows if row.get("spans")]

    write_jsonl(args.output, consensus_rows)
    print(
        f"Wrote {len(consensus_rows)} consensus rows to {args.output} "
        f"from {len(pairs)} paired tasks ({annotator_a} vs {annotator_b})."
    )


if __name__ == "__main__":
    main()
