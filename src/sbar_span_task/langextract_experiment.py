from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Any

import srsly

SBAR_ALLOWED_LABELS = {
    "SITUATION",
    "BACKGROUND",
    "ASSESSMENT",
    "RECOMMENDATION",
}

DEFAULT_PROMPT_DESCRIPTION = (
    "Extract exact quote spans from the handover transcript and assign each span to one "
    "SBAR label: SITUATION, BACKGROUND, ASSESSMENT, or RECOMMENDATION. "
    "Do not paraphrase. Ignore text that does not belong to SBAR."
)


@dataclasses.dataclass(frozen=True)
class SbarItem:
    label: str
    quote: str


@dataclasses.dataclass(frozen=True)
class SbarRecord:
    text: str
    items: list[SbarItem]
    annotator_id: str | None


def _span_items_from_text_and_spans(
    text: str, raw_spans: list[dict[str, Any]] | None
) -> list[SbarItem]:
    """Return valid SBAR span items from text and raw spans."""
    spans = raw_spans or []

    valid_spans = [
        span
        for span in spans
        if isinstance(span, dict) and span.get("label") in SBAR_ALLOWED_LABELS
    ]
    valid_spans.sort(key=lambda span: (span.get("start", -1), span.get("end", -1)))

    items: list[SbarItem] = []
    seen: set[tuple[str, str]] = set()
    text_len = len(text)

    for span in valid_spans:
        start = span.get("start")
        end = span.get("end")
        label = span.get("label")

        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > text_len:
            continue

        quote = text[start:end]
        if not quote:
            continue

        key = (str(label), quote)
        if key in seen:
            continue
        seen.add(key)
        items.append(SbarItem(label=str(label), quote=quote))

    return items


def span_items_from_record(record: dict[str, Any]) -> list[SbarItem]:
    """Return valid SBAR span items from a raw Prodigy-style record."""
    text = str(record.get("text") or "")
    return _span_items_from_text_and_spans(
        text=text, raw_spans=record.get("spans") or []
    )


def load_sbar_records(path: str, annotator_id: str | None = None) -> list[SbarRecord]:
    records: list[SbarRecord] = []
    for row in srsly.read_jsonl(path):
        if not isinstance(row, dict):
            continue
        if annotator_id is not None and row.get("_annotator_id") != annotator_id:
            continue

        text = str(row.get("text") or "")
        if not text:
            continue

        items = span_items_from_record(row)
        if not items:
            continue

        records.append(
            SbarRecord(
                text=text,
                items=items,
                annotator_id=row.get("_annotator_id"),
            )
        )

    return records


def exact_match_metrics(gold_items: list[SbarItem], pred_items: list[SbarItem]) -> dict[str, float]:
    """Compute exact-match precision/recall/f1 over (label, quote)."""
    gold_set = {(item.label, item.quote) for item in gold_items}
    pred_set = {(item.label, item.quote) for item in pred_items}

    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return {
        "true_positives": float(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _require_langextract() -> Any:
    try:
        import langextract as lx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "langextract is not installed. Install it first, for example: "
            "uv pip install langextract"
        ) from exc
    return lx


def _to_langextract_example(lx: Any, record: SbarRecord) -> Any:
    return lx.data.ExampleData(
        text=record.text,
        extractions=[
            lx.data.Extraction(
                extraction_text=item.quote,
                extraction_class=item.label,
                attributes={},
            )
            for item in record.items
        ],
    )


def _extract_items_from_prediction(prediction: Any) -> list[SbarItem]:
    docs: list[Any]
    if isinstance(prediction, list):
        docs = prediction
    else:
        docs = [prediction]

    items: list[SbarItem] = []
    seen: set[tuple[str, str]] = set()

    for doc in docs:
        if isinstance(doc, dict):
            extractions = doc.get("extractions")
        else:
            extractions = getattr(doc, "extractions", None)

        if extractions is None:
            extractions = [doc]

        for extraction in extractions:
            if isinstance(extraction, dict):
                label = extraction.get("extraction_class") or extraction.get("label")
                quote = (
                    extraction.get("extraction_text")
                    or extraction.get("quote")
                    or extraction.get("text")
                )
                attrs = extraction.get("attributes") or {}
            else:
                label = getattr(extraction, "extraction_class", None) or getattr(
                    extraction, "label", None
                )
                quote = (
                    getattr(extraction, "extraction_text", None)
                    or getattr(extraction, "quote", None)
                    or getattr(extraction, "text", None)
                )
                attrs = getattr(extraction, "attributes", None) or {}

            if not label and isinstance(attrs, dict):
                label = attrs.get("label")

            if not isinstance(label, str) or label not in SBAR_ALLOWED_LABELS:
                continue
            if not isinstance(quote, str) or not quote:
                continue

            key = (label, quote)
            if key in seen:
                continue
            seen.add(key)
            items.append(SbarItem(label=label, quote=quote))

    return items


def _call_extract_api(
    lx: Any,
    *,
    text: str,
    prompt_description: str,
    examples: list[Any],
    model_id: str,
    api_key: str | None,
    fence_output: bool | None,
    use_schema_constraints: bool,
) -> Any:
    kwargs = {
        "prompt_description": prompt_description,
        "examples": examples,
        "model_id": model_id,
        "api_key": api_key,
        "fence_output": fence_output,
        "use_schema_constraints": use_schema_constraints,
    }
    try:
        return lx.extract(text_or_documents=text, **kwargs)
    except TypeError:
        return lx.extract(text=text, **kwargs)


def _items_to_dicts(items: list[SbarItem]) -> list[dict[str, str]]:
    return [dataclasses.asdict(item) for item in items]


def _records_from_dspy_examples(examples: list[Any]) -> list[SbarRecord]:
    records: list[SbarRecord] = []
    for ex in examples:
        text = str(getattr(ex, "text", "") or "")
        if not text:
            continue
        try:
            spans = ex["gold_spans"]
        except Exception:
            spans = []

        items = _span_items_from_text_and_spans(text=text, raw_spans=spans)
        if not items:
            continue
        records.append(SbarRecord(text=text, items=items, annotator_id=None))

    return records


def run_langextract_sbar_experiment(
    *,
    data_file: str,
    output_file: str,
    model_id: str,
    train_examples: int = 24,
    eval_examples: int = 20,
    annotator_id: str | None = None,
    seed: int = 339,
    prompt_description: str = DEFAULT_PROMPT_DESCRIPTION,
    api_key: str | None = None,
    fence_output: bool | None = None,
    use_schema_constraints: bool = True,
    use_dataset_test_split: bool = False,
    dry_run: bool = False,
) -> dict[str, float]:
    if train_examples < 1 or eval_examples < 1:
        raise ValueError("train_examples and eval_examples must both be >= 1.")

    if use_dataset_test_split:
        from data.dataset import prepare_dataset_sbar_span

        trainset, testset = prepare_dataset_sbar_span(
            data_file, annotator_id=annotator_id
        )
        training_pool = _records_from_dspy_examples(trainset)
        held_out_pool = _records_from_dspy_examples(testset)

        if not training_pool:
            raise ValueError("No SBAR training records found in dataset train split.")
        if not held_out_pool:
            raise ValueError("No SBAR test records found in dataset test split.")

        train_count = min(train_examples, len(training_pool))
        eval_count = min(eval_examples, len(held_out_pool))

        training_records = training_pool[:train_count]
        held_out_records = held_out_pool[:eval_count]
    else:
        pool = load_sbar_records(data_file, annotator_id=annotator_id)
        if len(pool) < 2:
            raise ValueError("Not enough SBAR records found to run an experiment.")

        rng = random.Random(seed)
        rng.shuffle(pool)

        train_count = min(train_examples, max(1, len(pool) - 1))
        eval_count = min(eval_examples, len(pool) - train_count)
        if eval_count < 1:
            raise ValueError(
                "No held-out records left for evaluation. Reduce --train-examples."
            )

        training_records = pool[:train_count]
        held_out_records = pool[train_count : train_count + eval_count]

    rows: list[dict[str, Any]] = []
    f1_sum = 0.0

    if dry_run:
        for record in held_out_records:
            rows.append(
                {
                    "text": record.text,
                    "gold_items": _items_to_dicts(record.items),
                    "pred_items": [],
                    "metrics": exact_match_metrics(record.items, []),
                    "model_id": model_id,
                    "dry_run": True,
                }
            )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        srsly.write_jsonl(output_file, rows)
        return {
            "num_train_examples": float(train_count),
            "num_eval_examples": float(eval_count),
            "average_f1": 0.0,
        }

    lx = _require_langextract()
    few_shot_examples = [
        _to_langextract_example(lx, record) for record in training_records
    ]

    for record in held_out_records:
        raw_prediction = _call_extract_api(
            lx,
            text=record.text,
            prompt_description=prompt_description,
            examples=few_shot_examples,
            model_id=model_id,
            api_key=api_key,
            fence_output=fence_output,
            use_schema_constraints=use_schema_constraints,
        )
        pred_items = _extract_items_from_prediction(raw_prediction)
        metrics = exact_match_metrics(record.items, pred_items)
        f1_sum += metrics["f1"]

        rows.append(
            {
                "text": record.text,
                "gold_items": _items_to_dicts(record.items),
                "pred_items": _items_to_dicts(pred_items),
                "metrics": metrics,
                "model_id": model_id,
            }
        )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(output_file, rows)

    return {
        "num_train_examples": float(train_count),
        "num_eval_examples": float(eval_count),
        "average_f1": f1_sum / eval_count,
    }
