from __future__ import annotations

import dataclasses
import random
import time
from pathlib import Path
from typing import Any

import srsly
from span_metric.soft_f1 import label_aware_soft_f1

SBAR_ALLOWED_LABELS = {
    "SITUATION",
    "BACKGROUND",
    "ASSESSMENT",
    "RECOMMENDATION",
}

DEFAULT_PROMPT_DESCRIPTION = (
    """
    The task is to extract quotes from the text of a clinical handover transcript that aligns with the SBAR framework (SITUATION, BACKGROUND, ASSESSMENT, RECOMMENDATION) and label them accordingly.

    Instructions
    - If the transcript contains multiple different pieces of information within the same SBAR category from different sections of the text, extract each as a separate quote.
    - **Do not combine parts from different sections of the text into a single quote.**
    - Extracted quotes must appear **exactly as written in the original transcript**, preserving spelling, capitalization, punctuation, and phrasing. Do not paraphrase. Do not summarize. Do not correct incorrectly spelled words. Do not correct errors. Do not rephrase.
    - Assign quotes to the correct SBAR label based on its content, regardless of its position in the transcript.
    - If parts of the text do not fit into any SBAR category, do not extract it.
    - Quotes should only be assigned to one of the SBAR labels.
    """
)


@dataclasses.dataclass(frozen=True)
class SbarItem:
    label: str
    quote: str


@dataclasses.dataclass(frozen=True)
class SbarRecord:
    text: str
    items: list[SbarItem]
    gold_spans: list[dict[str, Any]]
    annotator_id: str | None


def _valid_gold_spans_from_text_and_spans(
    text: str, raw_spans: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Return valid SBAR gold spans with explicit char boundaries."""
    spans = raw_spans or []
    text_len = len(text)
    valid_gold_spans: list[dict[str, Any]] = []

    for span in spans:
        if not isinstance(span, dict):
            continue
        label = span.get("label")
        start = span.get("start")
        end = span.get("end")

        if label not in SBAR_ALLOWED_LABELS:
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > text_len:
            continue

        valid_gold_spans.append(
            {
                "start": start,
                "end": end,
                "label": str(label),
            }
        )

    valid_gold_spans.sort(key=lambda span: (span["start"], span["end"]))
    return valid_gold_spans


def _span_items_from_gold_spans(
    text: str, raw_spans: list[dict[str, Any]] | None
) -> list[SbarItem]:
    """Return de-duplicated SBAR label/quote items from raw spans."""
    valid_spans = _valid_gold_spans_from_text_and_spans(text=text, raw_spans=raw_spans)

    items: list[SbarItem] = []
    seen: set[tuple[str, str]] = set()

    for span in valid_spans:
        quote = text[span["start"] : span["end"]]
        if not quote:
            continue

        key = (span["label"], quote)
        if key in seen:
            continue
        seen.add(key)
        items.append(SbarItem(label=span["label"], quote=quote))

    return items


def span_items_from_record(record: dict[str, Any]) -> list[SbarItem]:
    """Return valid SBAR span items from a raw Prodigy-style record."""
    text = str(record.get("text") or "")
    return _span_items_from_gold_spans(
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

        gold_spans = _valid_gold_spans_from_text_and_spans(
            text=text,
            raw_spans=row.get("spans") or [],
        )
        if not gold_spans:
            continue
        items = _span_items_from_gold_spans(text=text, raw_spans=gold_spans)
        if not items:
            continue

        records.append(
            SbarRecord(
                text=text,
                items=items,
                gold_spans=gold_spans,
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
    prompt_validation_level: Any,
    prompt_validation_strict: bool,
    show_progress: bool,
    lm_timeout_seconds: int | None,
) -> Any:
    language_model_params: dict[str, Any] = {}
    if lm_timeout_seconds is not None:
        language_model_params["timeout"] = int(lm_timeout_seconds)

    kwargs = {
        "prompt_description": prompt_description,
        "examples": examples,
        "model_id": model_id,
        "api_key": api_key,
        "fence_output": fence_output,
        "use_schema_constraints": use_schema_constraints,
        "prompt_validation_level": prompt_validation_level,
        "prompt_validation_strict": prompt_validation_strict,
        "show_progress": show_progress,
        "language_model_params": language_model_params or None,
    }
    try:
        return lx.extract(text_or_documents=text, **kwargs)
    except TypeError:
        return lx.extract(text=text, **kwargs)


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

        gold_spans = _valid_gold_spans_from_text_and_spans(text=text, raw_spans=spans)
        if not gold_spans:
            continue
        items = _span_items_from_gold_spans(text=text, raw_spans=gold_spans)
        if not items:
            continue
        records.append(
            SbarRecord(text=text, items=items, gold_spans=gold_spans, annotator_id=None)
        )

    return records


def _parse_prompt_validation_level(lx: Any, level: str) -> Any:
    normalized = level.strip().lower()
    mapping = {
        "off": lx.prompt_validation.PromptValidationLevel.OFF,
        "warning": lx.prompt_validation.PromptValidationLevel.WARNING,
        "error": lx.prompt_validation.PromptValidationLevel.ERROR,
    }
    if normalized not in mapping:
        raise ValueError(
            "prompt_validation_level must be one of: off, warning, error."
        )
    return mapping[normalized]


def _items_to_pred_items(items: list[SbarItem]) -> list[dict[str, str]]:
    return [{"label": item.label, "quote": item.quote} for item in items]


def _eval_row_for_record(
    *,
    record: SbarRecord,
    pred_items: list[SbarItem],
    span_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build an eval JSONL row with the same schema as run_eval_sbar_span.py."""
    return {
        "example": {
            "text": record.text,
            "gold_spans": record.gold_spans,
        },
        "prediction": {
            "pred_spans": _items_to_pred_items(pred_items),
            "span_metrics": span_metrics,
        },
        "score": span_metrics["f1"],
    }


def iou_span_metrics(
    *, text: str, gold_spans: list[dict[str, Any]], pred_items: list[SbarItem]
) -> dict[str, Any]:
    """Compute IoU-based metrics using the same function as gepa_span_metric."""
    out = label_aware_soft_f1(
        text=text,
        gold_spans=gold_spans,
        pred_items=_items_to_pred_items(pred_items),
        fuzzy_threshold=0.6,
        iou_threshold=None,
        require_label_match=True,
    )
    metrics = dict(out)
    metrics["true_positives"] = out["tp"]

    # Match gepa_span_metric behavior.
    if not out["detailed"]["golds"] and not out["detailed"]["preds"]:
        metrics["f1"] = 1.0

    return metrics


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
    prompt_validation_level: str = "warning",
    prompt_validation_strict: bool = False,
    show_progress: bool = True,
    lm_timeout_seconds: int | None = None,
    max_retries: int = 1,
    retry_delay_seconds: float = 1.5,
    use_dataset_test_split: bool = False,
    dry_run: bool = False,
) -> dict[str, float]:
    if train_examples < 1 or eval_examples < 1:
        raise ValueError("train_examples and eval_examples must both be >= 1.")
    if lm_timeout_seconds is not None and lm_timeout_seconds < 1:
        raise ValueError("lm_timeout_seconds must be >= 1 when provided.")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1.")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be >= 0.")

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
            metrics = iou_span_metrics(
                text=record.text,
                gold_spans=record.gold_spans,
                pred_items=[],
            )
            rows.append(
                _eval_row_for_record(
                    record=record,
                    pred_items=[],
                    span_metrics=metrics,
                )
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
    requested_validation_level = _parse_prompt_validation_level(
        lx, prompt_validation_level
    )
    if requested_validation_level != lx.prompt_validation.PromptValidationLevel.OFF:
        report = lx.prompt_validation.validate_prompt_alignment(
            examples=few_shot_examples,
            aligner=lx.resolver.WordAligner(),
        )
        lx.prompt_validation.handle_alignment_report(
            report,
            level=requested_validation_level,
            strict_non_exact=prompt_validation_strict,
        )

    for record in held_out_records:
        last_error: Exception | None = None
        raw_prediction: Any | None = None

        for attempt in range(1, max_retries + 1):
            try:
                raw_prediction = _call_extract_api(
                    lx,
                    text=record.text,
                    prompt_description=prompt_description,
                    examples=few_shot_examples,
                    model_id=model_id,
                    api_key=api_key,
                    fence_output=fence_output,
                    use_schema_constraints=use_schema_constraints,
                    prompt_validation_level=lx.prompt_validation.PromptValidationLevel.OFF,
                    prompt_validation_strict=prompt_validation_strict,
                    show_progress=show_progress,
                    lm_timeout_seconds=lm_timeout_seconds,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    print(
                        f"LangExtract SBAR call failed (attempt {attempt}/{max_retries}): {exc}"
                    )
                    if retry_delay_seconds > 0:
                        time.sleep(retry_delay_seconds)

        if raw_prediction is None:
            assert last_error is not None
            raise last_error

        pred_items = _extract_items_from_prediction(raw_prediction)
        metrics = iou_span_metrics(
            text=record.text,
            gold_spans=record.gold_spans,
            pred_items=pred_items,
        )
        f1_sum += metrics["f1"]

        rows.append(
            _eval_row_for_record(
                record=record,
                pred_items=pred_items,
                span_metrics=metrics,
            )
        )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(output_file, rows)

    return {
        "num_train_examples": float(train_count),
        "num_eval_examples": float(eval_count),
        "average_f1": f1_sum / eval_count,
    }
