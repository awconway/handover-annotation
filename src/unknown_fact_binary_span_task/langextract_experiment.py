from __future__ import annotations

import dataclasses
import json
import random
import time
from pathlib import Path
from typing import Any

import srsly
from span_metric.soft_f1 import label_aware_soft_f1

UNKNOWN_FACT_LABEL = "Unknown fact"

DEFAULT_PROMPT_DESCRIPTION = (
    """
    Extract quotes from the text of a clinical handover transcript that explicitly state
    unknown facts. Unknown fact statements include open uncertainty about missing information.

    Examples:
    - "I don't know his allergies."
    - "Not sure if consent's been signed."

    Instructions
    - Extract each unknown-fact statement as a separate quote.
    - Label all extracted quotes as "Unknown fact".
    - Do not extract hedging or vague language unless it is an explicit lack of knowledge.
    - Do not combine text from different parts of the text into a single quote.
    - Extracted quotes must appear exactly as written in the original transcript.
    - If no unknown-fact statements are present, return an empty list.
    """
)


@dataclasses.dataclass(frozen=True)
class UnknownFactItem:
    label: str
    quote: str


@dataclasses.dataclass(frozen=True)
class UnknownFactRecord:
    text: str
    items: list[UnknownFactItem]
    gold_spans: list[dict[str, Any]]
    annotator_id: str | None


def _valid_gold_spans_from_text_and_spans(
    text: str, raw_spans: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    spans = raw_spans or []
    text_len = len(text)
    valid_gold_spans: list[dict[str, Any]] = []

    for span in spans:
        if not isinstance(span, dict):
            continue
        label = span.get("label")
        start = span.get("start")
        end = span.get("end")

        if label != UNKNOWN_FACT_LABEL:
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > text_len:
            continue

        valid_gold_spans.append({"start": start, "end": end, "label": str(label)})

    valid_gold_spans.sort(key=lambda span: (span["start"], span["end"], span["label"]))
    return valid_gold_spans


def _span_items_from_gold_spans(
    text: str, raw_spans: list[dict[str, Any]] | None
) -> list[UnknownFactItem]:
    valid_spans = _valid_gold_spans_from_text_and_spans(text=text, raw_spans=raw_spans)
    items: list[UnknownFactItem] = []
    seen: set[tuple[str, str]] = set()

    for span in valid_spans:
        quote = text[span["start"] : span["end"]]
        if not quote:
            continue

        key = (span["label"], quote)
        if key in seen:
            continue
        seen.add(key)
        items.append(UnknownFactItem(label=span["label"], quote=quote))

    return items


def span_items_from_record(record: dict[str, Any]) -> list[UnknownFactItem]:
    text = str(record.get("text") or "")
    return _span_items_from_gold_spans(text=text, raw_spans=record.get("spans") or [])


def load_unknown_fact_records(
    path: str, annotator_id: str | None = None
) -> list[UnknownFactRecord]:
    records: list[UnknownFactRecord] = []
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
        records.append(
            UnknownFactRecord(
                text=text,
                items=items,
                gold_spans=gold_spans,
                annotator_id=row.get("_annotator_id"),
            )
        )

    return records


def _require_langextract() -> Any:
    try:
        import langextract as lx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "langextract is not installed. Install it first, for example: "
            "uv pip install langextract"
        ) from exc
    return lx


def _to_langextract_example(lx: Any, record: UnknownFactRecord) -> Any:
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


def _extract_items_from_prediction(prediction: Any) -> list[UnknownFactItem]:
    docs: list[Any]
    if isinstance(prediction, list):
        docs = prediction
    else:
        docs = [prediction]

    items: list[UnknownFactItem] = []
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

            normalized_label = str(label).strip().lower() if label is not None else ""
            if normalized_label != UNKNOWN_FACT_LABEL.lower():
                continue
            if not isinstance(quote, str) or not quote:
                continue

            key = (UNKNOWN_FACT_LABEL, quote)
            if key in seen:
                continue
            seen.add(key)
            items.append(UnknownFactItem(label=UNKNOWN_FACT_LABEL, quote=quote))

    return items


def _ensure_positive_few_shot_examples(
    training_records: list[UnknownFactRecord],
    fallback_records: list[UnknownFactRecord] | None,
    train_examples: int,
) -> list[UnknownFactRecord]:
    positive_records = [r for r in training_records if r.gold_spans]
    if not positive_records and fallback_records:
        positive_records = [r for r in fallback_records if r.gold_spans]
        if positive_records:
            print(
                "Warning: no unknown-fact examples in selected training slice; "
                "falling back to full available pool for few-shot examples."
            )

    if not positive_records:
        raise ValueError(
            "No unknown-fact examples found in selected training examples for LangExtract few-shot examples."
        )

    if len(positive_records) >= train_examples:
        return positive_records[:train_examples]

    negative_records = [r for r in training_records if not r.gold_spans]
    if not negative_records:
        return positive_records

    return positive_records + negative_records[: train_examples - len(positive_records)]


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
    max_workers: int | None,
    lm_timeout_seconds: int | None,
    lm_num_threads: int | None,
    lm_max_output_tokens: int | None,
) -> Any:
    language_model_params: dict[str, Any] = {}
    if lm_timeout_seconds is not None:
        language_model_params["timeout"] = int(lm_timeout_seconds)
    if lm_num_threads is not None:
        language_model_params["num_threads"] = int(lm_num_threads)
    if lm_max_output_tokens is not None:
        language_model_params["max_output_tokens"] = int(lm_max_output_tokens)

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
        "max_workers": max_workers,
        "language_model_params": language_model_params or None,
    }
    try:
        return lx.extract(text_or_documents=text, **kwargs)
    except TypeError:
        return lx.extract(text=text, **kwargs)


def _records_from_dspy_examples(examples: list[Any]) -> list[UnknownFactRecord]:
    records: list[UnknownFactRecord] = []
    for ex in examples:
        text = str(getattr(ex, "text", "") or "")
        if not text:
            continue
        try:
            spans = ex["gold_spans"]
        except Exception:
            spans = []

        gold_spans = _valid_gold_spans_from_text_and_spans(text=text, raw_spans=spans)
        items = _span_items_from_gold_spans(text=text, raw_spans=gold_spans)
        records.append(
            UnknownFactRecord(
                text=text,
                items=items,
                gold_spans=gold_spans,
                annotator_id=getattr(ex, "_annotator_id", None),
            )
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


def _items_to_pred_items(items: list[UnknownFactItem]) -> list[dict[str, str]]:
    return [{"label": item.label, "quote": item.quote} for item in items]


def _eval_row_for_record(
    *,
    record: UnknownFactRecord,
    pred_items: list[UnknownFactItem],
    span_metrics: dict[str, Any],
) -> dict[str, Any]:
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


def _load_existing_rows(out_file: str) -> list[dict[str, Any]]:
    path = Path(out_file)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line_no, row in enumerate(srsly.read_jsonl(str(path)), start=1):
        if not isinstance(row, dict):
            raise ValueError(
                "Existing eval file has non-object JSONL row "
                f"at line {line_no}: {out_file}"
            )
        rows.append(row)
    return rows


def iou_span_metrics(
    *, text: str, gold_spans: list[dict[str, Any]], pred_items: list[UnknownFactItem]
) -> dict[str, Any]:
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

    if not out["detailed"]["golds"] and not out["detailed"]["preds"]:
        metrics["f1"] = 1.0

    return metrics


def run_langextract_unknown_fact_experiment(
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
    max_workers: int | None = None,
    lm_timeout_seconds: int | None = None,
    lm_num_threads: int | None = None,
    lm_max_output_tokens: int | None = None,
    max_retries: int = 1,
    retry_delay_seconds: float = 1.5,
    use_dataset_test_split: bool = False,
    dry_run: bool = False,
) -> dict[str, float]:
    if train_examples < 1 or eval_examples < 1:
        raise ValueError("train_examples and eval_examples must both be >= 1.")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1 when provided.")
    if lm_timeout_seconds is not None and lm_timeout_seconds < 1:
        raise ValueError("lm_timeout_seconds must be >= 1 when provided.")
    if lm_num_threads is not None and lm_num_threads < 1:
        raise ValueError("lm_num_threads must be >= 1 when provided.")
    if lm_max_output_tokens is not None and lm_max_output_tokens < 1:
        raise ValueError("lm_max_output_tokens must be >= 1 when provided.")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1.")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be >= 0.")

    if use_dataset_test_split:
        from data.dataset import prepare_dataset_unknown_fact_binary_span

        trainset, testset = prepare_dataset_unknown_fact_binary_span(
            data_file, annotator_id=annotator_id
        )
        training_pool = _records_from_dspy_examples(trainset)
        held_out_pool = _records_from_dspy_examples(testset)

        if not training_pool:
            raise ValueError("No unknown-fact training records found in dataset train split.")
        if not held_out_pool:
            raise ValueError("No unknown-fact test records found in dataset test split.")

        train_count = min(train_examples, len(training_pool))
        eval_count = min(eval_examples, len(held_out_pool))

        training_records = training_pool[:train_count]
        held_out_records = held_out_pool[:eval_count]
    else:
        pool = load_unknown_fact_records(data_file, annotator_id=annotator_id)
        if len(pool) < 2:
            raise ValueError("Not enough unknown-fact records found to run an experiment.")

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

    output_path = Path(output_file)
    existing_rows = _load_existing_rows(output_file)
    start_idx = len(existing_rows)
    if start_idx > eval_count:
        raise ValueError(
            "Existing eval file has more rows than current eval set: "
            f"{start_idx} > {eval_count} ({output_file})"
        )

    f1_sum = 0.0
    for row in existing_rows:
        try:
            f1_sum += float(row.get("score", 0.0))
        except (TypeError, ValueError):
            pass

    if start_idx:
        print(f"Resuming from {start_idx}/{eval_count} completed records in {output_file}")
    if start_idx == eval_count:
        print("All records already evaluated. Skipping extraction loop.")
        return {
            "num_train_examples": float(train_count),
            "num_eval_examples": float(eval_count),
            "average_f1": f1_sum / eval_count,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if start_idx > 0 else "w"
    with output_path.open(mode, encoding="utf-8") as out_f:
        if mode == "a" and output_path.stat().st_size > 0:
            with output_path.open("rb") as check_f:
                check_f.seek(-1, 2)
                if check_f.read(1) != b"\n":
                    out_f.write("\n")

        if dry_run:
            for idx0, record in enumerate(held_out_records[start_idx:], start=start_idx):
                metrics = iou_span_metrics(
                    text=record.text,
                    gold_spans=record.gold_spans,
                    pred_items=[],
                )
                f1_sum += metrics["f1"]
                row = _eval_row_for_record(
                    record=record,
                    pred_items=[],
                    span_metrics=metrics,
                )
                out_f.write(json.dumps(row, ensure_ascii=False))
                out_f.write("\n")
                out_f.flush()
                print(f"Processed {idx0 + 1}/{eval_count} records (dry run)")
            return {
                "num_train_examples": float(train_count),
                "num_eval_examples": float(eval_count),
                "average_f1": f1_sum / eval_count,
            }

        lx = _require_langextract()
        few_shot_pool: list[UnknownFactRecord]
        if use_dataset_test_split:
            few_shot_pool = training_pool
        else:
            few_shot_pool = training_records

        few_shot_training_records = _ensure_positive_few_shot_examples(
            few_shot_pool, held_out_records, train_count
        )
        few_shot_examples = [
            _to_langextract_example(lx, record)
            for record in few_shot_training_records
        ]
        requested_validation_level = _parse_prompt_validation_level(
            lx, prompt_validation_level
        )
        if requested_validation_level != lx.prompt_validation.PromptValidationLevel.OFF:
            validation_examples = [
                ex
                for ex in few_shot_examples
                if getattr(ex, "extractions", None)
            ]
            if validation_examples:
                report = lx.prompt_validation.validate_prompt_alignment(
                    examples=validation_examples,
                    aligner=lx.resolver.WordAligner(),
                )
                lx.prompt_validation.handle_alignment_report(
                    report,
                    level=requested_validation_level,
                    strict_non_exact=prompt_validation_strict,
                )

        for idx0, record in enumerate(held_out_records[start_idx:], start=start_idx):
            last_error: Exception | None = None
            raw_prediction: Any | None = None
            prediction_error: str | None = None

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
                        max_workers=max_workers,
                        lm_timeout_seconds=lm_timeout_seconds,
                        lm_num_threads=lm_num_threads,
                        lm_max_output_tokens=lm_max_output_tokens,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < max_retries:
                        print(
                            "LangExtract unknown fact call failed "
                            f"(attempt {attempt}/{max_retries}): {exc}"
                        )
                        if retry_delay_seconds > 0:
                            time.sleep(retry_delay_seconds)

            if raw_prediction is None:
                assert last_error is not None
                prediction_error = f"{type(last_error).__name__}: {last_error}"
                print(
                    "LangExtract unknown fact giving empty prediction after "
                    f"{max_retries} attempts: {prediction_error}"
                )
                pred_items = []
            else:
                pred_items = _extract_items_from_prediction(raw_prediction)

            metrics = iou_span_metrics(
                text=record.text,
                gold_spans=record.gold_spans,
                pred_items=pred_items,
            )
            f1_sum += metrics["f1"]

            row = _eval_row_for_record(
                record=record,
                pred_items=pred_items,
                span_metrics=metrics,
            )
            if prediction_error is not None:
                row["error"] = {"prediction_error": prediction_error, "metric_error": None}

            out_f.write(json.dumps(row, ensure_ascii=False))
            out_f.write("\n")
            out_f.flush()
            print(
                f"Processed {idx0 + 1}/{eval_count} records "
                f"(score={metrics['f1']:.4f})"
            )

    return {
        "num_train_examples": float(train_count),
        "num_eval_examples": float(eval_count),
        "average_f1": f1_sum / eval_count,
    }
