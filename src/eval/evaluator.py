from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable

import dspy
import srsly

from checklist_task.metric_gepa import multilabel_f1_with_feedback
from span_metric.gepa_span_metric import gepa_span_metric

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY_SECONDS = 1.0


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion for JSONL outputs."""
    if hasattr(value, "toDict"):
        try:
            return value.toDict()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _reset_litellm_executor_if_shutdown(exc: Exception) -> bool:
    """
    LiteLLM uses a global executor internally. In some HPC runs this can be shut
    down prematurely, causing RuntimeError on later requests. Reset best-effort.
    """
    if "cannot schedule new futures after shutdown" not in str(exc):
        return False

    try:
        import litellm.utils as litellm_utils

        old_executor = getattr(litellm_utils, "executor", None)
        max_workers = getattr(old_executor, "_max_workers", 64)
        litellm_utils.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        return True
    except Exception:
        return False


def _predict_with_retries(
    predictor: Any,
    inputs: dict[str, Any],
    *,
    fallback_prediction_factory: Callable[[], Any],
    max_retries: int,
    retry_delay_seconds: float,
) -> tuple[Any, str | None]:
    errors: list[str] = []

    for attempt in range(1, max_retries + 1):
        try:
            return predictor(**inputs), None
        except Exception as exc:
            reset_done = _reset_litellm_executor_if_shutdown(exc)
            message = f"attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
            if reset_done:
                message += " [litellm executor reset]"
            errors.append(message)
            if attempt < max_retries:
                time.sleep(retry_delay_seconds * attempt)

    return fallback_prediction_factory(), " | ".join(errors)


def _metric_score(metric: Callable[..., Any], example: Any, pred: Any) -> tuple[float, str | None]:
    try:
        score = metric(example, pred)
        return float(score), None
    except Exception as exc:
        return 0.0, f"{type(exc).__name__}: {exc}"


def _run_eval(
    predictor: Any,
    testset: list[Any],
    out_file: str,
    *,
    metric: Callable[..., Any],
    fallback_prediction_factory: Callable[[], Any],
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = _DEFAULT_RETRY_DELAY_SECONDS,
) -> float:
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    error_count = 0
    total = len(testset)

    for idx, example in enumerate(testset, start=1):
        if hasattr(example, "inputs"):
            inputs = example.inputs()
        elif isinstance(example, dict):
            inputs = {k: v for k, v in example.items() if k != "labels" and k != "gold_spans"}
        else:
            inputs = {}

        pred, pred_error = _predict_with_retries(
            predictor,
            inputs,
            fallback_prediction_factory=fallback_prediction_factory,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        score, metric_error = _metric_score(metric, example, pred)
        scores.append(score)

        row: dict[str, Any] = {
            "example": _to_jsonable(example),
            "prediction": _to_jsonable(pred),
            "score": score,
        }

        if pred_error or metric_error:
            error_count += 1
            row["error"] = {
                "prediction_error": pred_error,
                "metric_error": metric_error,
            }

        rows.append(row)

        if idx % 10 == 0 or idx == total:
            print(f"Processed {idx}/{total} examples")

    srsly.write_jsonl(out_file, rows)

    avg = (sum(scores) / total) if total else 0.0
    print(f"Average Metric: {sum(scores):.6f} / {total} ({avg * 100:.1f}%)")
    if error_count:
        print(
            "Completed with "
            f"{error_count}/{total} example-level errors; failed examples were scored as 0."
        )

    return avg * 100.0


def evaluate_checklist(
    predictor: Any,
    testset: list[Any],
    out_file: str,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = _DEFAULT_RETRY_DELAY_SECONDS,
) -> float:
    return _run_eval(
        predictor,
        testset,
        out_file,
        metric=multilabel_f1_with_feedback,
        fallback_prediction_factory=lambda: dspy.Prediction(labels=[]),
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )


def evaluate_sbar(
    predictor: Any,
    testset: list[Any],
    out_file: str,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = _DEFAULT_RETRY_DELAY_SECONDS,
) -> float:
    return _run_eval(
        predictor,
        testset,
        out_file,
        metric=gepa_span_metric,
        fallback_prediction_factory=lambda: dspy.Prediction(pred_spans=[]),
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )


def evaluate(
    predictor: Any,
    testset: list[Any],
    out_file: str,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = _DEFAULT_RETRY_DELAY_SECONDS,
) -> float:
    """Backward-compatible alias used by run_eval.py."""
    return evaluate_checklist(
        predictor,
        testset,
        out_file,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )
