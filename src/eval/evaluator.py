from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path
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


def _load_existing_rows(out_file: str) -> list[dict[str, Any]]:
    path = Path(out_file)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line_no, row in enumerate(srsly.read_jsonl(str(path)), start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Existing eval file has non-object JSONL row at line {line_no}: {out_file}"
            )
        rows.append(row)
    return rows


def _run_eval(
    predictor: Any,
    testset: list[Any],
    out_file: str,
    *,
    metric: Callable[..., Any],
    fallback_prediction_factory: Callable[[], Any],
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = _DEFAULT_RETRY_DELAY_SECONDS,
    resume: bool = True,
    num_threads: int | None = None,
) -> float:
    scores: list[float] = []
    error_count = 0
    total = len(testset)
    out_path = Path(out_file)
    # Keep legacy behavior unless callers explicitly opt into parallelism.
    effective_num_threads = 1 if num_threads is None else num_threads
    if effective_num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {effective_num_threads}")

    def process_example(idx0: int, example: Any) -> tuple[int, dict[str, Any], float, bool, float]:
        started_at = time.perf_counter()
        try:
            if hasattr(example, "inputs"):
                inputs = example.inputs()
            elif isinstance(example, dict):
                inputs = {
                    k: v
                    for k, v in example.items()
                    if k != "labels" and k != "gold_spans"
                }
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

            row: dict[str, Any] = {
                "example": _to_jsonable(example),
                "prediction": _to_jsonable(pred),
                "score": score,
            }

            has_error = pred_error is not None or metric_error is not None
            if has_error:
                row["error"] = {
                    "prediction_error": pred_error,
                    "metric_error": metric_error,
                }
        except Exception as exc:
            score = 0.0
            has_error = True
            row = {
                "example": _to_jsonable(example),
                "prediction": _to_jsonable(fallback_prediction_factory()),
                "score": score,
                "error": {
                    "prediction_error": (
                        "internal_evaluator_error: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "metric_error": None,
                },
            }

        elapsed = time.perf_counter() - started_at
        return idx0, row, score, has_error, elapsed

    start_idx = 0
    if resume:
        existing_rows = _load_existing_rows(out_file)
        start_idx = len(existing_rows)
        if start_idx > total:
            raise ValueError(
                "Existing eval file has more rows than the current test set: "
                f"{start_idx} > {total} ({out_file})"
            )

        for row in existing_rows:
            try:
                scores.append(float(row.get("score", 0.0)))
            except (TypeError, ValueError):
                scores.append(0.0)
            if row.get("error") is not None:
                error_count += 1

        if start_idx:
            print(f"Resuming from {start_idx}/{total} completed examples in {out_file}")
    elif out_path.exists():
        print(f"Overwrite enabled; replacing existing results file: {out_file}")

    if start_idx == total:
        print("All examples are already evaluated. Skipping prediction loop.")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume and start_idx > 0 else "w"
        with out_path.open(mode, encoding="utf-8") as f:
            if mode == "a" and out_path.stat().st_size > 0:
                with out_path.open("rb") as check_f:
                    check_f.seek(-1, 2)
                    if check_f.read(1) != b"\n":
                        f.write("\n")

            pending = list(enumerate(testset[start_idx:], start=start_idx))
            if effective_num_threads == 1:
                for idx0, example in pending:
                    _, row, score, has_error, elapsed = process_example(idx0, example)
                    scores.append(score)
                    if has_error:
                        error_count += 1

                    idx = idx0 + 1
                    f.write(json.dumps(row, ensure_ascii=False))
                    f.write("\n")
                    f.flush()
                    print(
                        f"Processed {idx}/{total} examples in {elapsed:.2f}s "
                        f"(score={score:.4f})"
                    )
            else:
                print(f"Running evaluation with {effective_num_threads} threads.")
                next_to_write = start_idx
                buffered_results: dict[int, tuple[dict[str, Any], float, bool, float]] = {}

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=effective_num_threads
                ) as executor:
                    futures = [
                        executor.submit(process_example, idx0, example)
                        for idx0, example in pending
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        idx0, row, score, has_error, elapsed = future.result()
                        buffered_results[idx0] = (row, score, has_error, elapsed)

                        while next_to_write in buffered_results:
                            (
                                next_row,
                                next_score,
                                next_has_error,
                                next_elapsed,
                            ) = buffered_results.pop(next_to_write)
                            scores.append(next_score)
                            if next_has_error:
                                error_count += 1

                            idx = next_to_write + 1
                            f.write(json.dumps(next_row, ensure_ascii=False))
                            f.write("\n")
                            f.flush()
                            print(
                                f"Processed {idx}/{total} examples in {next_elapsed:.2f}s "
                                f"(score={next_score:.4f})"
                            )
                            next_to_write += 1

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
    resume: bool = True,
    num_threads: int | None = None,
) -> float:
    return _run_eval(
        predictor,
        testset,
        out_file,
        metric=multilabel_f1_with_feedback,
        fallback_prediction_factory=lambda: dspy.Prediction(labels=[]),
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        resume=resume,
        num_threads=num_threads,
    )


def evaluate_sbar(
    predictor: Any,
    testset: list[Any],
    out_file: str,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = _DEFAULT_RETRY_DELAY_SECONDS,
    resume: bool = True,
    num_threads: int | None = None,
) -> float:
    return _run_eval(
        predictor,
        testset,
        out_file,
        metric=gepa_span_metric,
        fallback_prediction_factory=lambda: dspy.Prediction(pred_spans=[]),
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        resume=resume,
        num_threads=num_threads,
    )


def evaluate(
    predictor: Any,
    testset: list[Any],
    out_file: str,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = _DEFAULT_RETRY_DELAY_SECONDS,
    resume: bool = True,
    num_threads: int | None = None,
) -> float:
    """Backward-compatible alias used by run_eval.py."""
    return evaluate_checklist(
        predictor,
        testset,
        out_file,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        resume=resume,
        num_threads=num_threads,
    )
