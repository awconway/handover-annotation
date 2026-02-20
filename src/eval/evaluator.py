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
) -> tuple[Any, str | None, dict[str, Any]]:
    errors: list[str] = []
    attempt_durations: list[float] = []
    backoff_sleep_seconds = 0.0
    prediction_started_at = time.perf_counter()

    for attempt in range(1, max_retries + 1):
        attempt_started_at = time.perf_counter()
        try:
            pred = predictor(**inputs)
            attempt_durations.append(time.perf_counter() - attempt_started_at)
            total_seconds = time.perf_counter() - prediction_started_at
            return pred, None, {
                "attempts": attempt,
                "attempt_durations": attempt_durations,
                "backoff_sleep_seconds": backoff_sleep_seconds,
                "total_seconds": total_seconds,
                "succeeded": True,
            }
        except Exception as exc:
            attempt_durations.append(time.perf_counter() - attempt_started_at)
            reset_done = _reset_litellm_executor_if_shutdown(exc)
            message = f"attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
            if reset_done:
                message += " [litellm executor reset]"
            errors.append(message)
            if attempt < max_retries:
                sleep_seconds = retry_delay_seconds * attempt
                backoff_sleep_seconds += sleep_seconds
                time.sleep(sleep_seconds)

    total_seconds = time.perf_counter() - prediction_started_at
    return fallback_prediction_factory(), " | ".join(errors), {
        "attempts": max_retries,
        "attempt_durations": attempt_durations,
        "backoff_sleep_seconds": backoff_sleep_seconds,
        "total_seconds": total_seconds,
        "succeeded": False,
    }


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
    timing_logs: bool = False,
    timing_log_every: int = 1,
    timing_slow_threshold_seconds: float | None = None,
) -> float:
    scores: list[float] = []
    error_count = 0
    total = len(testset)
    out_path = Path(out_file)
    # Keep legacy behavior unless callers explicitly opt into parallelism.
    effective_num_threads = 1 if num_threads is None else num_threads
    if effective_num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {effective_num_threads}")
    if timing_log_every < 1:
        raise ValueError(f"timing_log_every must be >= 1, got {timing_log_every}")
    if (
        timing_slow_threshold_seconds is not None
        and timing_slow_threshold_seconds < 0
    ):
        raise ValueError(
            "timing_slow_threshold_seconds must be >= 0 when provided, "
            f"got {timing_slow_threshold_seconds}"
        )

    def process_example(
        idx0: int, example: Any
    ) -> tuple[int, dict[str, Any], float, bool, float, dict[str, Any], float]:
        started_at = time.perf_counter()
        timing: dict[str, Any] = {
            "inputs_seconds": 0.0,
            "prediction_seconds": 0.0,
            "metric_seconds": 0.0,
            "row_build_seconds": 0.0,
            "prediction_attempts": 0,
            "prediction_attempt_durations": [],
            "prediction_backoff_seconds": 0.0,
            "prediction_succeeded": False,
        }
        try:
            inputs_started_at = time.perf_counter()
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
            timing["inputs_seconds"] = time.perf_counter() - inputs_started_at

            pred, pred_error, prediction_timing = _predict_with_retries(
                predictor,
                inputs,
                fallback_prediction_factory=fallback_prediction_factory,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
            timing["prediction_seconds"] = prediction_timing["total_seconds"]
            timing["prediction_attempts"] = prediction_timing["attempts"]
            timing["prediction_attempt_durations"] = prediction_timing[
                "attempt_durations"
            ]
            timing["prediction_backoff_seconds"] = prediction_timing[
                "backoff_sleep_seconds"
            ]
            timing["prediction_succeeded"] = prediction_timing["succeeded"]

            metric_started_at = time.perf_counter()
            score, metric_error = _metric_score(metric, example, pred)
            timing["metric_seconds"] = time.perf_counter() - metric_started_at

            row_build_started_at = time.perf_counter()
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
            timing["row_build_seconds"] = time.perf_counter() - row_build_started_at
        except Exception as exc:
            score = 0.0
            has_error = True
            row_build_started_at = time.perf_counter()
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
            timing["row_build_seconds"] = time.perf_counter() - row_build_started_at

        completed_at = time.perf_counter()
        elapsed = completed_at - started_at
        timing["total_seconds"] = elapsed
        return idx0, row, score, has_error, elapsed, timing, completed_at

    def should_log_timing(example_idx: int, total_seconds: float) -> bool:
        if not timing_logs:
            return False
        if (
            timing_slow_threshold_seconds is not None
            and total_seconds >= timing_slow_threshold_seconds
        ):
            return True
        return (example_idx % timing_log_every) == 0

    timing_totals = {
        "total_seconds": 0.0,
        "inputs_seconds": 0.0,
        "prediction_seconds": 0.0,
        "metric_seconds": 0.0,
        "row_build_seconds": 0.0,
        "write_seconds": 0.0,
        "queue_wait_seconds": 0.0,
        "prediction_backoff_seconds": 0.0,
    }
    timing_prediction_attempts = 0.0
    timing_examples_count = 0
    timing_slow_count = 0

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
                    (
                        _,
                        row,
                        score,
                        has_error,
                        elapsed,
                        timing,
                        completed_at,
                    ) = process_example(idx0, example)
                    scores.append(score)
                    if has_error:
                        error_count += 1

                    idx = idx0 + 1
                    queue_wait_seconds = max(0.0, time.perf_counter() - completed_at)
                    write_started_at = time.perf_counter()
                    f.write(json.dumps(row, ensure_ascii=False))
                    f.write("\n")
                    f.flush()
                    write_seconds = time.perf_counter() - write_started_at
                    print(
                        f"Processed {idx}/{total} examples in {elapsed:.2f}s "
                        f"(score={score:.4f})"
                    )

                    timing_examples_count += 1
                    timing_totals["total_seconds"] += timing.get("total_seconds", elapsed)
                    timing_totals["inputs_seconds"] += timing.get("inputs_seconds", 0.0)
                    timing_totals["prediction_seconds"] += timing.get(
                        "prediction_seconds", 0.0
                    )
                    timing_totals["metric_seconds"] += timing.get("metric_seconds", 0.0)
                    timing_totals["row_build_seconds"] += timing.get(
                        "row_build_seconds", 0.0
                    )
                    timing_totals["write_seconds"] += write_seconds
                    timing_totals["queue_wait_seconds"] += queue_wait_seconds
                    timing_totals["prediction_backoff_seconds"] += timing.get(
                        "prediction_backoff_seconds", 0.0
                    )
                    timing_prediction_attempts += float(
                        timing.get("prediction_attempts", 0.0)
                    )
                    if (
                        timing_slow_threshold_seconds is not None
                        and elapsed >= timing_slow_threshold_seconds
                    ):
                        timing_slow_count += 1

                    if should_log_timing(idx, elapsed):
                        attempt_durations = timing.get(
                            "prediction_attempt_durations", []
                        )
                        attempts_fmt = ", ".join(
                            f"{duration:.2f}s" for duration in attempt_durations
                        )
                        print(
                            "[timing] "
                            f"{idx}/{total}: total={elapsed:.2f}s "
                            f"inputs={timing.get('inputs_seconds', 0.0):.2f}s "
                            f"predict={timing.get('prediction_seconds', 0.0):.2f}s "
                            f"metric={timing.get('metric_seconds', 0.0):.2f}s "
                            f"row_build={timing.get('row_build_seconds', 0.0):.2f}s "
                            f"queue_wait={queue_wait_seconds:.2f}s "
                            f"write_flush={write_seconds:.2f}s "
                            f"attempts={timing.get('prediction_attempts', 0)} "
                            f"attempt_durations=[{attempts_fmt}] "
                            f"backoff_sleep={timing.get('prediction_backoff_seconds', 0.0):.2f}s "
                            f"prediction_ok={timing.get('prediction_succeeded', False)}"
                        )
            else:
                print(f"Running evaluation with {effective_num_threads} threads.")
                next_to_write = start_idx
                buffered_results: dict[
                    int, tuple[dict[str, Any], float, bool, float, dict[str, Any], float]
                ] = {}

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=effective_num_threads
                ) as executor:
                    futures = [
                        executor.submit(process_example, idx0, example)
                        for idx0, example in pending
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        idx0, row, score, has_error, elapsed, timing, completed_at = (
                            future.result()
                        )
                        buffered_results[idx0] = (
                            row,
                            score,
                            has_error,
                            elapsed,
                            timing,
                            completed_at,
                        )

                        while next_to_write in buffered_results:
                            (
                                next_row,
                                next_score,
                                next_has_error,
                                next_elapsed,
                                next_timing,
                                next_completed_at,
                            ) = buffered_results.pop(next_to_write)
                            scores.append(next_score)
                            if next_has_error:
                                error_count += 1

                            idx = next_to_write + 1
                            queue_wait_seconds = max(
                                0.0, time.perf_counter() - next_completed_at
                            )
                            write_started_at = time.perf_counter()
                            f.write(json.dumps(next_row, ensure_ascii=False))
                            f.write("\n")
                            f.flush()
                            write_seconds = time.perf_counter() - write_started_at
                            print(
                                f"Processed {idx}/{total} examples in {next_elapsed:.2f}s "
                                f"(score={next_score:.4f})"
                            )

                            timing_examples_count += 1
                            timing_totals["total_seconds"] += next_timing.get(
                                "total_seconds", next_elapsed
                            )
                            timing_totals["inputs_seconds"] += next_timing.get(
                                "inputs_seconds", 0.0
                            )
                            timing_totals["prediction_seconds"] += next_timing.get(
                                "prediction_seconds", 0.0
                            )
                            timing_totals["metric_seconds"] += next_timing.get(
                                "metric_seconds", 0.0
                            )
                            timing_totals["row_build_seconds"] += next_timing.get(
                                "row_build_seconds", 0.0
                            )
                            timing_totals["write_seconds"] += write_seconds
                            timing_totals["queue_wait_seconds"] += queue_wait_seconds
                            timing_totals["prediction_backoff_seconds"] += next_timing.get(
                                "prediction_backoff_seconds", 0.0
                            )
                            timing_prediction_attempts += float(
                                next_timing.get("prediction_attempts", 0.0)
                            )
                            if (
                                timing_slow_threshold_seconds is not None
                                and next_elapsed >= timing_slow_threshold_seconds
                            ):
                                timing_slow_count += 1

                            if should_log_timing(idx, next_elapsed):
                                attempt_durations = next_timing.get(
                                    "prediction_attempt_durations", []
                                )
                                attempts_fmt = ", ".join(
                                    f"{duration:.2f}s" for duration in attempt_durations
                                )
                                print(
                                    "[timing] "
                                    f"{idx}/{total}: total={next_elapsed:.2f}s "
                                    f"inputs={next_timing.get('inputs_seconds', 0.0):.2f}s "
                                    f"predict={next_timing.get('prediction_seconds', 0.0):.2f}s "
                                    f"metric={next_timing.get('metric_seconds', 0.0):.2f}s "
                                    f"row_build={next_timing.get('row_build_seconds', 0.0):.2f}s "
                                    f"queue_wait={queue_wait_seconds:.2f}s "
                                    f"write_flush={write_seconds:.2f}s "
                                    f"attempts={next_timing.get('prediction_attempts', 0)} "
                                    f"attempt_durations=[{attempts_fmt}] "
                                    f"backoff_sleep={next_timing.get('prediction_backoff_seconds', 0.0):.2f}s "
                                    f"prediction_ok={next_timing.get('prediction_succeeded', False)}"
                                )
                            next_to_write += 1

    if timing_logs and timing_examples_count:
        avg_total = timing_totals["total_seconds"] / timing_examples_count
        avg_inputs = timing_totals["inputs_seconds"] / timing_examples_count
        avg_prediction = timing_totals["prediction_seconds"] / timing_examples_count
        avg_metric = timing_totals["metric_seconds"] / timing_examples_count
        avg_row_build = timing_totals["row_build_seconds"] / timing_examples_count
        avg_write = timing_totals["write_seconds"] / timing_examples_count
        avg_queue_wait = timing_totals["queue_wait_seconds"] / timing_examples_count
        avg_backoff = timing_totals["prediction_backoff_seconds"] / timing_examples_count
        avg_attempts = timing_prediction_attempts / timing_examples_count

        print(
            "[timing-summary] "
            f"examples={timing_examples_count} "
            f"avg_total={avg_total:.2f}s "
            f"avg_inputs={avg_inputs:.2f}s "
            f"avg_predict={avg_prediction:.2f}s "
            f"avg_metric={avg_metric:.2f}s "
            f"avg_row_build={avg_row_build:.2f}s "
            f"avg_queue_wait={avg_queue_wait:.2f}s "
            f"avg_write_flush={avg_write:.2f}s "
            f"avg_backoff_sleep={avg_backoff:.2f}s "
            f"avg_prediction_attempts={avg_attempts:.2f}"
        )
        if timing_slow_threshold_seconds is not None:
            print(
                "[timing-summary] "
                f"slow_examples(>={timing_slow_threshold_seconds:.2f}s)="
                f"{timing_slow_count}/{timing_examples_count}"
            )

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
    timing_logs: bool = False,
    timing_log_every: int = 1,
    timing_slow_threshold_seconds: float | None = None,
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
        timing_logs=timing_logs,
        timing_log_every=timing_log_every,
        timing_slow_threshold_seconds=timing_slow_threshold_seconds,
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
    timing_logs: bool = False,
    timing_log_every: int = 1,
    timing_slow_threshold_seconds: float | None = None,
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
        timing_logs=timing_logs,
        timing_log_every=timing_log_every,
        timing_slow_threshold_seconds=timing_slow_threshold_seconds,
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
    timing_logs: bool = False,
    timing_log_every: int = 1,
    timing_slow_threshold_seconds: float | None = None,
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
        timing_logs=timing_logs,
        timing_log_every=timing_log_every,
        timing_slow_threshold_seconds=timing_slow_threshold_seconds,
    )
