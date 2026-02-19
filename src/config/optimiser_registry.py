import os

import dspy

from checklist_task.metric_gepa import multilabel_f1_with_feedback
from checklist_task.metric_miprov import multilabel_f1
from span_metric.gepa_span_metric import gepa_span_metric


def _resolve_gepa_num_threads() -> int:
    """
    Pick a conservative GEPA evaluation thread count.

    Priority:
    1) GEPA_NUM_THREADS env var (explicit override)
    2) Scheduler CPU allocation (PBS/SLURM), minus one core for overhead
    3) Safe fallback of 4 threads
    """
    override = os.getenv("GEPA_NUM_THREADS")
    if override:
        try:
            value = int(override)
        except ValueError as exc:
            raise ValueError("GEPA_NUM_THREADS must be an integer >= 1.") from exc
        if value < 1:
            raise ValueError("GEPA_NUM_THREADS must be >= 1.")
        return value

    detected_cpus = None
    for key in ("PBS_NCPUS", "SLURM_CPUS_PER_TASK", "NCPUS"):
        raw = os.getenv(key)
        if not raw:
            continue
        try:
            detected_cpus = int(raw)
        except ValueError:
            continue
        break

    if detected_cpus and detected_cpus > 0:
        # Keep one core free for Ollama/runtime overhead and cap by default.
        return max(1, min(6, detected_cpus - 1))

    return 4


GEPA_NUM_THREADS = _resolve_gepa_num_threads()


def build_reflection_lm() -> dspy.LM:
    return dspy.LM(model="openai/gpt-5.2")


OPTIM_REGISTRY = {
    "none": lambda p, t, v, gepa_log_dir=None: p,
    "mipro_light_checklist": lambda p, t, v, gepa_log_dir=None: dspy.MIPROv2(
        metric=multilabel_f1, auto="light", seed=749
    ).compile(p, trainset=t),
    "mipro_heavy_checklist": lambda p, t, v, gepa_log_dir=None: dspy.MIPROv2(
        metric=multilabel_f1, auto="heavy", seed=749
    ).compile(p, trainset=t),
    "gepa_light_checklist": lambda p, t, v, gepa_log_dir=None: dspy.GEPA(
        metric=multilabel_f1_with_feedback,
        auto="light",
        seed=749,
        num_threads=GEPA_NUM_THREADS,
        reflection_lm=build_reflection_lm(),
        log_dir=gepa_log_dir,
    ).compile(p, trainset=t, valset=v),
    "gepa_heavy_checklist": lambda p, t, v, gepa_log_dir=None: dspy.GEPA(
        metric=multilabel_f1_with_feedback,
        auto="heavy",
        seed=749,
        num_threads=GEPA_NUM_THREADS,
        reflection_lm=build_reflection_lm(),
        log_dir=gepa_log_dir,
    ).compile(p, trainset=t, valset=v),
    "gepa_light_span": lambda p, t, v, gepa_log_dir=None: dspy.GEPA(
        metric=gepa_span_metric,
        auto="light",
        seed=749,
        num_threads=GEPA_NUM_THREADS,
        reflection_lm=build_reflection_lm(),
        log_dir=gepa_log_dir,
    ).compile(p, trainset=t, valset=v),
    "gepa_heavy_span": lambda p, t, v, gepa_log_dir=None: dspy.GEPA(
        metric=gepa_span_metric,
        auto="heavy",
        seed=749,
        num_threads=GEPA_NUM_THREADS,
        reflection_lm=build_reflection_lm(),
        log_dir=gepa_log_dir,
    ).compile(p, trainset=t, valset=v),
}


def load_optimiser(name: str):
    if name not in OPTIM_REGISTRY:
        raise ValueError(f"Unknown optimiser: {name}")
    return OPTIM_REGISTRY[name]
