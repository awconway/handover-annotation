import dspy

from checklist_task.metric_gepa import multilabel_f1_with_feedback
from checklist_task.metric_miprov import multilabel_f1
from span_metric.gepa_span_metric import gepa_span_metric

OPTIM_REGISTRY = {
    "none": lambda p, t, v: p,
    "mipro_light_checklist": lambda p, t: dspy.MIPROv2(
        metric=multilabel_f1, auto="light", seed=749
    ).compile(p, trainset=t),
    "mipro_heavy_checklist": lambda p, t, v: dspy.MIPROv2(
        metric=multilabel_f1, auto="heavy", seed=749
    ).compile(p, trainset=t),
    "gepa_light_checklist": lambda p, t, v: dspy.GEPA(
        metric=multilabel_f1_with_feedback,
        auto="light",
        seed=749,
        num_threads=20,
        reflection_lm=dspy.LM(model="gpt-5.1", temperature=1.0, max_tokens=32000),
    ).compile(p, trainset=t, valset=v),
    "gepa_heavy_checklist": lambda p, t, v: dspy.GEPA(
        metric=multilabel_f1_with_feedback,
        auto="heavy",
        seed=749,
        num_threads=20,
        reflection_lm=dspy.LM(model="gpt-5.1", temperature=1.0, max_tokens=32000),
    ).compile(p, trainset=t, valset=v),
    "gepa_light_span": lambda p, t, v: dspy.GEPA(
        metric=gepa_span_metric,
        auto="light",
        seed=749,
        num_threads=20,
        reflection_lm=dspy.LM(model="gpt-5.1", temperature=1.0, max_tokens=32000),
    ).compile(p, trainset=t, valset=v),
    "gepa_heavy_span": lambda p, t, v: dspy.GEPA(
        metric=gepa_span_metric,
        auto="heavy",
        seed=749,
        num_threads=20,
        reflection_lm=dspy.LM(model="gpt-5.1", temperature=1.0, max_tokens=32000),
    ).compile(p, trainset=t, valset=v),
}


def load_optimiser(name: str):
    if name not in OPTIM_REGISTRY:
        raise ValueError(f"Unknown optimiser: {name}")
    return OPTIM_REGISTRY[name]
