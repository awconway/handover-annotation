from sys import meta_path

import dspy
import srsly

from checklist_task.metric_gepa import multilabel_f1_with_feedback
from span_metric.gepa_span_metric import gepa_span_metric


def evaluate_checklist(predictor, testset, out_file):
    ev = dspy.Evaluate(
        devset=testset,
        metric=multilabel_f1_with_feedback,
        display_progress=True,
        num_threads=20,
        provide_traceback=True,
    )

    result = ev(predictor, devset=testset)

    rows = []
    for ex, pred, score in result.results:
        rows.append(
            {
                "example": ex,
                "prediction": pred,
                "score": score,
            }
        )

    srsly.write_jsonl(out_file, rows)
    return result.score


def evaluate_sbar(predictor, testset, out_file):
    ev = dspy.Evaluate(
        devset=testset,
        metric=gepa_span_metric,
        display_progress=True,
        num_threads=20,
        provide_traceback=True,
    )

    result = ev(predictor, devset=testset)

    rows = []
    for ex, pred, score in result.results:
        rows.append(
            {
                "example": ex,
                "prediction": pred,
                "score": score,
            }
        )
    srsly.write_jsonl(out_file, rows)
    return result.score
