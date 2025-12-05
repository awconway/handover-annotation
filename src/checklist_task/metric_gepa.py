import dspy


def feedback_multilabel(gold_labels, pred_labels):
    """
    gold_labels, pred_labels: list of strings (labels)
    Returns: (feedback_text, score_float)
    """
    gold = set(gold_labels or [])
    got = set(pred_labels or [])

    correctly_included = gold & got
    incorrectly_included = got - gold
    incorrectly_excluded = gold - got

    # Edge case: no labels at all
    if not gold and not got:
        score = 1.0
        fb_text = (
            "The category classification is perfect. "
            "You correctly identified that the text does not fall under any category."
        )
        return fb_text, score

    tp = len(correctly_included)
    fp = len(incorrectly_included)
    fn = len(incorrectly_excluded)

    if tp == 0:
        score = 0.0
        fb_text = (
            "The category classification is not perfect. "
            "None of the correctly applicable categories were identified."
        )
        if incorrectly_included:
            fb_text += (
                f" You incorrectly identified the following categories: "
                f"`{', '.join(sorted(incorrectly_included))}`. "
                "The message does NOT fall under these categories."
            )
        if incorrectly_excluded:
            fb_text += (
                f" You also missed the following categories that actually apply: "
                f"`{', '.join(sorted(incorrectly_excluded))}`."
            )
        fb_text += " Think about how you could have reasoned to get the correct category labels."
        return fb_text, score

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    score = 2 * p * r / (p + r) if (p + r) else 0.0  # F1

    if score == 1.0:
        fb_text = (
            "The category classification is perfect. You correctly identified that the text "
            f"falls under the following categories: `{', '.join(sorted(gold))}`."
        )
    else:
        fb_text = (
            "The category classification is not perfect. "
            "You correctly identified that the message falls under the following categories: "
            f"`{', '.join(sorted(correctly_included))}`.\n"
        )
        if incorrectly_included:
            fb_text += (
                "However, you incorrectly identified that the message falls under the "
                f"following categories: `{', '.join(sorted(incorrectly_included))}`. "
                "The message DOES NOT fall under these categories.\n"
            )
        if incorrectly_excluded:
            prefix = "Additionally, " if incorrectly_included else "However, "
            fb_text += (
                f"{prefix}you didn't identify the following categories that the message actually "
                f"falls under: `{', '.join(sorted(incorrectly_excluded))}`.\n"
            )
        fb_text += "Think about how you could have reasoned to get the correct category labels."

    return fb_text, score


def multilabel_f1_with_feedback(
    example, pred, trace=None, pred_name=None, pred_trace=None
):
    """
    Single-module metric with feedback for multilabel classification.

    - example: gold example (dspy.Example or dict)
    - pred: model prediction (dspy.Prediction-like, with .labels)
    """
    gold_labels = example["labels"]
    pred_labels = pred["labels"]

    fb_text, score = feedback_multilabel(gold_labels, pred_labels)

    # GEPA uses this function in two modes:
    # 1) scoring: pred_name is None -> return numeric score
    # 2) feedback for a specific module: pred_name is e.g. "classifier.predict"
    if pred_name is None:
        return score

    return dspy.Prediction(score=score, feedback=fb_text)
