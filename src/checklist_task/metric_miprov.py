def get_labels(x):
    if hasattr(x, "toDict"):
        x = x.toDict()
    return set((x or {}).get("labels", []) or [])


def multilabel_f1(example, pred, trace=None):
    gold = get_labels(example)
    got = get_labels(pred)

    if not gold and not got:
        return 1.0

    tp = len(gold & got)
    fp = len(got - gold)
    fn = len(gold - got)

    if tp == 0:
        return 0.0

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0

    return 2 * p * r / (p + r) if (p + r) else 0
