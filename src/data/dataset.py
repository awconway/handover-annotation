import random

import dspy
import srsly


def _matches_annotator(line: dict, annotator_id: str | None) -> bool:
    if annotator_id is None:
        return True
    return line.get("_annotator_id") == annotator_id


def prepare_dataset(path: str, annotator_id: str | None = None):
    rng = random.Random(339)  # local, deterministic RNG
    train, test = [], []

    for line in srsly.read_jsonl(path):
        if not isinstance(line, dict):
            continue
        if not _matches_annotator(line, annotator_id):
            continue

        ex = dspy.Example(
            text=line.get("text"),
            labels=line.get("accept"),
        ).with_inputs("text")

        if rng.random() < 0.75:
            train.append(ex)
        else:
            test.append(ex)

    return train, test


SBAR_ALLOWED_LABELS = {
    "SITUATION",
    "BACKGROUND",
    "ASSESSMENT",
    "RECOMMENDATION",
}


def prepare_dataset_sbar_span(path: str, annotator_id: str | None = None):
    rng = random.Random(339)  # local, deterministic RNG
    train, test = [], []

    for line in srsly.read_jsonl(path):
        if not isinstance(line, dict):
            continue
        if not _matches_annotator(line, annotator_id):
            continue

        # Filter out spans with disallowed labels
        spans = [
            span
            for span in (line.get("spans") or [])
            if span.get("label") in SBAR_ALLOWED_LABELS
        ]

        # Optionally: skip examples that end up with no valid spans
        if not spans:
            continue

        ex = dspy.Example(
            text=line.get("text"),
            gold_spans=spans,
        ).with_inputs("text")

        if rng.random() < 0.75:
            train.append(ex)
        else:
            test.append(ex)

    return train, test


UNCERTAINTY_ALLOWED_LABELS = {
    "Vagueness",
    "Hedging",
    "Unknown fact",
    "Indefinite timing",
    "Source uncertainty",
    "Procedural uncertainty",
    "Responsibility uncertainty",
}


def prepare_dataset_uncertainty_span(path: str, annotator_id: str | None = None):
    rng = random.Random(339)  # local, deterministic RNG
    train, test = [], []

    for line in srsly.read_jsonl(path):
        if not isinstance(line, dict):
            continue
        if not _matches_annotator(line, annotator_id):
            continue

        # Filter out spans with disallowed labels
        spans = [
            span
            for span in (line.get("spans") or [])
            if span.get("label") in UNCERTAINTY_ALLOWED_LABELS
        ]

        # Optionally: skip examples that end up with no valid spans
        if not spans:
            continue

        ex = dspy.Example(
            text=line.get("text"),
            labels=spans,
        ).with_inputs("text")

        if rng.random() < 0.75:
            train.append(ex)
        else:
            test.append(ex)

    return train, test
