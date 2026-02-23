import srsly

from data.dataset import (
    UNKNOWN_FACT_LABEL,
    UNCERTAINTY_COLLAPSED_LABEL,
    _collapse_unknown_fact_spans,
    _collapse_uncertainty_spans,
    prepare_dataset_unknown_fact_binary_span,
    prepare_dataset_uncertainty_binary_span,
)


def test_collapse_uncertainty_spans_dedupes_and_maps_labels():
    spans = [
        {"start": 0, "end": 5, "label": "Hedging"},
        {"start": 0, "end": 5, "label": "Vagueness"},  # duplicate by span
        {"start": 6, "end": 10, "label": "OTHER"},  # disallowed
        {"start": 11, "end": 15, "label": "Indefinite timing"},
    ]

    collapsed = _collapse_uncertainty_spans(spans)

    assert len(collapsed) == 2
    assert {tuple((s["start"], s["end"])) for s in collapsed} == {(0, 5), (11, 15)}
    assert {s["label"] for s in collapsed} == {UNCERTAINTY_COLLAPSED_LABEL}


def test_prepare_dataset_uncertainty_binary_span_filters_and_collapses(tmp_path):
    data = [
        {
            "text": "Only allowed",
            "spans": [
                {"start": 0, "end": 5, "label": "Vagueness"},
                {"start": 6, "end": 13, "label": "Hedging"},
            ],
        },
        {
            "text": "Mixed labels",
            "spans": [
                {"start": 0, "end": 12, "label": "Unknown fact"},
                {"start": 0, "end": 12, "label": "Hedging"},  # duplicate span
                {"start": 13, "end": 18, "label": "OTHER"},  # disallowed
                {"start": 19, "end": 36, "label": "Source uncertainty"},
            ],
        },
        {
            "text": "Only disallowed",
            "spans": [
                {"start": 0, "end": 4, "label": "FOO"},
                {"start": 5, "end": 9, "label": "BAR"},
            ],
        },
    ]

    jsonl_path = tmp_path / "data_uncertainty_binary.jsonl"
    srsly.write_jsonl(jsonl_path, data)

    train, test = prepare_dataset_uncertainty_binary_span(str(jsonl_path))
    examples = train + test

    # Only disallowed examples should be removed entirely
    assert all(e.text != "Only disallowed" for e in examples)

    # All spans should be collapsed to the single UNCERTAIN label
    for ex in examples:
        assert {span["label"] for span in ex["gold_spans"]} == {
            UNCERTAINTY_COLLAPSED_LABEL
        }

    mixed = next(e for e in examples if e.text == "Mixed labels")
    mixed_spans = {(s["start"], s["end"]) for s in mixed["gold_spans"]}
    assert mixed_spans == {(0, 12), (19, 36)}


def test_collapse_unknown_fact_spans_keeps_only_unknown_fact():
    spans = [
        {"start": 0, "end": 12, "label": "Hedging"},
        {"start": 0, "end": 12, "label": "Unknown fact"},
        {"start": 13, "end": 20, "label": "Unknown fact"},
        {"start": 13, "end": 20, "label": "Vagueness"},
    ]

    collapsed = _collapse_unknown_fact_spans(spans)
    assert {(s["start"], s["end"]) for s in collapsed} == {(0, 12), (13, 20)}
    assert {s["label"] for s in collapsed} == {UNKNOWN_FACT_LABEL}


def test_prepare_dataset_unknown_fact_binary_span_includes_negatives(tmp_path):
    data = [
        {
            "text": "Unknown fact and hedging overlap",
            "spans": [
                {"start": 0, "end": 12, "label": "Unknown fact"},
                {"start": 0, "end": 12, "label": "Hedging"},
            ],
        },
        {
            "text": "No unknown fact labels",
            "spans": [
                {"start": 0, "end": 8, "label": "Hedging"},
                {"start": 10, "end": 19, "label": "Vagueness"},
            ],
        },
        {
            "text": "No spans at all",
            "spans": [],
        },
        {
            "text": "Only disallowed labels",
            "spans": [{"start": 0, "end": 3, "label": "OTHER"}],
        },
    ]

    jsonl_path = tmp_path / "data_unknown_fact_binary.jsonl"
    srsly.write_jsonl(jsonl_path, data)

    train, test = prepare_dataset_unknown_fact_binary_span(str(jsonl_path))
    examples = train + test

    assert {e.text for e in examples} == {
        "Unknown fact and hedging overlap",
        "No unknown fact labels",
    }

    mixed = next(e for e in examples if e.text == "Unknown fact and hedging overlap")
    assert mixed["gold_spans"] == [
        {"start": 0, "end": 12, "label": UNKNOWN_FACT_LABEL}
    ]

    no_unknown = next(e for e in examples if e.text == "No unknown fact labels")
    assert no_unknown["gold_spans"] == []

    assert all(e.text != "No spans at all" for e in examples)
    assert all(e.text != "Only disallowed labels" for e in examples)
