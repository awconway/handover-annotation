import dspy
import srsly

from data.dataset import prepare_dataset_sbar_span


def test_prepare_dataset_sbar_span_filters_disallowed_labels(tmp_path):
    data = [
        {
            "text": "Only allowed",
            "spans": [
                {"start": 0, "end": 5, "label": "SITUATION"},
                {"start": 6, "end": 10, "label": "BACKGROUND"},
            ],
        },
        {
            "text": "Mixed labels",
            "spans": [
                {"start": 0, "end": 7, "label": "SITUATION"},
                {"start": 8, "end": 15, "label": "OTHER"},  # disallowed
                {"start": 16, "end": 20, "label": "ASSESSMENT"},
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

    jsonl_path = tmp_path / "data.jsonl"
    srsly.write_jsonl(jsonl_path, data)

    train, test = prepare_dataset_sbar_span(str(jsonl_path))
    examples = train + test

    allowed_labels = {
        "SITUATION",
        "BACKGROUND",
        "ASSESSMENT",
        "RECOMMENDATION",
    }

    # All remaining examples should be dspy.Example
    for ex in examples:
        assert isinstance(ex, dspy.Example)

    # All spans in all remaining examples must have allowed labels
    for ex in examples:
        for span in ex["gold_spans"]:
            assert span["label"] in allowed_labels

    # The disallowed label "OTHER" should be removed from "Mixed labels"
    mixed = next(e for e in examples if e.text == "Mixed labels")
    mixed_labels = {span["label"] for span in mixed["gold_spans"]}
    assert "OTHER" not in mixed_labels
    assert {"SITUATION", "ASSESSMENT"}.issubset(mixed_labels)

    # The example with only disallowed labels should not be present
    assert all(e.text != "Only disallowed" for e in examples)
