import srsly

from sbar_span_task.langextract_experiment import (
    exact_match_metrics,
    load_sbar_records,
    run_langextract_sbar_experiment,
    span_items_from_record,
)


def test_span_items_from_record_filters_invalid_and_disallowed_spans():
    text = "alpha bravo charlie delta"
    record = {
        "text": text,
        "spans": [
            {"start": 0, "end": 5, "label": "SITUATION"},
            {"start": 0, "end": 5, "label": "SITUATION"},  # duplicate
            {"start": 6, "end": 11, "label": "BACKGROUND"},
            {"start": 12, "end": 19, "label": "OTHER"},  # disallowed
            {"start": -1, "end": 2, "label": "ASSESSMENT"},  # invalid
            {"start": 3, "end": 3, "label": "ASSESSMENT"},  # invalid
            {"start": 3, "end": 999, "label": "ASSESSMENT"},  # invalid
        ],
    }

    items = span_items_from_record(record)
    assert [(i.label, i.quote) for i in items] == [
        ("SITUATION", "alpha"),
        ("BACKGROUND", "bravo"),
    ]


def test_exact_match_metrics_returns_expected_values():
    gold = [
        {"label": "SITUATION", "quote": "A"},
        {"label": "BACKGROUND", "quote": "B"},
    ]
    pred = [
        {"label": "SITUATION", "quote": "A"},
        {"label": "ASSESSMENT", "quote": "C"},
    ]

    from sbar_span_task.langextract_experiment import SbarItem

    gold_items = [SbarItem(**item) for item in gold]
    pred_items = [SbarItem(**item) for item in pred]
    metrics = exact_match_metrics(gold_items, pred_items)

    assert metrics["true_positives"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_load_sbar_records_respects_annotator_filter(tmp_path):
    rows = [
        {
            "text": "A B C",
            "_annotator_id": "handover_db-user1",
            "spans": [{"start": 0, "end": 1, "label": "SITUATION"}],
        },
        {
            "text": "D E F",
            "_annotator_id": "handover_db-user2",
            "spans": [{"start": 0, "end": 1, "label": "BACKGROUND"}],
        },
        {
            "text": "X Y Z",
            "_annotator_id": "handover_db-user1",
            "spans": [{"start": 0, "end": 1, "label": "OTHER"}],  # ignored
        },
    ]
    data_file = tmp_path / "data.jsonl"
    srsly.write_jsonl(data_file, rows)

    user1_records = load_sbar_records(str(data_file), annotator_id="handover_db-user1")
    all_records = load_sbar_records(str(data_file), annotator_id=None)

    assert len(user1_records) == 1
    assert user1_records[0].text == "A B C"
    assert len(all_records) == 2


def test_use_dataset_test_split_matches_prepare_dataset_test_split(tmp_path):
    from data.dataset import prepare_dataset_sbar_span

    rows = []
    for i in range(20):
        text = f"sample_{i}"
        rows.append(
            {
                "text": text,
                "spans": [{"start": 0, "end": len(text), "label": "SITUATION"}],
            }
        )

    data_file = tmp_path / "data.jsonl"
    output_file = tmp_path / "out.jsonl"
    srsly.write_jsonl(data_file, rows)

    _, testset = prepare_dataset_sbar_span(str(data_file))
    expected_test_texts = [ex.text for ex in testset]
    assert expected_test_texts  # sanity: deterministic split should include test rows

    summary = run_langextract_sbar_experiment(
        data_file=str(data_file),
        output_file=str(output_file),
        model_id="gpt-5.2",
        train_examples=999,
        eval_examples=999,
        use_dataset_test_split=True,
        dry_run=True,
    )

    out_rows = list(srsly.read_jsonl(output_file))
    out_texts = [row["text"] for row in out_rows]

    assert int(summary["num_eval_examples"]) == len(expected_test_texts)
    assert out_texts == expected_test_texts
