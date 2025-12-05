import json
import tempfile

from search import find_missing_user2_annotations


def test_find_missing_user2_annotations():
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmp:
        # Example dataset:
        # - hash 1: user1 and user2 (should NOT be included)
        # - hash 2: user1 only (should be included)
        # - hash 3: user2 only (should NOT be included)
        # - hash 4: user1, user3 (should be included)

        data = [
            {"_input_hash": 1, "_annotator_id": "handover_db-user1"},
            {"_input_hash": 1, "_annotator_id": "handover_db-user2"},
            {"_input_hash": 2, "_annotator_id": "handover_db-user1"},
            {"_input_hash": 3, "_annotator_id": "handover_db-user2"},
            {"_input_hash": 4, "_annotator_id": "handover_db-user1"},
            {"_input_hash": 4, "_annotator_id": "handover_db-user3"},
        ]

        for obj in data:
            tmp.write(json.dumps(obj) + "\n")

        tmp.flush()

        # Run the function
        result = find_missing_user2_annotations(tmp.name)

    # Expected hashes: 2 and 4
    assert set(result) == {2, 4}
