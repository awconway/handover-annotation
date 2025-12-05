import spacy


def test_compare_tokenization():
    nlp = spacy.blank("en")

    text1 = "I’ve escalated to the medical registrar—awaiting review."
    text2 = "I\u2019ve escalated to the medical registrar\u2014awaiting review."

    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # Basic structural checks
    assert len(doc1) == len(doc2), "Token count mismatch"

    for t1, t2 in zip(doc1, doc2):
        # Compare raw token text
        assert t1.text == t2.text, f"Text mismatch: {t1.text!r} vs {t2.text!r}"

        # Orth IDs are deterministic for identical strings
        assert t1.orth == t2.orth, f"Orth mismatch for {t1.text!r}"

        # Token lengths should match
        assert len(t1) == len(t2), f"Length mismatch for {t1.text}"

        # Whitespace flags
        assert bool(t1.whitespace_) == bool(t2.whitespace_), (
            f"Whitespace mismatch after {t1.text!r}"
        )
