from ml.evaluate_models import citation_rate, content_words, grounding_rate, word_f1


def test_word_f1_handles_overlap_and_empty_output():
    assert word_f1("friction opposes motion", "friction opposes motion") == 1.0
    assert word_f1("friction opposes motion", "") == 0.0
    assert 0 < word_f1("force mass acceleration", "force causes acceleration") < 1


def test_citation_and_grounding_rates():
    rows = [
        {"source": {"original_name": "forces.md", "text": "Friction opposes motion and makes heat."}},
        {"source": {"original_name": "forces.md", "text": "Friction opposes motion and makes heat."}},
    ]
    outputs = [
        "Friction opposes motion and makes heat.\nSource: forces.md | page 2",
        "Bananas are yellow.\nSource: biology.md | page 1",
    ]
    assert citation_rate(rows, outputs) == 0.5
    assert grounding_rate(rows, outputs) == 0.5
    assert "friction" in content_words("Friction and motion")
    assert "the" not in content_words("the friction")
