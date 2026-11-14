from ml.synthetic_students import build_personas, group_split, summarise


def test_personas_are_deterministic_and_balanced():
    first = build_personas(per_label=3, seed=7)
    second = build_personas(per_label=3, seed=7)
    assert first == second
    labels = [persona["intended_label"] for persona in first]
    assert labels.count("Weak") == 3
    assert labels.count("Unknown") == 3
    for persona in first:
        features = persona["features"]
        assert 0 <= features["mean_accuracy"] <= 1
        assert 0 <= features["hint_rate"] <= 1
        if persona["intended_label"] == "Unknown":
            assert features["unknown"] is True


def test_summary_matches_attempts():
    attempts = [
        {"index": 1, "difficulty": "easy", "outcome": "correct", "hint_used": False, "error_type": ""},
        {"index": 2, "difficulty": "hard", "outcome": "wrong", "hint_used": True, "error_type": "sign_error"},
        {"index": 3, "difficulty": "medium", "outcome": "partly correct", "hint_used": False, "error_type": "sign_error"},
    ]
    features = summarise(attempts)
    assert features["mean_accuracy"] == 0.5
    assert features["hint_rate"] == round(1 / 3, 4)
    assert features["repeated_errors"] == 1
    assert features["mean_difficulty"] == 2.0


def test_group_split_keeps_personas_together():
    personas = build_personas(per_label=4, seed=3)
    splits = group_split(personas)
    seen = set()
    for split, items in splits.items():
        for persona in items:
            assert persona["persona_id"] not in seen
            seen.add(persona["persona_id"])
    assert len(seen) == len(personas)
    assert all(splits.values())
