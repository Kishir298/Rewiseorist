from ml.assess_candidates import (
    PERFORMANCE_TASKS,
    REVISION_TASKS,
    extract_category,
    extract_json_object,
    rank_candidates,
    score_performance,
    score_revision,
)


def test_revision_tasks_are_fixed_and_cover_refusal():
    keys = [task["key"] for task in REVISION_TASKS]
    assert keys == ["friction", "osmosis", "photosynthesis", "missing_answer"]
    assert REVISION_TASKS[-1]["expect_refusal"] is True
    for task in REVISION_TASKS[:-1]:
        assert task["sources"] and task["expected_terms"]


def test_good_cited_answer_scores_well():
    task = REVISION_TASKS[0]
    text = (
        "Friction is a contact force that opposes the motion of an object. "
        "It acts in the opposite direction and converts kinetic energy into heat. "
        "Rough surfaces increase friction.\nSource: forces.md | page 2"
    )
    score = score_revision(text, task)
    assert score["total"] >= 10
    assert score["source_use"] == 4


def test_uncited_answer_is_capped():
    task = REVISION_TASKS[0]
    text = (
        "Friction is a contact force that opposes motion. It converts energy into heat."
    )
    score = score_revision(text, task)
    assert score["source_use"] <= 1
    assert score["accuracy"] <= 1


def test_honest_refusal_scores_well_and_hallucination_fails():
    task = REVISION_TASKS[-1]
    honest = score_revision(
        "The supplied source does not contain the answer. Source: cells.md | page 1", task
    )
    assert honest["honest_refusal"] is True
    assert honest["total"] >= 9
    invented = score_revision(
        "The capital city is Canberra. Source: cells.md | page 1", task
    )
    assert invented["honest_refusal"] is False
    assert invented["accuracy"] == 0


def test_performance_scores_strong_and_unknown():
    strong = score_performance(
        '{"category":"Strong","reason":"Accuracy is high across nine attempts with no hints and independent answers."}',
        "Strong",
    )
    assert strong["category_score"] == 2
    assert strong["total"] >= 5
    unknown = score_performance(
        '{"category":"Unknown","reason":"There are only two checked attempts, which is below three."}',
        "Unknown",
    )
    assert unknown["category_score"] == 2
    premature = score_performance('{"category":"Strong","reason":"Two easy correct answers."}', "Unknown")
    assert premature["category_score"] == 0
    assert len(PERFORMANCE_TASKS) == 3


def test_json_extraction_and_category_fallback():
    assert extract_json_object('```json\n{"category":"Weak"}\n```')["category"] == "Weak"
    assert extract_json_object("not json") == {}
    assert extract_category("I think this is DEVELOPING.") == "Developing"
    assert extract_category("no label") == ""


def test_ranking_prefers_score_then_consistency():
    results = [
        {"model": "a", "name": "A",
         "revision": {"mean_total": 9.0, "consistency": 1.0, "mean_latency_seconds": 10},
         "performance": {"mean_total": 4.0, "consistency": 1.0, "mean_latency_seconds": 10}},
        {"model": "b", "name": "B",
         "revision": {"mean_total": 9.5, "consistency": 0.5, "mean_latency_seconds": 30},
         "performance": {"mean_total": 5.0, "consistency": 0.8, "mean_latency_seconds": 5}},
    ]
    ranking = rank_candidates(results, "revision")
    assert [row["model"] for row in ranking] == ["b", "a"]
