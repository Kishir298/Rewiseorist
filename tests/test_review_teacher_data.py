import json

from ml.review_teacher_data import (
    review_performance,
    review_revision,
    topic_splits,
)


def revision_row(answer: str, ident: str = "revision-1") -> dict:
    return {
        "id": ident,
        "task": "revision",
        "teacher": "qwen25-3b",
        "teacher_name": "Qwen",
        "question_id": "forces-q1",
        "topic": "forces",
        "question": "What is friction?",
        "source": {
            "original_name": "forces.md",
            "source_ref": "forces.md | sample",
            "text": "Friction is a contact force that opposes motion and produces heat on rough surfaces.",
        },
        "answer": answer,
        "error": "",
    }


def performance_row(answer: dict, distinct: int = 6, persona: str = "student-strong-01") -> dict:
    return {
        "id": f"performance-{persona}",
        "task": "performance",
        "teacher": "qwen25-3b",
        "model_name": "Qwen",
        "persona_id": persona,
        "topic": "Forces",
        "intended_label": "Strong",
        "features": {"distinct_questions": distinct, "mean_accuracy": 0.9},
        "history": "attempt 1: hard, correct, no hint",
        "raw_answer": json.dumps(answer),
        "error": "",
    }


def test_revision_review_accepts_grounded_cited_answer_and_repairs_citation():
    grounded = (
        "Friction is a contact force that opposes motion. It produces heat on rough surfaces.\n"
        "Source: forces.md | sample"
    )
    accepted, rejected = review_revision([revision_row(grounded)])
    assert len(accepted) == 1 and not rejected
    assert accepted[0]["prompt"].startswith("<user>")
    assert accepted[0]["citation"] == "forces.md | sample"

    uncited = "Friction is a contact force that opposes motion. It produces heat on rough surfaces."
    accepted, rejected = review_revision([revision_row(uncited, "revision-2")])
    assert len(accepted) == 1
    assert accepted[0]["citation_repaired"] is True
    assert "Source: forces.md" in accepted[0]["answer"]


def test_revision_review_rejects_ungrounded_or_wrong_citation():
    invented = "Bananas are yellow and rich in potassium. Source: forces.md | sample"
    accepted, rejected = review_revision([revision_row(invented)])
    assert not accepted
    assert any("not grounded" in reason for reason in rejected[0]["reasons"])

    wrong = (
        "Friction is a contact force that opposes motion and produces heat.\n"
        "Source: biology.md | page 1"
    )
    accepted, rejected = review_revision([revision_row(wrong, "revision-3")])
    assert not accepted
    assert any("different source" in reason for reason in rejected[0]["reasons"])


def test_performance_review_enforces_unknown_rule_and_dedupes():
    good = {"category": "Strong", "reason": "High accuracy across six attempts with independent answers."}
    accepted, rejected = review_performance([performance_row(good)])
    assert len(accepted) == 1
    assert accepted[0]["label"] == "Strong"
    assert not rejected

    premature = {"category": "Strong", "reason": "Two easy correct attempts suggest strong performance."}
    accepted, rejected = review_performance([performance_row(premature, distinct=2, persona="u1")])
    assert not accepted
    assert any("fewer than three" in reason for reason in rejected[0]["reasons"])

    weak_reason = {"category": "Strong", "reason": "ok"}
    accepted, rejected = review_performance([performance_row(weak_reason, persona="u2")])
    assert not accepted
    assert any("reason" in reason for reason in rejected[0]["reasons"])


def test_topic_splits_keep_topics_disjoint():
    topics = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    splits = topic_splits(topics)
    all_topics = set().union(*splits.values())
    assert all_topics == set(topics)
    assert sum(len(value) for value in splits.values()) == len(topics)
    assert not splits["train"] & splits["test"]
    assert not splits["train"] & splits["val"]
