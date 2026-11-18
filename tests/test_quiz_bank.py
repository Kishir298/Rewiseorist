"""Validate the sample quiz bank: 90 questions, unique ids, grounded sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
RESOURCES = SAMPLES / "resources"

EXPECTED_TOPICS = [
    "fractions", "forces", "chemical-bonding", "indices", "electric-circuits",
    "industrial-revolution", "algebra", "photosynthesis", "particle-model",
]


@pytest.fixture(scope="module")
def bank() -> dict:
    return json.loads((SAMPLES / "quiz_keys.json").read_text(encoding="utf-8"))


def test_topic_keys_are_exactly_the_sample_corpus(bank: dict) -> None:
    assert list(bank.keys()) == EXPECTED_TOPICS


def test_every_topic_has_ten_questions(bank: dict) -> None:
    counts = {topic: len(items) for topic, items in bank.items()}
    assert all(count == 10 for count in counts.values()), counts
    assert sum(counts.values()) == 90


def test_question_ids_are_unique(bank: dict) -> None:
    ids = [q["id"] for items in bank.values() for q in items]
    assert len(ids) == len(set(ids)) == 90


def test_required_fields_present(bank: dict) -> None:
    required = ("id", "type", "question", "answer", "hint", "explanation", "difficulty", "source")
    for topic, items in bank.items():
        for q in items:
            for field in required:
                assert str(q.get(field, "")).strip(), f"{q.get('id')}: empty {field}"


def test_difficulty_and_type_values_are_valid(bank: dict) -> None:
    for items in bank.values():
        for q in items:
            assert q["difficulty"] in {"easy", "medium", "hard"}, q["id"]
            assert q["type"] in {"mcq", "short"}, q["id"]


def test_every_topic_covers_all_three_difficulties(bank: dict) -> None:
    for topic, items in bank.items():
        covered = {q["difficulty"] for q in items}
        assert {"easy", "medium", "hard"} <= covered, f"{topic}: {sorted(covered)}"


def test_mcq_shape_is_valid(bank: dict) -> None:
    for items in bank.values():
        for q in items:
            if q["type"] != "mcq":
                continue
            options = q.get("options") or []
            assert len(options) == 4, f"{q['id']}: {len(options)} options"
            assert len(set(options)) == 4, f"{q['id']}: duplicate options"
            assert q["answer"] in options, f"{q['id']}: answer is not an option"
            assert "accepted_keywords" not in q, f"{q['id']}: mcq carries keywords"


def test_short_answer_shape_is_valid(bank: dict) -> None:
    for items in bank.values():
        for q in items:
            if q["type"] != "short":
                continue
            assert q.get("accepted_keywords"), f"{q['id']}: no accepted_keywords"
            assert "options" not in q, f"{q['id']}: short answer carries options"


def test_sources_exist_and_match_topic(bank: dict) -> None:
    for topic, items in bank.items():
        for q in items:
            assert q["source"] == f"{topic}.md", f"{q['id']}: {q['source']}"
            assert (RESOURCES / q["source"]).exists(), f"{q['id']}: missing source file"


def test_bank_is_balanced_between_types(bank: dict) -> None:
    all_questions = [q for items in bank.values() for q in items]
    mcq = [q for q in all_questions if q["type"] == "mcq"]
    short = [q for q in all_questions if q["type"] == "short"]
    assert len(mcq) == 45, len(mcq)
    assert len(short) == 45, len(short)


def test_difficulty_distribution_is_reasonable(bank: dict) -> None:
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for items in bank.values():
        for q in items:
            counts[q["difficulty"]] += 1
    assert all(value >= 25 for value in counts.values()), counts


def test_new_questions_ground_in_their_source_text(bank: dict) -> None:
    """For every topic, at least one added question cites terms found in its source note."""
    for topic, items in bank.items():
        source_text = (RESOURCES / f"{topic}.md").read_text(encoding="utf-8").lower()
        grounded = 0
        for q in items:
            if q["type"] != "short" or int(q["id"].rsplit("-q", 1)[1]) < 4:
                continue
            if any(keyword.lower() in source_text for keyword in q["accepted_keywords"]):
                grounded += 1
        assert grounded >= 1, f"{topic}: no added question grounded in {topic}.md"
