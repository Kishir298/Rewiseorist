from datetime import date, timedelta
from pathlib import Path

import pytest

import storage
import study_tools


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "test.db"
    storage.init_db(path)
    return path


def test_marking_short_and_mcq():
    short = {"type": "short", "answer": "Friction opposes motion.",
             "accepted_keywords": ["friction", "opposes", "motion"]}
    result = study_tools.mark_answer(short, "Friction opposes motion")
    assert result["correct"] is True
    assert result["marks"] == 1.0
    partial = study_tools.mark_answer(short, "Friction")
    assert partial["marks"] == 0.5
    blank = study_tools.mark_answer(short, "")
    assert blank["error_type"] == "blank"
    mcq = {"type": "mcq", "answer": "Push or pull", "options": ["Mass", "Push or pull"]}
    assert study_tools.mark_answer(mcq, "Push or pull")["correct"] is True
    assert study_tools.mark_answer(mcq, "1")["correct"] is True
    assert study_tools.mark_answer(mcq, "Mass")["correct"] is False


def test_priority_and_plan(db_path):
    profile = storage.create_profile("Planner", db_path)
    topic = storage.add_topic(profile["id"], "Forces", difficulty="hard", db_path=db_path)
    deadline = storage.add_deadline(profile["id"], "Test", (date.today() + timedelta(days=2)).isoformat(),
                                    topic["id"], db_path=db_path)
    priority = study_tools.calculate_priority(topic, deadline, [], today=date.today())
    assert priority["level"] == "Unknown"
    assert priority["urgency"] > 0.3
    storage.add_topic(profile["id"], "Algebra", db_path=db_path)
    plan = study_tools.propose_plan(
        profile["id"],
        available=[(date.today().isoformat(), 60), ((date.today() + timedelta(days=1)).isoformat(), 30)],
        db_path=db_path,
    )
    assert plan["blocks"]
    assert "Revision plan" in plan["markdown"]
    assert plan["summary"]


def test_submit_rewards_and_improvement(db_path):
    profile = storage.create_profile("Student", db_path)
    topic = storage.add_topic(profile["id"], "Fractions", db_path=db_path)
    question = storage.save_question(
        topic["id"], "q1", "What is a fraction?", "A part of a whole.",
        db_path=db_path, accepted_keywords=["part", "whole"],
    )
    first = study_tools.submit_answer(profile["id"], question, "A part of a whole", db_path=db_path)
    assert first["correct"] is True
    assert first["rewards"]
    second = study_tools.submit_answer(profile["id"], question, "A part of a whole",
                                       db_path=db_path, attempt_key="duplicate-key")
    # The question-level reward is not paid twice; completion still gets a small XP award.
    assert not any(r["kind"] == "coins" for r in second["rewards"])
    for i, wrong_answer in enumerate(["wrong first try", "still not sure"]):
        study_tools.submit_answer(
            profile["id"], question, wrong_answer, db_path=db_path,
            attempt_key=f"wrong-{i}", hint_used=True, independent=False,
        )
    steps = study_tools.improvement_steps(profile["id"], topic["id"], db_path)
    assert steps["ready"] is True
    assert steps["steps"]
    progress = study_tools.progress_summary(profile["id"], "all", db_path=db_path)
    assert progress["attempts"] == 3
    assert progress["chart"]
    assert progress["adherence_label"] == "N/A"


def test_sample_bank_seed(tmp_path, db_path):
    profile = storage.create_profile("Bank", db_path)
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "quiz_keys.json").write_text(
        '{"forces": [{"id":"f1","type":"mcq","question":"A force is?",'
        '"options":["A push or pull","A mass"],"answer":"A push or pull",'
        '"accepted_keywords":[],"hint":"Think contact.","explanation":"Force is a push or pull.",'
        '"difficulty":"easy","source":"forces.md"}]}',
        encoding="utf-8",
    )
    seeded = study_tools.seed_sample_questions(profile["id"], "Forces", db_path, samples)
    assert len(seeded) == 1
    assert seeded[0]["question_key"] == "f1"
