from pathlib import Path

import pytest

import storage


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "test.db"
    storage.init_db(path)
    return path


def test_profiles_and_settings(db_path):
    profile = storage.create_profile("Alex", db_path)
    assert profile["name"] == "Alex"
    assert storage.get_profile_by_name("Alex", db_path)["id"] == profile["id"]
    settings = storage.save_settings(profile["id"], db_path, personality="Formal", theme="midnight")
    assert settings["personality"] == "Formal"
    assert settings["theme"] == "midnight"


def test_topics_resources_and_chunks(db_path):
    profile = storage.create_profile("Sam", db_path)
    subject = storage.add_subject(profile["id"], "Science", db_path)
    topic = storage.add_topic(profile["id"], "Forces", subject["id"], "medium", db_path)
    resource = storage.add_resource(
        profile["id"], "forces.md", "data/uploads/forces.md", "md", 100,
        topic_id=topic["id"], processed_text="Force equals mass times acceleration.", db_path=db_path,
    )
    count = storage.add_chunks(resource["id"], [
        {"text": "Force equals mass times acceleration.", "page_ref": "1", "source_ref": "forces.md|1"},
        {"text": "Friction opposes motion.", "page_ref": "2", "source_ref": "forces.md|2"},
    ], db_path)
    assert count == 2
    assert len(storage.list_chunks(profile["id"], db_path)) == 2
    assert storage.chunks_for_topic(profile["id"], topic["id"], db_path)[0]["original_name"] == "forces.md"


def test_sessions_and_messages(db_path):
    profile = storage.create_profile("Jo", db_path)
    session = storage.create_session(profile["id"], "Forces revision", db_path=db_path)
    storage.add_message(session["id"], "user", "Explain friction", db_path=db_path)
    storage.add_message(session["id"], "assistant", "Friction opposes motion.", db_path=db_path,
                        sources=[{"file": "forces.md", "page": "2"}], verified=True)
    messages = storage.list_messages(session["id"], db_path)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["sources"][0]["page"] == "2"


def test_attempts_and_unknown_rule(db_path):
    profile = storage.create_profile("Mia", db_path)
    topic = storage.add_topic(profile["id"], "Algebra", db_path=db_path)
    for i in range(3):
        storage.record_attempt(
            profile["id"], f"key-{i}", None, "answer", 1, 1, True,
            db_path=db_path, topic_id=topic["id"],
        )
    attempts = storage.list_attempts(profile["id"], topic["id"], db_path)
    assert len(attempts) == 3
    # Duplicate attempt keys do not create new evidence.
    storage.record_attempt(profile["id"], "key-0", None, "answer", 1, 1, True,
                           db_path=db_path, topic_id=topic["id"])
    assert len(storage.list_attempts(profile["id"], topic["id"], db_path)) == 3


def test_plan_versions_and_restore(db_path):
    profile = storage.create_profile("Noah", db_path)
    plan = storage.create_plan(profile["id"], "Week 1", {"summary": "v1", "blocks": []}, db_path)
    updated = storage.add_plan_version(plan["id"], {"summary": "v2", "blocks": []}, db_path,
                                       reason="Adapted", after_md="v2")
    assert len(updated["versions"]) == 2
    storage.accept_plan(plan["id"], 2, db_path)
    restored = storage.restore_plan_version(plan["id"], 1, db_path)
    assert restored["versions"][-1]["plan"]["summary"] == "v1"


def test_rewards_dedupe_and_store(db_path):
    profile = storage.create_profile("Ava", db_path)
    first = storage.award_reward(profile["id"], "lesson-1", "xp", 10, "Lesson", db_path)
    second = storage.award_reward(profile["id"], "lesson-1", "xp", 10, "Lesson", db_path)
    assert first["created"] is True
    assert second["created"] is False
    assert storage.get_profile(profile["id"], db_path)["xp"] == 10
    storage.award_reward(profile["id"], "coins-1", "coins", 100, "Test", db_path)
    result = storage.purchase_item(profile["id"], "ai-owl", db_path)
    assert result["charged"] == 45
    again = storage.purchase_item(profile["id"], "ai-owl", db_path)
    assert again["charged"] == 0
    assert storage.get_profile(profile["id"], db_path)["coins"] == 55


def test_inactivity_and_backup(db_path):
    profile = storage.create_profile("Eli", db_path)
    state = storage.touch_activity(profile["id"], db_path)
    assert state["last_real_activity"]
    storage.set_inactivity_marker(profile["id"], "warmup_stopped_at", storage.utc_now(), db_path)
    assert storage.get_inactivity_state(profile["id"], db_path)["warmup_stopped_at"]
    archive = storage.create_backup(db_path, db_path.parent / "backups")
    assert archive.exists()
    assert archive.stat().st_size > 0
