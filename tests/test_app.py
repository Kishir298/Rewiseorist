import io
import json
from pathlib import Path

import pytest

import app as app_module
import storage


class FakeRuntime:
    def __init__(self):
        self.unloaded = 0
        self.stopped = 0
        self.generated = 0
        self.classified = 0

    def status(self):
        return {"ready": True, "loaded": True, "mode": "local", "ollama_required": False,
                "cloud_required": False, "language_model": {"version": "fake-lm"},
                "performance_model": {"version": "fake-clf"}}

    def generate_revision(self, question, sources, personality="Friendly"):
        self.generated += 1
        source = sources[0]["source_ref"] if sources else ""
        return {"text": "Friction opposes motion. " + (f"Source: {source}" if source else ""),
                "model_version": "fake-lm", "verified": bool(sources), "sources": sources,
                "model": {"name": "fake", "version": "fake-lm"}}

    def classify_performance(self, features):
        self.classified += 1
        return {"category": "Developing", "reason": "Accuracy is improving but hints are still used.",
                "model_version": "fake-clf", "features": features}

    def warmup(self):
        return {"warm": True}

    def unload(self):
        self.unloaded += 1
        return {"unloaded": True}

    def shutdown(self):
        self.stopped += 1
        return {"stopped": True}


class FakeOllama:
    """Stands in for the Ollama SmolLM2 revision client."""

    def __init__(self):
        self.generated = 0
        self.warmed = 0
        self.unloaded = 0
        self.raise_on_generate = None
        self.answer = "Friction is a contact force that opposes motion.\nSource: forces.md | sample"

    def status(self):
        return {"available": True, "model_present": True, "model": "smollm2:360m",
                "label": "SmolLM2 360M", "base_url": "http://127.0.0.1:11434",
                "num_ctx": 2048, "max_tokens": 200, "temperature": 0.2,
                "ollama_required": True, "cloud_required": False, "error": None}

    def generate_revision(self, question, sources, personality="Friendly"):
        self.generated += 1
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return {
            "text": self.answer,
            "sources": list(sources[:3]),
            "verified": bool(sources),
            "model_version": "smollm2:360m",
            "model": {"name": "SmolLM2 360M", "version": "smollm2:360m",
                      "provider": "ollama", "local": True},
            "grounding": {"status": "verified" if sources else "no-source"},
        }

    def warmup(self):
        self.warmed += 1
        return {"warm": True, "model": "smollm2:360m"}

    def unload(self):
        self.unloaded += 1
        return {"unloaded": True, "message": "SmolLM2 360M released from memory."}


@pytest.fixture()
def client(tmp_path):
    runtime = FakeRuntime()
    ollama = FakeOllama()
    app = app_module.create_app({
        "TESTING": True,
        "DATABASE": str(tmp_path / "app.db"),
        "START_INACTIVITY_THREAD": False,
        "INACTIVITY_THRESHOLDS": {"warmup": 2, "worker": 7, "media": 10},
    }, runtime_override=runtime, ollama_override=ollama)
    with app.test_client() as test_client:
        yield test_client, runtime, tmp_path


def fake_ollama(client):
    return client.application.extensions["ollama"]


def json_of(response):
    assert response.is_json, response.data[:300]
    return response.get_json()


def make_profile(client, name="Student"):
    response = client.post("/api/profiles", json={"name": name})
    return json_of(response)["profile"]


def test_profile_settings_and_help(client):
    client, _, _ = client
    profile = make_profile(client)
    response = client.post("/api/settings", json={
        "profile_id": profile["id"], "personality": "Formal", "theme": "midnight",
    })
    assert json_of(response)["settings"]["personality"] == "Formal"
    help_response = json_of(client.post("/api/help", json={
        "profile_id": profile["id"], "message": "Where are my folders?",
    }))
    assert help_response["action"]["page"] == "resources"
    theme_response = json_of(client.post("/api/help", json={
        "profile_id": profile["id"], "message": "How do I change my theme?",
    }))
    assert theme_response["action"]["page"] == "settings"


def test_full_study_flow(client):
    client, runtime, tmp_path = client
    profile = make_profile(client, "Flow")
    topic = json_of(client.post("/api/topics", json={
        "profile_id": profile["id"], "name": "Forces", "difficulty": "medium",
    }))["topic"]
    upload = client.post(
        "/api/resources/upload",
        data={"profile_id": str(profile["id"]), "topic_name": "Forces",
              "files": (io.BytesIO(b"Friction opposes motion. A force is a push or pull."), "forces.md")},
        content_type="multipart/form-data",
    )
    assert json_of(upload)["count"] == 1
    chat = json_of(client.post("/api/chat", json={
        "profile_id": profile["id"], "topic_id": topic["id"],
        "message": "What is friction?",
    }))
    assert chat["message"]["role"] == "assistant"
    assert chat["sources"]
    session_id = chat["session_id"]
    reopened = json_of(client.get(f"/api/sessions/{session_id}"))
    assert len(reopened["messages"]) >= 2
    quiz = json_of(client.get(f"/api/quiz?profile_id={profile['id']}&topic_id={topic['id']}"))
    assert quiz["questions"]
    assert "answer" not in quiz["questions"][0]
    question = quiz["questions"][0]
    answer = json_of(client.post("/api/quiz/answer", json={
        "profile_id": profile["id"], "question_id": question["id"], "answer": "test",
    }))
    assert "result" in answer
    plan = json_of(client.post("/api/plans/propose", json={
        "profile_id": profile["id"], "horizon_days": 2, "daily_minutes": 30,
    }))
    assert plan["plan"]["blocks"]
    plan_id = plan["stored"]["id"]
    accepted = json_of(client.post(f"/api/plans/{plan_id}/accept", json={
        "profile_id": profile["id"],
    }))
    assert accepted["plan"]["status"] == "accepted"
    progress = json_of(client.get(f"/api/progress?profile_id={profile['id']}&period=all"))
    assert progress["progress"]["attempts"] >= 1
    assert fake_ollama(client).generated >= 1
    assert runtime.generated == 0


def test_classifier_unknown_and_fake(client):
    client, runtime, _ = client
    profile = make_profile(client, "Classify")
    topic = json_of(client.post("/api/topics", json={
        "profile_id": profile["id"], "name": "Algebra",
    }))["topic"]
    unknown = json_of(client.post("/api/classify", json={
        "profile_id": profile["id"], "topic_id": topic["id"],
    }))
    assert unknown["category"] == "Unknown"
    # Seed three distinct checked questions and attempts.
    for i in range(3):
        q = storage.save_question(topic["id"], f"q{i}", f"Question {i}", f"Answer {i}",
                                  db_path=Path(client.application.config["DATABASE"]))
        storage.record_attempt(profile["id"], f"k{i}", q["id"], f"Answer {i}", 1, 1, True,
                               db_path=Path(client.application.config["DATABASE"]),
                               topic_id=topic["id"], active_seconds=20)
    result = json_of(client.post("/api/classify", json={
        "profile_id": profile["id"], "topic_id": topic["id"],
    }))
    assert result["category"] == "Developing"
    assert runtime.classified == 1


def test_store_notes_and_backup(client):
    client, _, _ = client
    profile = make_profile(client, "Store")
    items = json_of(client.get(f"/api/store?profile_id={profile['id']}"))["items"]
    assert items
    buy = json_of(client.post("/api/store/buy", json={
        "profile_id": profile["id"], "item_key": "ai-owl",
    }))
    assert buy["ok"] is False or buy["charged"] == 0  # starting coins are 0
    note = json_of(client.post("/api/notes", json={
        "profile_id": profile["id"], "target_type": "quiz", "target_id": "1",
        "text": "Remember units", "strokes": [{"x": 1, "y": 2}],
    }))
    assert note["note"]["text"] == "Remember units"
    backup = client.get(f"/api/backup?profile_id={profile['id']}")
    assert backup.status_code == 200
    assert backup.data[:2] == b"PK"


def test_external_api_requires_permission_and_never_echoes_key(client, monkeypatch):
    client, runtime, _ = client
    profile = make_profile(client, "API")
    topic = json_of(client.post("/api/topics", json={
        "profile_id": profile["id"], "name": "Forces",
    }))["topic"]
    client.post(
        "/api/resources/upload",
        data={"profile_id": str(profile["id"]), "topic_name": "Forces",
              "files": (io.BytesIO(b"Friction is a contact force that opposes motion."), "forces.md")},
        content_type="multipart/form-data",
    )
    settings = json_of(client.post("/api/settings", json={
        "profile_id": profile["id"], "ai_mode": "api",
        "api_base": "https://example.invalid/v1", "api_model": "test-teacher",
    }))["settings"]
    assert settings["ai_mode"] == "api"
    assert json_of(client.post("/api/settings/api-key", json={
        "profile_id": profile["id"], "api_key": "test-secret-key",
    }))["api_key_present"] is True

    denied = client.post("/api/chat", json={
        "profile_id": profile["id"], "topic_id": topic["id"],
        "message": "What is friction?",
    })
    assert denied.status_code == 403
    assert "permission" in json_of(denied)["error"].lower()
    assert runtime.generated == 0

    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": (
                "Friction is a contact force that opposes motion.\\n"
                "Source: forces.md | sample"
            )}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    allowed = json_of(client.post("/api/chat", json={
        "profile_id": profile["id"], "topic_id": topic["id"],
        "message": "What is friction?", "api_permission": True,
    }))
    assert allowed["model"]["local"] is False
    assert allowed["model"]["version"] == "external:test-teacher"
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-secret-key"
    assert captured["json"]["model"] == "test-teacher"
    assert "test-secret-key" not in json.dumps(allowed)
    assert runtime.generated == 0


def test_inactivity_boundaries(client):
    client, runtime, _ = client
    profile = make_profile(client, "Timer")
    # 2 seconds => unload, not shutdown
    state = json_of(client.post("/api/inactivity/simulate", json={
        "profile_id": profile["id"], "seconds_ago": 2,
    }))
    assert "weights_unloaded" in state["actions"]
    assert runtime.unloaded == 1
    assert runtime.stopped == 0
    # reset then 7 seconds => shutdown
    client.post("/api/activity", json={"profile_id": profile["id"]})
    state = json_of(client.post("/api/inactivity/simulate", json={
        "profile_id": profile["id"], "seconds_ago": 7,
    }))
    assert "worker_stopped" in state["actions"]
    assert runtime.stopped >= 1
    # reset then 10 seconds => media stop
    client.post("/api/activity", json={"profile_id": profile["id"]})
    state = json_of(client.post("/api/inactivity/simulate", json={
        "profile_id": profile["id"], "seconds_ago": 10,
    }))
    assert "media_stopped" in state["actions"]
