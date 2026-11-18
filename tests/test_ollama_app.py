"""Ollama SmolLM2 integration tests for the Flask app (HTTP fully mocked)."""

import io
from pathlib import Path

import app as app_module
import storage
from tests.test_app import FakeOllama, FakeRuntime, fake_ollama, json_of, make_profile
import pytest


@pytest.fixture()
def client(tmp_path):
    runtime = FakeRuntime()
    ollama = FakeOllama()
    app = app_module.create_app({
        "TESTING": True,
        "DATABASE": str(tmp_path / "app.db"),
        "START_INACTIVITY_THREAD": False,
    }, runtime_override=runtime, ollama_override=ollama)
    with app.test_client() as test_client:
        yield test_client, runtime, tmp_path


def test_chat_uses_smollm2_and_persists_sources(client):
    client, runtime, tmp_path = client
    profile = make_profile(client, "OllamaChat")
    topic = json_of(client.post("/api/topics", json={
        "profile_id": profile["id"], "name": "Forces",
    }))["topic"]
    client.post(
        "/api/resources/upload",
        data={"profile_id": str(profile["id"]), "topic_name": "Forces",
              "files": (io.BytesIO(b"Friction is a contact force that opposes motion."), "forces.md")},
        content_type="multipart/form-data",
    )
    chat = json_of(client.post("/api/chat", json={
        "profile_id": profile["id"], "topic_id": topic["id"],
        "message": "What is friction?",
    }))
    assert chat["model"]["provider"] == "ollama"
    assert chat["model"]["version"] == "smollm2:360m"
    assert chat["message"]["model_version"] == "smollm2:360m"
    assert bool(chat["message"]["verified"]) is True
    assert runtime.generated == 0
    session_id = chat["session_id"]
    reopened = json_of(client.get(f"/api/sessions/{session_id}"))
    assistant = [m for m in reopened["messages"] if m["role"] == "assistant"][-1]
    assert assistant["model_version"] == "smollm2:360m"
    assert assistant["sources"]
    restarted = app_module.create_app({
        "TESTING": True, "DATABASE": str(tmp_path / "app.db"),
        "START_INACTIVITY_THREAD": False,
    }, runtime_override=runtime, ollama_override=fake_ollama(client))
    with restarted.test_client() as fresh:
        again = json_of(fresh.get(f"/api/sessions/{session_id}"))
        assert any(m["role"] == "assistant" and m["model_version"] == "smollm2:360m"
                   for m in again["messages"])


def test_chat_ollama_failure_modes(client):
    from ml.ollama_client import (OllamaEmptyAnswer, OllamaModelMissing,
                                  OllamaTimeout, OllamaUnavailable)
    client, runtime, _ = client
    profile = make_profile(client, "OllamaErrors")
    topic = json_of(client.post("/api/topics", json={
        "profile_id": profile["id"], "name": "Forces",
    }))["topic"]
    ollama = fake_ollama(client)
    cases = [
        (OllamaUnavailable("Ollama is not running. Start it with `ollama serve`."), 503, "ollama serve"),
        (OllamaModelMissing("Run `ollama pull smollm2:360m` and retry."), 503, "ollama pull smollm2:360m"),
        (OllamaTimeout("did not answer within 90 seconds"), 504, "did not answer"),
        (OllamaEmptyAnswer("returned an empty answer"), 502, "empty answer"),
    ]
    for exc, status, needle in cases:
        ollama.raise_on_generate = exc
        response = client.post("/api/chat", json={
            "profile_id": profile["id"], "topic_id": topic["id"],
            "message": "What is friction?",
        })
        assert response.status_code == status, (exc, response.status_code)
        body = json_of(response)
        assert needle in body["error"], body
    ollama.raise_on_generate = None


def test_model_status_reports_ollama_and_lifecycle(client):
    client, _, _ = client
    status = json_of(client.get("/api/models/status"))
    assert status["ollama"]["model"] == "smollm2:360m"
    assert status["ollama"]["num_ctx"] == 2048
    assert status["ollama"]["max_tokens"] == 200
    assert status["ollama"]["temperature"] == 0.2
    health = json_of(client.get("/api/health"))
    assert health["ollama"]["available"] is True
    warm = json_of(client.post("/api/models/warmup", json={}))
    assert warm["result"]["ollama"]["warm"] is True
    unload = json_of(client.post("/api/models/unload", json={}))
    assert unload["result"]["ollama"]["unloaded"] is True


def test_classify_uses_rule_fallback_when_classifier_missing(client):
    client, _, _ = client
    profile = make_profile(client, "Fallback")
    topic = json_of(client.post("/api/topics", json={
        "profile_id": profile["id"], "name": "Algebra",
    }))["topic"]
    db_path = Path(client.application.config["DATABASE"])
    for i in range(3):
        q = storage.save_question(topic["id"], f"fb{i}", f"Question {i}", f"Answer {i}",
                                  db_path=db_path)
        storage.record_attempt(profile["id"], f"fbk{i}", q["id"], f"Answer {i}", 1, 1, True,
                               db_path=db_path, topic_id=topic["id"], active_seconds=20)

    class MissingRuntime:
        def missing_files(self):
            return ["performance_model.joblib"]

        def status(self):
            return {"ready": False, "missing_files": self.missing_files()}

        def classify_performance(self, features):
            from ml.runtime import ModelUnavailable
            raise ModelUnavailable(
                "The trained local model files are missing: performance_model.joblib")

        def warmup(self):
            from ml.runtime import ModelUnavailable
            raise ModelUnavailable("missing")

        def unload(self):
            return {"unloaded": True}

        def shutdown(self):
            return {"stopped": True}

    client.application.extensions["runtime"] = MissingRuntime()
    result = json_of(client.post("/api/classify", json={
        "profile_id": profile["id"], "topic_id": topic["id"],
    }))
    assert result["model_version"] == "rule-fallback-v1"
    assert result["fallback"] is True
    assert result["category"] in {"Weak", "Developing", "Strong"}
    assert "Rule-based fallback" in result["reason"]
