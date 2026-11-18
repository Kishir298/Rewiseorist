"""Live revision-chat test against the real Ollama SmolLM2 service.

Skipped unless REWISEORIST_LIVE_OLLAMA=1 is set. Run with:
    REWISEORIST_LIVE_OLLAMA=1 pytest tests/test_ollama_live.py -q -s
Answers and latencies are written to evidence/verification/ollama-live-answers.json.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import pytest

import app as app_module

pytestmark = pytest.mark.skipif(
    os.environ.get("REWISEORIST_LIVE_OLLAMA") != "1",
    reason="live Ollama test not requested (set REWISEORIST_LIVE_OLLAMA=1)",
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "verification" / "ollama-live-answers.json"

NONCE_NOTE = """# Rewisium notes

The fictitious element rewisium has symbol Rw and atomic number 137.
It was named after this coursework project and does not exist in nature.
Rewisium melts at 451 degrees Celsius according to these notes.
"""


@pytest.fixture()
def client(tmp_path):
    app = app_module.create_app({
        "TESTING": True,
        "DATABASE": str(tmp_path / "live.db"),
        "START_INACTIVITY_THREAD": False,
    })
    with app.test_client() as test_client:
        yield test_client


def _json(response):
    assert response.is_json, response.data[:300]
    return response.get_json()


def test_live_smollm2_answers_from_uploaded_material(client):
    profile = _json(client.post("/api/profiles", json={"name": "LiveTest"}))["profile"]
    topic = _json(client.post("/api/topics", json={
        "profile_id": profile["id"], "name": "Chemistry",
    }))["topic"]
    upload = client.post(
        "/api/resources/upload",
        data={"profile_id": str(profile["id"]), "topic_name": "Chemistry",
              "files": (io.BytesIO(NONCE_NOTE.encode()), "rewisium-notes.md")},
        content_type="multipart/form-data",
    )
    assert _json(upload)["count"] == 1

    record = {"model": "smollm2:360m", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # 1. Answerable question: the fact exists ONLY in the uploaded file.
    started = time.time()
    answerable = _json(client.post("/api/chat", json={
        "profile_id": profile["id"], "topic_id": topic["id"],
        "message": "According to the uploaded notes, what is the atomic number of rewisium?",
    }))
    latency = round(time.time() - started, 2)
    text = answerable["message"]["content"]
    record["answerable"] = {
        "question": "According to the uploaded notes, what is the atomic number of rewisium?",
        "answer": text, "latency_s": latency,
        "verified": bool(answerable["message"]["verified"]),
        "sources": [s.get("original_name") for s in answerable["message"].get("sources", [])],
    }
    assert text.strip(), "empty answer"
    assert "137" in text, f"answer must contain the uploaded fact: {text}"
    assert answerable["message"]["sources"], "answer must carry source references"
    assert bool(answerable["message"]["verified"]) is True, f"answer not grounded: {text}"
    assert answerable["model"]["version"] == "smollm2:360m"

    # 2. Unsupported question: not in any uploaded source.
    started = time.time()
    unsupported = _json(client.post("/api/chat", json={
        "profile_id": profile["id"], "topic_id": topic["id"],
        "session_id": answerable["session_id"],
        "message": "Who won the FIFA World Cup in 1998 and what was the final score?",
    }))
    latency2 = round(time.time() - started, 2)
    text2 = unsupported["message"]["content"]
    record["unsupported"] = {
        "question": "Who won the FIFA World Cup in 1998 and what was the final score?",
        "answer": text2, "latency_s": latency2,
        "verified": bool(unsupported["message"]["verified"]),
    }
    # Grounding check must not mark it as verified: the topic has no such source.
    assert bool(unsupported["message"]["verified"]) is False, (
        f"unsupported answer must not be marked verified: {text2}"
    )

    # 3. Persistence: reopen the session and confirm both answers with sources.
    session_id = answerable["session_id"]
    reopened = _json(client.get(f"/api/sessions/{session_id}"))
    assistants = [m for m in reopened["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 2
    assert all(m["model_version"] == "smollm2:360m" for m in assistants)

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
