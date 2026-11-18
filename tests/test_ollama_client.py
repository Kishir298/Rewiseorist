"""Unit tests for the Ollama SmolLM2 revision client (all HTTP mocked)."""

import json

import pytest
import requests

from ml import ollama_client
from ml.ollama_client import (
    MODEL_NAME,
    NUM_CTX,
    MAX_TOKENS,
    TEMPERATURE,
    OllamaClient,
    OllamaEmptyAnswer,
    OllamaHTTPError,
    OllamaModelMissing,
    OllamaTimeout,
    OllamaUnavailable,
    lexical_overlap,
    pick_sources,
    validate_citation,
)
from ml.prompts import build_ollama_revision_messages

FRICTION = "Friction is a contact force that opposes motion and produces heat."
PHOTOS = "Chlorophyll inside chloroplasts absorbs light energy for photosynthesis."


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)
        self.ok = 200 <= status_code < 300

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def make_client(**overrides):
    defaults = dict(read_timeout=5.0)
    defaults.update(overrides)
    return OllamaClient(**defaults)


def friction_sources(count=1):
    sources = []
    for index in range(count):
        sources.append({
            "original_name": f"notes{index}.md",
            "source_ref": f"notes{index}.md | p{index}",
            "text": FRICTION if index == 0 else PHOTOS,
        })
    return sources


def test_pick_sources_caps_count_length_and_deduplicates():
    sources = [
        {"original_name": "a.md", "source_ref": "a.md | p1", "text": "x" * 900},
        {"original_name": "a.md", "source_ref": "a.md | p1", "text": "duplicate"},
        {"original_name": "b.md", "source_ref": "b.md | p1", "text": "second"},
        {"original_name": "c.md", "source_ref": "c.md | p1", "text": "third"},
        {"original_name": "d.md", "source_ref": "d.md | p1", "text": "fourth"},
    ]
    picked = pick_sources(sources)
    assert len(picked) == 3
    assert picked[0]["original_name"] == "a.md"
    assert picked[1]["original_name"] == "b.md"
    assert picked[2]["original_name"] == "c.md"
    assert len(picked[0]["text"]) == 600
    assert "duplicate" not in [p["text"] for p in picked]


def test_messages_are_short_and_use_at_most_three_sources():
    sources = friction_sources(5)
    messages = build_ollama_revision_messages("What is friction?", sources)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "smollm" not in messages[0]["content"].lower()
    for index in range(1, 6):
        if index <= 3:
            assert f"[{index}]" in messages[1]["content"]
        else:
            assert f"[{index}]" not in messages[1]["content"]
    assert len(messages[1]["content"]) < 2500
    assert messages[0]["content"].endswith("Source: <file> | <section>.")


def test_generate_revision_success_uses_exact_configuration(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(payload={
            "message": {"role": "assistant", "content": FRICTION},
            "model": MODEL_NAME, "eval_count": 42, "prompt_eval_count": 120,
        })

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    client = make_client()
    result = client.generate_revision("What is friction?", friction_sources())

    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["model"] == "smollm2:360m"
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"] == {
        "num_ctx": 2048, "num_predict": 200, "temperature": 0.2,
    }
    assert captured["timeout"] == (3.05, 5.0)
    assert result["verified"] is True
    assert result["grounding"]["status"] == "verified"
    assert result["model_version"] == "smollm2:360m"
    assert result["model"]["name"] == "SmolLM2 360M"
    assert result["model"]["provider"] == "ollama"
    assert result["model"]["local"] is True
    assert result["text"].endswith("Source: notes0.md | notes0.md | p0")
    assert result["sources"][0]["original_name"] == "notes0.md"
    assert result["usage"]["eval_count"] == 42
    assert result["latency_s"] >= 0


def test_generate_revision_drops_fabricated_citation(monkeypatch):
    content = f"{FRICTION}\nSource: invented-file.md | page 9"
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeResponse(payload={"message": {"content": content}}),
    )
    result = make_client().generate_revision("What is friction?", friction_sources())
    assert "invented-file.md" not in result["text"]
    assert "Source: notes0.md" in result["text"]
    assert result["verified"] is True


def test_generate_revision_marks_unrelated_answer_unverified(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeResponse(
            payload={"message": {"content": "The capital of France is Paris."}}
        ),
    )
    result = make_client().generate_revision("What is friction?", friction_sources())
    assert result["verified"] is False
    assert result["grounding"]["status"] == "unverified"
    assert "Source:" not in result["text"]


def test_generate_revision_without_sources_is_no_source(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeResponse(
            payload={"message": {"content": "No uploaded material was supplied."}}
        ),
    )
    result = make_client().generate_revision("What is friction?", [])
    assert result["verified"] is False
    assert result["grounding"]["status"] == "no-source"
    assert result["sources"] == []


def test_empty_answer_raises(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeResponse(payload={"message": {"content": "   "}}),
    )
    with pytest.raises(OllamaEmptyAnswer):
        make_client().generate_revision("Question?", friction_sources())


def test_connection_failure_raises_actionable_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(ollama_client.requests, "post", boom)
    with pytest.raises(OllamaUnavailable) as exc:
        make_client().generate_revision("Question?", friction_sources())
    assert "ollama serve" in str(exc.value)


def test_timeout_raises_actionable_error(monkeypatch):
    def slow(*a, **k):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(ollama_client.requests, "post", slow)
    with pytest.raises(OllamaTimeout) as exc:
        make_client().generate_revision("Question?", friction_sources())
    assert "did not answer" in str(exc.value)


def test_missing_model_raises_pull_instruction(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeResponse(
            status_code=404,
            payload={"error": f"model '{MODEL_NAME}' not found"},
        ),
    )
    with pytest.raises(OllamaModelMissing) as exc:
        make_client().generate_revision("Question?", friction_sources())
    assert "ollama pull smollm2:360m" in str(exc.value)


def test_http_error_raises(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeResponse(status_code=500, payload={"error": "boom"}),
    )
    with pytest.raises(OllamaHTTPError) as exc:
        make_client().generate_revision("Question?", friction_sources())
    assert exc.value.status == 500


def test_error_body_with_not_found_is_model_missing(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeResponse(
            payload={"error": f"model '{MODEL_NAME}' not found"}
        ),
    )
    with pytest.raises(OllamaModelMissing):
        make_client().generate_revision("Question?", friction_sources())


def test_status_reports_model_present(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "get",
        lambda *a, **k: FakeResponse(payload={"models": [{"name": MODEL_NAME}]}),
    )
    status = make_client().status()
    assert status["available"] is True
    assert status["model_present"] is True
    assert status["error"] is None
    assert status["cloud_required"] is False


def test_status_reports_missing_model_with_pull_command(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "get",
        lambda *a, **k: FakeResponse(payload={"models": [{"name": "llama3:8b"}]}),
    )
    status = make_client().status()
    assert status["available"] is True
    assert status["model_present"] is False
    assert "ollama pull smollm2:360m" in status["error"]


def test_status_reports_daemon_down(monkeypatch):
    def refused(*a, **k):
        raise requests.ConnectionError("no daemon")

    monkeypatch.setattr(ollama_client.requests, "get", refused)
    status = make_client().status()
    assert status["available"] is False
    assert "not running" in status["error"]


def test_status_reports_timeout(monkeypatch):
    def slow(*a, **k):
        raise requests.Timeout("slow")

    monkeypatch.setattr(ollama_client.requests, "get", slow)
    status = make_client().status()
    assert status["available"] is False
    assert "did not respond" in status["error"]


def test_from_config_reads_explicit_flask_settings():
    client = OllamaClient.from_config({
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434/",
        "OLLAMA_MODEL": MODEL_NAME,
        "OLLAMA_NUM_CTX": NUM_CTX,
        "OLLAMA_MAX_TOKENS": MAX_TOKENS,
        "OLLAMA_TEMPERATURE": TEMPERATURE,
    })
    assert client.base_url == "http://127.0.0.1:11434"
    assert client.model == "smollm2:360m"
    assert client.num_ctx == 2048
    assert client.max_tokens == 200
    assert client.temperature == 0.2


def test_warmup_and_unload(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return FakeResponse(payload={"message": {"content": "OK"}})

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    monkeypatch.setattr(
        ollama_client.requests, "get",
        lambda *a, **k: FakeResponse(payload={"models": [{"name": MODEL_NAME}]}),
    )
    client = make_client()
    assert client.warmup()["warm"] is True
    assert client.unload()["unloaded"] is True
    assert calls[1]["keep_alive"] == 0


def test_lexical_overlap_and_citation_helpers():
    assert lexical_overlap(FRICTION, FRICTION) >= 5
    assert lexical_overlap("Paris capital France", FRICTION) == 0
    text, cited = validate_citation(
        f"{FRICTION}\nSource: notes0.md | p0",
        [{"original_name": "notes0.md", "source_ref": "notes0.md | p0", "text": FRICTION}],
    )
    assert cited["original_name"] == "notes0.md"
    text, cited = validate_citation(
        "Answer.\nSource: ghost.md | p1",
        [{"original_name": "notes0.md", "source_ref": "notes0.md | p0", "text": FRICTION}],
    )
    assert cited is None
    assert "ghost.md" not in text
