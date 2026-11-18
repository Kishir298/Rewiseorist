"""Guard: revision chat runs on local Ollama with exactly smollm2:360m.

The old "never use Ollama" policy is officially replaced: Ollama is now the
local inference backend, pinned to the ``smollm2:360m`` model on localhost
only. The Python ``ollama`` SDK is still forbidden (the client uses
``requests``), and no hardcoded cloud LLM endpoint may appear anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".js", ".html", ".css", ".ps1", ".cmd", ".bat", ".sh"}
SKIP_DIRS = {
    ".git", ".venv", "node_modules", "teacher_models", "data", "models",
    "__pycache__", ".pytest_cache",
}
SELF_FILE = Path(__file__).resolve()

SDK_IMPORT = re.compile(r"^\s*(?:import|from)\s+ollama\b", re.MULTILINE)
CLOUD_HOSTS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
)


def iter_source_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.resolve() == SELF_FILE:
            continue
        yield path


def test_model_tag_is_exact_and_configured():
    from ml import ollama_client

    assert ollama_client.MODEL_NAME == "smollm2:360m"
    assert ollama_client.NUM_CTX == 2048
    assert ollama_client.MAX_TOKENS == 200
    assert ollama_client.TEMPERATURE == 0.2
    base_url = ollama_client.DEFAULT_BASE_URL
    assert "127.0.0.1" in base_url or "localhost" in base_url


def test_app_config_declares_explicit_ollama_settings(tmp_path):
    import app as app_module

    flask_app = app_module.create_app({
        "TESTING": True,
        "DATABASE": str(tmp_path / "p.db"),
        "START_INACTIVITY_THREAD": False,
    })
    config = flask_app.config
    assert config["OLLAMA_MODEL"] == "smollm2:360m"
    assert config["OLLAMA_NUM_CTX"] == 2048
    assert config["OLLAMA_MAX_TOKENS"] == 200
    assert config["OLLAMA_TEMPERATURE"] == 0.2
    base_url = config["OLLAMA_BASE_URL"]
    assert base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost")


def test_no_python_ollama_sdk_and_no_cloud_llm_calls():
    offenders = []
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        if SDK_IMPORT.search(text):
            offenders.append(f"{rel}: Python ollama SDK import")
        for host in CLOUD_HOSTS:
            if host in text:
                offenders.append(f"{rel}: hardcoded cloud endpoint {host}")
    assert not offenders, "Forbidden Ollama/cloud usage found:\n" + "\n".join(offenders)


def test_ollama_client_only_targets_configured_base_url():
    from ml import ollama_client

    assert ollama_client.DEFAULT_BASE_URL.startswith("http://127.0.0.1")
    client = ollama_client.OllamaClient()
    assert client.base_url == ollama_client.DEFAULT_BASE_URL


def test_runtime_still_cloud_free_and_teacher_uses_llama_cpp():
    from ml.runtime import ModelRuntime
    from ml.teacher_client import SERVER_BINARY, build_command, TEACHER_SPECS

    assert ModelRuntime(ROOT / "models").status()["cloud_required"] is False
    assert SERVER_BINARY.name == "llama-server.exe"
    command = build_command(TEACHER_SPECS[0])
    assert "llama-server" in command[0].lower()
