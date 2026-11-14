"""Guard: the project must never depend on Ollama at any stage."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE_SUFFIXES = {".py", ".js", ".html", ".css", ".cmd", ".bat", ".ps1", ".sh"}
SKIP_DIRS = {".git", ".venv", "node_modules", "teacher_models", "data", "models", "__pycache__"}
SKIP_FILES = {"tests/test_no_ollama.py"}

FORBIDDEN = [
    re.compile(r"\b11434\b"),
    re.compile(r"^\s*(?:import|from)\s+ollama\b", re.MULTILINE),
    re.compile(r"\bollama\.(?:chat|generate|list|pull|push|create|embed)\b", re.IGNORECASE),
    re.compile(r"\bOLLAMA_HOST\b"),
    re.compile(r"api/ollama", re.IGNORECASE),
]


def iter_source_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXECUTABLE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.relative_to(ROOT).as_posix() in SKIP_FILES:
            continue
        yield path


def test_no_ollama_calls_or_port():
    offenders = []
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)} matched {pattern.pattern}")
    assert not offenders, "Ollama dependency found:\n" + "\n".join(offenders)


def test_runtime_declares_no_ollama_and_local_only():
    from ml.runtime import ModelRuntime

    status = ModelRuntime(ROOT / "models").status()
    assert status["ollama_required"] is False
    assert status["cloud_required"] is False
    assert status["mode"] == "local"


def test_teacher_runtime_uses_llama_cpp_binary():
    from ml.teacher_client import SERVER_BINARY, build_command, TEACHER_SPECS

    assert SERVER_BINARY.name == "llama-server.exe"
    command = build_command(TEACHER_SPECS[0])
    assert "llama-server" in command[0].lower()
    assert "--host" in command
