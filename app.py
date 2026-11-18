"""Rewiseorist Flask application.

Run with:
    .venv\\Scripts\\python.exe app.py
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from flask import Flask, current_app, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

import resources
import storage
import study_tools
from ml.ollama_client import (
    MODEL_NAME as OLLAMA_MODEL_NAME,
    OllamaClient,
    OllamaEmptyAnswer,
    OllamaError,
    OllamaHTTPError,
    OllamaModelMissing,
    OllamaTimeout,
    OllamaUnavailable,
)
from ml.prompts import build_revision_prompt, clean_teacher_answer, extract_citation, strip_chat_markers
from ml.runtime import ModelRuntime, ModelUnavailable

ROOT = Path(__file__).resolve().parent
APP_VERSION = "1.0.0"
DEFAULT_INACTIVITY = {"warmup": 120, "worker": 420, "media": 600}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


class InactivityManager:
    """Server-side inactivity controller with exact 2/7/10 minute boundaries."""

    def __init__(self, runtime: ModelRuntime, db_path: str | Path,
                 thresholds: dict | None = None) -> None:
        self.runtime = runtime
        self.db_path = Path(db_path)
        self.thresholds = {**DEFAULT_INACTIVITY, **(thresholds or {})}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def touch(self, profile_id: int) -> dict:
        with self._lock:
            return storage.touch_activity(profile_id, self.db_path)

    def check(self, profile_id: int, now: datetime | None = None) -> dict:
        now = now or _now()
        state = storage.get_inactivity_state(profile_id, self.db_path)
        last = _parse_iso(state.get("last_real_activity"))
        if last is None:
            last = now
        elapsed = max((now - last).total_seconds(), 0)
        actions = []
        if elapsed >= self.thresholds["media"] and not state.get("media_stop_at"):
            storage.set_inactivity_marker(profile_id, "media_stop_at", now.isoformat(), self.db_path)
            actions.append("media_stopped")
            storage.log_activity(profile_id, "inactivity_media_stop", f"{elapsed:.0f}s", self.db_path)
            state = storage.get_inactivity_state(profile_id, self.db_path)
        if elapsed >= self.thresholds["worker"] and not state.get("worker_stopped_at"):
            self.runtime.shutdown()
            storage.set_inactivity_marker(profile_id, "worker_stopped_at", now.isoformat(), self.db_path)
            actions.append("worker_stopped")
            storage.log_activity(profile_id, "inactivity_worker_stop", f"{elapsed:.0f}s", self.db_path)
            state = storage.get_inactivity_state(profile_id, self.db_path)
        if elapsed >= self.thresholds["warmup"] and not state.get("warmup_stopped_at"):
            self.runtime.unload()
            storage.set_inactivity_marker(profile_id, "warmup_stopped_at", now.isoformat(), self.db_path)
            actions.append("weights_unloaded")
            storage.log_activity(profile_id, "inactivity_unload", f"{elapsed:.0f}s", self.db_path)
            state = storage.get_inactivity_state(profile_id, self.db_path)
        return{"elapsed_seconds": round(elapsed, 1), "actions": actions,
               "state": state, "thresholds": self.thresholds}

    def tick_all(self, now: datetime | None = None) -> list[dict]:
        results = []
        for profile in storage.list_profiles(self.db_path):
            results.append(self.check(profile["id"], now))
        return results

    def _loop(self) -> None:
        while not self._stop.wait(5):
            try:
                self.tick_all()
            except Exception:
                # A timer must never crash the website.
                continue

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="rewiseorist-inactivity", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def _db() -> Path:
    return Path(current_app.config["DATABASE"])


def _runtime() -> ModelRuntime:
    return current_app.extensions["runtime"]


def _ollama() -> OllamaClient:
    return current_app.extensions["ollama"]


def _inactivity() -> InactivityManager:
    return current_app.extensions["inactivity"]


def _body() -> dict:
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return {key: value for key, value in request.form.items()}


def _profile_id(required: bool = True) -> int | None:
    data = _body()
    value = data.get("profile_id")
    if value is None:
        value = request.args.get("profile_id")
    if value is None:
        value = request.headers.get("X-Profile-Id")
    if value in (None, ""):
        if required:
            raise ValueError("profile_id is required")
        return None
    try:
        profile_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("profile_id must be an integer") from exc
    if not storage.get_profile(profile_id, _db()):
        raise LookupError("Profile not found")
    return profile_id


def _error(message: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": message, **extra}), status


def _ok(**payload):
    return jsonify({"ok": True, **payload})


def _secrets_path() -> Path:
    return _db().parent / "secrets.json"


def _load_secrets() -> dict:
    path = _secrets_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _save_secrets(data: dict) -> None:
    path = _secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return str(value)


def _public_question(question: dict) -> dict:
    return {
        "id": question["id"],
        "question_key": question["question_key"],
        "type": question["type"],
        "difficulty": question["difficulty"],
        "prompt": question["prompt"],
        "options": question.get("options", []),
        "hint": question.get("hint", ""),
        "source_ref": question.get("source_ref", ""),
        "topic_id": question.get("topic_id"),
    }


def _classifier_features(profile_id: int, topic_id: int) -> dict:
    attempts = [a for a in storage.list_attempts(profile_id, topic_id, _db()) if a["verified"]]
    # One attempt per distinct question: re-answering the same question is not
    # new evidence about the learner's level.
    evidence = study_tools.evidence_attempts(attempts)
    distinct_questions = study_tools.distinct_question_count(evidence)
    if distinct_questions < 3:
        return {"unknown": True, "distinct_attempts": len(evidence),
                "distinct_questions": distinct_questions}
    ratios = [a["marks_awarded"] / max(a["max_marks"], 1) for a in evidence]
    recent = ratios[-3:]
    earlier = ratios[:-3]
    mean_accuracy = sum(ratios) / len(ratios)
    recent_accuracy = sum(recent) / len(recent)
    earlier_accuracy = sum(earlier) / len(earlier) if earlier else mean_accuracy
    trend = recent_accuracy - earlier_accuracy
    hint_rate = sum(a["hint_used"] for a in evidence) / len(evidence)
    independent_rate = sum(a["independent"] for a in evidence) / len(evidence)
    difficulty_values = {"easy": 1, "medium": 2, "hard": 3}
    difficulty = 0.0
    active = 0
    repeated_errors = 0
    seen_errors = set()
    for attempt in evidence:
        difficulty += difficulty_values.get(str(attempt.get("difficulty") or "medium"), 2)
        active += attempt["active_seconds"]
        if attempt["error_type"]:
            if attempt["error_type"] in seen_errors:
                repeated_errors += 1
            seen_errors.add(attempt["error_type"])
    return {
        "unknown": False,
        "mean_accuracy": round(mean_accuracy, 4),
        "recent_accuracy": round(recent_accuracy, 4),
        "accuracy_trend": round(trend, 4),
        "hint_rate": round(hint_rate, 4),
        "independent_rate": round(independent_rate, 4),
        "mean_difficulty": round(difficulty / len(evidence), 3),
        "mean_active_seconds": round(active / len(evidence), 1),
        "distinct_attempts": len(evidence),
        "distinct_questions": distinct_questions,
        "repeated_errors": repeated_errors,
    }


def _help_response(message: str, page: str) -> dict:
    text = (message or "").lower()
    if "folder" in text or "file" in text and "where" in text:
        return {
            "answer": "Open Resources, then choose the Files or Folders tab. Uploaded files keep their folder paths and topic groups.",
            "action": {"type": "navigate", "page": "resources", "tab": "folders"},
        }
    if "theme" in text or "dark" in text or "contrast" in text:
        return {
            "answer": "Open Settings, then choose a theme or contrast option. Your choice is saved to this profile.",
            "action": {"type": "navigate", "page": "settings", "focus": "appearance"},
        }
    if "avatar" in text or "equip" in text or "custom" in text:
        return {
            "answer": "Open Store, choose a student or AI avatar item, then equip it. Items are bought with earned coins only.",
            "action": {"type": "navigate", "page": "store", "focus": "avatars"},
        }
    if "xp" in text or "coin" in text or "reward" in text:
        return {
            "answer": "Earn XP and coins by completing lessons, checked practice and mastery. Repeated easy answers are not rewarded again. See Progress for totals.",
            "action": {"type": "navigate", "page": "progress", "focus": "rewards"},
        }
    if "new chat" in text or "start a chat" in text or "start chat" in text or "chat" in text:
        return {
            "answer": "Open Study and use the message box, or type a prompt on Home. Your prompt and attachments transfer to a new topic-named session.",
            "action": {"type": "navigate", "page": "study", "action": "new-chat"},
        }
    if "plan" in text or "timetable" in text or "schedule" in text:
        return {
            "answer": "In Study, open the plan panel and choose Generate plan. Review the before/after summary, then Accept or Reject. Previous versions can be restored.",
            "action": {"type": "navigate", "page": "study", "focus": "plans"},
        }
    if "upload" in text or "import" in text:
        return {
            "answer": "Open Resources and use Upload files or Upload folder. Individual files are limited to 20 MB; folders to 50 files and about 200 MB.",
            "action": {"type": "navigate", "page": "resources", "action": "upload"},
        }
    if "api" in text or "key" in text or "password" in text:
        return {
            "answer": "Open Settings → AI mode. Local mode is the default and needs no key. External API mode is optional and asks permission before content leaves the device.",
            "action": {"type": "navigate", "page": "settings", "focus": "ai-mode"},
        }
    return {
        "answer": "I can help with folders, themes, avatars, XP, new chats, plans, uploads and settings. Tell me which one you need, or open the page from the sidebar.",
        "action": None,
    }


def _external_revision(settings: dict, api_key: str, question: str,
                       sources: list[dict], personality: str) -> dict:
    """Call an explicitly configured external OpenAI-compatible endpoint.

    Only reached when the profile selected API mode *and* ticked per-message
    permission. The API key never appears in logs or responses.
    """
    base = str(settings.get("api_base", "")).strip().rstrip("/")
    model_name = str(settings.get("api_model", "")).strip()
    if not base or not model_name:
        raise ValueError("External API mode needs a base URL and model name in Settings.")
    if not api_key:
        raise ValueError("External API mode needs a saved API key in Settings.")
    prompt = strip_chat_markers(build_revision_prompt(question, sources, personality))
    response = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are Rewiseorist, a source-grounded revision tutor."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 320,
        },
        timeout=90,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"External API returned HTTP {response.status_code}.")
    payload = response.json()
    text = clean_teacher_answer(payload["choices"][0]["message"]["content"])
    picked = []
    for source in sources[:4]:
        label = source.get("original_name") or "source"
        ref = source.get("source_ref") or source.get("page_ref") or "source"
        picked.append({"original_name": label, "source_ref": ref, "text": source.get("text", "")})
    if not extract_citation(text) and picked:
        text = f"{text}\nSource: {picked[0]['original_name']} | {picked[0]['source_ref']}".strip()
    return {
        "text": text,
        "sources": picked or sources,
        "model_version": f"external:{model_name}",
        "verified": False,
        "model": {"name": model_name, "version": f"external:{model_name}", "local": False},
        "external": True,
    }


def create_app(test_config: dict | None = None, runtime_override: ModelRuntime | None = None,
               ollama_override: OllamaClient | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config.update(
        SECRET_KEY=os.environ.get("REWISEORIST_SECRET", secrets.token_hex(24)),
        DATABASE=str(storage.DEFAULT_DB),
        MAX_CONTENT_LENGTH=220 * 1024 * 1024,
        JSON_SORT_KEYS=False,
        START_INACTIVITY_THREAD=True,
        # Revision chat: pretrained SmolLM2 360M through Ollama's local API.
        OLLAMA_BASE_URL="http://127.0.0.1:11434",
        OLLAMA_MODEL=OLLAMA_MODEL_NAME,
        OLLAMA_NUM_CTX=2048,
        OLLAMA_MAX_TOKENS=200,
        OLLAMA_TEMPERATURE=0.2,
        OLLAMA_READ_TIMEOUT=90.0,
    )
    if test_config:
        app.config.update(test_config)
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    storage.init_db(app.config["DATABASE"])
    runtime = runtime_override or ModelRuntime(ROOT / "models")
    ollama = ollama_override or OllamaClient.from_config(app.config)
    manager = InactivityManager(runtime, app.config["DATABASE"], app.config.get("INACTIVITY_THRESHOLDS"))
    app.extensions["runtime"] = runtime
    app.extensions["ollama"] = ollama
    app.extensions["inactivity"] = manager
    app.config["APP_VERSION"] = APP_VERSION
    if app.config.get("START_INACTIVITY_THREAD") and not app.config.get("TESTING"):
        manager.start()

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(ROOT / "frontend", "index.html")

    @app.get("/stylingscripts/<path:filename>")
    def styling(filename: str):
        return send_from_directory(ROOT / "stylingscripts", filename)

    @app.get("/frontend/<path:filename>")
    def frontend_file(filename: str):
        return send_from_directory(ROOT / "frontend", filename)

    @app.get("/samples/<path:filename>")
    def sample_file(filename: str):
        return send_from_directory(ROOT / "samples", filename)

    # ------------------------------------------------------------------
    # Health, profiles and settings
    # ------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return _ok(app=APP_VERSION, database=str(_db()), runtime=_runtime().status(),
                   ollama=_ollama().status(), inactivity=_inactivity().thresholds)

    @app.get("/api/profiles")
    def profiles():
        return _ok(profiles=storage.list_profiles(_db()))

    @app.post("/api/profiles")
    def create_profile():
        data = _body()
        name = str(data.get("name", "")).strip()
        if not name:
            return _error("A profile name is required.")
        if len(name) > 60:
            return _error("Profile names must be 60 characters or fewer.")
        profile = storage.get_or_create_profile(name, _db())
        return _ok(profile=profile)

    @app.get("/api/profiles/<int:profile_id>")
    def get_profile(profile_id: int):
        profile = storage.get_profile(profile_id, _db())
        if not profile:
            return _error("Profile not found", 404)
        return _ok(profile=profile, settings=storage.get_settings(profile_id, _db()))

    @app.get("/api/settings")
    def get_settings():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        return _ok(settings=storage.get_settings(profile_id, _db()),
                   secrets={"api_key_present": bool(_load_secrets().get(str(profile_id), {}).get("api_key"))})

    @app.post("/api/settings")
    def save_settings():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        allowed = {k: data[k] for k in (
            "personality", "theme", "contrast", "ai_mode", "api_base",
            "api_model", "web_permission_default",
        ) if k in data}
        if "personality" in allowed and allowed["personality"] not in study_tools.PERSONALITIES:
            return _error("Personality must be Friendly, Formal or Concise.")
        if "ai_mode" in allowed and allowed["ai_mode"] not in {"local", "api"}:
            return _error("AI mode must be local or api.")
        settings = storage.save_settings(profile_id, _db(), **allowed)
        return _ok(settings=settings)

    @app.post("/api/settings/api-key")
    def save_api_key():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        key = str(data.get("api_key", "")).strip()
        secrets_data = _load_secrets()
        entry = secrets_data.setdefault(str(profile_id), {})
        if key:
            entry["api_key"] = key
        else:
            entry.pop("api_key", None)
        _save_secrets(secrets_data)
        storage.save_settings(profile_id, _db(), api_key_present=bool(key))
        return _ok(api_key_present=bool(key))

    # ------------------------------------------------------------------
    # Subjects, topics, deadlines
    # ------------------------------------------------------------------
    @app.get("/api/subjects")
    def subjects():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        return _ok(subjects=storage.list_subjects(profile_id, _db()))

    @app.post("/api/subjects")
    def create_subject():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        name = str(_body().get("name", "")).strip()
        if not name:
            return _error("Subject name is required.")
        return _ok(subject=storage.add_subject(profile_id, name, _db()))

    @app.get("/api/topics")
    def topics():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        return _ok(topics=storage.list_topics(profile_id, _db()))

    @app.post("/api/topics")
    def create_topic():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        name = str(data.get("name", "")).strip()
        if not name:
            return _error("Topic name is required.")
        subject_id = data.get("subject_id")
        topic = storage.add_topic(
            profile_id, name, int(subject_id) if subject_id not in (None, "") else None,
            str(data.get("difficulty", "medium")), _db(),
        )
        return _ok(topic=topic)

    @app.get("/api/deadlines")
    def deadlines():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        return _ok(deadlines=storage.list_deadlines(profile_id, _db()))

    @app.post("/api/deadlines")
    def create_deadline():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        if not data.get("title") or not data.get("due_date"):
            return _error("Deadline title and due date are required.")
        topic_id = data.get("topic_id")
        return _ok(deadline=storage.add_deadline(
            profile_id, str(data["title"]), str(data["due_date"]),
            int(topic_id) if topic_id not in (None, "") else None,
            int(data.get("marks", 0) or 0), str(data.get("notes", "")), _db(),
        ))

    # ------------------------------------------------------------------
    # Sessions and chat
    # ------------------------------------------------------------------
    @app.get("/api/sessions")
    def sessions():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        limit = min(int(request.args.get("limit", 100)), 250)
        return _ok(sessions=storage.list_sessions(profile_id, limit, _db()))

    @app.post("/api/sessions")
    def create_session():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        title = str(data.get("title", "")).strip() or "New study session"
        topic_id = data.get("topic_id")
        session = storage.create_session(
            profile_id, title, int(topic_id) if topic_id not in (None, "") else None, _db()
        )
        return _ok(session=session)

    @app.get("/api/sessions/<int:session_id>")
    def get_session(session_id: int):
        session = storage.get_session(session_id, _db())
        if not session:
            return _error("Session not found", 404)
        return _ok(session=session, messages=storage.list_messages(session_id, _db()))

    @app.post("/api/sessions/<int:session_id>/draft")
    def save_draft(session_id: int):
        data = _body()
        session = storage.update_session(session_id, _db(), draft=str(data.get("draft", "")))
        return _ok(session=session)

    @app.post("/api/chat")
    def chat():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        message = str(data.get("message", "")).strip()
        if not message:
            return _error("Type a question before sending.")
        topic_id = data.get("topic_id")
        topic_id = int(topic_id) if topic_id not in (None, "") else None
        topic = next((t for t in storage.list_topics(profile_id, _db()) if t["id"] == topic_id), None)
        session_id = data.get("session_id")
        if session_id in (None, ""):
            title = (topic["name"] if topic else message[:60]).strip() or "Study session"
            session = storage.create_session(profile_id, title, topic_id, _db())
            session_id = session["id"]
        else:
            session_id = int(session_id)
            session = storage.get_session(session_id, _db())
            if not session:
                return _error("Session not found", 404)
        settings = storage.get_settings(profile_id, _db())
        personality = settings.get("personality", "Friendly")
        attachments = data.get("attachments", [])
        storage.add_message(session_id, "user", message, _db(), personality=personality,
                            attachments=attachments, verified=True)
        chunks = storage.chunks_for_topic(profile_id, topic_id, _db())
        sources = resources.retrieve(message, chunks, top_k=5)
        if settings.get("ai_mode") == "api":
            if not bool(data.get("api_permission")):
                return _error(
                    "External API mode needs per-message permission. Tick the permission box "
                    "or switch back to local mode in Settings.",
                    403, sources=sources,
                )
            api_key = _load_secrets().get(str(profile_id), {}).get("api_key", "")
            try:
                result = _external_revision(settings, api_key, message, sources, personality)
            except (ValueError, RuntimeError, KeyError, requests.RequestException) as exc:
                return _error(f"The external API could not answer: {exc}", 503, sources=sources)
        else:
            try:
                result = _ollama().generate_revision(message, sources, personality)
            except OllamaModelMissing as exc:
                return _error(str(exc), 503, sources=sources)
            except OllamaUnavailable as exc:
                return _error(str(exc), 503, sources=sources)
            except OllamaTimeout as exc:
                return _error(str(exc), 504, sources=sources)
            except OllamaEmptyAnswer as exc:
                return _error(str(exc), 502, sources=sources)
            except OllamaHTTPError as exc:
                return _error(str(exc), 502, sources=sources)
            except OllamaError as exc:
                return _error(str(exc), 503, sources=sources)
            except Exception as exc:
                return _error(f"Revision chat could not answer: {exc}", 503, sources=sources)
        answer = str(result.get("text", "")).strip()
        if not answer:
            return _error("The revision model returned an empty answer. Try rephrasing.", 502, sources=sources)
        assistant = storage.add_message(
            session_id, "assistant", answer, _db(), personality=personality,
            model_version=str(result.get("model_version", "")),
            sources=result.get("sources", sources), verified=bool(result.get("verified", True)),
        )
        if topic:
            storage.update_session(session_id, _db(), title=session["title"])
            storage.update_topic(topic_id, _db(), last_reviewed_at=storage.utc_now())
            resources.group_chat_logs(profile_id, topic_id, topic["name"], _db().parent, _db())
        _inactivity().touch(profile_id)
        return _ok(session=storage.get_session(session_id, _db()),
                   session_id=session_id, message=assistant, sources=sources,
                   model=result.get("model", {}), fallback=bool(result.get("fallback", False)))

    # ------------------------------------------------------------------
    # Resources and plugins
    # ------------------------------------------------------------------
    @app.get("/api/resources")
    def list_resources():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        items = storage.list_resources(profile_id, _db())
        for item in items:
            item["status_label"] = resources.resource_status(item)
        return _ok(resources=items, plugins=storage.list_plugins(_db()))

    @app.post("/api/resources/upload")
    def upload_resources():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        files = request.files.getlist("files")
        if not files:
            return _error("Choose at least one file or folder.")
        topic_name = str(request.form.get("topic_name", "")).strip() or "General"
        topic_id = request.form.get("topic_id")
        topic = None
        if topic_id not in (None, ""):
            topic = next((t for t in storage.list_topics(profile_id, _db())
                          if t["id"] == int(topic_id)), None)
            topic_id = int(topic_id)
        if not topic:
            topic = storage.add_topic(profile_id, topic_name, db_path=_db())
            topic_id = topic["id"]
            topic_name = topic["name"]
        pairs = []
        relative_paths = request.form.getlist("relative_paths")
        for index, file in enumerate(files):
            if not file.filename:
                continue
            data = file.read()
            name = relative_paths[index] if index < len(relative_paths) else file.filename
            pairs.append((name, data))
        if not pairs:
            return _error("No readable files were selected.")
        try:
            imported = resources.import_files(
                profile_id, topic_id, topic_name, pairs, _db().parent, _db(),
                source_kind="file" if len(pairs) == 1 else "folder",
            )
        except resources.ResourceError as exc:
            return _error(str(exc), 413 if "limit" in str(exc).lower() else 400)
        total = sum(item["size_bytes"] for item in imported)
        storage.update_topic(topic_id, _db(), content_length=total)
        _inactivity().touch(profile_id)
        return _ok(resources=imported, topic=topic, count=len(imported))

    @app.get("/api/plugins")
    def plugins():
        return _ok(plugins=storage.list_plugins(_db()), ocr_available=resources.ocr_available())

    @app.post("/api/plugins/<plugin_key>")
    def toggle_plugin(plugin_key: str):
        if plugin_key not in {"ocr", "web-research", "cloud-import"}:
            return _error("Unknown plugin", 404)
        data = _body()
        enabled = bool(data.get("enabled"))
        status = str(data.get("status", ""))
        if plugin_key == "ocr" and enabled and not resources.ocr_available():
            return _error("Tesseract OCR is not installed. Install it before enabling OCR.")
        if plugin_key == "cloud-import" and enabled:
            return _error("No cloud credentials are configured on this laptop.")
        if not status:
            status = "Enabled" if enabled else "Disabled"
        return _ok(plugin=storage.set_plugin(plugin_key, enabled, status, _db()))

    @app.post("/api/research")
    def research():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        if not data.get("permission"):
            return _error("Web research needs explicit permission for this request.", 403)
        url = str(data.get("url", "")).strip()
        if not url:
            return _error("A URL is required.")
        try:
            page = resources.fetch_readable(url)
        except Exception as exc:
            return _error(f"Could not read that page: {exc}", 502)
        source = storage.add_web_source(
            profile_id, page["url"], page["title"], page["text"], _db(),
            int(data["topic_id"]) if data.get("topic_id") not in (None, "") else None,
        )
        return _ok(source=source)

    # ------------------------------------------------------------------
    # Quizzes and classification
    # ------------------------------------------------------------------
    @app.get("/api/quiz")
    def quiz():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        topic_id = request.args.get("topic_id")
        if topic_id in (None, ""):
            return _error("topic_id is required.")
        try:
            questions = study_tools.ensure_quiz(
                profile_id, int(topic_id), _db(), int(request.args.get("count", 5))
            )
        except ValueError as exc:
            return _error(str(exc), 404)
        return _ok(questions=[_public_question(q) for q in questions])

    @app.post("/api/quiz/answer")
    def answer_quiz():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        question_id = data.get("question_id")
        if question_id in (None, ""):
            return _error("question_id is required.")
        question = storage.get_question(int(question_id), _db())
        if not question:
            return _error("Question not found", 404)
        result = study_tools.submit_answer(
            profile_id, question, str(data.get("answer", "")), _db(),
            session_id=int(data["session_id"]) if data.get("session_id") not in (None, "") else None,
            hint_used=bool(data.get("hint_used", False)),
            independent=not bool(data.get("hint_used", False)),
            active_seconds=int(data.get("active_seconds", 0) or 0),
        )
        attempts = [a for a in storage.list_attempts(profile_id, question["topic_id"], _db()) if a["verified"]]
        # A lesson is complete once three distinct checked questions have been
        # attempted; retrying one question does not count as three pieces of work.
        if study_tools.distinct_question_count(attempts) >= 3:
            study_tools.award_lesson_completion(profile_id, question["topic_id"], _db())
        _inactivity().touch(profile_id)
        return _ok(result=result, question=_public_question(question),
                   model_answer=question["answer"], explanation=question.get("explanation", ""),
                   attempts=len(attempts))

    @app.get("/api/quiz/improvement")
    def improvement():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        topic_id = request.args.get("topic_id")
        if topic_id in (None, ""):
            return _error("topic_id is required.")
        return _ok(improvement=study_tools.improvement_steps(profile_id, int(topic_id), _db()))

    @app.post("/api/classify")
    def classify():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        topic_id = data.get("topic_id")
        if topic_id in (None, ""):
            return _error("topic_id is required.")
        features = _classifier_features(profile_id, int(topic_id))
        if features.get("unknown"):
            return _ok(category="Unknown", reason=(
                "Fewer than three distinct checked questions have been attempted. "
                "Complete more comparable practice before estimating a category."
            ), features=features, model_version="rule:insufficient-evidence")
        try:
            result = _runtime().classify_performance(features)
        except ModelUnavailable as exc:
            if "missing" not in str(exc).lower():
                return _error(str(exc), 503, features=features)
            # Trained classifier files are absent in this delivery: fall back to
            # the transparent rule baseline and label it honestly.
            from ml.train_character import rule_label
            category = rule_label(features)
            reason = (
                f"Rule-based fallback (trained classifier files are absent): "
                f"mean accuracy {float(features.get('mean_accuracy', 0)):.0%}, "
                f"hint rate {float(features.get('hint_rate', 0)):.0%}, "
                f"{features.get('distinct_questions', 0)} distinct checked questions. "
                f"This is the documented baseline, not the trained model."
            )
            payload = {**features, "topic_id": int(topic_id)}
            return _ok(category=category, reason=reason, features=payload,
                       model_version="rule-fallback-v1", fallback=True, confidence=None)
        except Exception as exc:
            return _error(f"The performance model could not classify this history: {exc}", 503,
                          features=features)
        _inactivity().touch(profile_id)
        payload = {**result, "features": features, "topic_id": int(topic_id)}
        return _ok(**payload)

    # ------------------------------------------------------------------
    # Plans and progress
    # ------------------------------------------------------------------
    @app.post("/api/plans/propose")
    def propose_plan():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        available = data.get("available")
        if available is not None:
            try:
                available = [(str(item[0]), int(item[1])) for item in available]
            except (TypeError, ValueError, IndexError):
                return _error("available must be a list of [date, minutes] pairs.")
        plan = study_tools.propose_plan(
            profile_id, available, _db(),
            int(data.get("horizon_days", 7) or 7),
            int(data.get("daily_minutes", 45) or 45),
            str(data.get("title", "")).strip() or None,
        )
        title = str(data.get("title", "")).strip() or "Revision plan"
        stored = storage.create_plan(profile_id, title, plan, _db(), reason="Initial proposal")
        stored["plan"] = plan
        return _ok(plan=plan, stored=stored)

    @app.get("/api/plans")
    def plans():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        return _ok(plans=storage.list_plans(profile_id, _db()))

    @app.get("/api/plans/<int:plan_id>")
    def get_plan(plan_id: int):
        plan = storage.get_plan(plan_id, _db())
        if not plan:
            return _error("Plan not found", 404)
        return _ok(plan=plan)

    @app.post("/api/plans/<int:plan_id>/accept")
    def accept_plan(plan_id: int):
        data = _body()
        version = int(data.get("version", 0) or 0)
        plan = storage.get_plan(plan_id, _db())
        if not plan:
            return _error("Plan not found", 404)
        if not version:
            version = plan["versions"][-1]["version"]
        accepted = storage.accept_plan(plan_id, version, _db())
        return _ok(plan=accepted)

    @app.post("/api/plans/<int:plan_id>/reject")
    def reject_plan(plan_id: int):
        plan = storage.get_plan(plan_id, _db())
        if not plan:
            return _error("Plan not found", 404)
        storage.log_activity(plan["profile_id"], "plan_rejected", str(plan_id), _db())
        return _ok(plan=plan, message="Plan rejected; the previous accepted plan is unchanged.")

    @app.post("/api/plans/<int:plan_id>/restore")
    def restore_plan(plan_id: int):
        data = _body()
        if not data.get("confirm"):
            return _error("Confirm restoration before changing the current plan.")
        version = int(data.get("version", 0) or 0)
        if not version:
            return _error("A plan version is required.")
        try:
            plan = storage.restore_plan_version(plan_id, version, _db())
        except ValueError as exc:
            return _error(str(exc), 404)
        storage.accept_plan(plan_id, plan["versions"][-1]["version"], _db())
        return _ok(plan=storage.get_plan(plan_id, _db()))

    @app.get("/api/plans/<int:plan_id>/download")
    def download_plan(plan_id: int):
        plan = storage.get_plan(plan_id, _db())
        if not plan:
            return _error("Plan not found", 404)
        version = request.args.get("version")
        selected = None
        if version:
            selected = next((v for v in plan["versions"] if v["version"] == int(version)), None)
        if not selected:
            selected = next((v for v in plan["versions"] if v["accepted"]), plan["versions"][-1])
        text = selected.get("after_md") or selected["plan"].get("markdown", "")
        path = _db().parent / "exports" / f"plan-{plan_id}-v{selected['version']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.get("/api/progress")
    def progress():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        period = request.args.get("period", "all")
        if period not in {"all", "daily", "weekly", "monthly"}:
            return _error("period must be all, daily, weekly or monthly.")
        topic_id = request.args.get("topic_id")
        return _ok(progress=study_tools.progress_summary(
            profile_id, period,
            int(topic_id) if topic_id not in (None, "") else None, _db(),
        ))

    # ------------------------------------------------------------------
    # Store, notes and help
    # ------------------------------------------------------------------
    @app.get("/api/store")
    def store():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        return _ok(items=storage.list_store_items(_db()), balance=storage.get_balance(profile_id, _db()))

    @app.post("/api/store/buy")
    def buy_item():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        item_key = str(_body().get("item_key", "")).strip()
        if not item_key:
            return _error("item_key is required.")
        try:
            result = storage.purchase_item(profile_id, item_key, _db())
        except ValueError as exc:
            return _error(str(exc), 400)
        return _ok(**result, balance=storage.get_balance(profile_id, _db()))

    @app.post("/api/store/equip")
    def equip_item():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        item_key = str(_body().get("item_key", "")).strip()
        try:
            result = storage.equip_item(profile_id, item_key, _db())
        except ValueError as exc:
            return _error(str(exc), 400)
        return _ok(**result, balance=storage.get_balance(profile_id, _db()))

    @app.get("/api/notes")
    def notes():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        return _ok(notes=storage.list_notes(profile_id, _db()))

    @app.post("/api/notes")
    def save_note():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        target_type = str(data.get("target_type", "resource"))
        target_id = str(data.get("target_id", "general"))
        image_data = data.get("image_data")
        image_path = ""
        if image_data:
            try:
                header, encoded = str(image_data).split(",", 1)
                if "image/png" not in header:
                    raise ValueError("PNG image data expected")
                raw = base64.b64decode(encoded)
                path = _db().parent / "notes" / str(profile_id) / f"{target_type}-{target_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                image_path = str(path)
            except Exception:
                return _error("The drawing image could not be saved.")
        note = storage.save_note(
            profile_id, target_type, target_id, _db(),
            text=str(data.get("text", "")), strokes=data.get("strokes", []),
            image_path=image_path,
        )
        return _ok(note=note)

    @app.post("/api/help")
    def help_assistant():
        try:
            _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        return _ok(**_help_response(str(data.get("message", "")), str(data.get("page", ""))))

    # ------------------------------------------------------------------
    # Models, inactivity, backup
    # ------------------------------------------------------------------
    @app.get("/api/models/status")
    def model_status():
        return _ok(runtime=_runtime().status(), ollama=_ollama().status())

    @app.post("/api/models/warmup")
    def model_warmup():
        results: dict[str, Any] = {}
        try:
            results["ollama"] = _ollama().warmup()
        except OllamaError as exc:
            results["ollama"] = {"warm": False, "error": str(exc)}
        try:
            results["classifier"] = _runtime().warmup()
        except ModelUnavailable as exc:
            results["classifier"] = {
                "warm": False, "error": str(exc),
                "note": "Trained classifier files are absent; the rule-based fallback is used.",
            }
        return _ok(result=results, runtime=_runtime().status(), ollama=_ollama().status())

    @app.post("/api/models/unload")
    def model_unload():
        results: dict[str, Any] = {}
        try:
            results["ollama"] = _ollama().unload()
        except OllamaError as exc:
            results["ollama"] = {"unloaded": False, "error": str(exc)}
        try:
            results["classifier"] = _runtime().unload()
        except ModelUnavailable as exc:
            results["classifier"] = {"unloaded": True, "error": str(exc)}
        return _ok(result=results, runtime=_runtime().status(), ollama=_ollama().status())

    @app.post("/api/activity")
    def activity():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        return _ok(state=_inactivity().touch(profile_id))

    @app.get("/api/inactivity")
    def inactivity():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        return _ok(**_inactivity().check(profile_id))

    @app.post("/api/inactivity/simulate")
    def simulate_inactivity():
        if not (current_app.config.get("TESTING") or os.environ.get("FLASK_TESTING") == "1"):
            return _error("This test hook is disabled.", 404)
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        seconds = int(_body().get("seconds_ago", 0) or 0)
        with storage.db(_db()) as conn:
            conn.execute(
                "UPDATE inactivity_state SET last_real_activity = ?, warmup_stopped_at=NULL, "
                "worker_stopped_at=NULL, media_stop_at=NULL WHERE profile_id = ?",
                ((_now() - timedelta(seconds=seconds)).isoformat(), profile_id),
            )
        return _ok(**_inactivity().check(profile_id))

    @app.get("/api/backup")
    def backup():
        archive = storage.create_backup(_db(), _db().parent / "backups")
        return send_file(archive, as_attachment=True, download_name=archive.name)

    @app.post("/api/api/test")
    def test_api():
        try:
            profile_id = _profile_id()
        except (ValueError, LookupError) as exc:
            return _error(str(exc), 400 if isinstance(exc, ValueError) else 404)
        data = _body()
        if not data.get("permission"):
            return _error("External API checks need explicit permission.", 403)
        base = str(data.get("api_base", "")).strip().rstrip("/")
        if not base:
            return _error("An API base URL is required.")
        key = str(data.get("api_key", "")).strip() or _load_secrets().get(str(profile_id), {}).get("api_key", "")
        if not key:
            return _error("No API key is configured.")
        import requests
        try:
            response = requests.get(
                f"{base}/models", timeout=12,
                headers={"Authorization": f"Bearer {key}"},
            )
            return _ok(status_code=response.status_code, reachable=response.ok,
                       message="API endpoint reachable" if response.ok else response.text[:200])
        except Exception as exc:
            return _error(f"API endpoint could not be reached: {exc}", 502)

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return _error("The upload is too large. Individual files are limited to 20 MB and folders to about 200 MB.", 413)

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return _error("API route not found", 404)
        return send_from_directory(ROOT / "frontend", "index.html")

    @app.errorhandler(500)
    def server_error(error):
        current_app.logger.exception("Unhandled error: %s", error)
        return _error("The server hit an unexpected error. Your saved work is still in the database.", 500)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
