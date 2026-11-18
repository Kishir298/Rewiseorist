"""SQLite persistence for Rewiseorist.

The module deliberately keeps the schema and queries readable: it is a local,
single-user coursework application, not a distributed service.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "rewiseorist.db"
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json(value: Any, default: Any = None) -> str:
    return json.dumps(value if value is not None else default, ensure_ascii=False)


def _loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    xp INTEGER NOT NULL DEFAULT 0,
    coins INTEGER NOT NULL DEFAULT 0,
    avatar_item TEXT NOT NULL DEFAULT 'student-default'
);

CREATE TABLE IF NOT EXISTS settings (
    profile_id INTEGER PRIMARY KEY REFERENCES profiles(id) ON DELETE CASCADE,
    personality TEXT NOT NULL DEFAULT 'Friendly',
    theme TEXT NOT NULL DEFAULT 'light',
    contrast TEXT NOT NULL DEFAULT 'normal',
    ai_mode TEXT NOT NULL DEFAULT 'local',
    api_base TEXT NOT NULL DEFAULT '',
    api_model TEXT NOT NULL DEFAULT '',
    api_key_present INTEGER NOT NULL DEFAULT 0,
    web_permission_default INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(profile_id, name)
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    subject_id INTEGER REFERENCES subjects(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    difficulty TEXT NOT NULL DEFAULT 'medium',
    content_length INTEGER NOT NULL DEFAULT 0,
    estimated_level TEXT NOT NULL DEFAULT 'Unknown',
    last_reviewed_at TEXT,
    next_review_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(profile_id, name)
);

CREATE TABLE IF NOT EXISTS deadlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    due_date TEXT NOT NULL,
    marks INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    original_name TEXT NOT NULL,
    relative_path TEXT NOT NULL DEFAULT '',
    stored_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL DEFAULT '',
    source_kind TEXT NOT NULL DEFAULT 'file',
    plugin TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    error TEXT NOT NULL DEFAULT '',
    processed_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resource_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_ref TEXT NOT NULL DEFAULT '',
    location_ref TEXT NOT NULL DEFAULT '',
    source_ref TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    draft TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    personality TEXT NOT NULL DEFAULT 'Friendly',
    model_version TEXT NOT NULL DEFAULT '',
    attachments_json TEXT NOT NULL DEFAULT '[]',
    sources_json TEXT NOT NULL DEFAULT '[]',
    verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    question_key TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'short',
    difficulty TEXT NOT NULL DEFAULT 'medium',
    prompt TEXT NOT NULL,
    options_json TEXT NOT NULL DEFAULT '[]',
    answer TEXT NOT NULL,
    accepted_keywords_json TEXT NOT NULL DEFAULT '[]',
    hint TEXT NOT NULL DEFAULT '',
    explanation TEXT NOT NULL DEFAULT '',
    source_ref TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'template',
    created_at TEXT NOT NULL,
    UNIQUE(topic_id, question_key)
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    question_id INTEGER REFERENCES questions(id) ON DELETE SET NULL,
    session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    answer TEXT NOT NULL DEFAULT '',
    marks_awarded REAL NOT NULL DEFAULT 0,
    max_marks REAL NOT NULL DEFAULT 1,
    correct INTEGER NOT NULL DEFAULT 0,
    hint_used INTEGER NOT NULL DEFAULT 0,
    independent INTEGER NOT NULL DEFAULT 1,
    error_type TEXT NOT NULL DEFAULT '',
    difficulty TEXT NOT NULL DEFAULT 'medium',
    question_type TEXT NOT NULL DEFAULT 'short',
    active_seconds INTEGER NOT NULL DEFAULT 0,
    attempt_key TEXT NOT NULL UNIQUE,
    verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    before_md TEXT NOT NULL DEFAULT '',
    after_md TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    accepted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(plan_id, version)
);

CREATE TABLE IF NOT EXISTS rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    event_key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    amount INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS badges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    badge_key TEXT NOT NULL,
    name TEXT NOT NULL,
    earned_at TEXT NOT NULL,
    UNIQUE(profile_id, badge_key)
);

CREATE TABLE IF NOT EXISTS store_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    asset TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS owned_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    item_key TEXT NOT NULL,
    purchased_at TEXT NOT NULL,
    equipped INTEGER NOT NULL DEFAULT 0,
    UNIQUE(profile_id, item_key)
);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    strokes_json TEXT NOT NULL DEFAULT '[]',
    image_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(profile_id, target_type, target_id)
);

CREATE TABLE IF NOT EXISTS plugins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plugin_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'not configured',
    config_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS web_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    fetched_at TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    permission TEXT NOT NULL DEFAULT 'explicit'
);

CREATE TABLE IF NOT EXISTS inactivity_state (
    profile_id INTEGER PRIMARY KEY REFERENCES profiles(id) ON DELETE CASCADE,
    last_real_activity TEXT NOT NULL,
    warmup_stopped_at TEXT,
    worker_stopped_at TEXT,
    media_stop_at TEXT,
    pending_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    task TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER REFERENCES profiles(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""

STORE_ITEMS = [
    ("student-default", "Default student", "student_avatar", 0, "The starting student avatar.", "avatar-student"),
    ("student-glasses", "Study glasses", "student_avatar", 30, "A focused student avatar accessory.", "avatar-glasses"),
    ("student-cap", "Graduation cap", "student_avatar", 80, "A cap for consistent revision.", "avatar-cap"),
    ("ai-default", "Default AI tutor", "ai_avatar", 0, "The starting AI tutor avatar.", "avatar-ai"),
    ("ai-owl", "Owl tutor", "ai_avatar", 45, "A wise owl tutor avatar.", "avatar-owl"),
    ("ai-robot", "Robot tutor", "ai_avatar", 90, "A friendly robot tutor avatar.", "avatar-robot"),
    ("theme-midnight", "Midnight theme", "theme", 60, "A dark navy study theme.", "theme-midnight"),
    ("theme-ocean", "Ocean theme", "theme", 70, "A turquoise-tinted study theme.", "theme-ocean"),
    ("background-grid", "Grid study background", "background", 50, "A subtle grid background.", "bg-grid"),
    ("background-paper", "Paper background", "background", 40, "A warm paper background.", "bg-paper"),
]


# ---------------------------------------------------------------------------
# Connection and initialization
# ---------------------------------------------------------------------------

@contextmanager
def db(db_path: str | Path = DEFAULT_DB):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate_attempts(conn) -> None:
    """Add attempt difficulty/type columns to databases created before them.

    Existing rows are backfilled from the question they answered, so the
    performance classifier sees the same evidence shape as new attempts.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(attempts)")}
    if not columns:
        return
    added = False
    if "difficulty" not in columns:
        conn.execute("ALTER TABLE attempts ADD COLUMN difficulty TEXT NOT NULL DEFAULT 'medium'")
        added = True
    if "question_type" not in columns:
        conn.execute("ALTER TABLE attempts ADD COLUMN question_type TEXT NOT NULL DEFAULT 'short'")
        added = True
    if added:
        conn.execute(
            """
            UPDATE attempts
               SET difficulty = COALESCE(
                       (SELECT difficulty FROM questions WHERE questions.id = attempts.question_id),
                       'medium'),
                   question_type = COALESCE(
                       (SELECT type FROM questions WHERE questions.id = attempts.question_id),
                       'short')
             WHERE question_id IS NOT NULL
            """
        )


def init_db(db_path: str | Path = DEFAULT_DB) -> None:
    with db(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate_attempts(conn)
        now = utc_now()
        for item in STORE_ITEMS:
            conn.execute(
                """
                INSERT INTO store_items(item_key, name, category, price, description, asset)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_key) DO UPDATE SET
                    name=excluded.name, category=excluded.category,
                    price=excluded.price, description=excluded.description,
                    asset=excluded.asset
                """,
                item,
            )
        conn.execute(
            """
            INSERT INTO plugins(plugin_key, name, enabled, status, config_json, updated_at)
            VALUES ('ocr', 'Optional OCR', 0, 'Tesseract not detected; text-based files work', '{}', ?)
            ON CONFLICT(plugin_key) DO NOTHING
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO plugins(plugin_key, name, enabled, status, config_json, updated_at)
            VALUES ('web-research', 'Permission-based web research', 0, 'Requires explicit permission per question', '{}', ?)
            ON CONFLICT(plugin_key) DO NOTHING
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO plugins(plugin_key, name, enabled, status, config_json, updated_at)
            VALUES ('cloud-import', 'Cloud import connectors', 0, 'No OneDrive/Google Drive/Teams credentials configured', '{}', ?)
            ON CONFLICT(plugin_key) DO NOTHING
            """,
            (now,),
        )


# ---------------------------------------------------------------------------
# Profiles and settings
# ---------------------------------------------------------------------------

def create_profile(name: str, db_path: str | Path = DEFAULT_DB) -> dict:
    now = utc_now()
    with db(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO profiles(name, created_at, updated_at) VALUES (?, ?, ?)",
            (name.strip(), now, now),
        )
        profile_id = cur.lastrowid
        conn.execute(
            "INSERT INTO settings(profile_id, updated_at) VALUES (?, ?)",
            (profile_id, now),
        )
        conn.execute(
            "INSERT INTO inactivity_state(profile_id, last_real_activity) VALUES (?, ?)",
            (profile_id, now),
        )
    return get_profile(profile_id, db_path)


def list_profiles(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute("SELECT * FROM profiles ORDER BY name").fetchall()
    return [dict(row) for row in rows]


def get_profile(profile_id: int, db_path: str | Path = DEFAULT_DB) -> dict | None:
    with db(db_path) as conn:
        row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    return dict(row) if row else None


def get_profile_by_name(name: str, db_path: str | Path = DEFAULT_DB) -> dict | None:
    with db(db_path) as conn:
        row = conn.execute("SELECT * FROM profiles WHERE name = ?", (name.strip(),)).fetchone()
    return dict(row) if row else None


def get_or_create_profile(name: str, db_path: str | Path = DEFAULT_DB) -> dict:
    return get_profile_by_name(name, db_path) or create_profile(name, db_path)


def update_profile(profile_id: int, db_path: str | Path = DEFAULT_DB, **fields) -> dict:
    allowed = {"name", "avatar_item", "xp", "coins"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_profile(profile_id, db_path)
    sets = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [utc_now(), profile_id]
    with db(db_path) as conn:
        conn.execute(f"UPDATE profiles SET {sets}, updated_at = ? WHERE id = ?", values)
    return get_profile(profile_id, db_path)


def get_settings(profile_id: int, db_path: str | Path = DEFAULT_DB) -> dict:
    with db(db_path) as conn:
        row = conn.execute("SELECT * FROM settings WHERE profile_id = ?", (profile_id,)).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO settings(profile_id, updated_at) VALUES (?, ?)",
                (profile_id, utc_now()),
            )
            row = conn.execute("SELECT * FROM settings WHERE profile_id = ?", (profile_id,)).fetchone()
    return dict(row) if row else {}


def save_settings(profile_id: int, db_path: str | Path = DEFAULT_DB, **fields) -> dict:
    allowed = {
        "personality", "theme", "contrast", "ai_mode", "api_base",
        "api_model", "api_key_present", "web_permission_default",
    }
    current = get_settings(profile_id, db_path)
    data = {k: fields.get(k, current.get(k)) for k in allowed}
    with db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO settings(profile_id, personality, theme, contrast, ai_mode, api_base,
                                 api_model, api_key_present, web_permission_default, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id) DO UPDATE SET
                personality=excluded.personality, theme=excluded.theme,
                contrast=excluded.contrast, ai_mode=excluded.ai_mode,
                api_base=excluded.api_base, api_model=excluded.api_model,
                api_key_present=excluded.api_key_present,
                web_permission_default=excluded.web_permission_default,
                updated_at=excluded.updated_at
            """,
            (
                profile_id, data["personality"], data["theme"], data["contrast"],
                data["ai_mode"], data["api_base"], data["api_model"],
                int(bool(data["api_key_present"])), int(bool(data["web_permission_default"])),
                utc_now(),
            ),
        )
    return get_settings(profile_id, db_path)


# ---------------------------------------------------------------------------
# Subjects, topics, deadlines
# ---------------------------------------------------------------------------

def add_subject(profile_id: int, name: str, db_path: str | Path = DEFAULT_DB) -> dict:
    now = utc_now()
    with db(db_path) as conn:
        conn.execute(
            "INSERT INTO subjects(profile_id, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(profile_id, name) DO NOTHING",
            (profile_id, name.strip(), now),
        )
        row = conn.execute(
            "SELECT * FROM subjects WHERE profile_id = ? AND name = ?",
            (profile_id, name.strip()),
        ).fetchone()
    return dict(row)


def list_subjects(profile_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM subjects WHERE profile_id = ? ORDER BY name", (profile_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def add_topic(profile_id: int, name: str, subject_id: int | None = None,
              difficulty: str = "medium", db_path: str | Path = DEFAULT_DB) -> dict:
    now = utc_now()
    with db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO topics(profile_id, subject_id, name, difficulty, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, name) DO UPDATE SET
                subject_id=COALESCE(excluded.subject_id, topics.subject_id),
                difficulty=excluded.difficulty
            """,
            (profile_id, subject_id, name.strip(), difficulty, now),
        )
        row = conn.execute(
            "SELECT * FROM topics WHERE profile_id = ? AND name = ?",
            (profile_id, name.strip()),
        ).fetchone()
    return dict(row)


def list_topics(profile_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM topics WHERE profile_id = ? ORDER BY name", (profile_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_topic(topic_id: int, db_path: str | Path = DEFAULT_DB, **fields) -> dict:
    allowed = {"name", "difficulty", "content_length", "estimated_level",
               "last_reviewed_at", "next_review_at", "subject_id"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if updates:
        sets = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [topic_id]
        with db(db_path) as conn:
            conn.execute(f"UPDATE topics SET {sets} WHERE id = ?", values)
    with db(db_path) as conn:
        row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
    return dict(row) if row else {}


def add_deadline(profile_id: int, title: str, due_date: str,
                 topic_id: int | None = None, marks: int = 0, notes: str = "",
                 db_path: str | Path = DEFAULT_DB) -> dict:
    with db(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO deadlines(profile_id, topic_id, title, due_date, marks, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (profile_id, topic_id, title, due_date, marks, notes, utc_now()),
        )
        row = conn.execute("SELECT * FROM deadlines WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_deadlines(profile_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM deadlines WHERE profile_id = ? ORDER BY due_date", (profile_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def add_resource(profile_id: int, original_name: str, stored_path: str,
                 file_type: str, size_bytes: int = 0, sha256: str = "",
                 topic_id: int | None = None, relative_path: str = "",
                 source_kind: str = "file", plugin: str = "", status: str = "ready",
                 error: str = "", processed_text: str = "",
                 db_path: str | Path = DEFAULT_DB) -> dict:
    with db(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO resources(profile_id, topic_id, original_name, relative_path,
               stored_path, file_type, size_bytes, sha256, source_kind, plugin, status,
               error, processed_text, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (profile_id, topic_id, original_name, relative_path, stored_path, file_type,
             size_bytes, sha256, source_kind, plugin, status, error, processed_text, utc_now()),
        )
        row = conn.execute("SELECT * FROM resources WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_resources(profile_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM resources WHERE profile_id = ? ORDER BY created_at DESC", (profile_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_resource(resource_id: int, db_path: str | Path = DEFAULT_DB) -> dict | None:
    with db(db_path) as conn:
        row = conn.execute("SELECT * FROM resources WHERE id = ?", (resource_id,)).fetchone()
    return dict(row) if row else None


def add_chunks(resource_id: int, chunks: Iterable[dict], db_path: str | Path = DEFAULT_DB) -> int:
    now = utc_now()
    count = 0
    with db(db_path) as conn:
        for index, chunk in enumerate(chunks):
            conn.execute(
                """INSERT INTO resource_chunks(resource_id, chunk_index, text, page_ref,
                   location_ref, source_ref, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (resource_id, index, chunk["text"], chunk.get("page_ref", ""),
                 chunk.get("location_ref", ""), chunk.get("source_ref", ""), now),
            )
            count += 1
    return count


def list_chunks(profile_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            """SELECT c.*, r.original_name, r.relative_path, r.topic_id, r.file_type
               FROM resource_chunks c JOIN resources r ON r.id = c.resource_id
               WHERE r.profile_id = ? ORDER BY c.resource_id, c.chunk_index""",
            (profile_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def chunks_for_topic(profile_id: int, topic_id: int | None,
                     db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        if topic_id is None:
            rows = conn.execute(
                """SELECT c.*, r.original_name, r.relative_path, r.topic_id, r.file_type
                   FROM resource_chunks c JOIN resources r ON r.id = c.resource_id
                   WHERE r.profile_id = ? ORDER BY c.resource_id, c.chunk_index""",
                (profile_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT c.*, r.original_name, r.relative_path, r.topic_id, r.file_type
                   FROM resource_chunks c JOIN resources r ON r.id = c.resource_id
                   WHERE r.profile_id = ? AND r.topic_id = ?
                   ORDER BY c.resource_id, c.chunk_index""",
                (profile_id, topic_id),
            ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Sessions and messages
# ---------------------------------------------------------------------------

def create_session(profile_id: int, title: str, topic_id: int | None = None,
                   db_path: str | Path = DEFAULT_DB) -> dict:
    now = utc_now()
    with db(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO sessions(profile_id, topic_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (profile_id, topic_id, title.strip(), now, now),
        )
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_sessions(profile_id: int, limit: int = 100,
                  db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            """SELECT s.*, t.name AS topic_name FROM sessions s
               LEFT JOIN topics t ON t.id = s.topic_id
               WHERE s.profile_id = ? ORDER BY s.updated_at DESC LIMIT ?""",
            (profile_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: int, db_path: str | Path = DEFAULT_DB) -> dict | None:
    with db(db_path) as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def update_session(session_id: int, db_path: str | Path = DEFAULT_DB, **fields) -> dict:
    allowed = {"title", "status", "draft", "topic_id"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    updates["updated_at"] = utc_now()
    sets = ", ".join(f"{k} = ?" for k in updates)
    with db(db_path) as conn:
        conn.execute(f"UPDATE sessions SET {sets} WHERE id = ?",
                     list(updates.values()) + [session_id])
    return get_session(session_id, db_path)


def add_message(session_id: int, role: str, content: str, db_path: str | Path = DEFAULT_DB,
                personality: str = "Friendly", model_version: str = "",
                attachments: Any = None, sources: Any = None,
                verified: bool = False) -> dict:
    with db(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO messages(session_id, role, content, personality, model_version,
               attachments_json, sources_json, verified, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, role, content, personality, model_version,
             _json(attachments, []), _json(sources, []), int(verified), utc_now()),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (utc_now(), session_id))
        row = conn.execute("SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)).fetchone()
    result = dict(row)
    result["attachments"] = _loads(result.pop("attachments_json"), [])
    result["sources"] = _loads(result.pop("sources_json"), [])
    return result


def list_messages(session_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    messages = []
    for row in rows:
        item = dict(row)
        item["attachments"] = _loads(item.pop("attachments_json"), [])
        item["sources"] = _loads(item.pop("sources_json"), [])
        messages.append(item)
    return messages


# ---------------------------------------------------------------------------
# Questions and attempts
# ---------------------------------------------------------------------------

def save_question(topic_id: int, question_key: str, prompt: str, answer: str,
                  db_path: str | Path = DEFAULT_DB, qtype: str = "short",
                  difficulty: str = "medium", options: Any = None,
                  accepted_keywords: Any = None, hint: str = "",
                  explanation: str = "", source_ref: str = "",
                  origin: str = "template") -> dict:
    with db(db_path) as conn:
        conn.execute(
            """INSERT INTO questions(topic_id, question_key, type, difficulty, prompt,
               options_json, answer, accepted_keywords_json, hint, explanation,
               source_ref, origin, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(topic_id, question_key) DO UPDATE SET
                 prompt=excluded.prompt, answer=excluded.answer, hint=excluded.hint,
                 explanation=excluded.explanation, source_ref=excluded.source_ref,
                 options_json=excluded.options_json,
                 accepted_keywords_json=excluded.accepted_keywords_json""",
            (topic_id, question_key, qtype, difficulty, prompt, _json(options, []), answer,
             _json(accepted_keywords, []), hint, explanation, source_ref, origin, utc_now()),
        )
        row = conn.execute(
            "SELECT * FROM questions WHERE topic_id = ? AND question_key = ?",
            (topic_id, question_key),
        ).fetchone()
    item = dict(row)
    item["options"] = _loads(item.pop("options_json"), [])
    item["accepted_keywords"] = _loads(item.pop("accepted_keywords_json"), [])
    return item


def list_questions(topic_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM questions WHERE topic_id = ? ORDER BY id", (topic_id,)
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["options"] = _loads(item.pop("options_json"), [])
        item["accepted_keywords"] = _loads(item.pop("accepted_keywords_json"), [])
        result.append(item)
    return result


def get_question(question_id: int, db_path: str | Path = DEFAULT_DB) -> dict | None:
    with db(db_path) as conn:
        row = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["options"] = _loads(item.pop("options_json"), [])
    item["accepted_keywords"] = _loads(item.pop("accepted_keywords_json"), [])
    return item


def record_attempt(profile_id: int, attempt_key: str, question_id: int | None,
                   answer: str, marks_awarded: float, max_marks: float, correct: bool,
                   db_path: str | Path = DEFAULT_DB, topic_id: int | None = None,
                   session_id: int | None = None, hint_used: bool = False,
                   independent: bool = True, error_type: str = "",
                   active_seconds: int = 0, verified: bool = True,
                   difficulty: str = "medium", question_type: str = "short") -> dict:
    with db(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO attempts(profile_id, topic_id, question_id, session_id, answer,
               marks_awarded, max_marks, correct, hint_used, independent, error_type,
               difficulty, question_type, active_seconds, attempt_key, verified, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(attempt_key) DO NOTHING""",
            (profile_id, topic_id, question_id, session_id, answer, marks_awarded,
             max_marks, int(correct), int(hint_used), int(independent), error_type,
             difficulty, question_type, active_seconds, attempt_key, int(verified),
             utc_now()),
        )
        row = conn.execute("SELECT * FROM attempts WHERE attempt_key = ?", (attempt_key,)).fetchone()
    return dict(row) if row else {}


def list_attempts(profile_id: int, topic_id: int | None = None,
                  db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        if topic_id is None:
            rows = conn.execute(
                "SELECT * FROM attempts WHERE profile_id = ? ORDER BY created_at", (profile_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM attempts WHERE profile_id = ? AND topic_id = ? ORDER BY created_at",
                (profile_id, topic_id),
            ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Plans and versions
# ---------------------------------------------------------------------------

def create_plan(profile_id: int, title: str, plan: dict,
                db_path: str | Path = DEFAULT_DB, reason: str = "Initial plan") -> dict:
    now = utc_now()
    with db(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO plans(profile_id, title, status, created_at, updated_at) VALUES (?, ?, 'draft', ?, ?)",
            (profile_id, title, now, now),
        )
        plan_id = cur.lastrowid
        conn.execute(
            """INSERT INTO plan_versions(plan_id, version, plan_json, before_md, after_md,
               reason, accepted, created_at) VALUES (?, 1, ?, '', ?, ?, 0, ?)""",
            (plan_id, _json(plan, {}), plan.get("summary", ""), reason, now),
        )
    return get_plan(plan_id, db_path)


def get_plan(plan_id: int, db_path: str | Path = DEFAULT_DB) -> dict | None:
    with db(db_path) as conn:
        row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
        if not row:
            return None
        versions = conn.execute(
            "SELECT * FROM plan_versions WHERE plan_id = ? ORDER BY version", (plan_id,)
        ).fetchall()
    plan = dict(row)
    plan["versions"] = []
    for version in versions:
        v = dict(version)
        v["plan"] = _loads(v.pop("plan_json"), {})
        plan["versions"].append(v)
    return plan


def list_plans(profile_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM plans WHERE profile_id = ? ORDER BY updated_at DESC", (profile_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def add_plan_version(plan_id: int, plan: dict, db_path: str | Path = DEFAULT_DB,
                     reason: str = "", before_md: str = "", after_md: str = "",
                     accepted: bool = False) -> dict:
    with db(db_path) as conn:
        current = conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM plan_versions WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()["v"]
        conn.execute(
            """INSERT INTO plan_versions(plan_id, version, plan_json, before_md, after_md,
               reason, accepted, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (plan_id, current + 1, _json(plan, {}), before_md, after_md, reason,
             int(accepted), utc_now()),
        )
        conn.execute("UPDATE plans SET updated_at = ? WHERE id = ?", (utc_now(), plan_id))
    return get_plan(plan_id, db_path)


def accept_plan(plan_id: int, version: int, db_path: str | Path = DEFAULT_DB) -> dict:
    with db(db_path) as conn:
        conn.execute("UPDATE plan_versions SET accepted = 0 WHERE plan_id = ?", (plan_id,))
        conn.execute(
            "UPDATE plan_versions SET accepted = 1 WHERE plan_id = ? AND version = ?",
            (plan_id, version),
        )
        conn.execute(
            "UPDATE plans SET status = 'accepted', updated_at = ? WHERE id = ?",
            (utc_now(), plan_id),
        )
    return get_plan(plan_id, db_path)


def restore_plan_version(plan_id: int, version: int,
                         db_path: str | Path = DEFAULT_DB) -> dict:
    plan = get_plan(plan_id, db_path)
    if not plan:
        raise ValueError("Plan not found")
    selected = next((v for v in plan["versions"] if v["version"] == version), None)
    if not selected:
        raise ValueError("Plan version not found")
    current = next((v for v in plan["versions"] if v["accepted"]), None)
    before = current["after_md"] if current else ""
    return add_plan_version(
        plan_id, selected["plan"], db_path,
        reason=f"Restored version {version}",
        before_md=before,
        after_md=selected["after_md"],
        accepted=True,
    )


# ---------------------------------------------------------------------------
# Rewards, store and badges
# ---------------------------------------------------------------------------

def award_reward(profile_id: int, event_key: str, kind: str, amount: int,
                 reason: str, db_path: str | Path = DEFAULT_DB) -> dict:
    now = utc_now()
    with db(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO rewards(profile_id, event_key, kind, amount, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(event_key) DO NOTHING""",
            (profile_id, event_key, kind, amount, reason, now),
        )
        created = cur.rowcount > 0
        if created:
            if kind == "xp":
                conn.execute("UPDATE profiles SET xp = xp + ?, updated_at = ? WHERE id = ?",
                             (amount, now, profile_id))
            elif kind == "coins":
                conn.execute("UPDATE profiles SET coins = coins + ?, updated_at = ? WHERE id = ?",
                             (amount, now, profile_id))
        row = conn.execute("SELECT * FROM rewards WHERE event_key = ?", (event_key,)).fetchone()
    result = dict(row) if row else {}
    result["created"] = created
    return result


def get_balance(profile_id: int, db_path: str | Path = DEFAULT_DB) -> dict:
    profile = get_profile(profile_id, db_path) or {}
    with db(db_path) as conn:
        badges = conn.execute(
            "SELECT * FROM badges WHERE profile_id = ? ORDER BY earned_at", (profile_id,)
        ).fetchall()
        owned = conn.execute(
            "SELECT * FROM owned_items WHERE profile_id = ?", (profile_id,)
        ).fetchall()
    return {
        "xp": profile.get("xp", 0),
        "coins": profile.get("coins", 0),
        "badges": [dict(b) for b in badges],
        "owned": [dict(o) for o in owned],
    }


def award_badge(profile_id: int, badge_key: str, name: str,
                db_path: str | Path = DEFAULT_DB) -> dict:
    with db(db_path) as conn:
        conn.execute(
            "INSERT INTO badges(profile_id, badge_key, name, earned_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(profile_id, badge_key) DO NOTHING",
            (profile_id, badge_key, name, utc_now()),
        )
        row = conn.execute(
            "SELECT * FROM badges WHERE profile_id = ? AND badge_key = ?",
            (profile_id, badge_key),
        ).fetchone()
    return dict(row)


def list_store_items(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute("SELECT * FROM store_items ORDER BY category, price, name").fetchall()
    return [dict(r) for r in rows]


def purchase_item(profile_id: int, item_key: str, db_path: str | Path = DEFAULT_DB) -> dict:
    now = utc_now()
    with db(db_path) as conn:
        item = conn.execute("SELECT * FROM store_items WHERE item_key = ?", (item_key,)).fetchone()
        if not item:
            raise ValueError("Store item not found")
        owned = conn.execute(
            "SELECT * FROM owned_items WHERE profile_id = ? AND item_key = ?",
            (profile_id, item_key),
        ).fetchone()
        if owned:
            return {"item": dict(item), "charged": 0, "owned": True, "equipped": bool(owned["equipped"])}
        profile = conn.execute("SELECT coins FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        if not profile or profile["coins"] < item["price"]:
            raise ValueError("Not enough coins")
        if item["price"]:
            conn.execute(
                "UPDATE profiles SET coins = coins - ?, updated_at = ? WHERE id = ?",
                (item["price"], now, profile_id),
            )
        conn.execute(
            "INSERT INTO owned_items(profile_id, item_key, purchased_at, equipped) VALUES (?, ?, ?, 1)",
            (profile_id, item_key, now),
        )
        # Equip only within the same category.
        conn.execute(
            """UPDATE owned_items SET equipped = 0
               WHERE profile_id = ? AND item_key IN (
                 SELECT item_key FROM store_items WHERE category = ?
               )""",
            (profile_id, item["category"]),
        )
        conn.execute(
            "UPDATE owned_items SET equipped = 1 WHERE profile_id = ? AND item_key = ?",
            (profile_id, item_key),
        )
    return {"item": dict(item), "charged": item["price"], "owned": True, "equipped": True}


def equip_item(profile_id: int, item_key: str, db_path: str | Path = DEFAULT_DB) -> dict:
    with db(db_path) as conn:
        owned = conn.execute(
            "SELECT * FROM owned_items WHERE profile_id = ? AND item_key = ?",
            (profile_id, item_key),
        ).fetchone()
        if not owned:
            raise ValueError("Item not owned")
        item = conn.execute("SELECT * FROM store_items WHERE item_key = ?", (item_key,)).fetchone()
        conn.execute(
            """UPDATE owned_items SET equipped = 0
               WHERE profile_id = ? AND item_key IN (
                 SELECT item_key FROM store_items WHERE category = ?
               )""",
            (profile_id, item["category"]),
        )
        conn.execute(
            "UPDATE owned_items SET equipped = 1 WHERE profile_id = ? AND item_key = ?",
            (profile_id, item_key),
        )
        if item["category"] == "theme":
            conn.execute(
                "UPDATE settings SET theme = ?, updated_at = ? WHERE profile_id = ?",
                (item["item_key"].replace("theme-", ""), utc_now(), profile_id),
            )
    return {"item": dict(item), "equipped": True}


# ---------------------------------------------------------------------------
# Notes / drawings
# ---------------------------------------------------------------------------

def save_note(profile_id: int, target_type: str, target_id: str,
              db_path: str | Path = DEFAULT_DB, text: str = "",
              strokes: Any = None, image_path: str = "") -> dict:
    now = utc_now()
    with db(db_path) as conn:
        conn.execute(
            """INSERT INTO notes(profile_id, target_type, target_id, text, strokes_json,
               image_path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(profile_id, target_type, target_id) DO UPDATE SET
                 text=excluded.text, strokes_json=excluded.strokes_json,
                 image_path=excluded.image_path, updated_at=excluded.updated_at""",
            (profile_id, target_type, target_id, text, _json(strokes, []),
             image_path, now, now),
        )
        row = conn.execute(
            "SELECT * FROM notes WHERE profile_id = ? AND target_type = ? AND target_id = ?",
            (profile_id, target_type, target_id),
        ).fetchone()
    item = dict(row)
    item["strokes"] = _loads(item.pop("strokes_json"), [])
    return item


def list_notes(profile_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM notes WHERE profile_id = ? ORDER BY updated_at DESC", (profile_id,)
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["strokes"] = _loads(item.pop("strokes_json"), [])
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Plugins, web sources, inactivity and model runs
# ---------------------------------------------------------------------------

def list_plugins(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute("SELECT * FROM plugins ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def set_plugin(plugin_key: str, enabled: bool, status: str,
               db_path: str | Path = DEFAULT_DB) -> dict:
    with db(db_path) as conn:
        conn.execute(
            "UPDATE plugins SET enabled = ?, status = ?, updated_at = ? WHERE plugin_key = ?",
            (int(enabled), status, utc_now(), plugin_key),
        )
        row = conn.execute("SELECT * FROM plugins WHERE plugin_key = ?", (plugin_key,)).fetchone()
    return dict(row)


def add_web_source(profile_id: int, url: str, title: str, text: str,
                   db_path: str | Path = DEFAULT_DB, topic_id: int | None = None) -> dict:
    with db(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO web_sources(profile_id, topic_id, url, title, fetched_at, text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (profile_id, topic_id, url, title, utc_now(), text),
        )
        row = conn.execute("SELECT * FROM web_sources WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_web_sources(profile_id: int, db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM web_sources WHERE profile_id = ? ORDER BY fetched_at DESC", (profile_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def touch_activity(profile_id: int, db_path: str | Path = DEFAULT_DB) -> dict:
    now = utc_now()
    with db(db_path) as conn:
        conn.execute(
            """INSERT INTO inactivity_state(profile_id, last_real_activity) VALUES (?, ?)
               ON CONFLICT(profile_id) DO UPDATE SET last_real_activity=excluded.last_real_activity,
                 warmup_stopped_at=NULL, worker_stopped_at=NULL, media_stop_at=NULL""",
            (profile_id, now),
        )
        row = conn.execute("SELECT * FROM inactivity_state WHERE profile_id = ?", (profile_id,)).fetchone()
    return dict(row)


def get_inactivity_state(profile_id: int, db_path: str | Path = DEFAULT_DB) -> dict:
    with db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM inactivity_state WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if not row:
            now = utc_now()
            conn.execute(
                "INSERT INTO inactivity_state(profile_id, last_real_activity) VALUES (?, ?)",
                (profile_id, now),
            )
            row = conn.execute(
                "SELECT * FROM inactivity_state WHERE profile_id = ?", (profile_id,)
            ).fetchone()
    return dict(row)


def set_inactivity_marker(profile_id: int, marker: str, value: str | None,
                          db_path: str | Path = DEFAULT_DB) -> dict:
    allowed = {"warmup_stopped_at", "worker_stopped_at", "media_stop_at", "last_real_activity", "pending_json"}
    if marker not in allowed:
        raise ValueError("Unknown inactivity marker")
    with db(db_path) as conn:
        conn.execute(f"UPDATE inactivity_state SET {marker} = ? WHERE profile_id = ?",
                     (value, profile_id))
        row = conn.execute(
            "SELECT * FROM inactivity_state WHERE profile_id = ?", (profile_id,)
        ).fetchone()
    return dict(row) if row else {}


def record_model_run(model_name: str, version: str, task: str, metrics: dict,
                     db_path: str | Path = DEFAULT_DB) -> dict:
    with db(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO model_runs(model_name, version, task, metrics_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (model_name, version, task, _json(metrics, {}), utc_now()),
        )
        row = conn.execute("SELECT * FROM model_runs WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def log_activity(profile_id: int | None, kind: str, detail: str = "",
                 db_path: str | Path = DEFAULT_DB) -> None:
    with db(db_path) as conn:
        conn.execute(
            "INSERT INTO activity_log(profile_id, kind, detail, created_at) VALUES (?, ?, ?, ?)",
            (profile_id, kind, detail, utc_now()),
        )


# ---------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------

def create_backup(db_path: str | Path = DEFAULT_DB, out_dir: str | Path | None = None) -> Path:
    """Create a consistent ZIP backup of the DB plus uploads/processed files.

    Secrets, the virtual environment and raw datasets are never included.
    """
    db_path = Path(db_path)
    data_dir = db_path.parent
    out_dir = Path(out_dir) if out_dir else data_dir / "backups"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = out_dir / f"rewiseorist-backup-{stamp}.zip"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db = Path(temp_dir) / "rewiseorist.db"
        source = sqlite3.connect(str(db_path))
        target = sqlite3.connect(str(temp_db))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(temp_db, "rewiseorist.db")
            for folder in ("uploads", "processed", "chatlogs", "plans"):
                base = data_dir / folder
                if not base.exists():
                    continue
                for path in base.rglob("*"):
                    if path.is_file() and "secret" not in path.name.lower():
                        zf.write(path, str(Path(folder) / path.relative_to(base)))
    return archive


def restore_backup(archive_path: str | Path, db_path: str | Path = DEFAULT_DB) -> Path:
    """Restore a backup into the data directory after extracting to a temp folder.

    The caller should stop the app before restoring. A safety copy of the current
    database is made first.
    """
    archive_path = Path(archive_path)
    db_path = Path(db_path)
    data_dir = db_path.parent
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(temp_dir)
        extracted_db = Path(temp_dir) / "rewiseorist.db"
        if not extracted_db.exists():
            raise ValueError("Backup does not contain rewiseorist.db")
        if db_path.exists():
            create_backup(db_path, data_dir / "backups")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(extracted_db.read_bytes())
    return db_path
