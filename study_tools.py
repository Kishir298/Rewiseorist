"""Quizzes, marking, feedback, revision plans, progress and rewards."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import storage

PERSONALITIES = {
    "Friendly": "You are a warm, encouraging study tutor. Use short sentences and one practical next step.",
    "Formal": "You are a precise academic tutor. Use formal language, define terms, and cite the supplied sources.",
    "Concise": "You are a concise tutor. Give the answer in the fewest useful words, then one next step.",
}

DIFFICULTY_MULTIPLIER = {"easy": 0.8, "medium": 1.0, "hard": 1.25}
PRIORITY_WEIGHTS = {"urgency": 0.45, "practice": 0.35, "overdue": 0.20}
PRACTICE_NEED = {"Weak": 1.0, "Developing": 0.5, "Strong": 0.0, "Unknown": 0.75}
REVIEW_INTERVAL_DAYS = 7
XP_COMPLETE = 5
XP_CORRECT = 10
COINS_CORRECT = 2
COINS_LESSON = 1


# ---------------------------------------------------------------------------
# Marking and feedback
# ---------------------------------------------------------------------------

def normalize_answer(value: str) -> str:
    value = (value or "").lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _similarity(a: str, b: str) -> float:
    a, b = normalize_answer(a), normalize_answer(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if b in a:
        return 0.92
    if a in b:
        # A short fragment of the model answer is partial evidence, not a full answer.
        return 0.72
    return SequenceMatcher(None, a, b).ratio()


def mark_answer(question: dict, answer: str) -> dict:
    """Mark one answer against a checked key and produce explanatory feedback."""
    answer = (answer or "").strip()
    qtype = question.get("type", "short")
    expected = str(question.get("answer", ""))
    keywords = question.get("accepted_keywords") or []
    options = question.get("options") or []
    if not answer:
        return {
            "correct": False, "marks": 0.0, "max_marks": 1.0, "error_type": "blank",
            "feedback": "No answer was submitted. Read the question again, then try one clear sentence.",
            "model_answer": expected,
        }
    if qtype == "mcq":
        submitted = normalize_answer(answer)
        correct = submitted == normalize_answer(expected) or _similarity(answer, expected) >= 0.85
        if not correct and options:
            try:
                index = int(answer)
                correct = 0 <= index < len(options) and normalize_answer(options[index]) == normalize_answer(expected)
            except (TypeError, ValueError):
                pass
        return {
            "correct": correct, "marks": 1.0 if correct else 0.0, "max_marks": 1.0,
            "error_type": "" if correct else "misconception",
            "feedback": (
                "Correct. You selected the statement that matches the source. Next: explain why it is true without looking."
                if correct else
                f"Not yet. The correct option is '{expected}'. Check the source, then explain the idea in your own words."
            ),
            "model_answer": expected,
        }

    keyword_hits = [
        kw for kw in keywords
        if normalize_answer(kw) and normalize_answer(kw) in normalize_answer(answer)
    ]
    keyword_ratio = len(keyword_hits) / max(len(keywords), 1)
    sim = _similarity(answer, expected)
    correct = sim >= 0.82 or (keywords and keyword_ratio >= 0.6)
    partial = not correct and (sim >= 0.45 or keyword_ratio >= 0.35)
    if correct:
        marks = 1.0
        error = ""
        feedback = (
            "Correct. Your answer matches the checked key. Next: write one sentence explaining the reasoning."
        )
    elif partial:
        marks = 0.5
        error = "incomplete"
        feedback = (
            "Partly correct. You included some relevant ideas, but the answer is missing a key part. "
            f"Compare it with the model answer: {expected}"
        )
    else:
        marks = 0.0
        error = "recall" if keywords else "misconception"
        feedback = (
            "Not yet. The answer does not match the checked key. "
            f"Use the hint, then compare your wording with: {expected}"
        )
    return {
        "correct": correct, "marks": marks, "max_marks": 1.0, "error_type": error,
        "feedback": feedback, "model_answer": expected,
        "keywords_matched": keyword_hits,
    }


# ---------------------------------------------------------------------------
# Quiz generation / sample bank
# ---------------------------------------------------------------------------

def load_sample_quiz_keys(samples_dir: str | Path | None = None) -> dict:
    root = Path(samples_dir) if samples_dir else Path(__file__).resolve().parent / "samples"
    path = root / "quiz_keys.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def seed_sample_questions(profile_id: int, topic_name: str, db_path: str | Path = storage.DEFAULT_DB,
                          samples_dir: str | Path | None = None) -> list[dict]:
    bank = load_sample_quiz_keys(samples_dir)
    key = topic_name.strip().lower().replace(" ", "-")
    items = bank.get(key) or bank.get(topic_name) or []
    if not items:
        for bank_key, bank_items in bank.items():
            if bank_key.replace("-", " ").lower() == topic_name.strip().lower():
                items = bank_items
                break
    if not items:
        return []
    topics = storage.list_topics(profile_id, db_path)
    topic = next((t for t in topics if t["name"].lower() == topic_name.lower()), None)
    if not topic:
        topic = storage.add_topic(profile_id, topic_name, difficulty="medium", db_path=db_path)
    saved = []
    for item in items:
        saved.append(storage.save_question(
            topic["id"], item["id"], item["question"], item["answer"], db_path=db_path,
            qtype=item.get("type", "short"), difficulty=item.get("difficulty", "medium"),
            options=item.get("options", []),
            accepted_keywords=item.get("accepted_keywords", []),
            hint=item.get("hint", ""), explanation=item.get("explanation", ""),
            source_ref=item.get("source", ""), origin="sample-bank",
        ))
    return saved


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]


def generate_questions_from_chunks(topic_id: int, topic_name: str, chunks: Sequence[dict],
                                   db_path: str | Path = storage.DEFAULT_DB,
                                   count: int = 5) -> list[dict]:
    """Create checked questions from source sentences when no sample bank exists."""
    sentences: list[tuple[str, dict]] = []
    for chunk in chunks:
        for sentence in _sentences(chunk.get("text", "")):
            sentences.append((sentence, chunk))
    if not sentences:
        return []
    questions = []
    for index, (sentence, chunk) in enumerate(sentences[:count]):
        source_ref = chunk.get("source_ref") or chunk.get("page_ref", "")
        words = re.findall(r"[A-Za-z][A-Za-z\-]+", sentence)
        key_terms = [w for w in words if len(w) > 5][:5]
        if " is " in sentence or " are " in sentence or " means " in sentence:
            prompt = f"Complete or explain this source statement: “{sentence[:180]}…”"
        else:
            prompt = f"State one accurate fact about {topic_name} from the source."
        answer = sentence
        questions.append(storage.save_question(
            topic_id, f"generated-{index + 1}", prompt, answer, db_path=db_path,
            qtype="short", difficulty="medium",
            accepted_keywords=key_terms, hint="Re-read the linked source passage and look for the key terms.",
            explanation="The model answer is the source sentence. A correct answer should preserve its meaning.",
            source_ref=source_ref, origin="template",
        ))
    return questions


def ensure_quiz(profile_id: int, topic_id: int, db_path: str | Path = storage.DEFAULT_DB,
                count: int = 5) -> list[dict]:
    existing = storage.list_questions(topic_id, db_path)
    if existing:
        return existing
    topic = next((t for t in storage.list_topics(profile_id, db_path) if t["id"] == topic_id), None)
    if not topic:
        raise ValueError("Topic not found")
    seeded = seed_sample_questions(profile_id, topic["name"], db_path)
    if seeded:
        return seeded
    chunks = storage.chunks_for_topic(profile_id, topic_id, db_path)
    return generate_questions_from_chunks(topic_id, topic["name"], chunks, db_path, count)


# ---------------------------------------------------------------------------
# Attempts, rewards and improvement
# ---------------------------------------------------------------------------

def submit_answer(profile_id: int, question: dict, answer: str,
                  db_path: str | Path = storage.DEFAULT_DB, session_id: int | None = None,
                  hint_used: bool = False, independent: bool = True,
                  active_seconds: int = 0, attempt_key: str | None = None) -> dict:
    result = mark_answer(question, answer)
    normalized = normalize_answer(answer)
    # Repeated submissions of the same answer are not new evidence.
    for existing in storage.list_attempts(profile_id, db_path=db_path):
        if (existing["question_id"] == question["id"]
                and existing["verified"]
                and normalize_answer(existing["answer"]) == normalized):
            return {
                **result, "attempt": existing, "rewards": [], "duplicate": True,
                "message": "That answer was already recorded; it was not counted as new evidence.",
            }
    key = attempt_key or f"{profile_id}:{question['id']}:{storage.utc_now()}"
    attempt = storage.record_attempt(
        profile_id, key, question["id"], answer, result["marks"], result["max_marks"],
        result["correct"], db_path=db_path, topic_id=question.get("topic_id"),
        session_id=session_id, hint_used=hint_used, independent=independent,
        error_type=result["error_type"], active_seconds=active_seconds, verified=True,
    )
    rewards = []
    if attempt:
        rewards.append(storage.award_reward(
            profile_id, f"attempt-complete:{attempt['id']}", "xp", XP_COMPLETE,
            "Completed a practice attempt", db_path,
        ))
        if result["correct"]:
            rewards.append(storage.award_reward(
                profile_id, f"question-correct:{question['id']}", "xp", XP_CORRECT,
                "Correct checked answer", db_path,
            ))
            rewards.append(storage.award_reward(
                profile_id, f"question-coins:{question['id']}", "coins", COINS_CORRECT,
                "Correct checked answer", db_path,
            ))
    return {
        **result, "attempt": attempt,
        "rewards": [r for r in rewards if r.get("created")],
    }


def improvement_steps(profile_id: int, topic_id: int,
                      db_path: str | Path = storage.DEFAULT_DB) -> dict:
    attempts = [a for a in storage.list_attempts(profile_id, topic_id, db_path) if a["verified"]]
    if len(attempts) < 3:
        return {
            "ready": False,
            "message": "Complete at least three distinct checked practice attempts on this topic first.",
            "steps": [],
        }
    recent = attempts[-5:]
    accuracy = sum(a["marks_awarded"] / max(a["max_marks"], 1) for a in recent) / len(recent)
    hint_rate = sum(a["hint_used"] for a in recent) / len(recent)
    independent_rate = sum(a["independent"] for a in recent) / len(recent)
    errors = defaultdict(int)
    for a in recent:
        if a["error_type"]:
            errors[a["error_type"]] += 1
    steps = []
    if accuracy < 0.5:
        steps.append("Re-teach the core idea with the linked source, then answer two easier recall questions.")
    elif accuracy < 0.8:
        steps.append("Review the missed key terms, then retry the same questions without hints.")
    else:
        steps.append("Move to a harder question set and explain your reasoning in writing.")
    if hint_rate > 0.4:
        steps.append("Do one unhinted practice set; use the hint only after writing what you already know.")
    if independent_rate < 0.7:
        steps.append("Close the source for the next attempt, then compare your answer with the model answer.")
    if errors:
        top_error = max(errors, key=errors.get)
        steps.append(f"Target the most common error type: {top_error.replace('_', ' ')}.")
    steps.append("Schedule a short review in 2–3 days and one mixed quiz before the deadline.")
    return {
        "ready": True, "accuracy": round(accuracy, 3), "hint_rate": round(hint_rate, 3),
        "independent_rate": round(independent_rate, 3),
        "error_counts": dict(errors), "steps": steps,
    }


# ---------------------------------------------------------------------------
# Revision plans
# ---------------------------------------------------------------------------

def _parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def calculate_priority(topic: dict, deadline: dict | None, attempts: Sequence[dict],
                       today: date | None = None) -> dict:
    today = today or date.today()
    if deadline:
        days_left = (_parse_date(deadline["due_date"]) - today).days
    else:
        days_left = 21
    urgency = 1.0 / (max(days_left, 0) + 1)
    recent = list(attempts)[-5:]
    if recent:
        ratio = sum(a["marks_awarded"] / max(a["max_marks"], 1) for a in recent) / len(recent)
        if ratio >= 0.8 and all(a["independent"] for a in recent):
            level = "Strong"
        elif ratio >= 0.55:
            level = "Developing"
        else:
            level = "Weak"
    else:
        level = "Unknown"
    practice = PRACTICE_NEED.get(level, 0.75)
    last_reviewed = topic.get("last_reviewed_at")
    if last_reviewed:
        try:
            days_since = max((today - _parse_date(last_reviewed)).days, 0)
        except ValueError:
            days_since = REVIEW_INTERVAL_DAYS
    else:
        days_since = REVIEW_INTERVAL_DAYS
    overdue = min(days_since / REVIEW_INTERVAL_DAYS, 1.0)
    score = (
        PRIORITY_WEIGHTS["urgency"] * urgency
        + PRIORITY_WEIGHTS["practice"] * practice
        + PRIORITY_WEIGHTS["overdue"] * overdue
    )
    return {
        "topic_id": topic["id"], "topic": topic["name"], "days_left": days_left,
        "urgency": round(urgency, 4), "level": level, "practice_need": practice,
        "overdue": round(overdue, 4), "priority": round(score, 4),
        "deadline": deadline["due_date"] if deadline else None,
    }


def estimate_minutes(topic: dict, level: str, pace_seconds: int | None = None) -> int:
    length = max(int(topic.get("content_length") or 800), 100)
    base = length / 220.0
    if pace_seconds:
        base = max(base, pace_seconds / 60.0 * 3)
    minutes = base * DIFFICULTY_MULTIPLIER.get(topic.get("difficulty", "medium"), 1.0)
    if level == "Weak":
        minutes *= 1.2
    return int(max(10, min(90, round(minutes))))


def propose_plan(profile_id: int, available: Sequence[tuple[str, int]] | None = None,
                 db_path: str | Path = storage.DEFAULT_DB, horizon_days: int = 7,
                 daily_minutes: int = 45, title: str | None = None) -> dict:
    """Create a feasible draft plan; uncovered tasks are shown instead of overbooking."""
    topics = storage.list_topics(profile_id, db_path)
    deadlines = storage.list_deadlines(profile_id, db_path)
    deadline_by_topic = {d["topic_id"]: d for d in deadlines if d.get("topic_id")}
    if not topics:
        return {
            "summary": "No topics yet. Add a subject and topic before generating a plan.",
            "blocks": [], "uncovered": [], "weights": PRIORITY_WEIGHTS, "horizon_days": horizon_days,
        }
    ranked = []
    for topic in topics:
        attempts = storage.list_attempts(profile_id, topic["id"], db_path)
        ranked.append(calculate_priority(topic, deadline_by_topic.get(topic["id"]), attempts))
    ranked.sort(key=lambda item: item["priority"], reverse=True)
    if available is None:
        start = date.today()
        available = [
            ((start + timedelta(days=i)).isoformat(),
             0 if (start + timedelta(days=i)).weekday() == 6 else daily_minutes)
            for i in range(horizon_days)
        ]
    blocks = []
    uncovered = []
    task_index = 0
    activities = ["Learn", "Recall", "Practice test"]
    for day, minutes_left in available:
        if minutes_left <= 0:
            continue
        cursor = 0
        while minutes_left >= 10 and ranked:
            item = ranked[task_index % len(ranked)]
            minutes = min(estimate_minutes(
                next(t for t in topics if t["id"] == item["topic_id"]),
                item["level"],
            ), minutes_left, 50)
            if minutes < 10:
                break
            activity = activities[task_index % len(activities)]
            blocks.append({
                "date": day, "start_minute": cursor, "minutes": minutes,
                "topic_id": item["topic_id"], "topic": item["topic"],
                "activity": activity, "priority": item["priority"],
                "reason": (
                    f"Priority {item['priority']:.2f}; urgency {item['urgency']:.2f}, "
                    f"practice need {item['practice_need']:.2f}, overdue {item['overdue']:.2f}."
                ),
                "covered": True,
            })
            minutes_left -= minutes
            cursor += minutes
            task_index += 1
        # Anything estimated beyond the day is listed as uncovered.
        if minutes_left < 10 and task_index < len(ranked):
            item = ranked[task_index % len(ranked)]
            uncovered.append({
                "topic_id": item["topic_id"], "topic": item["topic"],
                "minutes": estimate_minutes(
                    next(t for t in topics if t["id"] == item["topic_id"]), item["level"]
                ),
                "reason": "No remaining time in the selected study window.",
            })
    summary = (
        f"{len(blocks)} study blocks across {len(set(b['date'] for b in blocks))} days. "
        f"{len(uncovered)} task(s) need extra time."
    )
    plan = {
        "summary": summary, "generated_at": storage.utc_now(), "horizon_days": horizon_days,
        "weights": PRIORITY_WEIGHTS, "blocks": blocks, "uncovered": uncovered,
        "ranked_topics": ranked,
    }
    plan["markdown"] = format_plan_md(plan)
    return plan


def format_plan_md(plan: dict) -> str:
    lines = ["# Revision plan", "", plan.get("summary", ""), ""]
    if plan.get("ranked_topics"):
        lines.append("## Topic priority")
        for item in plan["ranked_topics"]:
            lines.append(
                f"- **{item['topic']}** — priority {item['priority']:.2f}; level {item['level']}; "
                f"{item['days_left']} day(s) left."
            )
        lines.append("")
    lines.append("## Scheduled blocks")
    if not plan.get("blocks"):
        lines.append("No blocks scheduled.")
    for block in plan.get("blocks", []):
        hour = 9 + block.get("start_minute", 0) // 60
        minute = block.get("start_minute", 0) % 60
        lines.append(
            f"- {block['date']} {hour:02d}:{minute:02d} — {block['topic']} — "
            f"{block['activity']} ({block['minutes']} min). {block['reason']}"
        )
    if plan.get("uncovered"):
        lines.append("")
        lines.append("## Needs extra time")
        for item in plan["uncovered"]:
            lines.append(f"- {item['topic']} — {item['minutes']} min. {item['reason']}")
    return "\n".join(lines)


def plan_diff(before: dict, after: dict) -> dict:
    before_keys = {(b["date"], b["topic_id"], b["activity"]) for b in before.get("blocks", [])}
    after_keys = {(b["date"], b["topic_id"], b["activity"]) for b in after.get("blocks", [])}
    return {
        "added": [k for k in sorted(after_keys - before_keys)],
        "removed": [k for k in sorted(before_keys - after_keys)],
        "before_summary": before.get("summary", ""),
        "after_summary": after.get("summary", ""),
    }


# ---------------------------------------------------------------------------
# Progress, adherence and charts
# ---------------------------------------------------------------------------

def _period_start(period: str, today: date | None = None) -> date | None:
    today = today or date.today()
    if period == "daily":
        return today
    if period == "weekly":
        return today - timedelta(days=today.weekday())
    if period == "monthly":
        return today.replace(day=1)
    return None


def progress_summary(profile_id: int, period: str = "all", topic_id: int | None = None,
                     db_path: str | Path = storage.DEFAULT_DB) -> dict:
    start = _period_start(period)
    attempts = storage.list_attempts(profile_id, topic_id, db_path)
    if start:
        attempts = [a for a in attempts if _parse_date(a["created_at"]) >= start]
    verified = [a for a in attempts if a["verified"]]
    total_marks = sum(a["marks_awarded"] for a in verified)
    max_marks = sum(a["max_marks"] for a in verified)
    accuracy = total_marks / max_marks if max_marks else 0
    active_seconds = sum(a["active_seconds"] for a in verified)
    search = storage.list_web_sources(profile_id, db_path)
    # Rewards in the selected period
    with storage.db(db_path) as conn:
        reward_rows = conn.execute(
            "SELECT * FROM rewards WHERE profile_id = ? ORDER BY created_at", (profile_id,)
        ).fetchall()
        profile = conn.execute("SELECT xp, coins FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    rewards = [dict(r) for r in reward_rows]
    if start:
        rewards = [r for r in rewards if _parse_date(r["created_at"]) >= start]
    xp = sum(r["amount"] for r in rewards if r["kind"] == "xp")
    coins = sum(r["amount"] for r in rewards if r["kind"] == "coins")
    topics = storage.list_topics(profile_id, db_path)
    topic_stats = []
    for topic in topics:
        topic_attempts = [a for a in verified if a["topic_id"] == topic["id"]]
        if topic_id is not None and topic["id"] != topic_id:
            continue
        marks = sum(a["marks_awarded"] for a in topic_attempts)
        possible = sum(a["max_marks"] for a in topic_attempts)
        topic_stats.append({
            "topic_id": topic["id"], "topic": topic["name"],
            "attempts": len(topic_attempts),
            "accuracy": round(marks / possible, 3) if possible else None,
            "active_seconds": sum(a["active_seconds"] for a in topic_attempts),
            "level": topic.get("estimated_level", "Unknown"),
        })
    # Daily chart buckets, always including today.
    buckets: dict[str, dict] = {}
    for attempt in verified:
        day = attempt["created_at"][:10]
        bucket = buckets.setdefault(day, {"date": day, "minutes": 0, "xp": 0, "coins": 0})
        bucket["minutes"] += round(attempt["active_seconds"] / 60, 1)
    for reward in rewards:
        day = reward["created_at"][:10]
        bucket = buckets.setdefault(day, {"date": day, "minutes": 0, "xp": 0, "coins": 0})
        if reward["kind"] == "xp":
            bucket["xp"] += reward["amount"]
        elif reward["kind"] == "coins":
            bucket["coins"] += reward["amount"]
    chart = [buckets[key] for key in sorted(buckets)]
    # Accepted current plan and adherence.
    plans = storage.list_plans(profile_id, db_path)
    current_plan = next((p for p in plans if p["status"] == "accepted"), None)
    adherence = None
    due_blocks = []
    if current_plan:
        full = storage.get_plan(current_plan["id"], db_path)
        accepted = next((v for v in full["versions"] if v["accepted"]), None) if full else None
        if accepted:
            blocks = accepted["plan"].get("blocks", [])
            today = date.today()
            due_blocks = [b for b in blocks if _parse_date(b["date"]) <= today]
            if due_blocks:
                completed = 0
                for block in due_blocks:
                    if any(
                        a["topic_id"] == block["topic_id"]
                        and _parse_date(a["created_at"]) <= _parse_date(block["date"])
                        for a in verified
                    ):
                        completed += 1
                adherence = round(completed / len(due_blocks), 3)
    return {
        "period": period, "accuracy": round(accuracy, 3) if max_marks else None,
        "attempts": len(verified), "active_seconds": active_seconds,
        "active_minutes": round(active_seconds / 60, 1),
        "xp": xp, "coins": coins,
        "total_xp": profile["xp"] if profile else 0,
        "total_coins": profile["coins"] if profile else 0,
        "topics": topic_stats, "chart": chart,
        "web_sources": len(search),
        "current_plan": current_plan,
        "adherence": adherence,
        "adherence_label": "N/A" if adherence is None else f"{adherence * 100:.0f}%",
        "due_blocks": len(due_blocks),
    }


def award_lesson_completion(profile_id: int, topic_id: int,
                            db_path: str | Path = storage.DEFAULT_DB) -> dict:
    today = date.today().isoformat()
    xp = storage.award_reward(
        profile_id, f"lesson-complete:{topic_id}:{today}", "xp", 15,
        "Completed a study activity", db_path,
    )
    coins = storage.award_reward(
        profile_id, f"lesson-coins:{topic_id}:{today}", "coins", COINS_LESSON,
        "Completed a study activity", db_path,
    )
    badge = storage.award_badge(profile_id, "first-lesson", "First lesson completed", db_path)
    return {"xp": xp, "coins": coins, "badge": badge}
