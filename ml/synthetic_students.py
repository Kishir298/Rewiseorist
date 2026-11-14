"""Deterministic synthetic student histories for classifier distillation.

The generated histories include an *intended* category for study design and
review agreement only. The classifier is trained on teacher labels from the
saved raw data, never on these intended labels.
"""

from __future__ import annotations

import random

TOPICS = [
    "Fractions", "Forces", "Chemical bonding", "Indices", "Electric circuits",
    "The Industrial Revolution", "Algebra", "Photosynthesis", "Particle model",
]

DIFFICULTY_VALUES = {"easy": 1, "medium": 2, "hard": 3}
OUTCOMES = {"correct": 1.0, "partly correct": 0.5, "wrong": 0.0}
ERROR_TYPES = ["sign_error", "unit_error", "definition_confusion", "calculation_slip", "missing_step"]

PROFILES = {
    "Strong": {"accuracy": 0.86, "trend": 0.07, "hints": 0.04,
               "difficulties": ["medium", "hard", "hard"], "questions": (7, 12)},
    "Developing": {"accuracy": 0.56, "trend": 0.02, "hints": 0.24,
                   "difficulties": ["easy", "medium", "medium", "hard"], "questions": (4, 9)},
    "Weak": {"accuracy": 0.24, "trend": -0.03, "hints": 0.62,
             "difficulties": ["easy", "easy", "medium"], "questions": (3, 7)},
    "Unknown": {"accuracy": 0.72, "trend": 0.0, "hints": 0.1,
                "difficulties": ["easy", "medium"], "questions": (2, 2)},
}


def _attempts_for(label: str, rng: random.Random) -> list[dict]:
    profile = PROFILES[label]
    count = rng.randint(*profile["questions"])
    attempts = []
    for index in range(count):
        progress = index / max(1, count - 1)
        base = profile["accuracy"] + profile["trend"] * (progress - 0.5)
        roll = rng.random()
        if roll < base:
            outcome = "correct"
        elif roll < base + 0.22:
            outcome = "partly correct"
        else:
            outcome = "wrong"
        hint = rng.random() < profile["hints"]
        difficulty = rng.choice(profile["difficulties"])
        error = ""
        if outcome != "correct":
            error = rng.choice(ERROR_TYPES)
        attempts.append({
            "index": index + 1,
            "difficulty": difficulty,
            "outcome": outcome,
            "hint_used": hint,
            "error_type": error,
        })
    return attempts


def summarise(attempts: list[dict]) -> dict:
    ratios = [OUTCOMES[item["outcome"]] for item in attempts]
    recent = ratios[-3:]
    earlier = ratios[:-3]
    mean_accuracy = sum(ratios) / len(ratios)
    recent_accuracy = sum(recent) / len(recent)
    earlier_accuracy = sum(earlier) / len(earlier) if earlier else mean_accuracy
    seen_errors: set[str] = set()
    repeated = 0
    for item in attempts:
        if item["error_type"]:
            if item["error_type"] in seen_errors:
                repeated += 1
            seen_errors.add(item["error_type"])
    unknown = len(attempts) < 3
    return {
        "mean_accuracy": round(mean_accuracy, 4),
        "recent_accuracy": round(recent_accuracy, 4),
        "accuracy_trend": round(recent_accuracy - earlier_accuracy, 4),
        "hint_rate": round(sum(item["hint_used"] for item in attempts) / len(attempts), 4),
        "independent_rate": round(
            sum(1 for item in attempts if item["outcome"] == "correct" and not item["hint_used"])
            / len(attempts), 4
        ),
        "mean_difficulty": round(
            sum(DIFFICULTY_VALUES[item["difficulty"]] for item in attempts) / len(attempts), 3
        ),
        "mean_active_seconds": round(
            sum(random.Random(item["index"]).uniform(25, 95) for item in attempts) / len(attempts), 1
        ),
        "distinct_attempts": len(attempts),
        "distinct_questions": len(attempts),
        "repeated_errors": repeated,
        "unknown": unknown,
    }


def history_text(attempts: list[dict]) -> str:
    parts = []
    for item in attempts:
        hint = "hint used" if item["hint_used"] else "no hint"
        parts.append(
            f"attempt {item['index']}: {item['difficulty']}, {item['outcome']}, {hint}"
        )
    return "; ".join(parts)


def build_personas(per_label: int = 15, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    personas = []
    for label in ("Weak", "Developing", "Strong", "Unknown"):
        for index in range(1, per_label + 1):
            attempts = _attempts_for(label, rng)
            personas.append({
                "persona_id": f"student-{label.lower()}-{index:02d}",
                "intended_label": label,
                "topic": TOPICS[(index + len(personas)) % len(TOPICS)],
                "features": summarise(attempts),
                "history": history_text(attempts),
                "attempts": attempts,
            })
    rng.shuffle(personas)
    return personas


def group_split(personas: list[dict]) -> dict[str, list[dict]]:
    """Split by persona and stratify by intended label."""
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    by_label: dict[str, list[dict]] = {}
    for persona in personas:
        by_label.setdefault(persona["intended_label"], []).append(persona)
    for label, items in by_label.items():
        for position, persona in enumerate(items):
            if position < max(1, int(len(items) * 0.7)):
                split = "train"
            elif position < max(2, int(len(items) * 0.85)):
                split = "val"
            else:
                split = "test"
            buckets[split].append(persona)
    return buckets
