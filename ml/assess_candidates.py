"""Repeated, rubric-based assessment of local llama.cpp teacher candidates.

Every candidate is run on the same fixed revision and performance tasks. Raw
outputs, timings and rubric scores are stored in ``evidence/ml/`` so the
selection can be checked later. Nothing here calls a hosted API or Ollama.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.prompts import (  # noqa: E402
    build_performance_prompt,
    build_revision_prompt,
    extract_citation,
)
from ml.teacher_client import (  # noqa: E402
    TEACHER_SPECS,
    LlamaCppServer,
    TeacherSpec,
    available_specs,
)

EVIDENCE_DIR = ROOT / "evidence" / "ml"

REVISION_TASKS = [
    {
        "key": "friction",
        "question": "What is friction and how does it affect a moving object?",
        "personality": "Friendly",
        "sources": [
            {
                "original_name": "forces.md",
                "source_ref": "forces.md | page 2",
                "text": (
                    "Friction is a contact force that opposes the motion of an object. "
                    "It acts in the opposite direction to movement and converts kinetic "
                    "energy into heat. Rough surfaces increase friction, while smooth "
                    "surfaces reduce it."
                ),
            }
        ],
        "expected_terms": ["contact force", "opposes", "motion", "heat", "surface"],
    },
    {
        "key": "osmosis",
        "question": "Explain osmosis in plant roots.",
        "personality": "Formal",
        "sources": [
            {
                "original_name": "transport.docx",
                "source_ref": "transport.docx | section 3",
                "text": (
                    "Osmosis is the movement of water from a dilute solution to a more "
                    "concentrated solution through a partially permeable membrane. Root "
                    "hair cells have a concentrated cell sap, so water enters them by "
                    "osmosis from the soil."
                ),
            }
        ],
        "expected_terms": ["water", "partially permeable", "concentrated", "root hair", "soil"],
    },
    {
        "key": "photosynthesis",
        "question": "State what photosynthesis needs and what it produces.",
        "personality": "Concise",
        "sources": [
            {
                "original_name": "plants.pptx",
                "source_ref": "plants.pptx | slide 7",
                "text": (
                    "Photosynthesis takes place in the chloroplasts. Carbon dioxide and "
                    "water are used, light energy is absorbed by chlorophyll, and glucose "
                    "and oxygen are produced."
                ),
            }
        ],
        "expected_terms": ["carbon dioxide", "water", "light", "glucose", "oxygen"],
    },
    {
        "key": "missing_answer",
        "question": "What is the capital city of Australia?",
        "personality": "Friendly",
        "sources": [
            {
                "original_name": "cells.md",
                "source_ref": "cells.md | page 1",
                "text": "Animal cells have a nucleus, cytoplasm and a cell membrane.",
            }
        ],
        "expected_terms": [],
        "expect_refusal": True,
    },
]

PERFORMANCE_TASKS = [
    {
        "key": "strong",
        "expected": "Strong",
        "features": {
            "distinct_questions": 9,
            "accuracy": 0.89,
            "mean_difficulty": 2.4,
            "hint_rate": 0.0,
            "recent_accuracy": 0.92,
            "earlier_accuracy": 0.79,
            "independent_rate": 0.89,
        },
        "history": (
            "attempt 1: hard, correct, no hint; attempt 2: hard, correct, no hint; "
            "attempt 3: medium, correct, no hint; attempt 4: hard, partly correct, no hint; "
            "attempt 5: hard, correct, no hint; attempt 6: hard, correct, no hint; "
            "attempt 7: medium, correct, no hint; attempt 8: hard, correct, no hint; "
            "attempt 9: hard, correct, no hint"
        ),
    },
    {
        "key": "developing",
        "expected": "Developing",
        "features": {
            "distinct_questions": 6,
            "accuracy": 0.55,
            "mean_difficulty": 1.8,
            "hint_rate": 0.33,
            "recent_accuracy": 0.6,
            "earlier_accuracy": 0.5,
            "independent_rate": 0.67,
        },
        "history": (
            "attempt 1: easy, correct, no hint; attempt 2: medium, wrong, hint used; "
            "attempt 3: medium, partly correct, no hint; attempt 4: medium, correct, hint used; "
            "attempt 5: hard, wrong, no hint; attempt 6: medium, correct, no hint"
        ),
    },
    {
        "key": "unknown_too_few",
        "expected": "Unknown",
        "features": {
            "distinct_questions": 2,
            "accuracy": 1.0,
            "mean_difficulty": 1.0,
            "hint_rate": 0.0,
            "recent_accuracy": 1.0,
            "earlier_accuracy": 1.0,
            "independent_rate": 1.0,
        },
        "history": "attempt 1: easy, correct, no hint; attempt 2: easy, correct, no hint",
    },
]


def _contains(text: str, terms: list[str]) -> int:
    lowered = (text or "").lower()
    return sum(1 for term in terms if re.search(r"\b" + re.escape(term.lower()) + r"\b", lowered))


def _citation_filename(text: str) -> str:
    citation = extract_citation(text)
    if not citation:
        return ""
    return citation.split("|", 1)[0].strip().lower()


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?]+", text or "") if part.strip()]


def score_revision(text: str, task: dict) -> dict:
    """Apply the 12-mark revision rubric as deterministic checks."""
    text = (text or "").strip()
    if not text:
        return {"accuracy": 0, "relevance": 0, "clarity": 0, "source_use": 0, "total": 0}

    filename = (task["sources"][0].get("original_name") or "").lower()
    cited = _citation_filename(text)

    if task.get("expect_refusal"):
        honest = bool(re.search(
            r"\b(not|no information|does not|cannot|can't|unable|isn't|is not)\b", text, re.IGNORECASE
        )) and not re.search(r"\b(canberra|sydney|melbourne)\b", text, re.IGNORECASE)
        accuracy = 4 if honest else 0
        relevance = 4 if honest else 1
        clarity = 4 if len(_sentences(text)) <= 6 and len(text) >= 15 else 2
        source_use = 2 if re.search(r"\b(source|notes|cells\.md)\b", text, re.IGNORECASE) else 1
        return {
            "accuracy": accuracy, "relevance": relevance, "clarity": clarity,
            "source_use": source_use, "total": accuracy + relevance + clarity + source_use,
            "honest_refusal": honest, "cited": cited,
        }

    hits = _contains(text, task["expected_terms"])
    accuracy = 4 if hits >= 4 else 3 if hits == 3 else 2 if hits == 2 else 1 if hits == 1 else 0

    question_words = {
        word for word in re.findall(r"[a-z]{4,}", task["question"].lower())
        if word not in {"what", "does", "explain", "state", "with", "from", "that", "this", "affect"}
    }
    question_hits = _contains(text, sorted(question_words))
    ratio = question_hits / max(1, len(question_words))
    relevance = 4 if ratio >= 0.5 else 3 if ratio >= 0.3 else 2 if ratio >= 0.2 else 1 if ratio >= 0.1 else 0

    sentences = _sentences(text)
    average_words = len(text.split()) / max(1, len(sentences))
    if 2 <= len(sentences) <= 8 and average_words <= 30 and len(text) >= 40:
        clarity = 4
    elif 1 <= len(sentences) <= 10 and average_words <= 40:
        clarity = 3
    elif len(text) >= 20:
        clarity = 2
    else:
        clarity = 1

    if cited and filename and cited == filename:
        source_use = 4
    elif cited:
        source_use = 3
    elif filename and filename.split(".")[0] in text.lower():
        source_use = 2
    elif re.search(r"\b(source|text|notes)\b", text, re.IGNORECASE):
        source_use = 1
    else:
        source_use = 0

    # Rubric rule: an unsupported claim caps accuracy and source use at 1.
    if not cited:
        accuracy = min(accuracy, 1)
        source_use = min(source_use, 1)

    return {
        "accuracy": accuracy, "relevance": relevance, "clarity": clarity,
        "source_use": source_use, "total": accuracy + relevance + clarity + source_use,
        "expected_term_hits": hits, "cited": cited,
    }


def extract_category(text: str) -> str:
    payload = extract_json_object(text)
    if payload and isinstance(payload.get("category"), str):
        category = payload["category"].strip().title()
        if category in {"Weak", "Developing", "Strong", "Unknown"}:
            return category
    match = re.search(r"\b(Weak|Developing|Strong|Unknown)\b", text or "", re.IGNORECASE)
    return match.group(1).title() if match else ""


def extract_json_object(text: str) -> dict:
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            continue
    return {}


def score_performance(text: str, expected: str) -> dict:
    """Apply the 6-mark performance rubric as deterministic checks."""
    text = (text or "").strip()
    category = extract_category(text)
    payload = extract_json_object(text)
    reason = str(payload.get("reason", "")) if payload else ""

    if category == expected:
        category_score = 2
    elif category and {category, expected} in ({"Weak", "Developing"}, {"Developing", "Strong"}):
        category_score = 1
    else:
        category_score = 0

    evidence_terms = [
        "accuracy", "correct", "attempt", "question", "trend", "improv", "hint",
        "independent", "difficulty", "error", "recent", "consistent", "evidence",
    ]
    lowered = text.lower()
    evidence_hits = sum(1 for term in evidence_terms if term in lowered)
    if reason and evidence_hits >= 3:
        reason_score = 2
    elif reason and evidence_hits >= 1:
        reason_score = 1
    else:
        reason_score = 0

    hint_mentioned = "hint" in lowered
    independent_mentioned = any(
        term in lowered for term in ("independent", "unaided", "without help", "no hint", "own")
    )
    hints_score = 2 if hint_mentioned and independent_mentioned else 1 if hint_mentioned or independent_mentioned else 0

    return {
        "category": category, "expected": expected,
        "category_score": category_score, "reason_score": reason_score,
        "hints_score": hints_score,
        "total": category_score + reason_score + hints_score,
        "reason": reason,
        "valid_json": bool(payload),
    }


def _run_revision_task(server: LlamaCppServer, task: dict, max_tokens: int) -> dict:
    prompt = build_revision_prompt(task["question"], task["sources"], task["personality"])
    response = server.chat(prompt, temperature=0.1, max_tokens=max_tokens)
    score = score_revision(response["text"], task)
    return {**response, "score": score}


def _run_performance_task(server: LlamaCppServer, task: dict, max_tokens: int) -> dict:
    prompt = build_performance_prompt(task["features"], task["history"])
    response = server.chat(prompt, temperature=0.0, max_tokens=max_tokens)
    score = score_performance(response["text"], task["expected"])
    return {**response, "score": score}


def assess_candidate(spec: TeacherSpec, repeats: int, port: int, max_tokens: int,
                     context: int, threads: int) -> dict:
    started = time.time()
    result: dict = {
        "model": spec.key,
        "name": spec.name,
        "manifest": spec.manifest(),
        "repeats": repeats,
        "revision_runs": [],
        "performance_runs": [],
    }
    with LlamaCppServer(spec, port=port, context=context, threads=threads) as server:
        for repeat in range(1, repeats + 1):
            for task in REVISION_TASKS:
                try:
                    runs = [_run_revision_task(server, task, max_tokens)]
                except Exception as exc:  # record and continue
                    runs = [{"text": "", "error": str(exc), "score": {"total": 0}}]
                result["revision_runs"].append({"repeat": repeat, "task": task["key"], **runs[0]})
            for task in PERFORMANCE_TASKS:
                try:
                    runs = [_run_performance_task(server, task, max_tokens)]
                except Exception as exc:
                    runs = [{"text": "", "error": str(exc), "score": {"total": 0}}]
                result["performance_runs"].append({"repeat": repeat, "task": task["key"], **runs[0]})
    result["total_seconds"] = round(time.time() - started, 1)
    result["revision"] = _aggregate(result["revision_runs"], [task["key"] for task in REVISION_TASKS])
    result["performance"] = _aggregate(
        result["performance_runs"], [task["key"] for task in PERFORMANCE_TASKS]
    )
    return result


def _aggregate(runs: list[dict], task_keys: list[str]) -> dict:
    valid = [run for run in runs if not run.get("error")]
    totals = [run.get("score", {}).get("total", 0) for run in runs]
    per_task = {}
    for key in task_keys:
        task_totals = [run.get("score", {}).get("total", 0) for run in runs if run["task"] == key]
        per_task[key] = round(statistics.mean(task_totals), 2) if task_totals else 0.0
    latency = [run.get("elapsed_seconds", 0) for run in valid]
    return {
        "mean_total": round(statistics.mean(totals), 2) if totals else 0.0,
        "consistency": round(len(valid) / len(runs), 3) if runs else 0.0,
        "mean_latency_seconds": round(statistics.mean(latency), 2) if latency else 0.0,
        "per_task": per_task,
        "maximum": 12 if task_keys and task_keys[0] in {t["key"] for t in REVISION_TASKS} else 6,
    }


def rank_candidates(results: list[dict], task_type: str) -> list[dict]:
    """Rank by rubric mean, then valid-output consistency, then latency."""
    ranked = sorted(
        results,
        key=lambda item: (
            -item[task_type]["mean_total"],
            -item[task_type]["consistency"],
            item[task_type]["mean_latency_seconds"],
        ),
    )
    return [
        {
            "rank": index + 1,
            "model": item["model"],
            "name": item["name"],
            **item[task_type],
        }
        for index, item in enumerate(ranked)
    ]


def write_report(results: list[dict], selected: dict, runtime: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "revision_tasks": [task["key"] for task in REVISION_TASKS],
        "performance_tasks": [task["key"] for task in PERFORMANCE_TASKS],
        "results": results,
        "revision_ranking": rank_candidates(results, "revision"),
        "performance_ranking": rank_candidates(results, "performance"),
        "selected": selected,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Teacher candidate assessment",
        "",
        f"Generated: {payload['generated_at']}",
        f"Runtime: {runtime.get('runtime_version', 'llama.cpp')} CPU",
        "",
        "## Revision task ranking (out of 12)",
        "",
        "| Rank | Model | Mean | Consistency | Latency (s) |",
        "|---|---|---|---|---|",
    ]
    for row in payload["revision_ranking"]:
        lines.append(
            f"| {row['rank']} | {row['name']} | {row['mean_total']} | "
            f"{row['consistency']} | {row['mean_latency_seconds']} |"
        )
    lines += [
        "",
        "## Performance task ranking (out of 6)",
        "",
        "| Rank | Model | Mean | Consistency | Latency (s) |",
        "|---|---|---|---|---|",
    ]
    for row in payload["performance_ranking"]:
        lines.append(
            f"| {row['rank']} | {row['name']} | {row['mean_total']} | "
            f"{row['consistency']} | {row['mean_latency_seconds']} |"
        )
    lines += [
        "",
        "## Selected teachers",
        "",
        f"- Revision: {', '.join(selected['revision'])}",
        f"- Performance: {', '.join(selected['performance'])}",
        "",
        "Selection rule: highest mean rubric score, then valid structured output, then latency.",
        "",
        "Raw outputs and per-task scores are in the JSON report next to this file.",
        "",
    ]
    (output.parent / "assessment_report.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess local llama.cpp teacher candidates.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Teacher keys. Default: every downloaded model.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--max-tokens", type=int, default=340)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--output", type=Path, default=EVIDENCE_DIR / "teacher_assessment.json")
    args = parser.parse_args()

    specs = available_specs()
    if args.models:
        specs = [spec for spec in specs if spec.key in set(args.models)]
    if not specs:
        raise SystemExit(
            "No usable teacher models. Download the GGUF files into teacher_models/ first. "
            f"Known keys: {[spec.key for spec in TEACHER_SPECS]}"
        )

    results = []
    for index, spec in enumerate(specs):
        port = args.port + index
        print(f"[{spec.key}] assessing on port {port}…", flush=True)
        results.append(
            assess_candidate(spec, args.repeats, port, args.max_tokens, args.context, args.threads)
        )
        summary = results[-1]
        print(
            f"[{spec.key}] revision {summary['revision']['mean_total']}/12, "
            f"performance {summary['performance']['mean_total']}/6, "
            f"{summary['total_seconds']}s",
            flush=True,
        )

    revision_ranking = rank_candidates(results, "revision")
    performance_ranking = rank_candidates(results, "performance")
    selected = {
        "revision": [row["model"] for row in revision_ranking[:2]],
        "performance": [row["model"] for row in performance_ranking[:2]],
    }
    runtime = {
        "name": "llama.cpp",
        "runtime_version": "b12375",
        "server": str(ROOT / "teacher_models" / "llama.cpp" / "llama-server.exe"),
        "cpu_threads": args.threads,
        "context": args.context,
        "ollama_used": False,
    }
    path = write_report(results, selected, runtime, args.output)
    print(f"Saved assessment to {path}")
    print(f"Selected revision teachers: {selected['revision']}")
    print(f"Selected performance teachers: {selected['performance']}")


if __name__ == "__main__":
    main()
