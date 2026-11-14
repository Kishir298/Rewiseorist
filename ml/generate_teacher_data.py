"""Generate teacher revision answers and performance labels with llama.cpp.

Raw outputs are written to ``data/teacher/raw/`` (ignored by Git) and can be
reviewed with ``ml/review_teacher_data.py``. Selection uses the saved assessment
report when it exists; otherwise every downloaded teacher is used.

Examples:
    python ml/generate_teacher_data.py revision --models qwen25-3b llama32-3b
    python ml/generate_teacher_data.py performance --models qwen25-3b
"""

from __future__ import annotations

import argparse
import json
import random
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
    clean_teacher_answer,
    extract_citation,
)
from ml.synthetic_students import build_personas  # noqa: E402
from ml.teacher_client import (  # noqa: E402
    TEACHER_SPECS,
    LlamaCppServer,
    TeacherSpec,
    available_specs,
)

RAW_DIR = ROOT / "data" / "teacher" / "raw"
REVIEWED_DIR = ROOT / "data" / "teacher" / "reviewed"
EVIDENCE_DIR = ROOT / "evidence" / "ml"
SAMPLE_DIR = ROOT / "samples" / "resources"


def selected_models(task: str, override: list[str] | None = None) -> list[TeacherSpec]:
    specs = available_specs()
    if override:
        by_key = {spec.key: spec for spec in specs}
        missing = [key for key in override if key not in by_key]
        if missing:
            raise SystemExit(f"Teacher(s) not available: {missing}")
        return [by_key[key] for key in override]
    report = EVIDENCE_DIR / "teacher_assessment.json"
    keys: list[str] = []
    if report.exists():
        payload = json.loads(report.read_text(encoding="utf-8"))
        keys = payload.get("selected", {}).get(task, [])
    selected = [spec for spec in specs if spec.key in keys]
    return selected or specs


def existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(str(json.loads(line).get("id", "")))
        except ValueError:
            continue
    return ids


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def revision_jobs(limit: int | None = None) -> list[dict]:
    keys = json.loads((ROOT / "samples" / "quiz_keys.json").read_text(encoding="utf-8"))
    jobs = []
    for topic, questions in keys.items():
        for question in questions:
            source_name = question.get("source") or f"{topic}.md"
            source_path = SAMPLE_DIR / source_name
            if not source_path.exists():
                continue
            text = source_path.read_text(encoding="utf-8", errors="replace")[:1600]
            jobs.append({
                "question_id": question["id"],
                "topic": topic,
                "question": question["question"],
                "source": {
                    "original_name": source_name,
                    "source_ref": f"{source_name} | sample",
                    "text": text,
                },
            })
    random.Random(11).shuffle(jobs)
    return jobs[:limit] if limit else jobs


def run_revision(spec: TeacherSpec, port: int, limit: int | None, max_tokens: int) -> int:
    target = RAW_DIR / "revision.jsonl"
    done = existing_ids(target)
    jobs = revision_jobs(limit)
    written = 0
    with LlamaCppServer(spec, port=port, threads=10, context=4096) as server:
        for index, job in enumerate(jobs, start=1):
            row_id = f"revision-{spec.key}-{job['question_id']}"
            if row_id in done:
                continue
            prompt = build_revision_prompt(job["question"], [job["source"]], "Friendly")
            try:
                response = server.chat(prompt, temperature=0.2, max_tokens=max_tokens)
                answer = clean_teacher_answer(response["text"])
                error = ""
            except Exception as exc:
                answer = ""
                error = str(exc)
                response = {"elapsed_seconds": 0}
            citation = extract_citation(answer)
            append_jsonl(target, [{
                "id": row_id,
                "task": "revision",
                "teacher": spec.key,
                "model_name": spec.name,
                "licence": spec.licence,
                "question_id": job["question_id"],
                "topic": job["topic"],
                "question": job["question"],
                "source": job["source"],
                "prompt": prompt,
                "answer": answer,
                "citation": citation,
                "elapsed_seconds": response.get("elapsed_seconds"),
                "error": error,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }])
            written += 1
            print(
                f"[revision {spec.key}] {index}/{len(jobs)} {job['question_id']} "
                f"({len(answer)} chars, {response.get('elapsed_seconds')}s)",
                flush=True,
            )
    print(f"[revision {spec.key}] wrote {written} row(s) to {target}", flush=True)
    return written


def run_performance(spec: TeacherSpec, port: int, limit: int | None,
                    max_tokens: int, per_label: int = 15) -> int:
    target = RAW_DIR / "performance.jsonl"
    done = existing_ids(target)
    personas = build_personas(per_label=per_label)
    if limit:
        personas = personas[:limit]
    written = 0
    with LlamaCppServer(spec, port=port, threads=10, context=4096) as server:
        for index, persona in enumerate(personas, start=1):
            row_id = f"performance-{spec.key}-{persona['persona_id']}"
            if row_id in done:
                continue
            prompt = build_performance_prompt(persona["features"], persona["history"])
            try:
                response = server.chat(prompt, temperature=0.0, max_tokens=max_tokens)
                raw_answer = clean_teacher_answer(response["text"])
                error = ""
            except Exception as exc:
                raw_answer = ""
                error = str(exc)
                response = {"elapsed_seconds": 0}
            append_jsonl(target, [{
                "id": row_id,
                "task": "performance",
                "teacher": spec.key,
                "model_name": spec.name,
                "persona_id": persona["persona_id"],
                "topic": persona["topic"],
                "intended_label": persona["intended_label"],
                "features": persona["features"],
                "history": persona["history"],
                "prompt": prompt,
                "raw_answer": raw_answer,
                "elapsed_seconds": response.get("elapsed_seconds"),
                "error": error,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }])
            written += 1
            print(
                f"[performance {spec.key}] {index}/{len(personas)} {persona['persona_id']} "
                f"({persona['intended_label']}, {response.get('elapsed_seconds')}s)",
                flush=True,
            )
    print(f"[performance {spec.key}] wrote {written} row(s) to {target}", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["revision", "performance"])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-label", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()

    specs = selected_models(args.stage, args.models)
    if not specs:
        raise SystemExit(
            "No teacher models available. Download the GGUF files first. "
            f"Known keys: {[spec.key for spec in TEACHER_SPECS]}"
        )
    print(f"Using teachers: {[spec.key for spec in specs]}", flush=True)
    started = time.time()
    total = 0
    for offset, spec in enumerate(specs):
        port = args.port + offset
        print(f"Starting {spec.name} on port {port}", flush=True)
        if args.stage == "revision":
            total += run_revision(spec, port, args.limit, args.max_tokens)
        else:
            total += run_performance(spec, port, args.limit, args.max_tokens, args.per_label)
    print(f"Generated {total} row(s) in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
