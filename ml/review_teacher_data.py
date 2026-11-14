"""Validate and group teacher outputs into reproducible reviewed datasets.

Checks:
- revision answers are non-empty, grounded in the supplied source and cite it;
- performance labels are valid and obey the three-distinct-question rule;
- duplicates are removed;
- splits are grouped by topic (language model) and persona (classifier).

Writes ``data/teacher/reviewed/*.jsonl`` and evidence reports.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.assess_candidates import extract_json_object  # noqa: E402
from ml.prompts import build_revision_prompt, extract_citation  # noqa: E402
from ml.synthetic_students import build_personas, group_split  # noqa: E402

RAW_DIR = ROOT / "data" / "teacher" / "raw"
REVIEWED_DIR = ROOT / "data" / "teacher" / "reviewed"
EVIDENCE_DIR = ROOT / "evidence" / "ml"
VALID_CATEGORIES = {"Weak", "Developing", "Strong", "Unknown"}

STOPWORDS = {
    "this", "that", "with", "from", "have", "does", "they", "your", "about", "when",
    "which", "what", "will", "their", "there", "these", "those", "into", "than", "then",
    "also", "because", "would", "could", "should", "more", "most", "such", "only",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            print(f"Skipping malformed line {number} in {path}: {exc}", flush=True)
    return rows


def _content_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z]{4,}", (text or "").lower())
        if word not in STOPWORDS
    }


def review_revision(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        reasons = []
        answer = str(row.get("answer", "")).strip()
        source = row.get("source") or {}
        source_name = str(source.get("original_name", "")).lower()
        source_ref = str(source.get("source_ref") or source_name or "sample")
        if row.get("error"):
            reasons.append("teacher error")
        if len(answer) < 40:
            reasons.append("answer too short")
        citation = extract_citation(answer)
        if not citation:
            answer = f"{answer}\nSource: {source_name} | {source_ref.split('|')[-1].strip()}".strip()
            citation = extract_citation(answer)
            row = {**row, "citation_repaired": True}
        cited_name = citation.split("|", 1)[0].strip().lower() if citation else ""
        if citation and source_name and cited_name != source_name:
            reasons.append("citation names a different source")
        overlap = len(_content_words(answer) & _content_words(source.get("text", "")))
        if overlap < 2:
            reasons.append("answer is not grounded in the source")

        key = (str(row.get("question_id", "")), re.sub(r"\s+", " ", answer.lower())[:240])
        if key in seen:
            reasons.append("duplicate answer")
        if reasons:
            rejected.append({**row, "reasons": reasons})
            continue
        seen.add(key)
        accepted.append({
            "id": row["id"],
            "task": "revision",
            "topic": row.get("topic", "General"),
            "question_id": row.get("question_id", ""),
            "question": row["question"],
            "source": source,
            "prompt": build_revision_prompt(row["question"], [source], "Friendly"),
            "answer": answer.rstrip(),
            "citation": citation,
            "teacher": row.get("teacher", ""),
            "teacher_model": row.get("model_name", ""),
            "grounding_overlap": overlap,
            "citation_repaired": bool(row.get("citation_repaired")),
        })
    return accepted, rejected


def review_performance(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    best_by_persona: dict[str, dict] = {}

    for row in rows:
        reasons = []
        payload = extract_json_object(row.get("raw_answer", ""))
        category = str(payload.get("category", "")).strip().title()
        reason = str(payload.get("reason", "")).strip()
        if not category:
            match = re.search(r"\b(Weak|Developing|Strong|Unknown)\b", row.get("raw_answer", ""), re.I)
            category = match.group(1).title() if match else ""
        if category not in VALID_CATEGORIES:
            reasons.append("missing or invalid category")
        features = row.get("features") or {}
        distinct = int(features.get("distinct_questions", features.get("distinct_attempts", 0)))
        if distinct < 3 and category in {"Weak", "Developing", "Strong"}:
            reasons.append("labelled a category with fewer than three distinct questions")
        if len(reason) < 20:
            reasons.append("reason missing or too short")
        if row.get("error"):
            reasons.append("teacher error")
        if reasons:
            rejected.append({**row, "category": category, "reasons": reasons})
            continue
        cleaned = {
            "id": row["id"],
            "task": "performance",
            "persona_id": row.get("persona_id", ""),
            "topic": row.get("topic", "General"),
            "intended_label": row.get("intended_label", ""),
            "features": features,
            "history": row.get("history", ""),
            "label": category,
            "reason": reason,
            "teacher": row.get("teacher", ""),
            "teacher_model": row.get("model_name", ""),
            "parsed_json": bool(payload),
        }
        persona_id = cleaned["persona_id"]
        prior = best_by_persona.get(persona_id)
        # Prefer parsed JSON, then a longer evidence-based reason.
        if prior is None or (cleaned["parsed_json"], len(cleaned["reason"])) > (
            prior["parsed_json"], len(prior["reason"])
        ):
            if prior is not None:
                rejected.append({**prior, "reasons": ["duplicate persona; kept better label"]})
            best_by_persona[persona_id] = cleaned
        else:
            rejected.append({**cleaned, "reasons": ["duplicate persona; kept better label"]})

    accepted = list(best_by_persona.values())
    return accepted, rejected


def topic_splits(topics: list[str], seed: int = 23) -> dict[str, set[str]]:
    ordered = sorted(set(topics))
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    train_count = max(1, int(count * 0.7))
    val_count = max(1, int(count * 0.15)) if count >= 3 else 0
    return {
        "train": set(ordered[:train_count]),
        "val": set(ordered[train_count:train_count + val_count]),
        "test": set(ordered[train_count + val_count:]),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=15)
    args = parser.parse_args()

    revision_rows = read_jsonl(RAW_DIR / "revision.jsonl")
    performance_rows = read_jsonl(RAW_DIR / "performance.jsonl")
    if not revision_rows and not performance_rows:
        raise SystemExit(f"No raw teacher data in {RAW_DIR}. Run generate_teacher_data.py first.")

    revision, rejected_revision = review_revision(revision_rows)
    performance, rejected_performance = review_performance(performance_rows)

    splits = topic_splits([row["topic"] for row in revision])
    revision_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    unassigned = []
    for row in revision:
        split = next((name for name, topics in splits.items() if row["topic"] in topics), None)
        if split:
            revision_by_split[split].append(row)
        else:
            unassigned.append(row)
    # Keep every reviewed row in at least one split.
    revision_by_split["train"].extend(unassigned)

    persona_split = {}
    for name, personas in group_split(build_personas(per_label=args.per_label)).items():
        for persona in personas:
            persona_split[persona["persona_id"]] = name
    performance_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in performance:
        performance_by_split[persona_split.get(row["persona_id"], "train")].append(row)

    for split in ("train", "val", "test"):
        write_jsonl(REVIEWED_DIR / f"{split}.jsonl", revision_by_split[split])
        write_jsonl(REVIEWED_DIR / f"performance_{split}.jsonl", performance_by_split[split])

    agreement = Counter()
    for row in performance:
        agreement["agree" if row["label"] == row["intended_label"] else "disagree"] += 1
    intent_distribution = Counter(row["intended_label"] for row in performance)
    label_distribution = Counter(row["label"] for row in performance)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "revision": {
            "raw": len(revision_rows),
            "accepted": len(revision),
            "rejected": len(rejected_revision),
            "rejection_reasons": dict(Counter(
                reason for row in rejected_revision for reason in row["reasons"]
            )),
            "split_counts": {name: len(rows) for name, rows in revision_by_split.items()},
            "topics": {name: sorted(topics) for name, topics in splits.items()},
            "teachers": sorted({row["teacher"] for row in revision}),
            "citation_repairs": sum(1 for row in revision if row["citation_repaired"]),
        },
        "performance": {
            "raw": len(performance_rows),
            "accepted": len(performance),
            "rejected": len(rejected_performance),
            "rejection_reasons": dict(Counter(
                reason for row in rejected_performance for reason in row["reasons"]
            )),
            "split_counts": {name: len(rows) for name, rows in performance_by_split.items()},
            "label_distribution": dict(label_distribution),
            "intended_distribution": dict(intent_distribution),
            "teacher_agreement": dict(agreement),
            "teachers": sorted({row["teacher"] for row in performance}),
        },
        "ollama_used": False,
    }
    REVIEWED_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEWED_DIR / "review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Teacher data review",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Revision",
        "",
        f"- Raw rows: {summary['revision']['raw']}",
        f"- Accepted: {summary['revision']['accepted']}",
        f"- Rejected: {summary['revision']['rejected']}",
        f"- Split counts: {summary['revision']['split_counts']}",
        f"- Teachers: {', '.join(summary['revision']['teachers']) or 'none'}",
        f"- Citation repairs: {summary['revision']['citation_repairs']}",
        "",
        "## Performance",
        "",
        f"- Raw rows: {summary['performance']['raw']}",
        f"- Accepted: {summary['performance']['accepted']}",
        f"- Rejected: {summary['performance']['rejected']}",
        f"- Split counts: {summary['performance']['split_counts']}",
        f"- Teacher labels: {summary['performance']['label_distribution']}",
        f"- Agreement with intended design label: {summary['performance']['teacher_agreement']}",
        "",
        "Rejected rows and reasons are recorded in `review_summary.json` counts; raw rows stay",
        "under the ignored `data/teacher/raw/` directory so the review is reproducible.",
        "",
    ]
    (EVIDENCE_DIR / "review_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Reviewed data written to {REVIEWED_DIR}")


if __name__ == "__main__":
    main()
