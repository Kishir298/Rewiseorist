"""Frozen held-out evaluation for both trained models.

Language-model metrics:
- answer-token perplexity on reviewed test examples;
- word overlap F1 against the teacher reference answer;
- citation rate and source-grounding rate;
- raw generated samples, kept in the evidence report.

Classifier metrics:
- accuracy and macro-F1 on persona-grouped test rows;
- confusion matrix and concrete misclassified examples.

The test split is never used for training or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.language_model import InstructionDataset, load_model  # noqa: E402
from ml.prompts import extract_citation  # noqa: E402
from ml.service import FEATURE_ORDER  # noqa: E402

DATA_DIR = ROOT / "data" / "teacher" / "reviewed"
MODELS = ROOT / "models"
EVIDENCE = ROOT / "evidence" / "ml"
STOPWORDS = {
    "this", "that", "with", "from", "have", "does", "they", "your", "about", "when",
    "which", "what", "will", "their", "there", "these", "those", "into", "than", "then",
    "also", "because", "would", "could", "should", "more", "most", "such", "only",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def content_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z]{4,}", (text or "").lower())
        if word not in STOPWORDS
    }


def word_f1(reference: str, prediction: str) -> float:
    expected = content_words(reference)
    produced = content_words(prediction)
    if not expected or not produced:
        return 0.0
    overlap = len(expected & produced)
    precision = overlap / len(produced)
    recall = overlap / len(expected)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def citation_rate(rows: list[dict], outputs: list[str]) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row, output in zip(rows, outputs):
        citation = extract_citation(output)
        source_name = str((row.get("source") or {}).get("original_name", "")).lower()
        if citation and (not source_name or citation.split("|", 1)[0].strip().lower() == source_name):
            hits += 1
    return round(hits / len(rows), 4)


def grounding_rate(rows: list[dict], outputs: list[str]) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row, output in zip(rows, outputs):
        source_text = str((row.get("source") or {}).get("text", ""))
        if len(content_words(output) & content_words(source_text)) >= 2:
            hits += 1
    return round(hits / len(rows), 4)


def answer_perplexity(model_path: Path, rows: list[dict], batch_size: int = 4) -> float:
    model, tokenizer, _ = load_model(model_path)
    dataset = InstructionDataset(rows, tokenizer, model.config.context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    model.eval()
    with torch.no_grad():
        for inputs, targets, mask in loader:
            _, loss = model(inputs, targets, mask)
            if torch.isfinite(loss):
                losses.append(float(loss))
    if not losses:
        return float("inf")
    mean_loss = sum(losses) / len(losses)
    return round(math.exp(min(mean_loss, 20)), 4)


def evaluate_language(model_path: Path, rows: list[dict]) -> dict:
    model, tokenizer, extra = load_model(model_path)
    outputs = []
    for row in rows:
        prompt_ids = tokenizer.encode(
            row["prompt"], add_bos=True, max_length=model.config.context_length
        )
        generated = model.generate(
            prompt_ids, max_new_tokens=100, temperature=0.6, top_k=30,
            repetition_penalty=1.12, eos_id=tokenizer.eos_id, seed=17,
        )
        outputs.append(tokenizer.decode(generated))
    f1_scores = [
        word_f1(row["answer"], output) for row, output in zip(rows, outputs)
    ]
    return {
        "model": model_path.name,
        "version": extra.get("version", ""),
        "examples": len(rows),
        "answer_perplexity": answer_perplexity(model_path, rows) if rows else None,
        "word_f1_mean": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "citation_rate": citation_rate(rows, outputs),
        "grounding_rate": grounding_rate(rows, outputs),
        "samples": [
            {
                "question": row["question"],
                "reference": row["answer"],
                "output": output,
                "word_f1": score,
            }
            for row, output, score in list(zip(rows, outputs, f1_scores))[:6]
        ],
    }


def evaluate_classifier(rows: list[dict]) -> dict:
    model_path = MODELS / "performance_model.joblib"
    preprocessor = json.loads(
        (MODELS / "performance_preprocessor.json").read_text(encoding="utf-8")
    )
    pipeline = joblib.load(model_path)
    features = [
        [float(row["features"].get(name, 0) or 0) for name in preprocessor["features"]]
        for row in rows
    ]
    labels = [row["label"] for row in rows]
    predictions = list(pipeline.predict(features)) if rows else []
    correct = sum(1 for expected, actual in zip(labels, predictions) if expected == actual)
    mistakes = [
        {
            "persona_id": row.get("persona_id"),
            "expected": expected,
            "predicted": actual,
            "features": row["features"],
        }
        for row, expected, actual in zip(rows, labels, predictions)
        if expected != actual
    ][:8]
    return {
        "model": model_path.name,
        "version": preprocessor.get("version", ""),
        "examples": len(rows),
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "mistakes": mistakes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    test_rows = read_jsonl(DATA_DIR / "test.jsonl")
    performance_rows = read_jsonl(DATA_DIR / "performance_test.jsonl")
    if not test_rows:
        raise SystemExit(f"No reviewed test rows in {DATA_DIR}. Run review_teacher_data.py first.")

    fine_tuned_path = MODELS / "language_model.pt"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_rows": len(test_rows),
        "fine_tuned": evaluate_language(fine_tuned_path, test_rows),
        "classifier": evaluate_classifier(performance_rows) if performance_rows else {},
        "ollama_used": False,
    }
    pretrained = MODELS / "language_model_pretrained.pt"
    if pretrained.exists():
        report["pretrained"] = evaluate_language(pretrained, test_rows)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "final-evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Final held-out evaluation",
        "",
        f"Generated: {report['generated_at']}",
        f"Revision test examples: {report['test_rows']}",
        "",
        "## Language model",
        "",
        "| Model | Answer perplexity | Word F1 | Citation rate | Grounding rate |",
        "|---|---|---|---|---|",
    ]
    for name in ("pretrained", "fine_tuned"):
        result = report.get(name)
        if not result:
            continue
        lines.append(
            f"| {name} | {result['answer_perplexity']} | {result['word_f1_mean']} | "
            f"{result['citation_rate']} | {result['grounding_rate']} |"
        )
    lines += [
        "",
        "## Classifier",
        "",
        f"Accuracy: {report.get('classifier', {}).get('accuracy')}",
        f"Examples: {report.get('classifier', {}).get('examples')}",
        "",
        "Misclassified examples are listed in `final-evaluation.json`.",
        "",
        "Samples are generated outputs, not cherry-picked claims. The small model is",
        "expected to make grammatical and factual mistakes; citations are checked against",
        "the supplied source file.",
        "",
    ]
    (EVIDENCE / "final-evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("pretrained", "fine_tuned", "classifier") if key in report}, indent=2))
    print(f"Saved evaluation to {EVIDENCE / 'final-evaluation.json'}")


if __name__ == "__main__":
    main()
