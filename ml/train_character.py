"""Train the student-performance classifier on reviewed teacher labels.

Compares a regularised logistic-regression classifier with two baselines
(most-frequent class and a transparent rule) on a held-out, persona-grouped
split, then saves the model and preprocessing metadata used by ``ml/service.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.service import FEATURE_ORDER  # noqa: E402

DATA_DIR = ROOT / "data" / "teacher" / "reviewed"
MODELS = ROOT / "models"
EVIDENCE = ROOT / "evidence" / "ml"
CATEGORIES = ("Weak", "Developing", "Strong", "Unknown")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def matrix(rows: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    features = []
    labels = []
    for row in rows:
        values = row.get("features", {})
        features.append([float(values.get(name, 0) or 0) for name in FEATURE_ORDER])
        labels.append(str(row["label"]))
    return np.array(features, dtype=float), labels, [row.get("persona_id", "") for row in rows]


def rule_label(features: dict) -> str:
    if int(features.get("distinct_questions", 0)) < 3:
        return "Unknown"
    accuracy = float(features.get("mean_accuracy", 0))
    hint_rate = float(features.get("hint_rate", 0))
    if accuracy >= 0.75 and hint_rate <= 0.35:
        return "Strong"
    if accuracy < 0.42:
        return "Weak"
    return "Developing"


def build_pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=c_value, max_iter=2000, class_weight="balanced", random_state=42,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="rewiseorist-classifier-1.0")
    args = parser.parse_args()

    train_rows = read_jsonl(DATA_DIR / "performance_train.jsonl")
    val_rows = read_jsonl(DATA_DIR / "performance_val.jsonl")
    test_rows = read_jsonl(DATA_DIR / "performance_test.jsonl")
    if not train_rows or not test_rows:
        raise SystemExit(
            f"Missing reviewed performance data in {DATA_DIR}. "
            "Run generate_teacher_data.py performance and review_teacher_data.py first."
        )

    x_train, y_train, _ = matrix(train_rows)
    x_val, y_val, _ = matrix(val_rows)
    x_test, y_test, _ = matrix(test_rows)
    print("Train labels:", dict(Counter(y_train)))
    print("Val labels:", dict(Counter(y_val)))
    print("Test labels:", dict(Counter(y_test)))

    # Choose regularisation on validation macro-F1.
    best_c, best_score = 1.0, -1.0
    for c_value in (0.05, 0.1, 0.3, 1.0, 3.0, 10.0):
        if not val_rows:
            break
        model = build_pipeline(c_value)
        model.fit(x_train, y_train)
        score = f1_score(y_val, model.predict(x_val), average="macro", zero_division=0)
        print(f"C={c_value:<4} val macro-F1={score:.4f}")
        if score > best_score:
            best_c, best_score = c_value, score

    # Refit on train+val for the final model; the test set stays untouched.
    x_fit = np.vstack([x_train, x_val]) if val_rows else x_train
    y_fit = list(y_train) + list(y_val)
    model = build_pipeline(best_c)
    model.fit(x_fit, y_fit)
    predictions = model.predict(x_test)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_fit, y_fit)
    dummy_predictions = dummy.predict(x_test)

    rule_predictions = [rule_label(row["features"]) for row in test_rows]

    metrics = {
        "model": {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "macro_f1": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4),
            "report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=list(CATEGORIES)).tolist(),
            "labels": list(CATEGORIES),
        },
        "baseline_most_frequent": {
            "accuracy": round(float(accuracy_score(y_test, dummy_predictions)), 4),
            "macro_f1": round(float(f1_score(y_test, dummy_predictions, average="macro", zero_division=0)), 4),
        },
        "baseline_rule": {
            "accuracy": round(float(accuracy_score(y_test, rule_predictions)), 4),
            "macro_f1": round(float(f1_score(y_test, rule_predictions, average="macro", zero_division=0)), 4),
        },
    }

    if len(set(y_train)) > 1 and min(Counter(y_train).values()) >= 2:
        folds = StratifiedKFold(n_splits=min(5, min(Counter(y_train).values())), shuffle=True, random_state=7)
        cv_scores = cross_val_score(
            build_pipeline(best_c), x_train, y_train, cv=folds, scoring="f1_macro"
        )
        metrics["cross_validation"] = {
            "folds": int(folds.get_n_splits()),
            "macro_f1_mean": round(float(cv_scores.mean()), 4),
            "macro_f1_std": round(float(cv_scores.std()), 4),
        }

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "performance_model.joblib")
    preprocessor = {
        "version": args.version,
        "features": FEATURE_ORDER,
        "classes": list(model.named_steps["model"].classes_),
        "categories": list(CATEGORIES),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_examples": len(y_fit),
        "test_examples": len(y_test),
        "selected_c": best_c,
        "metrics": metrics,
        "label_distribution": dict(Counter(y_fit)),
        "teachers": sorted({row.get("teacher", "") for row in train_rows + val_rows + test_rows}),
        "unknown_rule": "fewer than three distinct checked questions always returns Unknown",
        "rule_baseline": "accuracy >= .75 and hint rate <= .35 => Strong; accuracy < .42 => Weak; else Developing",
        "local_only": True,
        "ollama_used": False,
    }
    (MODELS / "performance_preprocessor.json").write_text(
        json.dumps(preprocessor, indent=2), encoding="utf-8"
    )

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "classifier-metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    lines = [
        "# Student-performance classifier",
        "",
        f"Version: {args.version}",
        f"Training examples (train+val): {len(y_fit)}",
        f"Test examples: {len(y_test)}",
        f"Selected regularisation C: {best_c}",
        "",
        "| Model | Test accuracy | Test macro-F1 |",
        "|---|---|---|",
        f"| Logistic regression | {metrics['model']['accuracy']} | {metrics['model']['macro_f1']} |",
        f"| Most-frequent baseline | {metrics['baseline_most_frequent']['accuracy']} | "
        f"{metrics['baseline_most_frequent']['macro_f1']} |",
        f"| Rule baseline | {metrics['baseline_rule']['accuracy']} | {metrics['baseline_rule']['macro_f1']} |",
        "",
        "Unknown is returned before the classifier when fewer than three distinct checked",
        "questions exist. Splits are grouped by synthetic persona, so a persona never appears",
        "in more than one split.",
        "",
    ]
    (EVIDENCE / "classifier-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODELS / 'performance_model.joblib'}")


if __name__ == "__main__":
    main()
