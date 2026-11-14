import json
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.language_model import CausalLanguageModel, ModelConfig
from ml.service import FEATURE_ORDER, ModelService
from ml.tokenizer import train_tokenizer
from ml.train_character import build_pipeline, rule_label


def make_models(tmp_path: Path) -> Path:
    text = (
        "Friction is a contact force that opposes motion and produces heat. "
        "Osmosis moves water through a partially permeable membrane. "
        "Photosynthesis uses carbon dioxide and water to make glucose and oxygen. "
        "A source citation names the file and section."
    )
    tokenizer = train_tokenizer([text], vocab_size=300, max_length=48)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=48,
                         d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0)
    model = CausalLanguageModel(config)
    model.save(tmp_path / "language_model.pt", tokenizer, {"version": "test-lm", "parameters": 1234})
    tokenizer.save(tmp_path / "tokenizer.json")

    rows = []
    for index in range(12):
        rows.append([0.1 + index * 0.01] * len(FEATURE_ORDER))
    classifier = DummyClassifier(strategy="prior").fit(rows, ["Weak", "Developing", "Strong", "Unknown"] * 3)
    joblib.dump(classifier, tmp_path / "performance_model.joblib")
    (tmp_path / "performance_preprocessor.json").write_text(json.dumps({
        "version": "test-classifier",
        "features": FEATURE_ORDER,
        "classes": ["Developing", "Strong", "Unknown", "Weak"],
    }), encoding="utf-8")
    return tmp_path


def test_service_revision_and_classification(tmp_path):
    service = ModelService(make_models(tmp_path))
    result = service.generate_revision(
        "What is friction?",
        [{"original_name": "forces.md", "source_ref": "forces.md | page 2",
          "text": "Friction is a contact force that opposes motion and produces heat."}],
    )
    assert result["text"]
    assert "Source: forces.md" in result["text"]
    assert result["sources"][0]["original_name"] == "forces.md"
    assert result["model_version"] == "test-lm"
    assert result["verified"] is True

    unknown = service.classify_performance({
        "unknown": True, "distinct_attempts": 2, "distinct_questions": 2,
    })
    assert unknown["category"] == "Unknown"
    assert unknown["confidence"] is None

    classified = service.classify_performance({
        "unknown": False, "distinct_questions": 6, "distinct_attempts": 6,
        "mean_accuracy": 0.8, "recent_accuracy": 0.8, "accuracy_trend": 0.1,
        "hint_rate": 0.1, "independent_rate": 0.8, "mean_difficulty": 2.0,
        "mean_active_seconds": 50, "repeated_errors": 0,
    })
    assert classified["category"] in {"Weak", "Developing", "Strong", "Unknown"}
    assert abs(sum(classified["probabilities"].values()) - 1) < 1e-6

    assert service.warmup()["warm"] is True


def test_rule_baseline_and_pipeline_shape():
    assert rule_label({"distinct_questions": 2, "mean_accuracy": 1.0}) == "Unknown"
    assert rule_label({"distinct_questions": 6, "mean_accuracy": 0.9, "hint_rate": 0.1}) == "Strong"
    assert rule_label({"distinct_questions": 6, "mean_accuracy": 0.2, "hint_rate": 0.7}) == "Weak"
    assert rule_label({"distinct_questions": 6, "mean_accuracy": 0.55, "hint_rate": 0.2}) == "Developing"

    pipeline = build_pipeline(1.0)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)
    assert len(pipeline.named_steps["scale"].get_params()) >= 0
