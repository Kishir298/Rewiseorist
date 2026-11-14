import json
from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier

from ml.language_model import CausalLanguageModel, ModelConfig
from ml.runtime import ModelRuntime, ModelUnavailable
from ml.service import FEATURE_ORDER
from ml.tokenizer import train_tokenizer


def write_tiny_models(directory: Path) -> Path:
    tokenizer = train_tokenizer(
        ["Friction opposes motion. Water moves by osmosis through a membrane."],
        vocab_size=200, max_length=32,
    )
    config = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=32,
                         d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0)
    CausalLanguageModel(config).save(
        directory / "language_model.pt", tokenizer,
        {"version": "runtime-test-lm", "parameters": 1000},
    )
    tokenizer.save(directory / "tokenizer.json")
    rows = [[0.5] * len(FEATURE_ORDER)] * 8
    classifier = DummyClassifier(strategy="prior").fit(rows, ["Developing"] * 8)
    joblib.dump(classifier, directory / "performance_model.joblib")
    (directory / "performance_preprocessor.json").write_text(json.dumps({
        "version": "runtime-test-classifier",
        "features": FEATURE_ORDER,
        "classes": ["Developing"],
    }), encoding="utf-8")
    return directory


def test_missing_models_raise_unavailable(tmp_path):
    runtime = ModelRuntime(tmp_path / "missing")
    assert runtime.status()["ready"] is False
    with pytest.raises(ModelUnavailable):
        runtime.generate_revision("What is friction?", [])


def test_worker_lifecycle_and_cold_start(tmp_path):
    models = write_tiny_models(tmp_path / "models")
    runtime = ModelRuntime(models, startup_timeout=60, request_timeout=120)
    try:
        status = runtime.status()
        assert status["ready"] is True
        assert status["loaded"] is False
        assert status["worker"] == "subprocess"

        runtime.restart()
        assert runtime.status()["worker_running"] is True

        result = runtime.classify_performance({
            "unknown": True, "distinct_attempts": 1, "distinct_questions": 1,
        })
        assert result["category"] == "Unknown"
        assert runtime.status()["loaded"] is True

        runtime.unload()
        status = runtime.status()
        assert status["loaded"] is False
        assert status["worker_running"] is True

        runtime.warmup()
        assert runtime.status()["loaded"] is True

        runtime.shutdown()
        status = runtime.status()
        assert status["worker_running"] is False
        assert status["stopped"] is True

        # The next AI request cold-starts a new worker and still answers.
        result = runtime.classify_performance({
            "unknown": True, "distinct_attempts": 2, "distinct_questions": 2,
        })
        assert result["category"] == "Unknown"
        assert runtime.status()["worker_running"] is True
    finally:
        runtime.shutdown()
