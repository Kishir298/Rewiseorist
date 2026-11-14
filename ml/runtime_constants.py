"""Shared runtime exceptions and constants.

Kept separate so the subprocess worker can import them without importing the
Flask-facing runtime module.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "models"
REQUIRED_FILES = [
    "language_model.pt",
    "tokenizer.json",
    "performance_model.joblib",
    "performance_preprocessor.json",
]


class ModelUnavailable(RuntimeError):
    """Raised when the trained local model files cannot answer a request."""
