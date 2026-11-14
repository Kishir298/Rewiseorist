"""Model runtime used by the Flask app.

The runtime keeps the trained models inside this local process. It exposes
unload/shutdown operations so the inactivity controller can release weights at
two minutes and stop the model service at seven minutes. The Flask server keeps
serving the website; the next AI request performs a cold reload.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "models"
REQUIRED_FILES = [
    "language_model.pt",
    "tokenizer.json",
    "performance_model.joblib",
    "performance_preprocessor.json",
]


class ModelUnavailable(RuntimeError):
    pass


class ModelRuntime:
    def __init__(self, models_dir: str | Path = DEFAULT_MODELS) -> None:
        self.models_dir = Path(models_dir)
        self._service = None
        self._lock = threading.RLock()
        self._stopped = False

    def missing_files(self) -> list[str]:
        return [name for name in REQUIRED_FILES if not (self.models_dir / name).exists()]

    def manifest(self) -> dict:
        path = self.models_dir / "model_manifest.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}

    def status(self) -> dict:
        missing = self.missing_files()
        manifest = self.manifest()
        with self._lock:
            loaded = self._service is not None
            stopped = self._stopped
        return {
            "ready": not missing,
            "missing_files": missing,
            "loaded": loaded,
            "stopped": stopped,
            "mode": "local",
            "language_model": manifest.get("language_model", {}),
            "performance_model": manifest.get("performance_model", {}),
            "ollama_required": False,
            "cloud_required": False,
        }

    def _load(self):
        with self._lock:
            if self._service is not None:
                return self._service
            missing = self.missing_files()
            if missing:
                raise ModelUnavailable(
                    "The trained local model files are missing: " + ", ".join(missing)
                    + ". Run the training scripts or use the packaged model files."
                )
            from ml.service import ModelService
            self._service = ModelService(self.models_dir)
            self._stopped = False
            return self._service

    def generate_revision(self, question: str, sources: list[dict],
                          personality: str = "Friendly") -> dict:
        service = self._load()
        return service.generate_revision(question, sources, personality)

    def classify_performance(self, features: dict) -> dict:
        service = self._load()
        return service.classify_performance(features)

    def warmup(self) -> dict:
        with self._lock:
            if self._service is None:
                return {"warm": False, "message": "model not loaded"}
            try:
                return self._service.warmup()
            except Exception as exc:  # pragma: no cover - defensive
                return {"warm": False, "message": str(exc)}

    def unload(self) -> dict:
        with self._lock:
            if self._service is None:
                return {"unloaded": True, "message": "model already unloaded"}
            self._service.unload()
            self._service = None
            return {"unloaded": True, "message": "resident model weights released"}

    def shutdown(self) -> dict:
        with self._lock:
            self._service = None
            self._stopped = True
            return {"stopped": True, "message": "local model service stopped"}
