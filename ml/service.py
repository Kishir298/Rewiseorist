"""Local inference service for the two models trained in this project.

The service loads a from-scratch causal language model and a scikit-learn
student-performance classifier. Both files are local; no network calls, no
Ollama and no hosted API are involved.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import joblib
import torch

from ml.language_model import load_model
from ml.prompts import (
    build_revision_prompt,
    clean_teacher_answer,
    extract_citation,
    strip_chat_markers,
)

FEATURE_ORDER = [
    "mean_accuracy",
    "recent_accuracy",
    "accuracy_trend",
    "hint_rate",
    "independent_rate",
    "mean_difficulty",
    "mean_active_seconds",
    "distinct_attempts",
    "distinct_questions",
    "repeated_errors",
]

CATEGORIES = ("Weak", "Developing", "Strong")


class ModelService:
    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self._lock = threading.RLock()
        self._warm = False
        self.language_model, self.tokenizer, self.lm_extra = load_model(
            self.models_dir / "language_model.pt"
        )
        self.classifier = joblib.load(self.models_dir / "performance_model.joblib")
        self.preprocessor = json.loads(
            (self.models_dir / "performance_preprocessor.json").read_text(encoding="utf-8")
        )
        self.feature_order = self.preprocessor.get("features", FEATURE_ORDER)
        self.lm_version = str(self.lm_extra.get("version", "local-language-model"))
        self.classifier_version = str(self.preprocessor.get("version", "local-classifier"))

    # ------------------------------------------------------------------
    # Revision chat
    # ------------------------------------------------------------------
    def _pick_sources(self, sources: list[dict]) -> list[dict]:
        cleaned = []
        for source in sources[:4]:
            cleaned.append({
                "original_name": source.get("original_name") or source.get("file") or "source",
                "source_ref": source.get("source_ref") or source.get("page_ref") or "",
                "text": str(source.get("text", ""))[:1400],
            })
        return cleaned

    @staticmethod
    def _overlap(answer: str, source_text: str) -> int:
        answer_words = {
            word for word in re.findall(r"[a-z]{4,}", answer.lower())
            if word not in {"this", "that", "with", "from", "have", "does", "they", "your", "about", "when"}
        }
        source_words = set(re.findall(r"[a-z]{4,}", source_text.lower()))
        return len(answer_words & source_words)

    def generate_revision(self, question: str, sources: list[dict],
                          personality: str = "Friendly") -> dict:
        with self._lock:
            picked = self._pick_sources(sources)
            prompt = build_revision_prompt(question, picked, personality)
            prompt_ids = self.tokenizer.encode(
                prompt, add_bos=True, max_length=self.language_model.config.context_length
            )
            generated = self.language_model.generate(
                prompt_ids,
                max_new_tokens=110,
                temperature=0.65,
                top_k=30,
                repetition_penalty=1.12,
                eos_id=self.tokenizer.eos_id,
            )
            text = clean_teacher_answer(self.tokenizer.decode(generated))
            self._warm = True

        used = [source for source in picked if self._overlap(text, source["text"]) >= 2]
        citations = []
        for source in used:
            label = source["original_name"]
            ref = source["source_ref"] or "source"
            citations.append(f"Source: {label} | {ref}")
        if not extract_citation(text) and picked:
            source = (used or picked)[0]
            ref = source["source_ref"] or "source"
            text = f"{text}\nSource: {source['original_name']} | {ref}".strip()
            if not used:
                used = [source]
        elif extract_citation(text) and not used and picked:
            used = [picked[0]]

        answer_sources = used or picked[:1]
        return {
            "text": text,
            "sources": answer_sources,
            "model_version": self.lm_version,
            "verified": bool(answer_sources),
            "model": {
                "name": "Rewiseorist language model",
                "version": self.lm_version,
                "parameters": int(self.lm_extra.get("parameters", 0)) or None,
                "local": True,
            },
        }

    # ------------------------------------------------------------------
    # Student-performance classification
    # ------------------------------------------------------------------
    def _feature_vector(self, features: dict) -> list[float]:
        vector = []
        for name in self.feature_order:
            value = features.get(name, 0)
            try:
                vector.append(float(value))
            except (TypeError, ValueError):
                vector.append(0.0)
        return vector

    def explain(self, category: str, features: dict) -> str:
        accuracy = float(features.get("mean_accuracy", 0))
        trend = float(features.get("accuracy_trend", 0))
        hint_rate = float(features.get("hint_rate", 0))
        independent = float(features.get("independent_rate", 0))
        attempts = int(features.get("distinct_questions", 0))
        trend_text = "improving" if trend > 0.08 else "declining" if trend < -0.08 else "steady"
        return (
            f"Across {attempts} distinct checked questions, mean accuracy is "
            f"{accuracy:.0%} and the recent trend is {trend_text}. "
            f"Hints were used on {hint_rate:.0%} of attempts and "
            f"{independent:.0%} were answered independently. This evidence fits {category}."
        )

    def classify_performance(self, features: dict) -> dict:
        if features.get("unknown") or int(features.get("distinct_questions", 0)) < 3:
            return {
                "category": "Unknown",
                "reason": (
                    "Fewer than three distinct checked questions are available, "
                    "so there is not enough evidence for a category."
                ),
                "confidence": None,
                "model_version": self.classifier_version,
                "unknown": True,
            }
        vector = self._feature_vector(features)
        with self._lock:
            probabilities = self.classifier.predict_proba([vector])[0]
            classes = list(getattr(self.classifier, "classes_", CATEGORIES))
        best_index = int(torch.tensor(probabilities).argmax())
        category = str(classes[best_index])
        confidence = float(probabilities[best_index])
        reason = self.explain(category, features)
        return {
            "category": category,
            "reason": reason,
            "confidence": round(confidence, 4),
            "probabilities": {
                str(label): round(float(value), 4) for label, value in zip(classes, probabilities)
            },
            "model_version": self.classifier_version,
            "unknown": False,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def warmup(self) -> dict:
        with self._lock:
            tokens = torch.tensor([[self.tokenizer.bos_id]], dtype=torch.long)
            with torch.no_grad():
                self.language_model(tokens)
            self._warm = True
        return {"warm": True, "model_version": self.lm_version}

    def unload(self) -> None:
        with self._lock:
            self.language_model = None
            self.classifier = None
            self._warm = False


def expected_prompt(prompt: str) -> str:
    """Expose the chat-body transformation for tests and debugging."""
    return strip_chat_markers(prompt)
