"""Local Ollama client for Rewiseorist revision chat.

Revision chat now answers through Ollama's local ``/api/chat`` endpoint with
the pretrained SmolLM2 360M model. This replaces the from-scratch language
model as the revision assistant for this delivery; the trained checkpoints in
``models/`` are preserved but are no longer used to answer questions.

Only localhost is contacted, the model tag is fixed, and the harness handles
timeouts, connection failures, missing models, empty answers and source
references so the Flask layer never sees a raw HTTP failure.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Sequence

import requests

from ml.prompts import build_ollama_revision_messages, extract_citation

# ---------------------------------------------------------------------------
# Explicit configuration for this delivery
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "smollm2:360m"
MODEL_LABEL = "SmolLM2 360M"
NUM_CTX = 2048
MAX_TOKENS = 200
TEMPERATURE = 0.2
KEEP_ALIVE = "5m"
CONNECT_TIMEOUT = 3.05
READ_TIMEOUT = 90.0
MAX_SOURCES = 3
MAX_SOURCE_CHARS = 600

_STOPWORDS = {
    "this", "that", "with", "from", "have", "does", "they", "your", "about",
    "when", "which", "what", "were", "been", "would", "could", "there", "their",
    "into", "than", "then", "these", "those", "will", "also", "just", "more",
    "most", "some", "such", "only", "other", "over", "under", "after", "before",
}


class OllamaError(RuntimeError):
    """Base class for Ollama harness failures."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class OllamaUnavailable(OllamaError):
    """The Ollama daemon is not reachable."""


class OllamaModelMissing(OllamaError):
    """Ollama is running but the configured model is not installed."""


class OllamaTimeout(OllamaError):
    """Ollama accepted the request but did not answer in time."""


class OllamaEmptyAnswer(OllamaError):
    """Ollama answered with empty content."""


class OllamaHTTPError(OllamaError):
    """Ollama returned an unexpected HTTP status or body."""


def pick_sources(sources: Sequence[dict] | None,
                 max_sources: int = MAX_SOURCES,
                 max_chars: int = MAX_SOURCE_CHARS) -> list[dict]:
    """Keep only the most relevant retrieved passages, in ranked order."""
    picked: list[dict] = []
    seen: set[str] = set()
    for source in list(sources or []):
        if len(picked) >= max_sources:
            break
        reference = str(
            source.get("source_ref") or source.get("page_ref")
            or source.get("original_name") or source.get("file") or ""
        ).strip()
        if reference and reference.casefold() in seen:
            continue
        if reference:
            seen.add(reference.casefold())
        text = re.sub(r"\s+", " ", str(source.get("text", ""))).strip()
        if not text:
            continue
        picked.append({
            "original_name": source.get("original_name") or source.get("file") or "source",
            "source_ref": source.get("source_ref") or source.get("page_ref") or "",
            "text": text[:max_chars],
        })
    return picked


def lexical_overlap(answer: str, source_text: str) -> int:
    """Count shared content words; two or more means the answer uses the source."""
    answer_words = {
        word for word in re.findall(r"[a-z]{4,}", (answer or "").lower())
        if word not in _STOPWORDS
    }
    source_words = set(re.findall(r"[a-z]{4,}", (source_text or "").lower()))
    return len(answer_words & source_words)


def validate_citation(text: str, picked: list[dict]) -> tuple[str, dict | None]:
    """Return the answer and the cited source, or drop a citation that names
    no real retrieved source."""
    citation = extract_citation(text)
    if not citation or not picked:
        return text, None
    name, _, ref = citation.partition("|")
    name = name.strip().casefold()
    ref = ref.strip().casefold()
    for source in picked:
        source_name = str(source.get("original_name", "")).strip().casefold()
        source_ref = str(source.get("source_ref", "")).strip().casefold()
        if name and (name == source_name or name in source_name or source_name in name):
            return text, source
        if ref and source_ref and (ref in source_ref or source_ref in ref):
            return text, source
    cleaned = re.sub(r"\n?\s*Source:\s*[^\n]*", "", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip(), None


class OllamaClient:
    """Thin, testable client for Ollama's local HTTP API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = MODEL_NAME,
                 num_ctx: int = NUM_CTX, max_tokens: int = MAX_TOKENS,
                 temperature: float = TEMPERATURE, read_timeout: float = READ_TIMEOUT,
                 keep_alive: str | int = KEEP_ALIVE) -> None:
        self.base_url = str(base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = str(model or MODEL_NAME)
        self.num_ctx = int(num_ctx)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.read_timeout = float(read_timeout)
        self.keep_alive = keep_alive
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, config) -> "OllamaClient":
        """Build a client from Flask configuration (explicit model name)."""
        return cls(
            base_url=config.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            model=config.get("OLLAMA_MODEL", MODEL_NAME),
            num_ctx=config.get("OLLAMA_NUM_CTX", NUM_CTX),
            max_tokens=config.get("OLLAMA_MAX_TOKENS", MAX_TOKENS),
            temperature=config.get("OLLAMA_TEMPERATURE", TEMPERATURE),
            read_timeout=config.get("OLLAMA_READ_TIMEOUT", READ_TIMEOUT),
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> dict:
        info = {
            "available": False,
            "model_present": False,
            "model": self.model,
            "label": MODEL_LABEL,
            "base_url": self.base_url,
            "num_ctx": self.num_ctx,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "ollama_required": True,
            "cloud_required": False,
            "error": None,
        }
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=(CONNECT_TIMEOUT, 3.0)
            )
        except requests.Timeout:
            info["error"] = "Ollama did not respond. Is the Ollama app running?"
            return info
        except requests.RequestException:
            info["error"] = (
                "Ollama is not running. Start it with `ollama serve` "
                "or open the Ollama app."
            )
            return info
        if response.status_code != 200:
            info["error"] = f"Ollama returned HTTP {response.status_code} on /api/tags."
            return info
        try:
            names = [
                str(model.get("name", ""))
                for model in response.json().get("models", [])
            ]
        except (ValueError, AttributeError):
            info["error"] = "Ollama returned an unreadable model list."
            return info
        info["available"] = True
        info["model_present"] = self.model in names
        if not info["model_present"]:
            info["error"] = (
                f"The {self.model} model is not installed. "
                f"Run `ollama pull {self.model}` and retry."
            )
        return info

    # ------------------------------------------------------------------
    # Revision chat
    # ------------------------------------------------------------------
    def generate_revision(self, question: str, sources: Sequence[dict] | None,
                          personality: str = "Friendly") -> dict:
        picked = pick_sources(sources)
        messages = build_ollama_revision_messages(
            question, picked, personality, MAX_SOURCES, MAX_SOURCE_CHARS
        )
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        started = time.time()
        with self._lock:
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat", json=payload,
                    timeout=(CONNECT_TIMEOUT, self.read_timeout),
                )
            except requests.Timeout as exc:
                raise OllamaTimeout(
                    f"{MODEL_LABEL} did not answer within "
                    f"{int(self.read_timeout)} seconds. Try a shorter question."
                ) from exc
            except requests.RequestException as exc:
                raise OllamaUnavailable(
                    "Ollama is not running. Start it with `ollama serve` "
                    "(or open the Ollama app), then retry."
                ) from exc
        latency = round(time.time() - started, 2)

        body_text = (response.text or "")[:300]
        if response.status_code == 404 or "not found" in body_text.lower():
            raise OllamaModelMissing(
                f"The {self.model} model is not installed. "
                f"Run `ollama pull {self.model}` and retry.",
                status=response.status_code,
            )
        if not response.ok:
            raise OllamaHTTPError(
                f"Ollama returned HTTP {response.status_code}: {body_text}",
                status=response.status_code,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaHTTPError(
                f"Ollama returned an unreadable response: {body_text}"
            ) from exc
        if isinstance(data, dict) and data.get("error"):
            error = str(data["error"])
            if "not found" in error.lower():
                raise OllamaModelMissing(
                    f"The {self.model} model is not installed. "
                    f"Run `ollama pull {self.model}` and retry.",
                    status=response.status_code,
                )
            raise OllamaHTTPError(f"Ollama error: {error}")

        text = str((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise OllamaEmptyAnswer(
                f"{MODEL_LABEL} returned an empty answer. Try rephrasing the question."
            )
        text, cited = validate_citation(text, picked)
        if not text:
            raise OllamaEmptyAnswer(
                f"{MODEL_LABEL} returned no usable answer. Try rephrasing the question."
            )

        overlaps = [lexical_overlap(text, source["text"]) for source in picked]
        best_overlap = max(overlaps) if overlaps else 0
        grounded = bool(picked) and (best_overlap >= 2 or (cited is not None and best_overlap >= 1))
        if grounded and cited is None:
            best = picked[overlaps.index(best_overlap)]
            reference = best.get("source_ref") or "source"
            text = f"{text}\nSource: {best['original_name']} | {reference}".strip()
            cited = best

        if not picked:
            status = "no-source"
        else:
            status = "verified" if grounded else "unverified"
        answer_sources = picked
        return {
            "text": text,
            "sources": answer_sources,
            "verified": grounded,
            "grounding": {
                "status": status,
                "overlap": int(best_overlap),
                "cited": cited is not None,
            },
            "model_version": self.model,
            "model": {
                "name": MODEL_LABEL,
                "version": self.model,
                "provider": "ollama",
                "local": True,
            },
            "latency_s": latency,
            "usage": {
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            },
        }

    # ------------------------------------------------------------------
    # Lifecycle helpers used by the model status endpoints
    # ------------------------------------------------------------------
    def warmup(self) -> dict:
        status = self.status()
        if not status["available"]:
            raise OllamaUnavailable(str(status["error"] or "Ollama is not running."))
        if not status["model_present"]:
            raise OllamaModelMissing(str(status["error"]))
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx, "num_predict": 4,
                        "temperature": self.temperature},
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload,
                timeout=(CONNECT_TIMEOUT, self.read_timeout),
            )
        except requests.Timeout as exc:
            raise OllamaTimeout(f"{MODEL_LABEL} did not warm up in time.") from exc
        except requests.RequestException as exc:
            raise OllamaUnavailable(
                "Ollama is not running. Start it with `ollama serve`."
            ) from exc
        if not response.ok:
            raise OllamaHTTPError(
                f"Ollama returned HTTP {response.status_code} during warmup.",
                status=response.status_code,
            )
        return {"warm": True, "model": self.model, "label": MODEL_LABEL}

    def unload(self) -> dict:
        """Ask Ollama to release the model after this request (keep_alive 0)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ok"}],
            "stream": False,
            "keep_alive": 0,
            "options": {"num_predict": 1},
        }
        try:
            requests.post(
                f"{self.base_url}/api/chat", json=payload,
                timeout=(CONNECT_TIMEOUT, 15.0),
            )
        except requests.RequestException:
            return {"unloaded": False, "message": "Ollama is not running; nothing to unload."}
        return {"unloaded": True, "message": f"{MODEL_LABEL} released from memory."}
