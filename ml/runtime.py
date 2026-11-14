"""Flask-facing runtime that controls the local model worker subprocess.

The website process stays lightweight. A child process (``ml/model_worker.py``)
owns PyTorch and scikit-learn memory. The inactivity controller can therefore:

* release resident weights after two idle minutes without killing the worker;
* stop the worker after seven idle minutes, returning model memory to the OS;
* cold-start a fresh worker on the next AI request.

There are no network or Ollama dependencies anywhere in this path.
"""

from __future__ import annotations

import atexit
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from ml.runtime_constants import (  # noqa: F401  (re-exported for app.py)
    DEFAULT_MODELS,
    REQUIRED_FILES,
    ROOT,
    ModelUnavailable,
)

LOG_DIR = ROOT / "data" / "logs"
WORKER_SCRIPT = ROOT / "ml" / "model_worker.py"


class ModelRuntime:
    def __init__(self, models_dir: str | Path = DEFAULT_MODELS,
                 startup_timeout: int = 45, request_timeout: int = 300) -> None:
        self.models_dir = Path(models_dir)
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._responses: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._resident = False
        self._stopped = False
        self._log_handle = None
        atexit.register(self._atexit)

    # ------------------------------------------------------------------
    # Status and files
    # ------------------------------------------------------------------
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
        manifest = self.manifest()
        with self._lock:
            alive = self._process is not None and self._process.poll() is None
            loaded = alive and self._resident
            stopped = self._stopped
        return {
            "ready": not self.missing_files(),
            "missing_files": self.missing_files(),
            "loaded": loaded,
            "worker_running": alive,
            "stopped": stopped,
            "mode": "local",
            "worker": "subprocess",
            "language_model": manifest.get("language_model", {}),
            "performance_model": manifest.get("performance_model", {}),
            "ollama_required": False,
            "cloud_required": False,
        }

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def _worker_command(self) -> list[str]:
        return [
            sys.executable, "-u", str(WORKER_SCRIPT),
            "--models", str(self.models_dir),
        ]

    def _log_path(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return LOG_DIR / "model_worker.log"

    def _start_worker(self) -> None:
        missing = self.missing_files()
        if missing:
            raise ModelUnavailable(
                "The trained local model files are missing: " + ", ".join(missing)
                + ". Run the training scripts or use the packaged model files."
            )
        self._terminate()
        self._responses = queue.Queue()
        self._log_handle = self._log_path().open("a", encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            self._worker_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creationflags,
        )
        process = self._process

        def reader() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                if line:
                    self._responses.put(line)
            self._responses.put(None)

        self._reader = threading.Thread(target=reader, name="model-worker-reader", daemon=True)
        self._reader.start()
        self._resident = False
        self._stopped = False
        self._send({"command": "ping"}, timeout=self.startup_timeout)

    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _send(self, request: dict, timeout: int | None = None) -> dict:
        if not self._alive():
            self._start_worker()
        process = self._process
        assert process is not None and process.stdin is not None
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._terminate()
            raise ModelUnavailable(
                f"The local model worker stopped unexpectedly: {exc}. "
                f"See {self._log_path()}."
            ) from exc
        try:
            line = self._responses.get(timeout=timeout or self.request_timeout)
        except queue.Empty as exc:
            self._terminate()
            raise ModelUnavailable(
                "The local model worker did not respond in time and was restarted."
            ) from exc
        if line is None:
            self._terminate()
            raise ModelUnavailable(
                f"The local model worker exited. See {self._log_path()} for details."
            )
        try:
            response = json.loads(line)
        except ValueError as exc:
            self._terminate()
            raise ModelUnavailable("The local model worker returned an invalid response.") from exc
        if not response.get("ok"):
            message = str(response.get("error", "The local model could not answer."))
            if response.get("code") == "unavailable":
                raise ModelUnavailable(message)
            raise RuntimeError(message)
        return response.get("result", {})

    def _terminate(self) -> None:
        process = self._process
        self._process = None
        self._resident = False
        if process is not None and process.poll() is None:
            try:
                if process.stdin:
                    process.stdin.close()
            except OSError:
                pass
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    def _atexit(self) -> None:
        with self._lock:
            self._terminate()

    # ------------------------------------------------------------------
    # Public model operations
    # ------------------------------------------------------------------
    def generate_revision(self, question: str, sources: list[dict],
                          personality: str = "Friendly") -> dict:
        with self._lock:
            result = self._send({
                "command": "generate_revision",
                "question": question,
                "sources": sources,
                "personality": personality,
            })
            self._resident = True
            return result

    def classify_performance(self, features: dict) -> dict:
        with self._lock:
            result = self._send({"command": "classify_performance", "features": features})
            self._resident = True
            return result

    def warmup(self) -> dict:
        with self._lock:
            result = self._send({"command": "warmup"}, timeout=max(self.startup_timeout, 120))
            self._resident = True
            return {"warm": True, **result}

    def unload(self) -> dict:
        with self._lock:
            if not self._alive():
                return {"unloaded": True, "message": "model worker is not running"}
            try:
                self._send({"command": "unload"}, timeout=60)
            finally:
                self._resident = False
            return {"unloaded": True, "message": "resident model weights released"}

    def shutdown(self) -> dict:
        with self._lock:
            if not self._alive():
                self._stopped = True
                return {"stopped": True, "message": "model worker already stopped"}
            try:
                self._send({"command": "shutdown"}, timeout=30)
            except (ModelUnavailable, RuntimeError):
                pass
            process = self._process
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._terminate()
            self._terminate()
            self._stopped = True
            return {"stopped": True, "message": "local model worker stopped; memory released"}

    def restart(self) -> dict:
        with self._lock:
            self._terminate()
            self._stopped = False
            started = time.time()
            self._start_worker()
            return {"started": True, "seconds": round(time.time() - started, 2)}
