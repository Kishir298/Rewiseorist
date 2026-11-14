"""Subprocess worker that holds the two trained local models.

It speaks one JSON object per line over stdin/stdout. The Flask process never
loads PyTorch or scikit-learn itself; when resident weights are released the
worker frees them, and when the worker is stopped the operating system reclaims
all model memory. The next request cold-starts a fresh worker.

Commands:
    {"command": "ping"}
    {"command": "warmup"}
    {"command": "generate_revision", "question": ..., "sources": [...], "personality": ...}
    {"command": "classify_performance", "features": {...}}
    {"command": "unload"}
    {"command": "shutdown"}
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.runtime_constants import ModelUnavailable  # noqa: E402


def log(message: str) -> None:
    print(f"[worker] {message}", file=sys.stderr, flush=True)


class Worker:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.service = None

    def load(self):
        if self.service is None:
            from ml.service import ModelService
            self.service = ModelService(self.models_dir)
        return self.service

    def handle(self, request: dict) -> dict:
        command = request.get("command")
        if command == "ping":
            return {"ok": True, "ready": True, "loaded": self.service is not None}
        if command == "warmup":
            result = self.load().warmup()
            return {"ok": True, "result": result}
        if command == "generate_revision":
            result = self.load().generate_revision(
                str(request.get("question", "")),
                list(request.get("sources") or []),
                str(request.get("personality", "Friendly")),
            )
            return {"ok": True, "result": result}
        if command == "classify_performance":
            result = self.load().classify_performance(dict(request.get("features") or {}))
            return {"ok": True, "result": result}
        if command == "unload":
            if self.service is not None:
                self.service.unload()
                self.service = None
                log("resident weights released")
            return {"ok": True, "result": {"unloaded": True}}
        if command == "shutdown":
            if self.service is not None:
                self.service.unload()
                self.service = None
            return {"ok": True, "result": {"stopped": True}, "shutdown": True}
        return {"ok": False, "error": f"unknown command: {command}", "code": "bad_request"}

    def run(self) -> int:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle(request)
            except FileNotFoundError as exc:
                response = {"ok": False, "error": str(exc), "code": "unavailable"}
            except ModelUnavailable as exc:
                response = {"ok": False, "error": str(exc), "code": "unavailable"}
            except Exception as exc:  # keep the protocol alive
                traceback.print_exc(file=sys.stderr)
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "code": "error"}
            print(json.dumps(response), flush=True)
            if response.get("shutdown"):
                return 0
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=str(ROOT / "models"))
    args = parser.parse_args()
    log(f"starting with models from {args.models}")
    raise SystemExit(Worker(Path(args.models)).run())


if __name__ == "__main__":
    main()
