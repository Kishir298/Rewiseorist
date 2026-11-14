"""Independent llama.cpp teacher runtime.

This module controls a local `llama-server.exe` process. It has no dependency on
any hosted API or model manager. Teacher models are development-only and are
never loaded by the Rewiseorist web application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import requests

ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "teacher_models" / "llama.cpp"
SERVER_BINARY = LLAMA_DIR / "llama-server.exe"
TEACHER_DIR = ROOT / "teacher_models"
LOG_DIR = TEACHER_DIR / "logs"


@dataclass(frozen=True)
class TeacherSpec:
    key: str
    name: str
    filename: str
    licence: str
    source_url: str

    @property
    def path(self) -> Path:
        return TEACHER_DIR / self.filename

    def manifest(self) -> dict:
        path = self.path
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        return {
            **asdict(self),
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256,
            "runtime": "llama.cpp",
            "runtime_binary": str(SERVER_BINARY),
        }


TEACHER_SPECS = [
    TeacherSpec(
        key="qwen25-3b",
        name="Qwen2.5-3B-Instruct",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        licence="Qwen2.5 model licence (see Hugging Face repository)",
        source_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF",
    ),
    TeacherSpec(
        key="llama32-3b",
        name="Llama-3.2-3B-Instruct",
        filename="llama-3.2-3b-instruct-q4_k_m.gguf",
        licence="Llama 3.2 Community Licence",
        source_url="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
    ),
]


def available_specs() -> list[TeacherSpec]:
    return [spec for spec in TEACHER_SPECS if spec.path.exists() and SERVER_BINARY.exists()]


def build_command(spec: TeacherSpec, port: int = 8080, context: int = 4096,
                  threads: int = 10) -> list[str]:
    return [
        str(SERVER_BINARY),
        "-m", str(spec.path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(context),
        "-t", str(threads),
        "-ngl", "0",
        "--log-disable",
    ]


class LlamaCppServer:
    def __init__(self, spec: TeacherSpec, port: int = 8080, context: int = 4096,
                 threads: int = 10, startup_timeout: int = 240) -> None:
        self.spec = spec
        self.port = port
        self.context = context
        self.threads = threads
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None
        self.log_path = LOG_DIR / f"{spec.key}-{port}.log"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "LlamaCppServer":
        if not SERVER_BINARY.exists():
            raise FileNotFoundError(
                f"llama-server.exe not found at {SERVER_BINARY}. "
                "Download the CPU release from https://github.com/ggml-org/llama.cpp/releases."
            )
        if not self.spec.path.exists():
            raise FileNotFoundError(f"Teacher model not found: {self.spec.path}")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            build_command(self.spec, self.port, self.context, self.threads),
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(LLAMA_DIR),
            creationflags=creationflags,
        )
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"llama-server exited during startup:\n{tail}")
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(1)
        self.stop()
        raise TimeoutError(f"llama-server did not become ready in {self.startup_timeout}s")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None

    def __enter__(self) -> "LlamaCppServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.2,
             max_tokens: int = 320, timeout: int = 300) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        started = time.time()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.spec.key,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=timeout,
        )
        elapsed = time.time() - started
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        return {
            "text": text,
            "model": self.spec.key,
            "model_name": self.spec.name,
            "elapsed_seconds": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }


def generate_with_spec(spec: TeacherSpec, prompt: str, system: str | None = None,
                       temperature: float = 0.2, max_tokens: int = 320,
                       port: int = 8080) -> dict:
    with LlamaCppServer(spec, port=port) as server:
        return server.chat(prompt, system=system, temperature=temperature,
                           max_tokens=max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen25-3b")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    spec = next((item for item in TEACHER_SPECS if item.key == args.model), None)
    if not spec:
        raise SystemExit(f"Unknown teacher. Choose from: {[s.key for s in TEACHER_SPECS]}")
    result = generate_with_spec(spec, args.prompt, temperature=args.temperature,
                                max_tokens=args.max_tokens)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
