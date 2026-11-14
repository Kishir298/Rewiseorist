"""Prepare a bounded, reproducible TinyStories subset.

TinyStories source:
  https://huggingface.co/datasets/roneneldan/TinyStories
Licence: CDLA-Sharing-1.0

This script downloads only the first part of the official training text with an
HTTP Range request, keeps complete stories, deduplicates them, and writes a
story-level train/validation/test split. Raw data stays under data/raw and is
ignored by Git.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "tinystories"
SOURCE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
LICENSE = "CDLA-Sharing-1.0"
DATASET_PAGE = "https://huggingface.co/datasets/roneneldan/TinyStories"
DEFAULT_BYTES = 40 * 1024 * 1024


def download_prefix(limit_bytes: int = DEFAULT_BYTES) -> bytes:
    headers = {"Range": f"bytes=0-{limit_bytes - 1}"}
    response = requests.get(SOURCE_URL, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    data = bytearray()
    for chunk in response.iter_content(1024 * 256):
        if not chunk:
            continue
        data.extend(chunk)
        if len(data) >= limit_bytes:
            break
    return bytes(data[:limit_bytes])


def split_stories(text: str, max_chars: int = 4000) -> list[str]:
    parts = re.split(r"<\|endoftext\|>", text)
    stories = []
    seen = set()
    for part in parts:
        story = re.sub(r"[ \t]+", " ", part).strip()
        if len(story) < 80 or len(story) > max_chars:
            continue
        digest = hashlib.sha256(story.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        stories.append(story)
    return stories


def write_split(stories: list[str], folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    n = len(stories)
    val_end = max(1, int(n * 0.05))
    test_end = max(val_end + 1, int(n * 0.10))
    splits = {
        "test": stories[:val_end],
        "val": stories[val_end:test_end],
        "train": stories[test_end:],
    }
    hashes = {}
    for name, items in splits.items():
        text = "\n\n<|endoftext|>\n\n".join(items) + "\n"
        path = folder / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        hashes[name] = {
            "path": str(path),
            "stories": len(items),
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    return hashes


def main(limit_bytes: int = DEFAULT_BYTES) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "TinyStoriesV2-GPT4-subset.txt"
    payload = download_prefix(limit_bytes)
    raw_path.write_bytes(payload)
    stories = split_stories(payload.decode("utf-8", errors="replace"))
    if len(stories) < 100:
        raise RuntimeError("Not enough complete TinyStories stories in the downloaded subset.")
    splits = write_split(stories, RAW_DIR)
    manifest = {
        "dataset": "TinyStoriesV2-GPT4",
        "source_url": SOURCE_URL,
        "dataset_page": DATASET_PAGE,
        "licence": LICENSE,
        "requested_bytes": limit_bytes,
        "downloaded_bytes": len(payload),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "total_stories": len(stories),
        "splits": splits,
        "note": "Raw dataset is excluded from Git and from the final ZIP; only this preparation script and manifest are shared.",
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
