"""A small tokenizer written from scratch.

It uses a word vocabulary plus single-character fallback. This keeps training
and inference deterministic and makes out-of-vocabulary technical words
representable without a pretrained tokenizer.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>", "<source>", "<end>"]

# Guaranteeing the printable ASCII characters means every input can be encoded,
# even when a word never appeared in the tokenizer training text.
PRINTABLE_ASCII = [chr(code) for code in range(33, 127)]

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\sa-z\d]", re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.lower()


def basic_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


class Tokenizer:
    def __init__(self, vocab: dict[str, int], max_length: int = 192) -> None:
        self.vocab = dict(vocab)
        self.inverse = {index: token for token, index in self.vocab.items()}
        self.max_length = max_length

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<eos>"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False,
               max_length: int | None = None) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for token in basic_tokens(text):
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Character fallback keeps the text readable and avoids a single
                # unknown token swallowing a whole technical word.
                for char in token:
                    ids.append(self.vocab.get(char, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        limit = max_length or self.max_length
        return ids[-limit:]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        pieces = []
        special = set(SPECIAL_TOKENS)
        for index in ids:
            token = self.inverse.get(int(index), "<unk>")
            if skip_special and token in special:
                continue
            pieces.append(token)
        # Single-character tokens (letters from the fallback, punctuation, digits)
        # are joined without spaces so words survive the round trip.
        text = ""
        for token in pieces:
            if len(token) == 1:
                text += token
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += token
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "word_char_fallback",
            "special_tokens": SPECIAL_TOKENS,
            "max_length": self.max_length,
            "vocab": self.vocab,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocab"], int(payload.get("max_length", 192)))


def train_tokenizer(texts: Iterable[str], vocab_size: int = 6000,
                    max_length: int = 192) -> Tokenizer:
    counter: Counter[str] = Counter()
    char_counter: Counter[str] = Counter()
    for text in texts:
        tokens = basic_tokens(text)
        counter.update(tokens)
        char_counter.update("".join(tokens))
    vocab: dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)
    # Every printable ASCII character gets an id for fallback.
    for char in PRINTABLE_ASCII:
        if char not in vocab:
            vocab[char] = len(vocab)
    for char, _ in char_counter.most_common():
        if char not in vocab:
            vocab[char] = len(vocab)
    for token, _ in counter.most_common():
        if len(vocab) >= vocab_size:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return Tokenizer(vocab, max_length=max_length)


def train_from_files(paths: Iterable[str | Path], vocab_size: int = 6000,
                     max_length: int = 192, max_chars: int = 6_000_000) -> Tokenizer:
    texts = []
    used = 0
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        texts.append(text)
        used += len(text)
        if used >= max_chars:
            break
    if not texts:
        raise ValueError("No tokenizer training text found.")
    return train_tokenizer(texts, vocab_size=vocab_size, max_length=max_length)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", nargs="+", required=True)
    parser.add_argument("--out", default="models/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=6000)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    tokenizer = train_from_files(args.text, args.vocab_size, args.max_length)
    tokenizer.save(args.out)
    print(f"Saved {tokenizer.vocab_size} tokens to {args.out}")
