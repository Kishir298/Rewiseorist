"""Pretrain and fine-tune the from-scratch Rewiseorist language model.

Examples (Windows):
    .venv\\Scripts\\python.exe ml\\train_language.py tokenizer
    .venv\\Scripts\\python.exe ml\\train_language.py pretrain --pilot
    .venv\\Scripts\\python.exe ml\\train_language.py pretrain --steps 1200 --max-chars 8000000
    .venv\\Scripts\\python.exe ml\\train_language.py finetune

Raw TinyStories data and model weights stay out of Git.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.language_model import (
    CausalLanguageModel,
    InstructionDataset,
    ModelConfig,
    PackedTokenDataset,
    load_model,
    param_summary,
)
from ml.tokenizer import Tokenizer, train_from_files

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tinystories"
MODELS = ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
METRICS = ROOT / "evidence" / "ml"
TOKENIZER_PATH = MODELS / "tokenizer.json"
STORY_SEPARATOR = "<|endoftext|>"

DEFAULT_VOCAB = 6000
DEFAULT_CONTEXT = 128
DEFAULT_CONFIG = dict(
    d_model=192, n_layers=4, n_heads=6, d_ff=768, dropout=0.1,
)


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def ensure_tokenizer(max_chars: int = 5_000_000, vocab_size: int = DEFAULT_VOCAB) -> Tokenizer:
    if TOKENIZER_PATH.exists():
        tokenizer = Tokenizer.load(TOKENIZER_PATH)
        log(f"Loaded tokenizer with {tokenizer.vocab_size} tokens")
        return tokenizer
    files = [RAW / "train.txt"]
    sample_files = sorted((ROOT / "samples" / "resources").glob("*.md"))
    tokenizer = train_from_files([*files, *sample_files], vocab_size=vocab_size,
                                 max_length=DEFAULT_CONTEXT, max_chars=max_chars)
    tokenizer.save(TOKENIZER_PATH)
    log(f"Trained tokenizer with {tokenizer.vocab_size} tokens -> {TOKENIZER_PATH}")
    return tokenizer


def encode_corpus(path: Path, tokenizer: Tokenizer, max_chars: int,
                  cache_name: str) -> list[int]:
    """Encode story-separated text, caching the integer stream under data/raw."""
    cache = RAW / f"{cache_name}.{max_chars}.ids.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            log(f"Loaded cached token stream {cache.name} ({len(data):,} tokens)")
            return data
        except (ValueError, OSError):
            cache.unlink(missing_ok=True)
    text = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    stories = text.split(STORY_SEPARATOR)
    ids: list[int] = []
    started = time.time()
    for index, story in enumerate(stories):
        if not story.strip():
            continue
        ids.extend(tokenizer.encode(story, add_bos=True, add_eos=True,
                                    max_length=10**12))
        if index % 2000 == 0 and index:
            log(f"  tokenized {index:,}/{len(stories):,} stories "
                f"({len(ids):,} tokens, {time.time() - started:.0f}s)")
    cache.write_text(json.dumps(ids), encoding="utf-8")
    log(f"Encoded {len(ids):,} tokens in {time.time() - started:.1f}s")
    return ids


def evaluate(model: CausalLanguageModel, dataset: PackedTokenDataset,
             batch_size: int, batches: int = 12) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(loader):
            if index >= batches:
                break
            _, loss = model(inputs, targets)
            losses.append(float(loss))
    model.train()
    return sum(losses) / max(len(losses), 1)


def lr_at(step: int, total: int, base_lr: float, warmup: int = 40) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


def save_manifest(payload: dict) -> None:
    path = MODELS / "model_manifest.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            existing = {}
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def train_pretrain(args: argparse.Namespace) -> dict:
    METRICS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 12)))

    tokenizer = ensure_tokenizer(args.tokenizer_chars, args.vocab_size)
    context = args.context
    batch_size = args.batch_size
    steps = args.steps
    max_chars = args.max_chars
    token_ids = encode_corpus(RAW / "train.txt", tokenizer, max_chars, "train")
    val_ids = encode_corpus(RAW / "val.txt", tokenizer, min(args.val_chars, 2_500_000), "val")

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=context,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = CausalLanguageModel(config)
    log(f"Model parameters: {model.parameter_count():,}")
    dataset = PackedTokenDataset(token_ids, context, stride=context // 2)
    val_dataset = PackedTokenDataset(val_ids, context)
    log(f"Train blocks: {len(dataset):,}; val blocks: {len(val_dataset):,}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    started = time.time()
    history = []
    best_val = float("inf")
    step = 0
    tokens_seen = 0
    model.train()
    while step < steps:
        for inputs, targets in loader:
            lr = lr_at(step, steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            tokens_seen += inputs.numel()
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - started
                log(f"step {step}/{steps} loss {loss.item():.4f} "
                    f"lr {lr:.2e} tokens {tokens_seen:,} "
                    f"({tokens_seen / max(elapsed, 1e-6):,.0f} tok/s)")
            if step % args.eval_every == 0 or step == steps:
                val_loss = evaluate(model, val_dataset, batch_size, batches=8)
                record = {"step": step, "train_loss": round(float(loss.item()), 5),
                          "val_loss": round(val_loss, 5), "lr": lr,
                          "tokens_seen": tokens_seen,
                          "elapsed_seconds": round(time.time() - started, 1)}
                history.append(record)
                log(f"  eval step {step}: val loss {val_loss:.4f}, "
                    f"perplexity {math.exp(min(val_loss, 20)):.1f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "config": asdict(config),
                        "state_dict": model.state_dict(),
                        "tokenizer": tokenizer.vocab,
                        "tokenizer_max_length": tokenizer.max_length,
                        "extra": {"version": args.version, "step": step,
                                  "val_loss": val_loss, "tokens_seen": tokens_seen},
                    }, MODELS / "language_model_pretrained.pt")
                torch.save({
                    "config": asdict(config),
                    "state_dict": model.state_dict(),
                    "tokenizer": tokenizer.vocab,
                    "tokenizer_max_length": tokenizer.max_length,
                    "extra": {"version": args.version, "step": step,
                              "val_loss": val_loss, "tokens_seen": tokens_seen},
                }, CHECKPOINTS / f"pretrain-step-{step:05d}.pt")
            if step >= steps:
                break

    elapsed = time.time() - started
    samples = []
    model.eval()
    for prompt in ["Once upon a time", "The little robot", "A girl found a key"]:
        prompt_ids = tokenizer.encode(prompt, add_bos=True, max_length=context)
        generated = model.generate(prompt_ids, max_new_tokens=70, temperature=0.85,
                                   top_k=40, eos_id=tokenizer.eos_id, seed=args.seed)
        samples.append({"prompt": prompt,
                        "output": tokenizer.decode(prompt_ids + generated)})
    metrics = {
        "task": "causal language model pretraining",
        "dataset": "TinyStoriesV2-GPT4 subset (CDLA-Sharing-1.0)",
        "version": args.version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "parameters": model.parameter_count(),
        "pilot": bool(args.pilot),
        "steps": steps,
        "batch_size": batch_size,
        "context_length": context,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(tokens_seen / max(elapsed, 1e-6), 1),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "best_val_loss": round(best_val, 5),
        "best_val_perplexity": round(math.exp(min(best_val, 20)), 3),
        "history": history,
        "samples": samples,
        "command": " ".join(["python", "ml/train_language.py", "pretrain", *(
            ["--pilot"] if args.pilot else [])]),
    }
    name = "pretrain-pilot" if args.pilot else "pretrain"
    (METRICS / f"{name}-metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (METRICS / f"{name}-loss-curve.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (METRICS / f"{name}-samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    torch.save({
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer.vocab,
        "tokenizer_max_length": tokenizer.max_length,
        "extra": {"version": args.version, "step": steps, "val_loss": best_val,
                  "tokens_seen": tokens_seen},
    }, MODELS / "language_model.pt")
    save_manifest({
        "language_model": {
            "version": args.version,
            "parameters": model.parameter_count(),
            "context_length": context,
            "vocab_size": tokenizer.vocab_size,
            "tokens_seen": tokens_seen,
            "best_val_loss": round(best_val, 5),
            "best_val_perplexity": round(math.exp(min(best_val, 20)), 3),
            "training_seconds": round(elapsed, 1),
            "dataset": "TinyStoriesV2-GPT4 subset",
            "licence": "CDLA-Sharing-1.0",
        }
    })
    log(f"Training complete in {elapsed / 60:.1f} min; val loss {best_val:.4f} -> {MODELS / 'language_model.pt'}")
    return metrics


def train_finetune(args: argparse.Namespace) -> dict:
    METRICS.mkdir(parents=True, exist_ok=True)
    source = MODELS / "language_model_pretrained.pt"
    if not source.exists():
        source = MODELS / "language_model.pt"
    if not source.exists():
        raise FileNotFoundError("No pretrained checkpoint found. Run pretrain first.")
    model, tokenizer, extra = load_model(source)
    model.train()
    data_dir = ROOT / "data" / "teacher" / "reviewed"
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing reviewed teacher data: {train_path}. Run the teacher pipeline first."
        )

    def read_jsonl(path: Path) -> list[dict]:
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items

    train_examples = read_jsonl(train_path)
    val_examples = read_jsonl(val_path) if val_path.exists() else train_examples[:8]
    train_dataset = InstructionDataset(train_examples, tokenizer, model.config.context_length)
    val_dataset = InstructionDataset(val_examples, tokenizer, model.config.context_length)
    log(f"Fine-tuning on {len(train_dataset)} examples, validating on {len(val_dataset)}")
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=0.01)
    started = time.time()
    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for inputs, targets, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss)
            batches += 1
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets, mask in val_loader:
                _, loss = model(inputs, targets, mask)
                val_losses.append(float(loss))
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        record = {"epoch": epoch, "train_loss": round(epoch_loss / max(batches, 1), 5),
                  "val_loss": round(val_loss, 5),
                  "elapsed_seconds": round(time.time() - started, 1)}
        history.append(record)
        log(f"epoch {epoch}/{args.epochs}: train {record['train_loss']:.4f}, val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            model.save(MODELS / "language_model.pt", tokenizer, {
                "version": args.version, "fine_tuned_examples": len(train_dataset),
                "val_loss": val_loss, "source": source.name,
            })
    samples = []
    model.eval()
    for example in val_examples[:5]:
        prompt_ids = tokenizer.encode(example["prompt"], add_bos=True,
                                      max_length=model.config.context_length)
        generated = model.generate(prompt_ids, max_new_tokens=80, temperature=0.7,
                                   top_k=30, eos_id=tokenizer.eos_id, seed=11)
        samples.append({"prompt": example["prompt"],
                        "reference": example["answer"],
                        "output": tokenizer.decode(generated)})
    metrics = {
        "task": "supervised fine-tuning on reviewed teacher data",
        "version": args.version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "examples": len(train_dataset),
        "epochs": args.epochs,
        "elapsed_seconds": round(time.time() - started, 1),
        "best_val_loss": round(best_val, 5),
        "history": history,
        "samples": samples,
        "teachers": ["selected revision teacher A", "selected revision teacher B"],
    }
    (METRICS / "finetune-metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (METRICS / "finetune-samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    save_manifest({"language_model": {
        **json.loads((MODELS / "model_manifest.json").read_text(encoding="utf-8")).get("language_model", {}),
        "version": args.version,
        "fine_tuned_examples": len(train_dataset),
        "fine_tune_val_loss": round(best_val, 5),
        "fine_tune_seconds": round(time.time() - started, 1),
    }})
    log(f"Fine-tuning complete -> {MODELS / 'language_model.pt'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    tok = sub.add_parser("tokenizer")
    tok.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    tok.add_argument("--max-chars", type=int, default=5_000_000)

    pre = sub.add_parser("pretrain")
    pre.add_argument("--pilot", action="store_true")
    pre.add_argument("--steps", type=int, default=1200)
    pre.add_argument("--batch-size", type=int, default=24)
    pre.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    pre.add_argument("--max-chars", type=int, default=8_000_000)
    pre.add_argument("--val-chars", type=int, default=2_000_000)
    pre.add_argument("--tokenizer-chars", type=int, default=5_000_000)
    pre.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    pre.add_argument("--d-model", type=int, default=DEFAULT_CONFIG["d_model"])
    pre.add_argument("--layers", type=int, default=DEFAULT_CONFIG["n_layers"])
    pre.add_argument("--heads", type=int, default=DEFAULT_CONFIG["n_heads"])
    pre.add_argument("--d-ff", type=int, default=DEFAULT_CONFIG["d_ff"])
    pre.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    pre.add_argument("--lr", type=float, default=3e-4)
    pre.add_argument("--log-every", type=int, default=20)
    pre.add_argument("--eval-every", type=int, default=100)
    pre.add_argument("--seed", type=int, default=1337)
    pre.add_argument("--version", default="rewiseorist-lm-0.1")

    fine = sub.add_parser("finetune")
    fine.add_argument("--epochs", type=int, default=6)
    fine.add_argument("--batch-size", type=int, default=8)
    fine.add_argument("--finetune-lr", type=float, default=8e-5)
    fine.add_argument("--version", default="rewiseorist-lm-1.0")

    args = parser.parse_args()
    if args.command == "tokenizer":
        ensure_tokenizer(args.max_chars, args.vocab_size)
        return
    if args.command == "pretrain":
        if args.pilot:
            args.steps = min(args.steps, 30)
            args.max_chars = min(args.max_chars, 600_000)
            args.batch_size = min(args.batch_size, 12)
            args.eval_every = min(args.eval_every, 10)
            args.version = "rewiseorist-lm-pilot"
        train_pretrain(args)
        return
    train_finetune(args)


if __name__ == "__main__":
    main()
