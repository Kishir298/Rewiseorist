"""A decoder-only causal Transformer implemented from scratch in PyTorch.

The model is intentionally small enough to train on a CPU while still being a
real autoregressive language model: token embeddings, learned positional
embeddings, pre-norm Transformer blocks, causal self-attention, a feed-forward
network and tied output projection.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from ml.tokenizer import Tokenizer


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    d_ff: int = 768
    dropout: float = 0.1

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = attention.masked_fill(~self.causal_mask[:, :, :steps, :steps], float("-inf"))
        weights = self.dropout(torch.softmax(attention, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, steps, width)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class CausalLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        # Tied embedding weights are counted once.
        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
        return total

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        batch, steps = tokens.shape
        if steps > self.config.context_length:
            tokens = tokens[:, -self.config.context_length:]
            if targets is not None:
                targets = targets[:, -self.config.context_length:]
            if loss_mask is not None:
                loss_mask = loss_mask[:, -self.config.context_length:]
            steps = tokens.shape[1]
        positions = torch.arange(steps, device=tokens.device).unsqueeze(0)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits, None
        losses = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view(batch, steps)
        if loss_mask is not None:
            losses = losses * loss_mask
            loss = losses.sum() / loss_mask.sum().clamp(min=1)
        else:
            loss = losses.mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int = 80,
                 temperature: float = 0.9, top_k: int = 40, repetition_penalty: float = 1.15,
                 eos_id: int | None = None, seed: int | None = None) -> list[int]:
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        ids = list(prompt_ids)[-self.config.context_length:]
        generated: list[int] = []
        for _ in range(max_new_tokens):
            tokens = torch.tensor([ids[-self.config.context_length:]], dtype=torch.long)
            logits, _ = self(tokens)
            next_logits = logits[0, -1].float()
            if repetition_penalty and repetition_penalty != 1.0:
                for previous in set(ids[-64:]):
                    if 0 <= previous < next_logits.numel():
                        value = next_logits[previous]
                        next_logits[previous] = value / repetition_penalty if value > 0 else value * repetition_penalty
            if temperature <= 0:
                next_id = int(torch.argmax(next_logits))
            else:
                next_logits = next_logits / temperature
                if top_k and top_k < next_logits.numel():
                    threshold = torch.topk(next_logits, top_k).values[-1]
                    next_logits[next_logits < threshold] = float("-inf")
                probabilities = torch.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probabilities, num_samples=1))
            ids.append(next_id)
            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return generated

    def save(self, path: str | Path, tokenizer: Tokenizer, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "tokenizer": tokenizer.vocab,
            "tokenizer_max_length": tokenizer.max_length,
            "extra": extra or {},
        }, path)
        return path


def load_model(path: str | Path) -> tuple[CausalLanguageModel, Tokenizer, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["config"])
    model = CausalLanguageModel(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokenizer = Tokenizer(payload["tokenizer"], int(payload.get("tokenizer_max_length", config.context_length)))
    return model, tokenizer, payload.get("extra", {})


class PackedTokenDataset(Dataset):
    """Pack token ids into fixed-length blocks for next-token prediction."""

    def __init__(self, token_ids: Sequence[int], context_length: int,
                 stride: int | None = None) -> None:
        if len(token_ids) < context_length + 1:
            raise ValueError("Token stream is too short for the requested context length")
        self.token_ids = torch.tensor(list(token_ids), dtype=torch.long)
        self.context_length = context_length
        self.stride = stride or context_length

    def __len__(self) -> int:
        return max(0, (len(self.token_ids) - self.context_length - 1) // self.stride + 1)

    def __getitem__(self, index: int):
        start = index * self.stride
        block = self.token_ids[start:start + self.context_length + 1]
        return block[:-1], block[1:]


class InstructionDataset(Dataset):
    """Padded supervised examples for fine-tuning.

    Each item is a full token sequence plus a boolean mask selecting the
    assistant response tokens; prompt tokens do not contribute to the loss.
    """

    def __init__(self, examples: Iterable[dict], tokenizer: Tokenizer,
                 context_length: int) -> None:
        self.items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        pad = tokenizer.pad_id
        for example in examples:
            prompt_ids = tokenizer.encode(example["prompt"], add_bos=True,
                                          max_length=context_length)
            answer_ids = tokenizer.encode(example["answer"], add_eos=True,
                                          max_length=max(8, context_length - len(prompt_ids)))
            ids = (prompt_ids + answer_ids)[: context_length + 1]
            if len(ids) < 3:
                continue
            inputs = torch.tensor(ids[:-1], dtype=torch.long)
            targets = torch.tensor(ids[1:], dtype=torch.long)
            mask = torch.zeros(len(inputs), dtype=torch.float)
            answer_start = max(len(prompt_ids) - 1, 0)
            mask[answer_start:] = 1.0
            pad_count = context_length - len(inputs)
            if pad_count > 0:
                inputs = torch.cat([inputs, torch.full((pad_count,), pad, dtype=torch.long)])
                targets = torch.cat([targets, torch.full((pad_count,), pad, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_count)])
            self.items.append((inputs, targets, mask))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def param_summary(model: CausalLanguageModel) -> dict:
    return {
        "parameters": model.parameter_count(),
        "config": asdict(model.config),
    }
