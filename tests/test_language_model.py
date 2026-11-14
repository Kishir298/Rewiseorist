import torch

from ml.language_model import (
    CausalLanguageModel,
    InstructionDataset,
    ModelConfig,
    PackedTokenDataset,
    load_model,
)
from ml.tokenizer import train_tokenizer


def tiny_config(vocab_size=120):
    return ModelConfig(vocab_size=vocab_size, context_length=16, d_model=32,
                       n_layers=2, n_heads=4, d_ff=64, dropout=0.0)


def test_forward_shapes_and_causality():
    torch.manual_seed(0)
    model = CausalLanguageModel(tiny_config())
    tokens = torch.randint(0, 120, (2, 16))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 16, 120)
    assert loss.item() > 0
    model.eval()
    first = model(tokens)[0]
    changed = tokens.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 120
    second = model(changed)[0]
    # Changing the final token must not alter earlier positions (causal mask).
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)
    assert not torch.allclose(first[:, -1], second[:, -1])


def test_loss_mask_ignores_prompt():
    torch.manual_seed(1)
    model = CausalLanguageModel(tiny_config())
    tokens = torch.randint(0, 120, (1, 16))
    mask = torch.zeros(1, 16)
    mask[:, 8:] = 1
    _, masked_loss = model(tokens, tokens, mask)
    assert torch.isfinite(masked_loss)


def test_generation_and_roundtrip(tmp_path):
    torch.manual_seed(2)
    tokenizer = train_tokenizer(["the cat sat on the mat"] * 20, vocab_size=120, max_length=16)
    config = tiny_config(tokenizer.vocab_size)
    model = CausalLanguageModel(config)
    generated = model.generate(tokenizer.encode("the cat", add_bos=True), max_new_tokens=5,
                               temperature=0.8, seed=3)
    assert len(generated) == 5
    path = model.save(tmp_path / "lm.pt", tokenizer, {"version": "test"})
    loaded, loaded_tokenizer, extra = load_model(path)
    assert extra["version"] == "test"
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded.parameter_count() == model.parameter_count()


def test_datasets():
    ids = list(range(200))
    dataset = PackedTokenDataset(ids, context_length=16)
    inputs, targets = dataset[0]
    assert len(dataset) > 0
    assert torch.equal(inputs[1:], targets[:-1])
    tokenizer = train_tokenizer(["hello world this is a test"] * 10, vocab_size=80, max_length=32)
    examples = [{"prompt": "<user> What is a force?", "answer": "A push or pull."}]
    instructions = InstructionDataset(examples, tokenizer, context_length=32)
    inputs, targets, mask = instructions[0]
    assert inputs.shape == targets.shape == mask.shape == (32,)
    assert mask.sum() > 0
