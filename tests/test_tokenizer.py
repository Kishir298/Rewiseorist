from pathlib import Path

from ml.tokenizer import Tokenizer, train_from_files, train_tokenizer


def test_train_encode_decode_roundtrip(tmp_path):
    tokenizer = train_tokenizer(
        ["A force is a push or pull. Friction opposes motion."] * 20,
        vocab_size=200, max_length=32,
    )
    ids = tokenizer.encode("A force is a push or pull.", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    text = tokenizer.decode(ids)
    assert "force" in text
    path = tokenizer.save(tmp_path / "tokenizer.json")
    loaded = Tokenizer.load(path)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("force") == tokenizer.encode("force")


def test_oov_character_fallback():
    tokenizer = train_tokenizer(["the cat sat on the mat"] * 10, vocab_size=100)
    ids = tokenizer.encode("the zzz cat")
    assert tokenizer.unk_id not in ids or len(ids) >= 3
    assert "zzz" in tokenizer.decode(ids)


def test_train_from_files(tmp_path):
    path = tmp_path / "stories.txt"
    path.write_text("Once upon a time there was a little robot. " * 50, encoding="utf-8")
    tokenizer = train_from_files([path], vocab_size=150)
    assert tokenizer.vocab_size <= 150
