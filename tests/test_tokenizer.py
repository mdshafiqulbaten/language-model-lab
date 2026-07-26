from lmlab.tokenizer import ByteBPETokenizer


def test_multilingual_round_trip(tmp_path):
    samples = ["AI শেখা 🔐", "Svenska språk", "def f(x): return x + 1"]
    tokenizer = ByteBPETokenizer.train(samples, vocab_size=280)
    for sample in samples:
        assert tokenizer.decode(tokenizer.encode(sample)) == sample
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.decode(loaded.encode(samples[0])) == samples[0]


def test_default_byte_tokenizer():
    tokenizer = ByteBPETokenizer()
    text = "Hello 🌍"
    assert tokenizer.decode(tokenizer.encode(text)) == text

