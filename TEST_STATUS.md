# Test Status

Verification environment:

- Python 3.12.13
- Linux x86_64
- Core installation: `python -m pip install -e ".[test]"`

Verified results:

- 13 core tests passed.
- The PyTorch test module skipped because PyTorch was not installed in this
  verification environment.
- `python -m lmlab.doctor` passed and reported the environment honestly.
- `python -m lmlab.smoke` passed.
- Multilingual and emoji text `AI শেখা 🔐` round-tripped correctly.
- Tokenizer training and evaluation commands passed.
- English, Bangla, Swedish, and code evaluation samples round-tripped.
- Scaling report generation passed.

Before describing the model tests as passed, install PyTorch and run:

```bash
python -m pip install -e ".[model,test]"
python -m pytest -q
python -m lmlab.model_demo --checkpoint reports/tiny_gpt.pt
```

The PyTorch suite checks model shape, tied weights, gradient flow, causal prefix
invariance, checkpoint reload, and reliability-fusion normalization.

