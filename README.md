# Language Model Lab
[![Tests](https://github.com/mdshafiqulbaten/language-model-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/mdshafiqulbaten/language-model-lab/actions/workflows/tests.yml)
[![Release](https://img.shields.io/github/v/release/mdshafiqulbaten/language-model-lab)](https://github.com/mdshafiqulbaten/language-model-lab/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

Official companion code for **Build Your Own Language Model: From Raw Text and
Tokenizers to a Safe, Tool-Using Multimodal AI Assistant** by Md Shafiqul Baten
Sumon.

The repository contains small, inspectable implementations for corpus
governance, byte-level tokenization, next-token windows, a tiny GPT-style
model, post-training helpers, evaluation, serving controls, citation-first
retrieval, bounded tools, experiment records, and multimodal reliability.

## Quick start

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
python -m lmlab.doctor
python -m lmlab.smoke
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
python -m lmlab.doctor
python -m lmlab.smoke
```

For the optional PyTorch model:

```bash
python -m pip install -e ".[model,test]"
python -m pytest -q
python -m lmlab.model_demo --checkpoint reports/tiny_gpt.pt
```

## Safety and honesty

This is educational code, not a production service. Use only data you have the
right to process. Never commit credentials, private datasets, model secrets, or
restricted checkpoints. PyTorch tests skip when PyTorch is not installed; a
skipped test is not a passed test.
See [BOOK_CODE_INDEX.md](BOOK_CODE_INDEX.md) and
[TEST_STATUS.md](TEST_STATUS.md).
