from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="reports/tiny_gpt.pt")
    args = parser.parse_args(argv)
    try:
        import torch
        from .model import TinyGPT
    except ImportError as exc:
        raise SystemExit("Install PyTorch with: python -m pip install -e '.[model]'") from exc
    torch.manual_seed(7)
    model = TinyGPT(vocab_size=64, context_length=16, width=32, layers=2, heads=4)
    token_ids = torch.randint(0, 64, (2, 8))
    logits, loss = model(token_ids, token_ids)
    path = Path(args.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)
    restored = TinyGPT(64, 16, 32, 2, 4)
    restored.load_state_dict(torch.load(path, map_location="cpu")["model_state"])
    restored_logits, _ = restored(token_ids)
    assert torch.allclose(logits, restored_logits)
    print(
        {
            "shape": list(logits.shape),
            "loss": float(loss),
            "checkpoint": str(path),
            "reload_equal": True,
        }
    )


if __name__ == "__main__":
    main()

