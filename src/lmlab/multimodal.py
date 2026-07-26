from __future__ import annotations

import random


def paired_bootstrap(a, b, repeats: int = 2000, seed: int = 7) -> dict:
    if len(a) != len(b) or not a or repeats < 100:
        raise ValueError("paired nonempty inputs and at least 100 repeats required")
    rng = random.Random(seed)
    count = len(a)
    differences = []
    for _ in range(repeats):
        indices = [rng.randrange(count) for _ in range(count)]
        differences.append(sum(b[i] - a[i] for i in indices) / count)
    differences.sort()
    return {
        "mean_difference": sum(y - x for x, y in zip(a, b)) / count,
        "ci_low": differences[int(0.025 * repeats)],
        "ci_high": differences[min(repeats - 1, int(0.975 * repeats))],
    }


def reliability_fusion(embeddings, reliability, available=None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("reliability_fusion requires PyTorch") from exc
    if embeddings.ndim != 3 or reliability.ndim != 2:
        raise ValueError("expected embeddings [B,M,D] and reliability [B,M]")
    if embeddings.shape[:2] != reliability.shape:
        raise ValueError("modality dimensions must match")
    if available is None:
        available = torch.ones_like(reliability, dtype=torch.bool)
    if (~available).all(dim=1).any():
        raise ValueError("every sample needs one available modality")
    weights = torch.softmax(reliability.masked_fill(~available, float("-inf")), dim=1)
    return (embeddings * weights.unsqueeze(-1)).sum(dim=1), weights

