from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class MixturePlan:
    requested_weights: dict[str, float]
    allocated_tokens: dict[str, int]
    total_tokens: int


def allocate_tokens(weights: dict[str, float], total_tokens: int) -> MixturePlan:
    if total_tokens < 0:
        raise ValueError("total_tokens must be nonnegative")
    if not weights or any(v < 0 for v in weights.values()):
        raise ValueError("weights must be nonempty and nonnegative")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("at least one weight must be positive")
    normalized = {k: v / total_weight for k, v in sorted(weights.items())}
    raw = {k: normalized[k] * total_tokens for k in normalized}
    allocated = {k: floor(v) for k, v in raw.items()}
    remaining = total_tokens - sum(allocated.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - allocated[k]), k))
    for key in order[:remaining]:
        allocated[key] += 1
    return MixturePlan(normalized, allocated, total_tokens)

