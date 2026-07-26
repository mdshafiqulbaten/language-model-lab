from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json


def _merge(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(sequence[i])
            i += 1
    return out


@dataclass
class ByteBPETokenizer:
    merges: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        self.merges = list(self.merges or [])

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 320) -> "ByteBPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        sequences = [list(text.encode("utf-8")) for text in texts]
        merges: list[tuple[int, int]] = []
        for new_id in range(256, vocab_size):
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            sequences = [_merge(seq, pair, new_id) for seq in sequences]
            merges.append(pair)
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        sequence = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            sequence = _merge(sequence, pair, 256 + rank)
        return sequence

    def decode(self, token_ids: list[int]) -> str:
        expanded: list[int] = []

        def expand(token_id: int) -> None:
            if 0 <= token_id < 256:
                expanded.append(token_id)
                return
            rank = token_id - 256
            if rank < 0 or rank >= len(self.merges):
                raise ValueError(f"unknown token id: {token_id}")
            left, right = self.merges[rank]
            expand(left)
            expand(right)

        for token_id in token_ids:
            expand(token_id)
        return bytes(expanded).decode("utf-8")

    def to_dict(self) -> dict:
        return {"type": "byte_bpe", "merges": [list(pair) for pair in self.merges]}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ByteBPETokenizer":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls([tuple(pair) for pair in payload["merges"]])

