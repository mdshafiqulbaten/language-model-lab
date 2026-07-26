from __future__ import annotations


def make_token_windows(
    token_ids: list[int], context_length: int, stride: int | None = None
) -> list[tuple[list[int], list[int]]]:
    if context_length < 1:
        raise ValueError("context_length must be positive")
    stride = context_length if stride is None else stride
    if stride < 1:
        raise ValueError("stride must be positive")
    windows = []
    for start in range(0, len(token_ids) - context_length, stride):
        block = token_ids[start : start + context_length + 1]
        if len(block) == context_length + 1:
            windows.append((block[:-1], block[1:]))
    return windows

