from __future__ import annotations


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def validate_conversation(messages: list[dict[str, str]]) -> bool:
    if not messages:
        return False
    for message in messages:
        if set(message) != {"role", "content"}:
            return False
        if message["role"] not in ALLOWED_ROLES:
            return False
        if not isinstance(message["content"], str) or not message["content"].strip():
            return False
    return any(m["role"] == "user" for m in messages) and any(
        m["role"] == "assistant" for m in messages
    )


def assistant_only_labels(
    token_ids: list[int],
    spans: list[tuple[str, int, int]],
    ignore_index: int = -100,
) -> list[int]:
    labels = [ignore_index] * len(token_ids)
    for role, start, end in spans:
        if role == "assistant":
            if start < 0 or end > len(token_ids) or start > end:
                raise ValueError("invalid span")
            labels[start:end] = token_ids[start:end]
    return labels


def dpo_loss(
    policy_chosen,
    policy_rejected,
    reference_chosen,
    reference_rejected,
    beta: float = 0.1,
):
    if beta <= 0:
        raise ValueError("beta must be positive")
    try:
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("dpo_loss requires the model extra: pip install -e '.[model]'") from exc
    advantage = (policy_chosen - policy_rejected) - (
        reference_chosen - reference_rejected
    )
    return -functional.logsigmoid(beta * advantage).mean()

