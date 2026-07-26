from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7


@dataclass(frozen=True)
class ServiceConfig:
    max_prompt_tokens: int = 4096
    max_new_tokens: int = 512


def validate_request(
    prompt_tokens: int, requested_new_tokens: int, config: ServiceConfig
) -> None:
    if not 1 <= prompt_tokens <= config.max_prompt_tokens:
        raise ValueError("prompt length outside allowed range")
    if not 1 <= requested_new_tokens <= config.max_new_tokens:
        raise ValueError("requested generation outside allowed range")


class TokenBucket:
    def __init__(self, capacity: float, refill_per_second: float):
        if capacity <= 0 or refill_per_second < 0:
            raise ValueError("invalid bucket parameters")
        self.capacity = capacity
        self.tokens = capacity
        self.refill_per_second = refill_per_second
        self.updated = time.monotonic()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(
            self.capacity,
            self.tokens + (now - self.updated) * self.refill_per_second,
        )
        self.updated = now
        if cost <= self.tokens:
            self.tokens -= cost
            return True
        return False


class ModelService:
    def __init__(self, generate, count_tokens=lambda text: len(text.split()), config=None):
        self.generate = generate
        self.count_tokens = count_tokens
        self.config = config or ServiceConfig()

    def handle(self, request: GenerationRequest):
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise ValueError("prompt must be nonempty")
        validate_request(
            self.count_tokens(request.prompt), request.max_new_tokens, self.config
        )
        return self.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

