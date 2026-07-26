from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    prompt: str
    expected: str
    category: str


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    category: str
    output: str
    passed: bool


def run_evaluation(cases, generate):
    results = []
    for case in cases:
        output = str(generate(case.prompt))
        results.append(
            EvaluationResult(
                case.case_id,
                case.category,
                output,
                case.expected.lower() in output.lower(),
            )
        )
    return results


def summarize(results) -> dict:
    rows = list(results)
    passed = sum(result.passed for result in rows)
    by_category = {}
    for result in rows:
        slot = by_category.setdefault(result.category, {"passed": 0, "count": 0})
        slot["count"] += 1
        slot["passed"] += int(result.passed)
    return {
        "case_count": len(rows),
        "passed": passed,
        "accuracy": passed / len(rows) if rows else 0.0,
        "by_category": by_category,
        "results": [asdict(result) for result in rows],
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)

