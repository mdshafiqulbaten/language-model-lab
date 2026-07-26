import json

import pytest

from lmlab.data import CorpusRecord, canonicalize, sensitive_findings
from lmlab.evaluation import EvaluationCase, run_evaluation, summarize, wilson_interval
from lmlab.experiments import ExperimentManifest, Variant, manifest_checksum, validate_single_component_ablations
from lmlab.lm_data import make_token_windows
from lmlab.mixture import allocate_tokens
from lmlab.posttraining import assistant_only_labels, validate_conversation
from lmlab.serving import GenerationRequest, ModelService, ServiceConfig, TokenBucket, validate_request


def test_corpus_record_and_sensitive_findings():
    record = CorpusRecord.create(
        "synthetic://one", "author-created", "en", "education", "  permitted   text "
    )
    assert record.text == "permitted text"
    assert len(record.sha256) == 64
    assert json.loads(record.to_json())["source"] == "synthetic://one"
    assert sensitive_findings("email a@example.com") == ["email"]


def test_mixture_exact_total():
    plan = allocate_tokens({"education": 0.45, "code": 0.15, "general": 0.4}, 101)
    assert sum(plan.allocated_tokens.values()) == 101


def test_windows_shift():
    windows = make_token_windows([10, 11, 12, 13, 14], 2)
    assert windows == [([10, 11], [11, 12]), ([12, 13], [13, 14])]


def test_conversation_and_mask():
    messages = [
        {"role": "user", "content": "Explain attention."},
        {"role": "assistant", "content": "It moves information."},
    ]
    assert validate_conversation(messages)
    assert assistant_only_labels(
        [1, 2, 3, 4], [("user", 0, 2), ("assistant", 2, 4)]
    ) == [-100, -100, 3, 4]


def test_evaluation_and_interval():
    cases = [EvaluationCase("math-1", "2+2?", "4", "math")]
    report = summarize(run_evaluation(cases, lambda _: "4"))
    assert report["accuracy"] == 1.0
    low, high = wilson_interval(8, 10)
    assert 0 <= low < high <= 1


def test_serving_boundaries():
    config = ServiceConfig(max_prompt_tokens=3, max_new_tokens=2)
    validate_request(1, 2, config)
    with pytest.raises(ValueError):
        validate_request(4, 1, config)
    service = ModelService(
        lambda prompt, **kwargs: prompt.upper(),
        count_tokens=lambda text: len(text.split()),
        config=config,
    )
    assert service.handle(GenerationRequest("hello", 1)) == "HELLO"
    bucket = TokenBucket(1, 0)
    assert bucket.allow()
    assert not bucket.allow()


def test_experiment_identity_and_ablation():
    manifest = ExperimentManifest("exp-1", "abc", "data-v1", {"layers": 2}, 7, {"python": "3.12"})
    assert len(manifest_checksum(manifest)) == 64
    full = Variant("full", {"gate": True, "retrieval": True})
    assert validate_single_component_ablations(
        full, (Variant("without-gate", {"gate": False, "retrieval": True}),)
    )
    with pytest.raises(ValueError):
        validate_single_component_ablations(
            full, (Variant("two-changes", {"gate": False, "retrieval": False}),)
        )

