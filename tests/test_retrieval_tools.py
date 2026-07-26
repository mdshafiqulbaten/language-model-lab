import pytest

from lmlab.retrieval import Document, LexicalIndex, build_grounded_context, chunk_document, validate_citations
from lmlab.tools import Field, ToolRegistry, ToolSpec, run_agent


def test_retrieval_and_citation_contract():
    document = Document(
        "guide", "A causal mask blocks future tokens during training.", "guide.md", "Guide"
    )
    chunks = chunk_document(document, words_per_chunk=6, overlap=1)
    hits = LexicalIndex(chunks).search("causal future", top_k=2)
    assert hits
    assert hits[0].chunk_id in build_grounded_context(hits)
    assert validate_citations({"claim": [hits[0].chunk_id]}, hits)["valid"]
    assert not validate_citations({"claim": ["unknown"]}, hits)["valid"]


def test_tool_permissions_and_approval():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "send",
            "Send reviewed text",
            "message.send",
            (Field("recipient", "string"), Field("text", "string")),
            changes_state=True,
        ),
        lambda recipient, text: f"{recipient}:{text}",
    )
    with pytest.raises(PermissionError):
        registry.execute(
            "send", {"recipient": "verified", "text": "hello"}, {"message.send"}
        )
    result = registry.execute(
        "send",
        {"recipient": "verified", "text": "hello"},
        {"message.send"},
        approved=True,
    )
    assert result == "verified:hello"


def test_bounded_agent_stops():
    registry = ToolRegistry()
    decisions = iter([{"type": "finish", "result": "done"}])
    result = run_agent(lambda _: next(decisions), registry, set(), max_steps=2)
    assert result["status"] == "completed"

