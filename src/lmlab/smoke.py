from __future__ import annotations

import json

from .data import CorpusRecord, sensitive_findings
from .lm_data import make_token_windows
from .mixture import allocate_tokens
from .retrieval import Document, LexicalIndex, chunk_document, validate_citations
from .tokenizer import ByteBPETokenizer
from .tools import Field, ToolRegistry, ToolSpec


def run() -> dict:
    sample = "AI শেখা 🔐"
    tokenizer = ByteBPETokenizer.train([sample, "Svenska språk"], vocab_size=280)
    token_ids = tokenizer.encode(sample)
    assert tokenizer.decode(token_ids) == sample

    record = CorpusRecord.create(
        "synthetic://lesson", "author-created", "bn", "education", sample
    )
    assert not sensitive_findings(record.text)
    assert make_token_windows([10, 11, 12, 13], 3) == [
        ([10, 11, 12], [11, 12, 13])
    ]
    mixture = allocate_tokens({"education": 0.7, "code": 0.3}, 100)

    document = Document("guide", "Causal masks block future tokens.", "guide.md", "Guide")
    chunks = chunk_document(document, words_per_chunk=8, overlap=1)
    hits = LexicalIndex(chunks).search("causal future", top_k=1)
    citation = validate_citations({"claim-1": [hits[0].chunk_id]}, hits)
    assert citation["valid"]

    registry = ToolRegistry()
    registry.register(
        ToolSpec("echo", "Return text", "echo.read", (Field("text", "string"),)),
        lambda text: text,
    )
    assert registry.execute("echo", {"text": "ok"}, {"echo.read"}) == "ok"
    return {
        "status": "ok",
        "round_trip_text": sample,
        "token_count": len(token_ids),
        "record_sha256": record.sha256,
        "allocated_tokens": mixture.allocated_tokens,
        "citation_valid": citation["valid"],
    }


def main():
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

