from __future__ import annotations

from dataclasses import dataclass
import math
import re


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source: str
    title: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    locator: str
    text: str
    source: str
    title: str


def chunk_document(
    document: Document, words_per_chunk: int = 120, overlap: int = 20
) -> list[Chunk]:
    if words_per_chunk < 1 or overlap < 0 or overlap >= words_per_chunk:
        raise ValueError("invalid chunk parameters")
    words = document.text.split()
    chunks, start, number = [], 0, 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        chunks.append(
            Chunk(
                f"{document.document_id}:{number}",
                document.document_id,
                f"words {start + 1}-{end}",
                " ".join(words[start:end]),
                document.source,
                document.title,
            )
        )
        if end == len(words):
            break
        start = end - overlap
        number += 1
    return chunks


class LexicalIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = list(chunks)
        self.term_sets = [_terms(chunk.text) for chunk in self.chunks]

    def search(self, query: str, top_k: int = 3, k: int | None = None) -> list[Chunk]:
        top_k = k if k is not None else top_k
        query_terms = _terms(query)
        scored = []
        for chunk, terms in zip(self.chunks, self.term_sets):
            score = len(query_terms & terms) / math.sqrt(max(1, len(terms)))
            scored.append((score, chunk.chunk_id, chunk))
        return [row[2] for row in sorted(scored, reverse=True)[:top_k] if row[0] > 0]


def build_grounded_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)


def validate_citations(
    claim_citations: dict[str, list[str]], allowed_chunks: list[Chunk]
) -> dict:
    allowed = {chunk.chunk_id for chunk in allowed_chunks}
    unknown = sorted(
        {
            citation
            for citations in claim_citations.values()
            for citation in citations
            if citation not in allowed
        }
    )
    unsupported = sorted(
        claim for claim, citations in claim_citations.items() if not citations
    )
    return {
        "valid": not unknown and not unsupported,
        "unknown": unknown,
        "unsupported_claims": unsupported,
    }

