from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import unicodedata


def canonicalize(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_sha256(text: str) -> str:
    return hashlib.sha256(canonicalize(text).encode("utf-8")).hexdigest()


def sensitive_findings(text: str) -> list[str]:
    findings = []
    if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
        findings.append("email")
    if re.search(r"\b(?:\d[ -]*?){13,16}\b", text):
        findings.append("long-number")
    if re.search(r"(?i)\b(api[_ -]?key|secret|password)\s*[:=]\s*\S+", text):
        findings.append("credential-pattern")
    return sorted(set(findings))


@dataclass(frozen=True)
class CorpusRecord:
    source: str
    permission: str
    language: str
    domain: str
    text: str
    sha256: str
    collected_at: str

    @classmethod
    def create(
        cls, source: str, permission: str, language: str, domain: str, text: str
    ) -> "CorpusRecord":
        fields = (source, permission, language, domain, text)
        if not all(isinstance(v, str) and v.strip() for v in fields):
            raise ValueError("all record fields must be nonempty strings")
        clean = canonicalize(text)
        return cls(
            source=source,
            permission=permission,
            language=language,
            domain=domain,
            text=clean,
            sha256=text_sha256(clean),
            collected_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

