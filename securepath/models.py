"""Immutable data contracts for research results and evidence packets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EvidenceState(str, Enum):
    """How a connector relates its answer to evidence."""

    FIXTURE = "fixture"
    PROVIDER_CITED = "provider-cited"
    UNSOURCED = "unsourced"


@dataclass(frozen=True, slots=True)
class Source:
    """One source made available to the answer pipeline."""

    id: str
    title: str
    uri: str
    excerpt: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "uri": self.uri,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True, slots=True)
class Claim:
    """A specific statement and the source identifiers attached to it."""

    text: str
    source_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "source_ids": list(self.source_ids)}


@dataclass(frozen=True, slots=True)
class ConnectorResult:
    """Normalized output from an offline or live connector."""

    answer: str
    claims: tuple[Claim, ...]
    sources: tuple[Source, ...]
    provider: str
    model: str
    evidence_state: EvidenceState
    captured_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """A deterministic fixture and the connector result it contains."""

    schema_version: int
    case_id: str
    question: str
    result: ConnectorResult


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """A policy-checked answer bound to its sources and integrity hash."""

    packet_id: str
    integrity_sha256: str
    case_id: str
    question: str
    answer: str
    claims: tuple[Claim, ...]
    sources: tuple[Source, ...]
    provider: str
    model: str
    evidence_state: EvidenceState
    captured_at: str
    policy_version: str
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def evidence_dict(self) -> dict[str, Any]:
        """Return the canonical payload covered by ``integrity_sha256``."""

        return {
            "case_id": self.case_id,
            "question": self.question,
            "answer": self.answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "sources": [source.to_dict() for source in self.sources],
            "provider": self.provider,
            "model": self.model,
            "evidence_state": self.evidence_state.value,
            "captured_at": self.captured_at,
            "policy_version": self.policy_version,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "packet_id": self.packet_id,
            "integrity_sha256": self.integrity_sha256,
            **self.evidence_dict(),
        }
