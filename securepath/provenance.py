"""Canonicalization and integrity hashing for evidence packets."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace
from urllib.parse import urlsplit, urlunsplit

from .models import Claim, ConnectorResult, EvidencePacket, EvidenceState, Source
from .policy import EvidencePolicy


def canonicalize_uri(uri: str) -> str:
    """Normalize the URI components relevant to source identity."""

    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if parsed.port and not (scheme == "https" and parsed.port == 443):
        hostname = f"{hostname}:{parsed.port}"
    path = parsed.path or ("/" if scheme == "https" else "")
    return urlunsplit((scheme, hostname, path, parsed.query, ""))


def normalize_result(result: ConnectorResult) -> ConnectorResult:
    """Deduplicate sources and assign stable identifiers by canonical URI."""

    source_by_uri: dict[str, Source] = {}
    old_id_to_uri: dict[str, str] = {}
    for source in result.sources:
        canonical_uri = canonicalize_uri(source.uri)
        old_id_to_uri[source.id] = canonical_uri
        source_by_uri.setdefault(
            canonical_uri,
            Source(
                id=source.id,
                title=source.title.strip(),
                uri=canonical_uri,
                excerpt=source.excerpt.strip(),
            ),
        )

    ordered_uris = sorted(source_by_uri)
    uri_to_id = {uri: f"src-{index}" for index, uri in enumerate(ordered_uris, start=1)}
    sources = tuple(
        replace(source_by_uri[uri], id=uri_to_id[uri]) for uri in ordered_uris
    )

    claims: list[Claim] = []
    for claim in result.claims:
        mapped = {
            uri_to_id[old_id_to_uri[source_id]]
            for source_id in claim.source_ids
            if source_id in old_id_to_uri
        }
        claims.append(
            Claim(
                text=claim.text.strip(),
                source_ids=tuple(
                    sorted(mapped, key=lambda item: int(item.split("-")[1]))
                ),
            )
        )

    return replace(
        result,
        answer=result.answer.strip(),
        claims=tuple(claims),
        sources=sources,
        provider=result.provider.strip(),
        model=result.model.strip(),
        metadata=dict(result.metadata),
    )


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def build_packet(
    *,
    case_id: str,
    question: str,
    result: ConnectorResult,
    policy: EvidencePolicy | None = None,
) -> EvidencePacket:
    """Policy-check, canonicalize, and hash one connector result."""

    active_policy = policy or EvidencePolicy()
    # Validate the connector's identifiers before normalization so malformed or
    # ambiguous references cannot be silently repaired by canonicalization.
    active_policy.validate(case_id=case_id, question=question, result=result)
    normalized = normalize_result(result)
    active_policy.validate(case_id=case_id, question=question, result=normalized)

    warning_by_state = {
        EvidenceState.FIXTURE: (
            "Synthetic fixture: this packet proves the replay pipeline, not current facts.",
        ),
        EvidenceState.PROVIDER_CITED: (
            "Provider-supplied citations were preserved but not independently verified.",
        ),
        EvidenceState.UNSOURCED: (
            "No provider-supplied citations were available; treat this output as unverified.",
        ),
    }
    unsigned = EvidencePacket(
        packet_id="",
        integrity_sha256="",
        case_id=case_id,
        question=question.strip(),
        answer=normalized.answer,
        claims=normalized.claims,
        sources=normalized.sources,
        provider=normalized.provider,
        model=normalized.model,
        evidence_state=normalized.evidence_state,
        captured_at=normalized.captured_at,
        policy_version=active_policy.version,
        warnings=warning_by_state[normalized.evidence_state],
        metadata=normalized.metadata,
    )
    digest = hashlib.sha256(_canonical_bytes(unsigned.evidence_dict())).hexdigest()
    return replace(unsigned, packet_id=f"sp_{digest[:16]}", integrity_sha256=digest)


def verify_packet(packet: EvidencePacket) -> bool:
    """Verify that a packet's public evidence payload still matches its hash."""

    expected = hashlib.sha256(_canonical_bytes(packet.evidence_dict())).hexdigest()
    expected_id = f"sp_{expected[:16]}"
    return hmac.compare_digest(
        packet.integrity_sha256, expected
    ) and hmac.compare_digest(packet.packet_id, expected_id)
