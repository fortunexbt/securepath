"""Stable JSON, Markdown, and Discord-safe presentation."""

from __future__ import annotations

import json

from .models import EvidencePacket, EvidenceState


_STATE_LABELS = {
    EvidenceState.FIXTURE: "OFFLINE FIXTURE — deterministic, no network",
    EvidenceState.PROVIDER_CITED: (
        "PROVIDER-CITED — citations supplied by provider, not independently verified"
    ),
    EvidenceState.UNSOURCED: "UNSOURCED LIVE OUTPUT — no citations available",
}


def escape_discord_mentions(value: str) -> str:
    """Prevent research text from triggering Discord mentions."""

    return value.replace("@", "@\u200b")


def _escape_markdown_label(value: str) -> str:
    escaped = escape_discord_mentions(value).replace("\r", " ").replace("\n", " ")
    escaped = escaped.replace("\\", "\\\\")
    for character in ("*", "_", "`", "~", "|", ">", "#", "[", "]"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def render_json(packet: EvidencePacket) -> str:
    """Render a reproducible, human-readable JSON packet."""

    return json.dumps(packet.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def render_markdown(packet: EvidencePacket) -> str:
    """Render an evidence packet with conspicuous provenance semantics."""

    source_numbers = {
        source.id: index for index, source in enumerate(packet.sources, start=1)
    }
    lines = [
        "# SecurePath evidence packet",
        "",
        f"**Evidence state:** {_STATE_LABELS[packet.evidence_state]}",
        f"**Provider/model:** {_escape_markdown_label(packet.provider)} / "
        f"{_escape_markdown_label(packet.model)}",
        f"**Packet:** `{packet.packet_id}`",
        f"**Captured:** {packet.captured_at}",
        "",
        "## Question",
        "",
        escape_discord_mentions(packet.question),
        "",
        "## Answer",
        "",
        escape_discord_mentions(packet.answer),
    ]

    if packet.claims:
        lines.extend(["", "## Claim map", ""])
        for claim in packet.claims:
            references = ", ".join(
                f"[{source_numbers[source_id]}]"
                for source_id in claim.source_ids
                if source_id in source_numbers
            )
            suffix = f" — {references}" if references else " — no source attached"
            lines.append(f"- {escape_discord_mentions(claim.text)}{suffix}")

    if packet.sources:
        lines.extend(["", "## Sources", ""])
        for index, source in enumerate(packet.sources, start=1):
            title = _escape_markdown_label(source.title)
            safe_uri = source.uri.replace("(", "%28").replace(")", "%29")
            lines.append(f"{index}. [{title}]({safe_uri})")
            if source.excerpt:
                excerpt = escape_discord_mentions(source.excerpt).replace("\n", " ")
                lines.append(f"   - {excerpt}")

    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {escape_discord_mentions(warning)}" for warning in packet.warnings)
    lines.extend(
        [
            "",
            f"Integrity SHA-256: `{packet.integrity_sha256}`",
            f"Policy: `{packet.policy_version}`",
        ]
    )
    return "\n".join(lines)


def split_discord_message(value: str, *, limit: int = 1_900) -> tuple[str, ...]:
    """Split text into bounded chunks, preferring line boundaries."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if not value:
        return ("",)

    chunks: list[str] = []
    current = ""
    for line in value.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                chunks.append(current.rstrip("\n"))
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit and current:
            chunks.append(current.rstrip("\n"))
            current = ""
        current += line
    if current or not chunks:
        chunks.append(current.rstrip("\n"))
    return tuple(chunks)
