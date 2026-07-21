"""Optional live provider adapters, imported only by explicit live commands."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from ..config import LiveSettings
from ..models import Claim, ConnectorResult, EvidenceState, Source


class ConnectorError(RuntimeError):
    """Raised for a safe, user-facing live connector failure."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _citation_items(payload: Mapping[str, Any]) -> list[object]:
    choices = payload.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    extras = first_choice.get("extras", {}) if isinstance(first_choice, Mapping) else {}
    items: list[object] = []
    if isinstance(extras, Mapping) and isinstance(extras.get("citations"), list):
        items.extend(extras["citations"])
    if isinstance(payload.get("citations"), list):
        items.extend(payload["citations"])
    if isinstance(payload.get("search_results"), list):
        items.extend(payload["search_results"])
    return items


def parse_perplexity_payload(
    payload: Mapping[str, Any],
    *,
    model: str,
    captured_at: str | None = None,
) -> ConnectorResult:
    """Normalize a Perplexity-compatible response without doing I/O."""

    choices = payload.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise ConnectorError("provider response did not contain a completion")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ConnectorError("provider response did not contain text content")
    answer = message["content"].strip()
    if not answer:
        raise ConnectorError("provider returned an empty answer")

    sources: list[Source] = []
    for item in _citation_items(payload):
        if isinstance(item, str):
            uri, title, excerpt = item, item, ""
        elif isinstance(item, Mapping):
            uri = str(item.get("url") or item.get("uri") or "").strip()
            title = str(item.get("title") or uri).strip()
            excerpt = str(item.get("snippet") or item.get("excerpt") or "").strip()
        else:
            continue
        if not uri:
            continue
        sources.append(
            Source(
                id=f"provider-{len(sources) + 1}",
                title=title or uri,
                uri=uri,
                excerpt=excerpt,
            )
        )

    state = EvidenceState.PROVIDER_CITED if sources else EvidenceState.UNSOURCED
    source_ids = tuple(source.id for source in sources)
    usage = payload.get("usage")
    metadata = {"usage": dict(usage)} if isinstance(usage, Mapping) else {}
    return ConnectorResult(
        answer=answer,
        claims=(Claim(text=answer, source_ids=source_ids),),
        sources=tuple(sources),
        provider="perplexity",
        model=model,
        evidence_state=state,
        captured_at=captured_at or _now_iso(),
        metadata=metadata,
    )


class PerplexityConnector:
    name = "perplexity"

    def __init__(self, settings: LiveSettings) -> None:
        self.settings = settings

    async def research(self, question: str) -> ConnectorResult:
        try:
            import aiohttp
        except ImportError as exc:
            raise ConnectorError(
                "Perplexity live mode requires the optional dependencies: pip install -e '.[live]'"
            ) from exc

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        request = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer carefully. Include provider-supported citations when available; "
                        "do not invent source URLs."
                    ),
                },
                {"role": "user", "content": question},
            ],
        }
        timeout = aiohttp.ClientTimeout(total=self.settings.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.settings.api_url,
                    headers=headers,
                    json=request,
                    allow_redirects=False,
                ) as response:
                    if not 200 <= response.status < 300:
                        raise ConnectorError(
                            f"Perplexity request failed with HTTP {response.status}"
                        )
                    try:
                        payload = await response.json(content_type=None)
                    except ValueError as exc:
                        raise ConnectorError(
                            "Perplexity returned an invalid JSON response"
                        ) from exc
        except asyncio.TimeoutError as exc:
            raise ConnectorError("Perplexity request timed out") from exc
        except aiohttp.ClientError as exc:
            raise ConnectorError(
                f"Perplexity request failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ConnectorError("Perplexity returned an invalid JSON object")
        return parse_perplexity_payload(payload, model=self.settings.model)


class OpenAIConnector:
    name = "openai"

    def __init__(self, settings: LiveSettings) -> None:
        self.settings = settings

    async def research(self, question: str) -> ConnectorResult:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ConnectorError(
                "OpenAI live mode requires the optional dependencies: pip install -e '.[live]'"
            ) from exc

        client = AsyncOpenAI(api_key=self.settings.api_key)
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
                completion = await client.chat.completions.create(
                    model=self.settings.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer carefully. This adapter cannot preserve source citations, "
                                "so do not fabricate URLs or claim independent verification."
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                )
        except TimeoutError as exc:
            raise ConnectorError("OpenAI request timed out") from exc
        except Exception as exc:
            raise ConnectorError(
                f"OpenAI request failed: {type(exc).__name__}"
            ) from exc
        finally:
            await client.close()

        answer = completion.choices[0].message.content if completion.choices else None
        if not isinstance(answer, str) or not answer.strip():
            raise ConnectorError("OpenAI returned an empty answer")
        return ConnectorResult(
            answer=answer.strip(),
            claims=(Claim(text=answer.strip()),),
            sources=(),
            provider="openai",
            model=self.settings.model,
            evidence_state=EvidenceState.UNSOURCED,
            captured_at=_now_iso(),
            metadata={},
        )


def build_live_connector(
    settings: LiveSettings,
) -> PerplexityConnector | OpenAIConnector:
    """Build only the provider selected at the validated live boundary."""

    if settings.provider == "perplexity":
        return PerplexityConnector(settings)
    return OpenAIConnector(settings)
