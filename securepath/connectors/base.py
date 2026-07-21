"""Connector contract shared by replay and live providers."""

from __future__ import annotations

from typing import Protocol

from ..models import ConnectorResult


class ResearchConnector(Protocol):
    """Anything that can normalize one question into an evidence result."""

    name: str

    async def research(self, question: str) -> ConnectorResult:
        """Research a question without presenting or persisting it."""
