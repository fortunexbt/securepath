"""Small orchestration layer joining connectors to evidence provenance."""

from __future__ import annotations

import hashlib

from .connectors.base import ResearchConnector
from .models import EvidencePacket
from .policy import EvidencePolicy
from .provenance import build_packet


class ResearchService:
    """Run a connector and return one validated, integrity-bound packet."""

    def __init__(
        self,
        connector: ResearchConnector,
        *,
        policy: EvidencePolicy | None = None,
    ) -> None:
        self.connector = connector
        self.policy = policy or EvidencePolicy()

    async def run(self, question: str, *, case_id: str | None = None) -> EvidencePacket:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        stable_case_id = case_id or (
            "query-"
            + hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()[:16]
        )
        self.policy.validate_request(
            case_id=stable_case_id,
            question=normalized_question,
        )
        result = await self.connector.research(normalized_question)
        return build_packet(
            case_id=stable_case_id,
            question=normalized_question,
            result=result,
            policy=self.policy,
        )
