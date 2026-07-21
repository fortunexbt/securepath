"""Deterministic, dependency-free fixture replay."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from ..models import Claim, ConnectorResult, EvidenceState, ReplayCase, Source


def _parse_case(payload: dict[str, Any]) -> ReplayCase:
    result = payload["result"]
    return ReplayCase(
        schema_version=int(payload["schema_version"]),
        case_id=str(payload["case_id"]),
        question=str(payload["question"]),
        result=ConnectorResult(
            answer=str(result["answer"]),
            claims=tuple(
                Claim(
                    text=str(item["text"]), source_ids=tuple(item.get("source_ids", ()))
                )
                for item in result["claims"]
            ),
            sources=tuple(
                Source(
                    id=str(item["id"]),
                    title=str(item["title"]),
                    uri=str(item["uri"]),
                    excerpt=str(item.get("excerpt", "")),
                )
                for item in result["sources"]
            ),
            provider=str(result["provider"]),
            model=str(result["model"]),
            evidence_state=EvidenceState(str(result["evidence_state"])),
            captured_at=str(result["captured_at"]),
            metadata=dict(result.get("metadata", {})),
        ),
    )


def load_replay_case(path: str | Path | None = None) -> ReplayCase:
    """Load either the packaged showcase fixture or an explicit fixture file."""

    if path is None:
        fixture = resources.files("securepath.fixtures").joinpath(
            "validator_controls.json"
        )
        payload = json.loads(fixture.read_text(encoding="utf-8"))
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    case = _parse_case(payload)
    if case.schema_version != 1:
        raise ValueError(f"unsupported replay schema: {case.schema_version}")
    return case


class FixtureConnector:
    """Replay exactly one captured case without network access."""

    name = "fixture"

    def __init__(self, case: ReplayCase) -> None:
        self.case = case

    @classmethod
    def default(cls) -> "FixtureConnector":
        return cls(load_replay_case())

    async def research(self, question: str) -> ConnectorResult:
        if question.strip() != self.case.question:
            raise LookupError(
                "the offline connector only replays its exact fixture question; "
                "use live-query for a new question"
            )
        return self.case.result
