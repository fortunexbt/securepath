"""Evidence policy and small runtime safeguards."""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime
from typing import TypeAlias
from urllib.parse import parse_qsl, urlsplit

from .models import ConnectorResult, EvidenceState


class PolicyViolation(ValueError):
    """Raised when an evidence result cannot be represented safely."""


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SENSITIVE_QUERY_KEYS = {
    "access_key",
    "api_key",
    "auth",
    "key",
    "password",
    "secret",
    "sig",
    "signature",
    "token",
    "x-amz-credential",
    "x-amz-signature",
}
RateDecision: TypeAlias = tuple[bool, float]


class EvidencePolicy:
    """Validate the bounded, citation-aware evidence contract."""

    version = "2026-07-21"
    max_question_chars = 2_000
    max_answer_chars = 40_000
    max_claim_chars = 8_000
    max_claims = 100
    max_sources = 50
    max_title_chars = 300
    max_excerpt_chars = 4_000
    max_uri_chars = 2_048
    max_metadata_bytes = 20_000

    def validate_request(self, *, case_id: str, question: str) -> None:
        """Reject malformed or oversized input before a connector incurs I/O."""

        self._bounded("case_id", case_id, 64)
        if not _SAFE_ID.fullmatch(case_id):
            raise PolicyViolation(
                "case_id must contain only letters, numbers, '.', '_' or '-'"
            )
        self._bounded("question", question, self.max_question_chars)

    def validate(self, *, case_id: str, question: str, result: ConnectorResult) -> None:
        self.validate_request(case_id=case_id, question=question)
        self._bounded("answer", result.answer, self.max_answer_chars)
        self._bounded("provider", result.provider, 100)
        self._bounded("model", result.model, 200)

        if not result.claims:
            raise PolicyViolation("results require at least one claim")
        if len(result.claims) > self.max_claims:
            raise PolicyViolation(f"claims exceed the limit of {self.max_claims}")
        if len(result.sources) > self.max_sources:
            raise PolicyViolation(f"sources exceed the limit of {self.max_sources}")

        source_ids: set[str] = set()
        for source in result.sources:
            if not _SAFE_ID.fullmatch(source.id):
                raise PolicyViolation(f"invalid source id: {source.id!r}")
            if source.id in source_ids:
                raise PolicyViolation(f"duplicate source id: {source.id}")
            source_ids.add(source.id)
            self._bounded("source title", source.title, self.max_title_chars)
            self._bounded("source URI", source.uri, self.max_uri_chars)
            if len(source.excerpt) > self.max_excerpt_chars:
                raise PolicyViolation(
                    f"source {source.id!r} excerpt exceeds {self.max_excerpt_chars} characters"
                )
            self._validate_uri(source.uri, result.evidence_state)

        if result.evidence_state in {
            EvidenceState.FIXTURE,
            EvidenceState.PROVIDER_CITED,
        }:
            if not result.sources:
                raise PolicyViolation(
                    f"{result.evidence_state.value} results require sources"
                )
        elif result.evidence_state is EvidenceState.UNSOURCED and result.sources:
            raise PolicyViolation("unsourced results must not attach sources")

        for claim in result.claims:
            self._bounded("claim", claim.text, self.max_claim_chars)
            if len(claim.source_ids) != len(set(claim.source_ids)):
                raise PolicyViolation("claim source ids must not contain duplicates")
            unknown = set(claim.source_ids) - source_ids
            if unknown:
                raise PolicyViolation(
                    f"claim references unknown source ids: {', '.join(sorted(unknown))}"
                )
            if result.evidence_state is EvidenceState.UNSOURCED and claim.source_ids:
                raise PolicyViolation("unsourced claims must not attach source ids")
            if (
                result.evidence_state is not EvidenceState.UNSOURCED
                and not claim.source_ids
            ):
                raise PolicyViolation(
                    "sourced claims must reference at least one source"
                )

        try:
            captured_at = datetime.fromisoformat(
                result.captured_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise PolicyViolation("captured_at must be an ISO-8601 timestamp") from exc
        if captured_at.tzinfo is None:
            raise PolicyViolation("captured_at must include a timezone")

        try:
            metadata_json = json.dumps(
                dict(result.metadata),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise PolicyViolation("metadata must be deterministic JSON data") from exc
        if len(metadata_json.encode("utf-8")) > self.max_metadata_bytes:
            raise PolicyViolation(
                f"metadata exceeds {self.max_metadata_bytes} encoded bytes"
            )

    @staticmethod
    def _bounded(name: str, value: str, maximum: int) -> None:
        if not isinstance(value, str) or not value.strip():
            raise PolicyViolation(f"{name} must not be empty")
        if len(value) > maximum:
            raise PolicyViolation(f"{name} exceeds {maximum} characters")

    @staticmethod
    def _validate_uri(uri: str, state: EvidenceState) -> None:
        if any(character.isspace() or ord(character) < 32 for character in uri):
            raise PolicyViolation(
                "source URIs must not contain whitespace or control characters"
            )
        if any(character in uri for character in '<>"{}|\\^`'):
            raise PolicyViolation("source URI contains an unsafe delimiter")
        try:
            parsed = urlsplit(uri)
            username = parsed.username
            password = parsed.password
            parsed.port
        except ValueError as exc:
            raise PolicyViolation(
                "source URI contains an invalid host or port"
            ) from exc
        if username or password:
            raise PolicyViolation("source URIs must not contain credentials")
        query_keys = {
            key.lower() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        }
        if query_keys & _SENSITIVE_QUERY_KEYS:
            raise PolicyViolation(
                "source URI query must not contain credential-like fields"
            )
        if state is EvidenceState.FIXTURE:
            if parsed.scheme != "fixture" or not (parsed.netloc or parsed.path):
                raise PolicyViolation("fixture sources must use a fixture:// URI")
            return
        if parsed.scheme != "https" or not parsed.netloc:
            raise PolicyViolation("live source citations must use absolute HTTPS URIs")


class SlidingWindowRateLimiter:
    """A deterministic, in-memory sliding-window limiter for the live bridge."""

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("rate limit and window must be greater than zero")
        self.limit = limit
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> RateDecision:
        """Return ``(allowed, retry_after_seconds)`` for one subject."""

        now = self._clock()
        events = self._events[key]
        cutoff = now - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= self.limit:
            retry_after = max(0.0, self.window_seconds - (now - events[0]))
            return False, float(math.ceil(retry_after))
        events.append(now)
        return True, 0.0
