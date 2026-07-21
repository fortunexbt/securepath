from __future__ import annotations

import unittest

from securepath.models import Claim, ConnectorResult, EvidenceState, Source
from securepath.policy import EvidencePolicy, PolicyViolation, SlidingWindowRateLimiter


def result(*, uri: str = "fixture://proof/source") -> ConnectorResult:
    return ConnectorResult(
        answer="Bounded answer",
        claims=(Claim("Bounded claim", ("source-a",)),),
        sources=(Source("source-a", "Proof", uri, "Synthetic excerpt"),),
        provider="fixture",
        model="v1",
        evidence_state=EvidenceState.FIXTURE,
        captured_at="2026-07-21T12:00:00Z",
    )


class EvidencePolicyTests(unittest.TestCase):
    def test_valid_fixture_contract(self) -> None:
        EvidencePolicy().validate(
            case_id="proof-v1", question="Question?", result=result()
        )

    def test_fixture_rejects_live_or_dangerous_uri(self) -> None:
        for uri in (
            "https://example.test",
            "javascript:alert(1)",
            "file:///tmp/evidence",
            "fixture://proof/bad path",
        ):
            with self.subTest(uri=uri), self.assertRaises(PolicyViolation):
                EvidencePolicy().validate(
                    case_id="proof-v1", question="Question?", result=result(uri=uri)
                )

    def test_live_citation_rejects_credentials(self) -> None:
        live = ConnectorResult(
            answer="Answer",
            claims=(Claim("Claim", ("source-a",)),),
            sources=(Source("source-a", "Proof", "https://user:pass@example.test"),),
            provider="provider",
            model="model",
            evidence_state=EvidenceState.PROVIDER_CITED,
            captured_at="2026-07-21T12:00:00+00:00",
        )
        with self.assertRaisesRegex(PolicyViolation, "credentials"):
            EvidencePolicy().validate(
                case_id="live-v1", question="Question?", result=live
            )

    def test_live_citation_rejects_secret_query_fields(self) -> None:
        live = ConnectorResult(
            answer="Answer",
            claims=(Claim("Claim", ("source-a",)),),
            sources=(
                Source(
                    "source-a",
                    "Proof",
                    "https://example.test/report?token=do-not-publish",
                ),
            ),
            provider="provider",
            model="model",
            evidence_state=EvidenceState.PROVIDER_CITED,
            captured_at="2026-07-21T12:00:00+00:00",
        )
        with self.assertRaisesRegex(PolicyViolation, "credential-like"):
            EvidencePolicy().validate(
                case_id="live-v1", question="Question?", result=live
            )

    def test_claim_must_reference_known_source(self) -> None:
        base = result()
        broken = ConnectorResult(
            answer=base.answer,
            claims=(Claim("Claim", ("missing",)),),
            sources=base.sources,
            provider=base.provider,
            model=base.model,
            evidence_state=base.evidence_state,
            captured_at=base.captured_at,
        )
        with self.assertRaisesRegex(PolicyViolation, "unknown source"):
            EvidencePolicy().validate(
                case_id="proof-v1", question="Question?", result=broken
            )

    def test_unsourced_result_cannot_smuggle_sources(self) -> None:
        unsourced = ConnectorResult(
            answer="Answer",
            claims=(Claim("Claim"),),
            sources=(Source("source-a", "Proof", "https://example.test"),),
            provider="provider",
            model="model",
            evidence_state=EvidenceState.UNSOURCED,
            captured_at="2026-07-21T12:00:00Z",
        )
        with self.assertRaisesRegex(PolicyViolation, "must not attach"):
            EvidencePolicy().validate(
                case_id="query-v1", question="Question?", result=unsourced
            )

    def test_result_requires_a_claim(self) -> None:
        unsourced = ConnectorResult(
            answer="Answer",
            claims=(),
            sources=(),
            provider="provider",
            model="model",
            evidence_state=EvidenceState.UNSOURCED,
            captured_at="2026-07-21T12:00:00Z",
        )
        with self.assertRaisesRegex(PolicyViolation, "at least one claim"):
            EvidencePolicy().validate(
                case_id="query-v1", question="Question?", result=unsourced
            )

    def test_metadata_is_size_bounded(self) -> None:
        base = result()
        oversized = ConnectorResult(
            answer=base.answer,
            claims=base.claims,
            sources=base.sources,
            provider=base.provider,
            model=base.model,
            evidence_state=base.evidence_state,
            captured_at=base.captured_at,
            metadata={"padding": "x" * 20_001},
        )
        with self.assertRaisesRegex(PolicyViolation, "metadata exceeds"):
            EvidencePolicy().validate(
                case_id="proof-v1", question="Question?", result=oversized
            )

    def test_timestamp_requires_timezone(self) -> None:
        base = result()
        broken = ConnectorResult(
            answer=base.answer,
            claims=base.claims,
            sources=base.sources,
            provider=base.provider,
            model=base.model,
            evidence_state=base.evidence_state,
            captured_at="2026-07-21T12:00:00",
        )
        with self.assertRaisesRegex(PolicyViolation, "timezone"):
            EvidencePolicy().validate(
                case_id="proof-v1", question="Question?", result=broken
            )


class RateLimiterTests(unittest.TestCase):
    def test_sliding_window_is_deterministic(self) -> None:
        now = [100.0]
        limiter = SlidingWindowRateLimiter(2, 10, clock=lambda: now[0])
        self.assertEqual(limiter.allow("alice"), (True, 0.0))
        self.assertEqual(limiter.allow("alice"), (True, 0.0))
        self.assertEqual(limiter.allow("bob"), (True, 0.0))
        self.assertEqual(limiter.allow("alice"), (False, 10.0))
        now[0] = 110.0
        self.assertEqual(limiter.allow("alice"), (True, 0.0))


if __name__ == "__main__":
    unittest.main()
