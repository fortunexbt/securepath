from __future__ import annotations

import unittest
from dataclasses import replace

from securepath.models import Claim, ConnectorResult, EvidenceState, Source
from securepath.policy import PolicyViolation
from securepath.provenance import build_packet, canonicalize_uri, verify_packet


class ProvenanceTests(unittest.TestCase):
    def test_canonicalization_removes_fragment_and_default_port(self) -> None:
        self.assertEqual(
            canonicalize_uri("HTTPS://EXAMPLE.TEST:443/docs?q=1#section"),
            "https://example.test/docs?q=1",
        )

    def test_duplicate_sources_are_collapsed_and_claims_remapped(self) -> None:
        result = ConnectorResult(
            answer="Answer",
            claims=(Claim("Claim", ("first", "duplicate")),),
            sources=(
                Source("first", "First", "https://EXAMPLE.test:443/docs#one"),
                Source("duplicate", "Duplicate", "https://example.test/docs#two"),
            ),
            provider="provider",
            model="model",
            evidence_state=EvidenceState.PROVIDER_CITED,
            captured_at="2026-07-21T12:00:00Z",
        )
        packet = build_packet(case_id="dedupe-v1", question="Question?", result=result)
        self.assertEqual(len(packet.sources), 1)
        self.assertEqual(packet.claims[0].source_ids, ("src-1",))
        self.assertTrue(verify_packet(packet))

    def test_integrity_detects_tampering(self) -> None:
        result = ConnectorResult(
            answer="Answer",
            claims=(Claim("Claim"),),
            sources=(),
            provider="provider",
            model="model",
            evidence_state=EvidenceState.UNSOURCED,
            captured_at="2026-07-21T12:00:00Z",
        )
        packet = build_packet(case_id="query-v1", question="Question?", result=result)
        self.assertTrue(verify_packet(packet))
        self.assertFalse(verify_packet(replace(packet, answer="Changed")))

    def test_live_http_citation_is_rejected(self) -> None:
        result = ConnectorResult(
            answer="Answer",
            claims=(Claim("Claim", ("source",)),),
            sources=(Source("source", "Source", "http://example.test"),),
            provider="provider",
            model="model",
            evidence_state=EvidenceState.PROVIDER_CITED,
            captured_at="2026-07-21T12:00:00Z",
        )
        with self.assertRaises(PolicyViolation):
            build_packet(case_id="query-v1", question="Question?", result=result)


if __name__ == "__main__":
    unittest.main()
