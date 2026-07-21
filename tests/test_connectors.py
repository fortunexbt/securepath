from __future__ import annotations

import unittest

from securepath.connectors.live import ConnectorError, parse_perplexity_payload
from securepath.models import EvidenceState


class PerplexityParserTests(unittest.TestCase):
    def test_parser_preserves_supported_citation_shapes(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {"content": "Evidence-aware answer"},
                    "extras": {
                        "citations": [
                            {"title": "Primary", "url": "https://docs.example.test/a"}
                        ]
                    },
                }
            ],
            "citations": ["https://docs.example.test/b"],
            "search_results": [
                {
                    "title": "Search result",
                    "url": "https://docs.example.test/c",
                    "snippet": "Useful excerpt",
                }
            ],
            "usage": {"total_tokens": 42},
        }
        result = parse_perplexity_payload(
            payload,
            model="sonar-test",
            captured_at="2026-07-21T12:00:00Z",
        )
        self.assertEqual(result.evidence_state, EvidenceState.PROVIDER_CITED)
        self.assertEqual(len(result.sources), 3)
        self.assertEqual(
            result.claims[0].source_ids, tuple(s.id for s in result.sources)
        )
        self.assertEqual(result.metadata["usage"]["total_tokens"], 42)

    def test_parser_labels_answer_without_citations_unsourced(self) -> None:
        result = parse_perplexity_payload(
            {"choices": [{"message": {"content": "No citations returned"}}]},
            model="sonar-test",
            captured_at="2026-07-21T12:00:00Z",
        )
        self.assertEqual(result.evidence_state, EvidenceState.UNSOURCED)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.claims[0].source_ids, ())

    def test_parser_rejects_missing_content(self) -> None:
        with self.assertRaises(ConnectorError):
            parse_perplexity_payload(
                {"choices": []},
                model="sonar-test",
                captured_at="2026-07-21T12:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
