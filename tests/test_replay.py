from __future__ import annotations

import unittest

from securepath.connectors.fixture import FixtureConnector, load_replay_case
from securepath.models import ConnectorResult, EvidenceState
from securepath.presentation import render_json, render_markdown
from securepath.provenance import verify_packet
from securepath.service import ResearchService


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_packaged_replay_is_deterministic(self) -> None:
        case = load_replay_case()
        service = ResearchService(FixtureConnector(case))
        first = await service.run(case.question, case_id=case.case_id)
        second = await service.run(case.question, case_id=case.case_id)
        self.assertEqual(first, second)
        self.assertEqual(render_json(first), render_json(second))
        self.assertEqual(render_markdown(first), render_markdown(second))
        self.assertTrue(verify_packet(first))
        self.assertEqual(first.evidence_state, EvidenceState.FIXTURE)
        self.assertEqual(len(first.sources), 3)

    async def test_fixture_refuses_to_answer_a_new_question(self) -> None:
        connector = FixtureConnector.default()
        with self.assertRaisesRegex(LookupError, "exact fixture question"):
            await connector.research("What is the price today?")

    async def test_stable_source_order_is_uri_based(self) -> None:
        case = load_replay_case()
        packet = await ResearchService(FixtureConnector(case)).run(
            case.question, case_id=case.case_id
        )
        self.assertEqual(
            [source.uri for source in packet.sources],
            sorted(source.uri for source in packet.sources),
        )

    async def test_oversized_question_is_rejected_before_connector_call(self) -> None:
        class CallTrackingConnector:
            name = "tracking"

            def __init__(self) -> None:
                self.called = False

            async def research(self, question: str) -> ConnectorResult:
                self.called = True
                raise AssertionError("connector should not have been called")

        connector = CallTrackingConnector()
        with self.assertRaisesRegex(ValueError, "question exceeds"):
            await ResearchService(connector).run("x" * 2_001, case_id="bounded-v1")
        self.assertFalse(connector.called)


if __name__ == "__main__":
    unittest.main()
