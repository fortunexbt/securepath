from __future__ import annotations

import unittest

from securepath.connectors.fixture import FixtureConnector, load_replay_case
from securepath.presentation import (
    escape_discord_mentions,
    render_markdown,
    split_discord_message,
)
from securepath.service import ResearchService


class PresentationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        case = load_replay_case()
        self.packet = await ResearchService(FixtureConnector(case)).run(
            case.question, case_id=case.case_id
        )

    async def test_markdown_makes_fixture_state_conspicuous(self) -> None:
        rendered = render_markdown(self.packet)
        self.assertIn("OFFLINE FIXTURE", rendered)
        self.assertIn("Synthetic fixture", rendered)
        self.assertIn("Integrity SHA-256", rendered)
        self.assertIn("## Sources", rendered)

    async def test_mentions_are_neutralized(self) -> None:
        self.assertEqual(
            escape_discord_mentions("@everyone @alice"), "@\u200beveryone @\u200balice"
        )

    async def test_discord_chunks_respect_limit_and_round_trip_lines(self) -> None:
        text = "alpha\n" + ("b" * 35) + "\nomega"
        chunks = split_discord_message(text, limit=10)
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))
        self.assertIn("alpha", chunks[0])
        self.assertIn("omega", chunks[-1])

    async def test_invalid_chunk_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_discord_message("text", limit=0)


if __name__ == "__main__":
    unittest.main()
