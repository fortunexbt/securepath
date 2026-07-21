from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    @staticmethod
    def _offline_env() -> dict[str, str]:
        environ = os.environ.copy()
        for name in ("PERPLEXITY_API_KEY", "OPENAI_API_KEY", "DISCORD_TOKEN"):
            environ.pop(name, None)
        return environ

    def test_json_replay_runs_without_live_secrets(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "securepath",
                "replay",
                "--format",
                "json",
                "--verify",
            ],
            cwd=ROOT,
            env=self._offline_env(),
            check=True,
            capture_output=True,
            text=True,
        )
        packet = json.loads(completed.stdout)
        self.assertEqual(packet["evidence_state"], "fixture")
        self.assertTrue(packet["packet_id"].startswith("sp_"))
        self.assertEqual(completed.stderr, "")

    def test_package_import_runs_without_live_dependencies_or_secrets(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import securepath, securepath.cli, sys; "
                    "assert not ({'aiohttp', 'discord', 'openai'} & set(sys.modules)); "
                    "print(securepath.__version__)"
                ),
            ],
            cwd=ROOT,
            env=self._offline_env(),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "0.2.0")

    def test_legacy_main_entry_point_defaults_to_replay(self) -> None:
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=ROOT,
            env=self._offline_env(),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("OFFLINE FIXTURE", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_live_validation_fails_cleanly_without_key(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "securepath",
                "validate-live",
                "--provider",
                "openai",
            ],
            cwd=ROOT,
            env=self._offline_env(),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("OPENAI_API_KEY", completed.stderr)
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
