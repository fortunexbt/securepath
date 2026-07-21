from __future__ import annotations

import unittest

from securepath.config import ConfigurationError, LiveSettings


class LiveSettingsTests(unittest.TestCase):
    def test_selected_provider_requires_only_its_key(self) -> None:
        settings = LiveSettings.from_env(
            {"OPENAI_API_KEY": "openai-secret"},
            provider="openai",
        )
        self.assertEqual(settings.provider, "openai")
        self.assertIsNone(settings.api_url)

    def test_missing_selected_key_is_named_without_values(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "PERPLEXITY_API_KEY"):
            LiveSettings.from_env({}, provider="perplexity")

    def test_discord_is_required_only_at_discord_boundary(self) -> None:
        settings = LiveSettings.from_env(
            {"OPENAI_API_KEY": "secret"}, provider="openai", require_discord=False
        )
        self.assertIsNone(settings.discord_token)
        with self.assertRaisesRegex(ConfigurationError, "DISCORD_TOKEN"):
            LiveSettings.from_env(
                {"OPENAI_API_KEY": "secret"},
                provider="openai",
                require_discord=True,
            )

    def test_perplexity_url_must_be_safe_https(self) -> None:
        for unsafe in (
            "http://api.example.test/v1",
            "https://user:pass@example.test/v1",
            "https://api.example.test/v1?api_key=redacted-value",
            "https://api.example.test:not-a-port/v1",
            "relative/path",
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(ConfigurationError):
                LiveSettings.from_env(
                    {"PERPLEXITY_API_KEY": "secret", "PERPLEXITY_API_URL": unsafe},
                    provider="perplexity",
                )

    def test_public_summary_does_not_contain_secrets(self) -> None:
        settings = LiveSettings.from_env(
            {"OPENAI_API_KEY": "DO-NOT-PRINT", "DISCORD_TOKEN": "ALSO-SECRET"},
            provider="openai",
        )
        rendered = repr(settings.public_summary())
        self.assertNotIn("DO-NOT-PRINT", rendered)
        self.assertNotIn("ALSO-SECRET", rendered)
        self.assertTrue(settings.public_summary()["discord_configured"])

    def test_bad_numeric_runtime_setting_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "greater than zero"):
            LiveSettings.from_env(
                {"OPENAI_API_KEY": "secret", "SECUREPATH_TIMEOUT_SECONDS": "0"},
                provider="openai",
            )


if __name__ == "__main__":
    unittest.main()
