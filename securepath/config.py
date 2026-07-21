"""Configuration loaded only at an explicit live integration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """Raised when a requested live integration is not configured safely."""


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class LiveSettings:
    """Validated settings for a selected provider and optional Discord bridge."""

    provider: str
    api_key: str
    model: str
    api_url: str | None
    timeout_seconds: int
    discord_token: str | None
    command_prefix: str
    rate_limit_count: int
    rate_limit_seconds: int

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str],
        *,
        provider: str | None = None,
        require_discord: bool = False,
    ) -> "LiveSettings":
        selected = (
            (provider or environ.get("SECUREPATH_PROVIDER", "perplexity"))
            .strip()
            .lower()
        )
        if selected not in {"perplexity", "openai"}:
            raise ConfigurationError("provider must be 'perplexity' or 'openai'")

        missing: list[str] = []
        if selected == "perplexity":
            api_key = environ.get("PERPLEXITY_API_KEY", "").strip()
            model = environ.get("PERPLEXITY_MODEL", "sonar-pro").strip()
            api_url = environ.get(
                "PERPLEXITY_API_URL",
                "https://api.perplexity.ai/chat/completions",
            ).strip()
            if not api_key:
                missing.append("PERPLEXITY_API_KEY")
            if any(character.isspace() or ord(character) < 32 for character in api_url):
                raise ConfigurationError(
                    "PERPLEXITY_API_URL must not contain whitespace or control characters"
                )
            try:
                parsed_url = urlsplit(api_url)
                parsed_url.port
            except ValueError as exc:
                raise ConfigurationError(
                    "PERPLEXITY_API_URL contains an invalid host or port"
                ) from exc
            if parsed_url.scheme != "https" or not parsed_url.netloc:
                raise ConfigurationError(
                    "PERPLEXITY_API_URL must be an absolute HTTPS URL"
                )
            if parsed_url.username or parsed_url.password:
                raise ConfigurationError(
                    "PERPLEXITY_API_URL must not contain credentials"
                )
            if parsed_url.query or parsed_url.fragment:
                raise ConfigurationError(
                    "PERPLEXITY_API_URL must not contain a query string or fragment"
                )
        else:
            api_key = environ.get("OPENAI_API_KEY", "").strip()
            model = environ.get("OPENAI_MODEL", "gpt-5").strip()
            api_url = None
            if not api_key:
                missing.append("OPENAI_API_KEY")

        if not model:
            missing.append(f"{selected.upper()}_MODEL")

        discord_token = environ.get("DISCORD_TOKEN", "").strip() or None
        if require_discord and not discord_token:
            missing.append("DISCORD_TOKEN")

        if missing:
            names = ", ".join(sorted(set(missing)))
            raise ConfigurationError(f"missing required live configuration: {names}")

        prefix = environ.get("SECUREPATH_COMMAND_PREFIX", "!").strip()
        if (
            not prefix
            or len(prefix) > 4
            or any(character.isspace() for character in prefix)
        ):
            raise ConfigurationError(
                "SECUREPATH_COMMAND_PREFIX must be 1-4 non-whitespace characters"
            )

        return cls(
            provider=selected,
            api_key=api_key,
            model=model,
            api_url=api_url,
            timeout_seconds=_positive_int(environ, "SECUREPATH_TIMEOUT_SECONDS", 45),
            discord_token=discord_token,
            command_prefix=prefix,
            rate_limit_count=_positive_int(environ, "SECUREPATH_RATE_LIMIT_COUNT", 5),
            rate_limit_seconds=_positive_int(
                environ, "SECUREPATH_RATE_LIMIT_SECONDS", 60
            ),
        )

    def public_summary(self) -> dict[str, object]:
        """Describe active live configuration without returning secret values."""

        host = urlsplit(self.api_url).hostname if self.api_url else "api.openai.com"
        return {
            "provider": self.provider,
            "model": self.model,
            "api_host": host,
            "timeout_seconds": self.timeout_seconds,
            "discord_configured": self.discord_token is not None,
            "command_prefix": self.command_prefix,
            "rate_limit": {
                "count": self.rate_limit_count,
                "seconds": self.rate_limit_seconds,
            },
        }
