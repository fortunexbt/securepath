"""Lazy Discord bridge for the live evidence service."""

from __future__ import annotations

import math

from ..config import LiveSettings
from ..connectors.live import build_live_connector
from ..policy import SlidingWindowRateLimiter
from ..presentation import render_markdown, split_discord_message
from ..service import ResearchService


def run_discord_bot(settings: LiveSettings) -> None:
    """Start a minimal ``ask`` bot; importing SecurePath never imports Discord."""

    try:
        import discord
        from discord.ext import commands
    except ImportError as exc:
        raise RuntimeError(
            "Discord mode requires the optional dependencies: pip install -e '.[live]'"
        ) from exc

    if not settings.discord_token:
        raise RuntimeError(
            "Discord mode requires validated DISCORD_TOKEN configuration"
        )

    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(
        command_prefix=settings.command_prefix, intents=intents, help_command=None
    )
    service = ResearchService(build_live_connector(settings))
    limiter = SlidingWindowRateLimiter(
        settings.rate_limit_count,
        settings.rate_limit_seconds,
    )
    mentions = discord.AllowedMentions.none()

    @bot.event
    async def on_ready() -> None:
        # Avoid logging tokens, guild details, or user identifiers.
        print(
            f"SecurePath Discord bridge ready as {bot.user}; provider={settings.provider}"
        )

    @bot.command(name="ask")
    async def ask(ctx: commands.Context, *, question: str = "") -> None:
        """Create a policy-checked live evidence packet."""

        if not question.strip():
            await ctx.send(
                f"Usage: `{settings.command_prefix}ask <question>`",
                allowed_mentions=mentions,
            )
            return
        allowed, retry_after = limiter.allow(str(ctx.author.id))
        if not allowed:
            await ctx.send(
                f"Rate limit reached; retry in {math.ceil(retry_after)} seconds.",
                allowed_mentions=mentions,
            )
            return
        async with ctx.typing():
            packet = await service.run(question, case_id=f"discord-{ctx.message.id}")
        for chunk in split_discord_message(render_markdown(packet)):
            await ctx.send(chunk, allowed_mentions=mentions)

    @bot.event
    async def on_command_error(ctx: commands.Context, error: Exception) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        original = getattr(error, "original", error)
        await ctx.send(
            f"SecurePath could not complete the request ({type(original).__name__}).",
            allowed_mentions=mentions,
        )

    bot.run(settings.discord_token, log_handler=None)
