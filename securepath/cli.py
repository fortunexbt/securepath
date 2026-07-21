"""Command-line interface for offline replay and explicit live boundaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence

from .config import ConfigurationError, LiveSettings
from .connectors.fixture import FixtureConnector, load_replay_case
from .models import EvidencePacket
from .policy import PolicyViolation
from .presentation import render_json, render_markdown
from .provenance import verify_packet
from .service import ResearchService


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="securepath",
        description="Replay and inspect citation-aware evidence packets.",
    )
    commands = parser.add_subparsers(dest="command")

    replay = commands.add_parser("replay", help="run the deterministic offline proof")
    replay.add_argument("--fixture", help="path to a schema-v1 replay fixture")
    replay.add_argument(
        "--verify", action="store_true", help="fail if integrity verification fails"
    )
    _add_format(replay)

    validate = commands.add_parser(
        "validate-live",
        help="validate selected live configuration without making a request",
    )
    validate.add_argument("--provider", choices=("perplexity", "openai"))
    validate.add_argument(
        "--discord", action="store_true", help="also require Discord config"
    )

    query = commands.add_parser(
        "live-query", help="perform an explicit network-backed query"
    )
    query.add_argument("--provider", choices=("perplexity", "openai"))
    query.add_argument("question")
    _add_format(query)

    discord = commands.add_parser(
        "live-discord", help="start the optional Discord bridge"
    )
    discord.add_argument("--provider", choices=("perplexity", "openai"))
    return parser


def _render(packet: EvidencePacket, format_name: str) -> str:
    if format_name == "json":
        return render_json(packet)
    return render_markdown(packet)


async def _replay(args: argparse.Namespace) -> int:
    case = load_replay_case(args.fixture)
    packet = await ResearchService(FixtureConnector(case)).run(
        case.question,
        case_id=case.case_id,
    )
    if args.verify and not verify_packet(packet):
        raise RuntimeError("fixture packet failed integrity verification")
    print(_render(packet, args.format))
    return 0


async def _live_query(args: argparse.Namespace) -> int:
    from .connectors.live import build_live_connector

    settings = LiveSettings.from_env(os.environ, provider=args.provider)
    packet = await ResearchService(build_live_connector(settings)).run(args.question)
    print(_render(packet, args.format))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run SecurePath, defaulting to the no-network replay proof."""

    arguments = list(argv) if argv is not None else sys.argv[1:]
    if not arguments:
        arguments = ["replay"]
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "replay":
            return asyncio.run(_replay(args))
        if args.command == "validate-live":
            settings = LiveSettings.from_env(
                os.environ,
                provider=args.provider,
                require_discord=args.discord,
            )
            print(json.dumps(settings.public_summary(), indent=2, sort_keys=True))
            return 0
        if args.command == "live-query":
            return asyncio.run(_live_query(args))
        if args.command == "live-discord":
            from .integrations.discord_bot import run_discord_bot

            settings = LiveSettings.from_env(
                os.environ,
                provider=args.provider,
                require_discord=True,
            )
            run_discord_bot(settings)
            return 0
        parser.print_help()
        return 2
    except (
        ConfigurationError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        LookupError,
        PolicyViolation,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"securepath: {exc}", file=sys.stderr)
        return 2
