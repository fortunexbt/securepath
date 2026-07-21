# SecurePath Evidence Lab

[![Evidence proof](https://github.com/fortunexbt/securepath/actions/workflows/test.yml/badge.svg)](https://github.com/fortunexbt/securepath/actions/workflows/test.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11%E2%80%933.13-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-4C7A6B.svg)](LICENSE)

SecurePath is an offline-first reference implementation for research systems that need to show **what evidence accompanied an answer, what did not, and whether the resulting packet changed**. The default demo is deterministic, dependency-free, and makes no network request.

This revival keeps the useful heart of the original Discord research bot—provider queries, citations, policy controls, and presentation—while replacing a 2,088-line runtime with small, testable modules. It is a lab and portfolio proof, not a claim that model output is true.

## What is real today

| Path | Network / secrets | Evidence semantics | Status |
| --- | --- | --- | --- |
| Packaged replay | None | Synthetic fixture, stable hash | Verified offline proof |
| Perplexity query | API key + network | Provider-supplied citations retained; not independently verified | Optional live adapter |
| OpenAI query | API key + network | Explicitly marked unsourced | Optional live adapter |
| Discord `!ask` | Provider key + bot token + network | Same packet policy, mention suppression, per-user rate limit | Optional live bridge |

Chart-image analysis, conversation memory, scheduled news summaries, PostgreSQL telemetry, owner commands, and performance claims from the legacy bot were intentionally not carried forward. They obscured the evidence boundary and had no maintained offline proof.

## Run the proof

Python 3.11 or newer is enough:

```bash
python -m securepath replay --verify
```

Or emit machine-readable output:

```bash
python -m securepath replay --format json --verify
```

No `.env`, Discord token, API key, third-party package, or internet access is required. `python main.py` remains as a compatibility entry point and also runs the replay.

The replay loads a clearly labeled synthetic validator-operations case, enforces the evidence policy, canonicalizes and deduplicates sources, maps claims to stable source IDs, and computes a SHA-256 integrity identity over the complete evidence payload. Repeating it produces the same packet ID.

## Architecture

```text
connector -> normalized result -> evidence policy -> provenance hash -> JSON / Markdown
    |                                  |
    + fixture (offline)                + URI, size, reference, timestamp checks
    + Perplexity (optional live)
    + OpenAI (optional live)
```

- `securepath/connectors/` isolates fixture and provider I/O.
- `securepath/policy.py` rejects broken references, unsafe citation schemes, credential-bearing URLs, oversized content, and ambiguous timestamps.
- `securepath/provenance.py` canonicalizes sources and binds accepted packets to a stable SHA-256 digest.
- `securepath/presentation.py` renders deterministic JSON or readable Markdown and suppresses Discord mentions.
- `securepath/integrations/` contains delivery code that is never imported by the offline path.

The hash proves that a packet has not changed since construction. It does **not** prove that a cited page is correct, that a provider interpreted it faithfully, or that content at a URL has remained unchanged.

## Optional live modes

Install live-only dependencies:

```bash
python -m pip install -e '.[live]'
```

Set values in your shell or secret manager; `.env.example` documents every supported name. SecurePath does not auto-load `.env` files and never needs both provider keys.

```bash
export SECUREPATH_PROVIDER=perplexity
export PERPLEXITY_API_KEY='...'
python -m securepath validate-live --provider perplexity
python -m securepath live-query --provider perplexity \
  'What operational risks should a validator team investigate?'
```

For OpenAI, set `OPENAI_API_KEY` and select `--provider openai`. That adapter is deliberately labeled `unsourced`: a normal chat completion does not provide a citation chain this lab can preserve.

The validation command prints only a redacted configuration summary. Provider keys are validated only for the provider selected at the live boundary. The Perplexity endpoint must be absolute HTTPS and cannot contain embedded credentials.

### Discord bridge

Set `DISCORD_TOKEN` plus the selected provider key, enable Message Content Intent for the bot, then run:

```bash
python -m securepath live-discord --provider perplexity
```

The bridge exposes one command, `!ask <question>`. It disables generated mentions, bounds messages, applies an in-memory per-user sliding-window limit, and returns the same conspicuously labeled evidence packet as the CLI. The rate limit is process-local, so a multi-instance deployment needs a shared limiter before it should be considered abuse-resistant.

## Configuration

| Variable | Used when | Default |
| --- | --- | --- |
| `SECUREPATH_PROVIDER` | Live mode without `--provider` | `perplexity` |
| `PERPLEXITY_API_KEY` | Perplexity only | Required |
| `PERPLEXITY_MODEL` | Perplexity only | `sonar-pro` |
| `PERPLEXITY_API_URL` | Perplexity only | Official HTTPS completions endpoint |
| `OPENAI_API_KEY` | OpenAI only | Required |
| `OPENAI_MODEL` | OpenAI only | `gpt-5` |
| `DISCORD_TOKEN` | Discord only | Required |
| `SECUREPATH_TIMEOUT_SECONDS` | Live providers | `45` |
| `SECUREPATH_COMMAND_PREFIX` | Discord | `!` |
| `SECUREPATH_RATE_LIMIT_COUNT` | Discord | `5` |
| `SECUREPATH_RATE_LIMIT_SECONDS` | Discord | `60` |

## Development

The full core test suite uses the standard library:

```bash
python -m compileall -q securepath tests main.py
python -m unittest discover -s tests -v
python -m securepath replay --format json --verify
```

CI runs those checks across supported Python versions and builds a wheel without pulling live dependencies.

## Security and scope

- Never commit live keys; `.env` is ignored and `.env.example` contains blanks only.
- Offline imports and replay do not read configuration or initialize a network client.
- Live errors report status or exception type, not provider response bodies or credentials.
- Citations are data supplied by a provider, not independent verification.
- The synthetic fixture is a software demonstration, not current staking guidance.

SecurePath provides educational research infrastructure, not financial, legal, security, or investment advice. Validate live claims against primary sources before acting.

## License

MIT — see [LICENSE](LICENSE).
