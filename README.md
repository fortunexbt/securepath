# securepath 🚀

> elite discord bot for crypto degens who actually know what they're doing

```
 ██████╗ ███████╗ ██████╗██╗   ██╗██████╗ ███████╗██████╗  █████╗ ████████╗██╗  ██╗
██╔════╝ ██╔════╝██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║  ██║
╚█████╗  █████╗  ██║     ██║   ██║██████╔╝█████╗  ██████╔╝███████║   ██║   ███████║
 ╚═══██╗ ██╔══╝  ██║     ██║   ██║██╔══██╗██╔══╝  ██╔═══╝ ██╔══██║   ██║   ██╔══██║
██████╔╝ ███████╗╚██████╗╚██████╔╝██║  ██║███████╗██║     ██║  ██║   ██║   ██║  ██║
╚═════╝  ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
```

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square&logo=python)](https://www.python.org)
[![Discord.py](https://img.shields.io/badge/discord.py-2.0+-5865f2.svg?style=flat-square&logo=discord)](https://discordpy.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

## what is this

high-performance discord bot that actually understands crypto. built for traders who are tired of basic bots that can't tell the difference between a rug and a gem.

- **ai-powered market analysis** - gpt-4/perplexity integration for real insights
- **chart vision** - upload any chart, get actual technical analysis
- **context-aware conversations** - remembers what you're talking about
- **rate limiting that doesn't suck** - handles api limits like a boss
- **rich logging** - debug in style with color-coded terminal output

## quick start

```bash
# clone it
git clone https://github.com/fortunexbt/securepath.git && cd securepath

# venv (because we're not savages)
python -m venv venv && source venv/bin/activate

# deps
pip install -r requirements.txt

# configure (see below)
cp .env.example .env && nano .env

# send it
python main.py
```

## config

minimal `.env`:

```env
DISCORD_TOKEN=your_bot_token
OWNER_ID=your_discord_id

# pick your fighter
OPENAI_API_KEY=sk-...
# or
PERPLEXITY_API_KEY=pplx-...
USE_PERPLEXITY_API=True
```

<details>
<summary>advanced config (for pros)</summary>

```env
# channels
LOG_CHANNEL_ID=123456789
SUMMARY_CHANNEL_ID=123456789
NEWS_CHANNEL_ID=123456789
CHARTIST_CHANNEL_ID=123456789

# rate limits
API_RATE_LIMIT_MAX=100
API_RATE_LIMIT_INTERVAL=60
DAILY_API_CALL_LIMIT=1000

# context
MAX_CONTEXT_MESSAGES=50
MAX_CONTEXT_AGE=3600

# logging
LOG_LEVEL=INFO
```
</details>

## commands

```
!ask <question>       # get insights on anything crypto
!vision <image>       # analyze charts like a quant
!summary              # generate channel summaries
!commands             # see all available commands
!stats                # check bot usage stats
```

## architecture

```
src/
├── ai/               # openai/perplexity services
├── bot/              # discord client & cogs
├── config/           # settings management
├── database/         # sqlite models & repos
├── services/         # rate limiting, context mgmt
└── utils/            # helpers & formatters
```

built with:
- **discord.py** - async discord api wrapper
- **sqlalchemy** - orm that doesn't get in the way
- **rich** - terminal output that doesn't hurt your eyes
- **asyncio** - because blocking is for boomers

## deployment

### local dev
```bash
python main.py
```

### production
- use systemd/supervisor for process management
- set `LOG_LEVEL=WARNING` in prod
- configure proper rate limits based on your api tier
- consider redis for distributed caching if scaling

## contributing

1. fork it
2. feature branch (`git checkout -b feature/sick-feature`)
3. commit (`git commit -am 'add sick feature'`)
4. push (`git push origin feature/sick-feature`)
5. pr

code style: black. no exceptions.

## license

MIT - do whatever you want with it

## disclaimer

nfa. dyor. if you lose money because of this bot, that's on you anon.

---

built by [@fortunexbt](https://github.com/fortunexbt) | [twitter](https://twitter.com/fortunexbt)