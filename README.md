<p align="center">
  <img width="120" height="120" alt="logo refresh small copy" src="https://github.com/user-attachments/assets/d4a94780-987d-41de-958c-7b9fa468dce1" />
</p>

<p align="center">
  <i>"show me the docs, show me the code" - elite crypto research bot that actually ships</i>
</p>

<p align="center">
<pre>
                   ███████╗███████╗ ██████╗██╗   ██╗██████╗ ███████╗██████╗  █████╗ ████████╗██╗  ██╗
                   ██╔════╝██╔════╝██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║  ██║
                   ███████╗█████╗  ██║     ██║   ██║██████╔╝█████╗  ██████╔╝███████║   ██║   ███████║
                   ╚════██║██╔══╝  ██║     ██║   ██║██╔══██╗██╔══╝  ██╔═══╝ ██╔══██║   ██║   ██╔══██║
                   ███████║███████╗╚██████╗╚██████╔╝██║  ██║███████╗██║     ██║  ██║   ██║   ██║  ██║
                   ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
</pre>



[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square&logo=python)](https://www.python.org)
[![Discord.py](https://img.shields.io/badge/discord.py-2.3+-5865f2.svg?style=flat-square&logo=discord)](https://discordpy.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/postgresql-asyncpg-336791.svg?style=flat-square&logo=postgresql)](https://www.postgresql.org/)
[![GPT-4.1](https://img.shields.io/badge/gpt--4.1-vision-00A67E.svg?style=flat-square&logo=openai)](https://openai.com)

## what

discord bot that cuts through crypto noise. real-time research, chart analysis, channel intel. built for degens who need alpha, not hype.

```python
# three commands that matter
!ask     # perplexity sonar-pro → elite sources only
!analyze # gpt-4.1 vision → technical chart breakdowns  
!summary # channel intelligence → 72hr market pulse
```

## features that actually work

- **real-time research** - perplexity sonar-pro api, filtered to ethereum.org, defillama, etherscan, coingecko, github
- **chart vision** - gpt-4.1 vision for technical analysis: patterns, levels, volume, risk/reward
- **channel summaries** - concurrent chunk processing, 72hr lookback, market sentiment extraction
- **context memory** - maintains conversation flow (50 msg / 1hr default)
- **postgres analytics** - full usage tracking, cost analysis, user metrics
- **rate limiting** - per-user limits, daily caps, graceful handling

## setup

```bash
# clone
git clone https://github.com/fortunexbt/securepath && cd securepath

# env
python -m venv venv && source venv/bin/activate

# deps
pip install -r requirements.txt

# config (.env)
DISCORD_TOKEN=your_token
OWNER_ID=your_id
PERPLEXITY_API_KEY=pplx-xxx
OPENAI_API_KEY=  # or OPENAI_API_KEY
DATABASE_URL=postgresql://...  # optional but recommended

# run
python main.py
```

## architecture (soon... refactoring is a WIP)

```
src/
├── ai/               # perplexity + openai services
├── bot/              
│   ├── client.py     # custom bot with context manager
│   └── cogs/         # ai_commands, summary_commands, admin
├── database/         
│   ├── models.py     # usage tracking schema
│   └── repositories/ # analytics & usage repos
└── services/         # rate limiter, context manager
```

tech stack:
- **discord.py 2.3+** - async discord wrapper
- **asyncpg** - postgres without the bloat
- **perplexity sonar-pro** - real-time web search
- **gpt-4.1 + vision** - when you need the heavy artillery
- **rich** - console output that doesn't suck

## deployment

optimized for heroku but runs anywhere:

```bash
# heroku
heroku create your-app
heroku addons:create heroku-postgresql:mini
heroku config:set DISCORD_TOKEN=xxx PERPLEXITY_API_KEY=xxx
git push heroku main

# docker/k8s
# Dockerfile included, scale horizontally with redis
```

includes:
- health check endpoint
- connection pooling
- concurrent processing
- token caching (reduces costs ~40%)
- single instance locking

## commands

```
PUBLIC
!ask <query>          # real-time crypto research with sources
!analyze [image]      # chart analysis (attach or use recent)
!summary #channel     # 72hr channel intelligence extraction
!commands             # help menu
!ping                 # latency + system status

ADMIN (owner only)
!stats                # comprehensive usage analytics
!token_usage          # cost breakdown by model
!cache_stats          # cache performance metrics
```

## pro tips

1. **sources matter** - bot filters to elite crypto sites only
2. **context aware** - maintains conversation history, ask follow-ups
3. **multi-channel summaries** - `!summary #general #trading #alpha`
4. **dm support** - full features work in dms with context

## contributing

prs welcome if you:
- improve performance
- add degen features
- fix bugs
- make it more based

```bash
git checkout -b feature/your-feature
# make changes
git commit -m "feat: add based feature"
git push origin feature/your-feature
# open pr
```

## license

MIT - ship it

---

built by [@fortunexbt](https://github.com/fortunexbt) | powered by gpt-4 + perplexity

*"in crypto, speed and accuracy determine survival. this bot delivers both."*
