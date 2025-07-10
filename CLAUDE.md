# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

SecurePath AI is a Discord bot designed for crypto/DeFi analysis. The codebase has two architectural versions:
- **Production (main branch)**: Monolithic architecture in `main.py`
- **Refactored (refactor/modular-architecture branch)**: Modular architecture in `src/` directory

## Key Commands

### Running the Bot

**Production version:**
```bash
python main.py
```

**Refactored version:**
```bash
python main_new.py
```

### Managing Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Branch Management

```bash
# Switch between branches using helper script
./switch_branch.sh main      # Switch to production
./switch_branch.sh refactor  # Switch to refactored version

# Or use git directly
git checkout main
git checkout refactor/modular-architecture
```

### Testing

```bash
# Run import tests (refactored version)
python testing_files/test_direct_imports.py
```

### Deployment

The bot is deployed on Heroku:
```bash
# Deploy to Heroku (from main branch)
git push heroku main
```

## Architecture Overview

### Production Architecture (main.py)
- Single 1,977-line file containing all functionality
- Global variables for state management
- Direct API calls to OpenAI/Perplexity
- PostgreSQL database integration via asyncpg
- Discord.py commands defined inline

### Refactored Architecture (src/)
```
src/
├── ai/              # AI service integrations
│   ├── ai_manager.py      # Coordinates AI operations
│   ├── openai_service.py  # OpenAI API wrapper
│   ├── perplexity_service.py # Perplexity API wrapper
│   └── vision_service.py  # Image analysis
├── bot/             # Discord bot core
│   ├── client.py         # Bot initialization
│   ├── events.py         # Event handlers
│   └── cogs/            # Command groups
│       ├── ai_commands.py     # !ask, !analyze
│       ├── admin_commands.py  # !stats, !ping
│       └── summary_commands.py # !summary
├── config/          # Configuration
│   ├── settings.py      # Settings with validation
│   └── constants.py     # Application constants
├── database/        # Data layer
│   ├── connection.py    # Connection pooling
│   ├── models.py        # Data models
│   └── repositories/    # Repository pattern
├── services/        # Business logic
│   ├── rate_limiter.py  # API rate limiting
│   └── context_manager.py # Conversation context
└── utils/           # Utilities
    ├── discord_helpers.py # Discord utilities
    ├── validators.py      # Input validation
    └── formatting.py      # Text formatting
```

## Key Design Patterns

### Configuration Management
- Production: Direct environment variable access via `config.py`
- Refactored: Dataclass-based `Settings` with validation and defaults

### Database Access
- Production: Direct SQL queries in `database.py`
- Refactored: Repository pattern with `UsageRepository` and `AnalyticsRepository`

### AI Service Integration
- Production: Inline API calls with global client instances
- Refactored: Service classes with dependency injection through `AIManager`

### Discord Commands
- Production: `@bot.command` decorators in main file
- Refactored: Cog-based organization for command groups

### Context Management
- Production: Global `user_contexts` dictionary
- Refactored: `ContextManager` singleton service

## Critical Files to Understand

### For Production:
1. `main.py` - Contains entire application logic
2. `config.py` - Environment configuration
3. `database.py` - Database operations

### For Refactored:
1. `src/bot/client.py` - Bot initialization and setup
2. `src/ai/ai_manager.py` - AI service coordination
3. `src/database/__init__.py` - Unified database interface
4. `src/config/settings.py` - Configuration management

## Environment Variables

Required:
- `DISCORD_TOKEN` - Discord bot token
- `OWNER_ID` - Bot owner's Discord ID
- `PERPLEXITY_API_KEY` or `OPENAI_API_KEY` - AI service credentials

Key Optional:
- `DATABASE_URL` - PostgreSQL connection string
- `LOG_CHANNEL_ID` - Discord channel for logs
- `USE_PERPLEXITY_API` - Toggle between AI providers

## Database Schema

The bot uses PostgreSQL with these main tables:
- `usage_tracking` - API usage logs
- `user_analytics` - User statistics
- `user_queries` - Query history
- `daily_usage_summary` - Aggregated daily stats

## API Integration Notes

### Perplexity API
- Uses domain filtering for crypto-focused results
- Configured for 90-day search recency
- Returns citations with responses

### OpenAI API
- GPT-4 for text, GPT-4 Vision for images
- Token usage tracking with cost calculation
- Cache hit rate monitoring

## Development Workflow

1. Always check current branch before making changes
2. Refactored code uses type hints and docstrings extensively
3. Database operations should use the repository pattern in refactored version
4. New commands should be added as cogs in refactored architecture
5. Run import tests after modifying module structure
6. Update both requirements.txt and requirements_new.txt if adding dependencies