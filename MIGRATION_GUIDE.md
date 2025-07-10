# 🔄 SecurePath Bot Migration Guide

This guide helps developers migrate from the old monolithic structure to the new modular architecture.

## 📍 Quick Reference: Where Did Everything Go?

### **Configuration**
- **Old**: `config.py` → **New**: `src/config/settings_simple.py`
- **Old**: Direct env vars → **New**: `Settings` dataclass with validation

### **Bot Core**
- **Old**: `main.py` lines 795-894 → **New**: `src/bot/client.py` + `src/bot/events.py`
- **Old**: Global bot instance → **New**: `create_bot()` factory function

### **Commands**
- **Old**: `@bot.command` in main.py → **New**: Organized in `src/bot/cogs/`
  - `!ask`, `!analyze` → `src/bot/cogs/ai_commands.py`
  - `!stats`, `!ping` → `src/bot/cogs/admin_commands.py`
  - `!summary` → `src/bot/cogs/summary_commands.py`

### **AI Services**
- **Old**: Mixed in main.py → **New**: `src/ai/` directory
  - Perplexity calls → `src/ai/perplexity_service.py`
  - OpenAI calls → `src/ai/openai_service.py`
  - Image analysis → `src/ai/vision_service.py`
  - Coordination → `src/ai/ai_manager.py`

### **Database**
- **Old**: `database.py` → **New**: `src/database/` with repositories
  - Connection → `src/database/connection.py`
  - Models → `src/database/models_simple.py`
  - Usage tracking → `src/database/repositories/usage_repository.py`
  - Analytics → `src/database/repositories/analytics_repository.py`

## 🔧 Common Migration Tasks

### **1. Importing Settings**
```python
# Old way
import config
token = config.DISCORD_TOKEN

# New way
from src.config import get_settings
settings = get_settings()
token = settings.discord_token
```

### **2. Using the Bot**
```python
# Old way
bot = Bot(command_prefix=config.BOT_PREFIX, intents=intents)

# New way
from src.bot import create_bot
bot = create_bot()
```

### **3. Database Operations**
```python
# Old way
from database import db_manager
await db_manager.log_usage(...)

# New way (same interface, different import)
from src.database import db_manager
await db_manager.log_usage(...)
```

### **4. Adding New Commands**
```python
# Old way: Add to main.py
@bot.command(name='mycommand')
async def mycommand(ctx):
    pass

# New way: Create/update a cog
# In src/bot/cogs/my_cog.py
from discord.ext import commands

class MyCog(commands.Cog):
    @commands.command(name='mycommand')
    async def mycommand(self, ctx):
        pass

async def setup(bot):
    await bot.add_cog(MyCog(bot))
```

### **5. Using AI Services**
```python
# Old way: Direct API calls in main.py
response = await aclient.chat.completions.create(...)

# New way: Use AI Manager
from src.ai import AIManager
ai_manager = AIManager(session=session)
result = await ai_manager.process_query(user_id, query)
```

## 📂 File Mapping

| Old File | New Location | Purpose |
|----------|--------------|---------|
| `main.py` (lines 1-200) | `src/bot/client.py` | Bot initialization |
| `main.py` (lines 201-794) | `src/ai/`, `src/services/` | AI and service logic |
| `main.py` (lines 795-1977) | `src/bot/cogs/` | Command handlers |
| `config.py` | `src/config/settings_simple.py` | Configuration |
| `database.py` | `src/database/` | Database operations |

## 🚀 Running the Refactored Bot

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment setup**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Run the bot**:
   ```bash
   # Old way
   python main.py
   
   # New way
   python main_new.py
   ```

## 🧪 Testing Your Changes

The new structure makes testing easier:

```python
# Test individual modules without Discord
from src.utils.validators import validate_query_length
from src.utils.formatting import format_currency

# Test with mocked dependencies
from src.services.rate_limiter import RateLimiter
limiter = RateLimiter(max_calls=10, interval=60)
```

## ⚠️ Breaking Changes

1. **Import paths**: All imports now start with `src.`
2. **Settings access**: Use `get_settings()` instead of direct `config.` access
3. **Bot creation**: Use `create_bot()` factory instead of direct instantiation
4. **Database models**: Now use dataclasses instead of dictionaries

## 🆘 Troubleshooting

### Import Errors
- Make sure you're in the project root directory
- Add `src` to Python path if needed
- Check that all dependencies are installed

### Configuration Issues
- Ensure `.env` file exists with all required variables
- Check that variable names match the new settings structure
- Verify `DATABASE_URL` format for PostgreSQL

### Command Not Found
- Verify the cog is loaded in `bot/client.py`
- Check command decorator syntax matches cog structure
- Ensure proper `async def setup(bot)` in cog file

## 📚 Additional Resources

- See `REFACTORING_RECAP.md` for detailed changes
- Check `test_direct_imports.py` for import examples
- Review individual module docstrings for usage

---

**Remember**: The core functionality remains the same - only the organization has improved!