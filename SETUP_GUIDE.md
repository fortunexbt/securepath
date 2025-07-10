# 🚀 SecurePath Bot Setup Guide

This guide will help you set up and run the refactored SecurePath Discord bot.

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- PostgreSQL (optional, for usage tracking)
- Discord Bot Token
- API Keys (OpenAI and/or Perplexity)

## 🔧 Quick Setup

### 1. Run the Automated Setup

```bash
# Make sure you're on the refactor branch
git checkout refactor/modular-architecture

# Run the setup script
python setup.py
```

The setup script will:
- ✅ Check Python version
- ✅ Verify Git branch
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Create configuration files
- ✅ Validate setup
- ✅ Test imports
- ✅ Initialize database (if configured)
- ✅ Create run scripts

### 2. Configure Environment

Edit the `.env` file with your credentials:

```bash
# Open .env in your editor
nano .env  # or vim, code, etc.
```

Required settings:
- `DISCORD_TOKEN` - Your Discord bot token
- `OWNER_ID` - Your Discord user ID
- `PERPLEXITY_API_KEY` - Your Perplexity API key

### 3. Run the Bot

```bash
# Using the run script (recommended)
./run.sh  # On Linux/Mac
# or
run.bat   # On Windows

# Or directly
python main_new.py
```

## 📝 Manual Setup (Alternative)

### 1. Clone and Switch Branch

```bash
git clone https://github.com/fortunexbt/securepath.git
cd securepath
git checkout refactor/modular-architecture
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example file
cp .env.example .env

# Edit with your values
nano .env
```

### 5. Initialize Database (Optional)

```python
# Run in Python
from src.database import db_manager
import asyncio

async def init():
    await db_manager.connect()
    
asyncio.run(init())
```

### 6. Run the Bot

```bash
python main_new.py
```

## 🔐 Getting Required Credentials

### Discord Bot Token

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to "Bot" section
4. Click "Add Bot"
5. Copy the token

### Discord User ID

1. Enable Developer Mode in Discord (Settings → Advanced)
2. Right-click your username
3. Click "Copy ID"

### Perplexity API Key

1. Go to [Perplexity AI](https://www.perplexity.ai/)
2. Sign up/Login
3. Go to API settings
4. Generate API key

### OpenAI API Key (Optional)

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up/Login
3. Go to API keys
4. Create new secret key

## 🗄️ Database Setup (Optional)

The bot works without a database, but you'll miss usage tracking features.

### PostgreSQL Setup

1. Install PostgreSQL:
```bash
# Ubuntu/Debian
sudo apt install postgresql

# Mac
brew install postgresql

# Windows
# Download from https://www.postgresql.org/download/windows/
```

2. Create database:
```sql
sudo -u postgres psql
CREATE DATABASE securepath;
CREATE USER botuser WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE securepath TO botuser;
```

3. Update DATABASE_URL in .env:
```
DATABASE_URL=postgresql://botuser:your_password@localhost:5432/securepath
```

## 🧪 Testing the Setup

### 1. Test Imports

```bash
python testing_files/test_direct_imports.py
```

### 2. Test Configuration

```python
from src.config import get_settings
settings = get_settings()
print(f"Bot prefix: {settings.bot_prefix}")
print(f"Discord token configured: {bool(settings.discord_token)}")
```

### 3. Test Bot Connection

```python
from src.bot import create_bot
import asyncio

async def test():
    bot = create_bot()
    # Don't actually run, just test creation
    print("Bot created successfully!")
    
asyncio.run(test())
```

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Make sure you're in venv
which python  # Should show venv path

# Reinstall dependencies
pip install -r requirements.txt
```

**2. Config Validation Failed**
- Check .env file exists
- Verify all required values are set
- No quotes needed around values

**3. Database Connection Failed**
- Check PostgreSQL is running
- Verify DATABASE_URL format
- Test connection separately

**4. Discord Connection Failed**
- Verify bot token is correct
- Check bot has proper permissions
- Ensure bot is invited to server

### Debug Mode

Add to .env for detailed logging:
```
LOG_LEVEL=DEBUG
```

## 🚢 Deployment

### Local Development
- Use the setup above
- Run with `./run.sh` or `python main_new.py`

### Production (Heroku)
```bash
# Create app
heroku create your-app-name

# Set config
heroku config:set DISCORD_TOKEN=your_token
heroku config:set PERPLEXITY_API_KEY=your_key
# ... set other vars

# Deploy
git push heroku refactor/modular-architecture:main
```

### Production (VPS)
1. Clone repo on server
2. Follow setup steps
3. Use systemd or supervisor for process management
4. Consider using nginx for webhooks

## 📊 Monitoring

### Check Bot Status
```python
# In Discord
!ping
!stats  # Admin only
```

### View Logs
- Check console output
- Review log files if configured
- Monitor LOG_CHANNEL_ID in Discord

## 🔄 Updating

```bash
# Pull latest changes
git pull origin refactor/modular-architecture

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart bot
# Ctrl+C to stop, then start again
```

## 🆘 Getting Help

1. Check `TROUBLESHOOTING.md`
2. Review error logs
3. Check existing issues on GitHub
4. Create new issue with:
   - Error message
   - Steps to reproduce
   - Environment details

---

**Happy botting! 🤖**