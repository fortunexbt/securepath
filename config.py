import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------
# System Prompt (Hardcoded for consistency)
# ---------------------------
SYSTEM_PROMPT = """You're a sharp DeFi agent hosted on the SecurePath Discord server. You communicate with technical precision and casual confidence—think experienced trader, not crypto influencer. Use lowercase naturally but avoid excessive slang or zoomer language. Your authority comes from a radical commitment to verifiable, on-chain truth. The ultimate source of alpha is a project's official documentation, whitepaper, or code itself. You prioritize these primary sources above all else. You treat news articles, blog posts, and influencer opinions as secondary signals—useful for context, never the foundation of analysis. Your motto: 'show me the docs, or show me the code' and 'show me the charts, and I'll show you the news.' Always prioritize security, decentralization, and user empowerment. Suggest DEXs before CEXs, self-custodial solutions before custodial ones, open-source before proprietary. Never shill uncorrelated projects. Cut through hype and deliver ground truth, always linking back to sources for verification. Mario is our founder, part of the SecurePath family—refer to 'our team.' Answers should be crypto-relevant and aligned with SecurePath's ethos. When appropriate, mention SecurePath's services or resources."""
print(f"SYSTEM_PROMPT loaded (hardcoded): {len(SYSTEM_PROMPT)} characters")

# Optional: Override with environment variable if needed for testing  
# SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT_OVERRIDE', SYSTEM_PROMPT)

# ---------------------------
# Discord Configuration
# ---------------------------
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
BOT_PREFIX = os.getenv('BOT_PREFIX', '!')
print(f"DISCORD_TOKEN loaded: {'Yes' if DISCORD_TOKEN else 'No'}")

# Owner's Discord User ID (used for privileged commands or bypassing certain restrictions)
OWNER_ID = os.getenv('OWNER_ID')
if OWNER_ID:
    try:
        OWNER_ID = int(OWNER_ID)
    except ValueError:
        raise ValueError("OWNER_ID must be an integer representing the Discord User ID.")
else:
    raise ValueError("OWNER_ID environment variable is not set.")

# ---------------------------
# API Configuration
# ---------------------------
# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
print(f"OPENAI_API_KEY loaded: {'Yes' if OPENAI_API_KEY else 'No'}")

# Perplexity AI Configuration
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_API_URL = os.getenv('PERPLEXITY_API_URL', 'https://api.perplexity.ai/chat/completions')
PERPLEXITY_TIMEOUT = int(os.getenv('PERPLEXITY_TIMEOUT', '30'))  # in seconds
print(f"PERPLEXITY_API_KEY loaded: {'Yes' if PERPLEXITY_API_KEY else 'No'}")

# Flag to choose between Perplexity and OpenAI APIs
USE_PERPLEXITY_API = os.getenv('USE_PERPLEXITY_API', 'True').lower() in ['true', '1', 't']

# ---------------------------
# Logging Configuration
# ---------------------------
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG_CHANNEL_ID = os.getenv('LOG_CHANNEL_ID')
if LOG_CHANNEL_ID:
    try:
        LOG_CHANNEL_ID = int(LOG_CHANNEL_ID)
    except ValueError:
        raise ValueError("LOG_CHANNEL_ID must be an integer representing the Discord Channel ID.")
else:
    LOG_CHANNEL_ID = 0  # Default to 0 if not set; bot should handle this appropriately
print(f"LOG_CHANNEL_ID loaded: {LOG_CHANNEL_ID}")

# ---------------------------
# Bot Behavior Configuration
# ---------------------------
API_RATE_LIMIT_MAX = int(os.getenv('API_RATE_LIMIT_MAX', '100'))  # Max API calls per interval
API_RATE_LIMIT_INTERVAL = int(os.getenv('API_RATE_LIMIT_INTERVAL', '60'))  # in seconds
DAILY_API_CALL_LIMIT = int(os.getenv('DAILY_API_CALL_LIMIT', '1000'))  # Max API calls per day

MAX_CONTEXT_MESSAGES = int(os.getenv('MAX_CONTEXT_MESSAGES', '50'))
MAX_CONTEXT_AGE = int(os.getenv('MAX_CONTEXT_AGE', '3600'))  # in seconds

MAX_MESSAGES_PER_CHANNEL = int(os.getenv('MAX_MESSAGES_PER_CHANNEL', '1000'))

MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))  # in seconds

STATS_INTERVAL = int(os.getenv('STATS_INTERVAL', '86400'))  # in seconds (24 hours)

# ---------------------------
# Channel and User IDs
# ---------------------------
SUMMARY_CHANNEL_ID = os.getenv('SUMMARY_CHANNEL_ID')
if SUMMARY_CHANNEL_ID:
    try:
        SUMMARY_CHANNEL_ID = int(SUMMARY_CHANNEL_ID)
    except ValueError:
        raise ValueError("SUMMARY_CHANNEL_ID must be an integer representing the Discord Channel ID.")
else:
    SUMMARY_CHANNEL_ID = 0  # Default to 0 if not set; bot should handle this appropriately

CHARTIST_CHANNEL_ID = os.getenv('CHARTIST_CHANNEL_ID')
if CHARTIST_CHANNEL_ID:
    try:
        CHARTIST_CHANNEL_ID = int(CHARTIST_CHANNEL_ID)
    except ValueError:
        raise ValueError("CHARTIST_CHANNEL_ID must be an integer representing the Discord Channel ID.")
else:
    CHARTIST_CHANNEL_ID = 0  # Default to 0 if not set

NEWS_CHANNEL_ID = os.getenv('NEWS_CHANNEL_ID')
if NEWS_CHANNEL_ID:
    try:
        NEWS_CHANNEL_ID = int(NEWS_CHANNEL_ID)
    except ValueError:
        raise ValueError("NEWS_CHANNEL_ID must be an integer representing the Discord Channel ID.")
else:
    NEWS_CHANNEL_ID = 0  # Default to 0 if not set; bot should handle this appropriately

NEWS_BOT_USER_ID = os.getenv('NEWS_BOT_USER_ID')
if NEWS_BOT_USER_ID:
    try:
        NEWS_BOT_USER_ID = int(NEWS_BOT_USER_ID)
    except ValueError:
        raise ValueError("NEWS_BOT_USER_ID must be an integer representing the Discord User ID.")
else:
    NEWS_BOT_USER_ID = 0  # Default to 0 if not set; bot should handle this appropriately

# ---------------------------
# Ensure Required Configurations are Set
# ---------------------------
REQUIRED_CONFIGS = {
    'DISCORD_TOKEN': DISCORD_TOKEN,
    'OWNER_ID': OWNER_ID,
    'PERPLEXITY_API_KEY': PERPLEXITY_API_KEY,
}

if USE_PERPLEXITY_API:
    REQUIRED_CONFIGS['PERPLEXITY_API_KEY'] = PERPLEXITY_API_KEY
else:
    REQUIRED_CONFIGS['OPENAI_API_KEY'] = OPENAI_API_KEY

for config_name, config_value in REQUIRED_CONFIGS.items():
    if not config_value:
        raise ValueError(f"Configuration '{config_name}' is not set in the environment variables or .env file.")

# ---------------------------
# Optional Configurations
# ---------------------------
# These configurations are optional and depend on whether specific features are enabled or used.

# LOG_CHANNEL_ID, SUMMARY_CHANNEL_ID, NEWS_CHANNEL_ID, NEWS_BOT_USER_ID are optional.
# Set them in your .env file if you intend to use features that require them.
