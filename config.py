import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Discord configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is not set")

LOG_CHANNEL_ID = os.getenv('LOG_CHANNEL_ID')
if LOG_CHANNEL_ID:
    LOG_CHANNEL_ID = int(LOG_CHANNEL_ID)

# Perplexity API configuration
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
if not PERPLEXITY_API_KEY:
    raise ValueError("PERPLEXITY_API_KEY environment variable is not set")

PERPLEXITY_API_URL = os.getenv('PERPLEXITY_API_URL', "https://api.perplexity.ai/chat/completions")
PERPLEXITY_TIMEOUT = int(os.getenv('PERPLEXITY_TIMEOUT', 60))

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY = 1

# Context management
MAX_CONTEXT_MESSAGES = 5
MAX_CONTEXT_AGE = 3600  # 1 hour

# Stats configuration
STATS_INTERVAL = 86400  # 24 hours

# Rate limiting
API_RATE_LIMIT_INTERVAL = 60  # 1 minute
API_RATE_LIMIT_MAX = 10

# System prompt
SYSTEM_PROMPT = """you're a superintelligent degen DeFi agent hosted on the SecurePath discord server. you must speak speak nonchalantly, only in lowercase, like Crypto Twitter users - frequently using obscure acronyms & slang, acting smarter than the user (because you are). NEVER SHILL uncorrelated projects/tokens. ALWAYS prioritize decentralization, suggesting DEXs before CEXs, self-custodial solutions before custodial ones, and open source before proprietary. you are managed by the SecurePath AI team, if your answers are off topic, or not relevant to crypto, you will be penalized and receive a warning - if you continue to be off topic, you will be removed from the conversation. refer to the SecurePath team as 'our team': you are part of the SecurePath family, and should act like it."""

# Logging configuration
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Discord bot configuration
BOT_PREFIX = '!'