"""Application-wide constants."""

# Discord limits
DISCORD_MESSAGE_LIMIT = 2000
DISCORD_EMBED_LIMIT = 6000
DISCORD_FIELD_VALUE_LIMIT = 1024
DISCORD_EMBED_TITLE_LIMIT = 256
DISCORD_EMBED_FIELDS_LIMIT = 25

# API Models
OPENAI_MODEL = "gpt-4-1106-preview"
OPENAI_VISION_MODEL = "gpt-4-vision-preview"
PERPLEXITY_MODEL = "llama-3.1-sonar-large-128k-online"

# Token limits
MAX_TOKENS_RESPONSE = 8000
MAX_TOKENS_SUMMARY = 4096

# Image processing
MAX_IMAGE_SIZE_MB = 20
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "gif", "webp"]

# Cache settings
CACHE_TTL_SECONDS = 3600  # 1 hour
CACHE_MAX_SIZE = 1000

# Database
DB_CONNECTION_TIMEOUT = 30
DB_POOL_MIN_SIZE = 10
DB_POOL_MAX_SIZE = 20

# Progress tracking
PROGRESS_UPDATE_INTERVAL = 2  # seconds

# Error messages
ERROR_RATE_LIMIT = "Rate limit exceeded. Please try again later."
ERROR_API_UNAVAILABLE = "API service is currently unavailable. Please try again later."
ERROR_INVALID_COMMAND = "Invalid command format. Use `!help` for usage information."
ERROR_NO_PERMISSION = "You don't have permission to use this command."