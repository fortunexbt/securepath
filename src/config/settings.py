"""Simple settings configuration without Pydantic dependencies."""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""
    
    # System Configuration
    system_prompt: str = """You're a sharp DeFi agent hosted on the SecurePath Discord server. Communicate with technical precision and casual confidence. Use lowercase naturally but avoid excessive slang. Your authority comes from verifiable, on-chain truth. Prioritize official docs, whitepapers, and code over news/sentiment. Your motto: 'show me the docs, or show me the code.' Always prioritize security, decentralization, and user empowerment. Suggest DEXs over CEXs, self-custody over custodial, open-source over proprietary. Cut through hype and deliver ground truth. Mario is our founder, part of the SecurePath family. 

CRITICAL FORMATTING RULES:
- NO TABLES whatsoever (Discord can't render them)
- Use bullet points and numbered lists only
- Keep responses under 400 words total
- Be concise and direct, no fluff
- Use [1], [2] format for citations when available"""
    
    # Discord Configuration
    discord_token: str = ""
    bot_prefix: str = "!"
    owner_id: int = 0
    
    # API Configuration
    openai_api_key: Optional[str] = None
    perplexity_api_key: str = ""
    perplexity_api_url: str = "https://api.perplexity.ai/chat/completions"
    perplexity_timeout: int = 30
    use_perplexity_api: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_channel_id: Optional[int] = None
    
    # Rate Limiting Configuration
    api_rate_limit_max: int = 100
    api_rate_limit_interval: int = 60
    daily_api_call_limit: int = 1000
    
    # Context Management
    max_context_messages: int = 50
    max_context_age: int = 3600
    max_messages_per_channel: int = 1000
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay: int = 5
    
    # Feature Configuration
    stats_interval: int = 86400
    
    # Channel IDs
    summary_channel_id: Optional[int] = None
    chartist_channel_id: Optional[int] = None
    news_channel_id: Optional[int] = None
    news_bot_user_id: Optional[int] = None
    
    # Database Configuration
    database_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        # Load from .env if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        def get_bool(key: str, default: bool = False) -> bool:
            """Parse boolean from environment variable."""
            value = os.getenv(key, str(default)).lower()
            return value in ['true', '1', 't', 'yes', 'y']
            
        def get_int(key: str, default: int = 0) -> int:
            """Parse integer from environment variable."""
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default
                
        def get_optional_int(key: str) -> Optional[int]:
            """Parse optional integer from environment variable."""
            value = os.getenv(key)
            if value:
                try:
                    return int(value)
                except ValueError:
                    pass
            return None
        
        return cls(
            # Discord Configuration
            discord_token=os.getenv('DISCORD_TOKEN', ''),
            bot_prefix=os.getenv('BOT_PREFIX', '!'),
            owner_id=get_int('OWNER_ID'),
            
            # API Configuration
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            perplexity_api_key=os.getenv('PERPLEXITY_API_KEY', ''),
            perplexity_api_url=os.getenv('PERPLEXITY_API_URL', 'https://api.perplexity.ai/chat/completions'),
            perplexity_timeout=get_int('PERPLEXITY_TIMEOUT', 30),
            use_perplexity_api=get_bool('USE_PERPLEXITY_API', True),
            
            # Logging Configuration
            log_level=os.getenv('LOG_LEVEL', 'INFO').upper(),
            log_format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            log_channel_id=get_optional_int('LOG_CHANNEL_ID'),
            
            # Rate Limiting
            api_rate_limit_max=get_int('API_RATE_LIMIT_MAX', 100),
            api_rate_limit_interval=get_int('API_RATE_LIMIT_INTERVAL', 60),
            daily_api_call_limit=get_int('DAILY_API_CALL_LIMIT', 1000),
            
            # Context Management
            max_context_messages=get_int('MAX_CONTEXT_MESSAGES', 50),
            max_context_age=get_int('MAX_CONTEXT_AGE', 3600),
            max_messages_per_channel=get_int('MAX_MESSAGES_PER_CHANNEL', 1000),
            
            # Retry Configuration
            max_retries=get_int('MAX_RETRIES', 3),
            retry_delay=get_int('RETRY_DELAY', 5),
            
            # Feature Configuration
            stats_interval=get_int('STATS_INTERVAL', 86400),
            
            # Channel IDs
            summary_channel_id=get_optional_int('SUMMARY_CHANNEL_ID'),
            chartist_channel_id=get_optional_int('CHARTIST_CHANNEL_ID'),
            news_channel_id=get_optional_int('NEWS_CHANNEL_ID'),
            news_bot_user_id=get_optional_int('NEWS_BOT_USER_ID'),
            
            # Database
            database_url=os.getenv('DATABASE_URL'),
        )


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings