"""Discord bot client setup and initialization."""
import logging
from typing import Optional

import discord
from discord.ext import commands
from discord.ext.commands import Bot

from ..config.settings import get_settings
from ..services.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class SecurePathBot(Bot):
    """Enhanced Discord bot with custom functionality."""
    
    def __init__(self):
        """Initialize the SecurePath bot."""
        settings = get_settings()
        
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(
            command_prefix=settings.bot_prefix,
            intents=intents,
            help_command=None  # We'll use custom help
        )
        
        self.settings = settings
        self.rate_limiter: Optional[RateLimiter] = None
        self._ready = False
        
    async def setup_hook(self) -> None:
        """Set up the bot before starting."""
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_calls=self.settings.api_rate_limit_max,
            interval=self.settings.api_rate_limit_interval
        )
        
        # Load cogs
        await self.load_extensions()
        
        logger.info("Bot setup completed")
        
    async def load_extensions(self) -> None:
        """Load all bot extensions/cogs."""
        extensions = [
            "src.bot.cogs.ai_commands",
            "src.bot.cogs.admin_commands",
            "src.bot.cogs.summary_commands",
        ]
        
        for ext in extensions:
            try:
                await self.load_extension(ext)
                logger.info(f"Loaded extension: {ext}")
            except Exception as e:
                logger.error(f"Failed to load extension {ext}: {e}")
                
    async def on_ready(self) -> None:
        """Called when the bot is ready."""
        if self._ready:
            return
            
        self._ready = True
        logger.info(f"{self.user} has connected to Discord!")
        logger.info(f"Active in {len(self.guilds)} guild(s)")
        
        # Start background tasks
        from .events import setup_background_tasks
        await setup_background_tasks(self)
        
    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages."""
        # Ignore bot messages
        if message.author.bot:
            return
            
        # Process commands
        await self.process_commands(message)
        
        # Handle DM conversations
        if isinstance(message.channel, discord.DMChannel) and not message.content.startswith(self.settings.bot_prefix):
            from .events import handle_dm_conversation
            await handle_dm_conversation(self, message)
            
    async def close(self) -> None:
        """Clean up bot resources."""
        logger.info("Shutting down SecurePath bot...")
        await super().close()


def create_bot() -> SecurePathBot:
    """Create and return a bot instance."""
    return SecurePathBot()