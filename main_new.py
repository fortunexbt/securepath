"""
SecurePath AI Discord Bot - Refactored Entry Point

A crypto-focused Discord bot with AI-powered analysis and research capabilities.
"""
import asyncio
import logging
import signal
import sys
from pathlib import Path

import aiohttp
from aiohttp import ClientSession, TCPConnector
from rich.console import Console
from rich.logging import RichHandler

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import get_settings
from src.bot.client import create_bot
from src.ai import AIManager
from src.database import db_manager

# Initialize logging
console = Console()
logger = logging.getLogger('SecurePathAgent')


def setup_logging() -> None:
    """Set up logging configuration."""
    settings = get_settings()
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level, 'INFO'),
        format=settings.log_format,
        handlers=[
            RichHandler(rich_tracebacks=True, console=console)
        ]
    )
    
    # Reduce Discord library noise
    for module in ['discord', 'discord.http', 'discord.gateway', 'aiohttp']:
        logging.getLogger(module).setLevel(logging.WARNING)
        
    logger.info("Logging configured successfully")


async def create_http_session() -> ClientSession:
    """Create HTTP session for API calls."""
    connector = TCPConnector(limit=10, limit_per_host=5)
    session = ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=30)
    )
    logger.info("HTTP session created")
    return session


async def setup_bot_services(bot, session: ClientSession) -> None:
    """Set up bot services and dependencies."""
    settings = get_settings()
    
    # Create AI manager
    ai_manager = AIManager(
        session=session,
        rate_limiter=bot.rate_limiter
    )
    
    # Attach to bot for access by cogs
    bot.ai_manager = ai_manager
    bot.session = session
    
    logger.info("Bot services configured")


async def startup_sequence() -> None:
    """Execute startup sequence."""
    logger.info("🚀 Starting SecurePath Agent...")
    
    # Load settings
    settings = get_settings()
    logger.info(f"Configuration loaded - Environment: {settings.log_level}")
    
    # Create HTTP session
    session = await create_http_session()
    
    try:
        # Connect to database
        db_connected = await db_manager.connect()
        if db_connected:
            logger.info("✅ Database connection established")
        else:
            logger.warning("⚠️ Database connection failed - limited functionality")
            
        # Create bot
        bot = create_bot()
        
        # Set up bot services
        await setup_bot_services(bot, session)
        
        # Set up signal handlers for graceful shutdown
        if sys.platform != "win32":
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_event_loop().add_signal_handler(
                    sig, lambda: asyncio.create_task(shutdown_sequence(bot, session))
                )
                
        logger.info("🎯 Bot initialization complete")
        
        # Start bot
        async with bot:
            await bot.start(settings.discord_token)
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        await shutdown_sequence(None, session)
        raise
    finally:
        await session.close()


async def shutdown_sequence(bot=None, session=None) -> None:
    """Execute graceful shutdown sequence."""
    logger.info("🛑 Initiating graceful shutdown...")
    
    # Cancel all running tasks
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    if tasks:
        logger.info(f"Cancelling {len(tasks)} running tasks...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Clean up AI manager
    if bot and hasattr(bot, 'ai_manager'):
        await bot.ai_manager.cleanup()
        
    # Close database connections
    if db_manager:
        await db_manager.disconnect()
        logger.info("Database connections closed")
    
    # Close HTTP session
    if session and not session.closed:
        await session.close()
        logger.info("HTTP session closed")
        
    # Close bot
    if bot and not bot.is_closed():
        await bot.close()
        logger.info("Bot connection closed")
        
    logger.info("✅ Shutdown complete")


def ensure_single_instance() -> None:
    """Ensure only one instance of the bot is running."""
    lock_file = '/tmp/securepath_bot.lock'
    try:
        import fcntl
        fp = open(lock_file, 'w')
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.debug(f"Acquired lock on {lock_file}")
        return fp
    except (IOError, ImportError):
        logger.warning("Could not acquire lock. Multiple instances may be running.")
        return None


def main() -> None:
    """Main entry point."""
    # Set up logging first
    setup_logging()
    
    # Ensure single instance
    lock_handle = ensure_single_instance()
    
    try:
        # Run the bot
        asyncio.run(startup_sequence())
    except KeyboardInterrupt:
        logger.info("👋 Bot shutdown requested by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if lock_handle:
            lock_handle.close()
        logger.info("🔒 Process lock released")


if __name__ == "__main__":
    main()