"""Bot event handlers and background tasks."""
import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import List

import discord
from discord import Activity, ActivityType
from discord.ext import tasks

from ..database import db_manager
from ..services.context_manager import ContextManager
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Status messages for rotation
STATUS_MESSAGES = [
    ("!ask", "real-time market insights", ActivityType.watching),
    ("!analyze", "chart patterns & signals", ActivityType.watching),
    ("!summary", "alpha extraction from channels", ActivityType.listening),
    ("!commands", "for all features", ActivityType.playing),
    ("defi", "on-chain truth over hype", ActivityType.watching),
    ("docs", "show me the code", ActivityType.watching),
]


async def setup_background_tasks(bot) -> None:
    """Set up all background tasks for the bot."""
    # Start status rotation
    if not change_status.is_running():
        change_status.start(bot)
        logger.info("Started status rotation task")
        
    # Start daily reset
    if not reset_daily_limits.is_running():
        reset_daily_limits.start(bot)
        logger.info("Started daily reset task")
        
    # Send startup notification
    await send_startup_notification(bot)
    
    # Initialize database
    db_connected = await db_manager.connect()
    if db_connected:
        logger.info("Database connection established")
    else:
        logger.error("Failed to connect to database")


@tasks.loop(minutes=15)
async def change_status(bot) -> None:
    """Rotate bot status messages."""
    try:
        status = random.choice(STATUS_MESSAGES)
        name, state, activity_type = status
        activity = Activity(type=activity_type, name=f"{name} • {state}")
        await bot.change_presence(activity=activity)
        logger.debug(f"Changed status to: {name} • {state}")
    except Exception as e:
        logger.error(f"Error changing status: {e}")


@tasks.loop(hours=24)
async def reset_daily_limits(bot) -> None:
    """Reset daily API call limits and usage data."""
    # This will be implemented when we create the usage tracking service
    logger.info("Daily limits reset")


async def send_startup_notification(bot) -> None:
    """Send startup notification to admin channel."""
    settings = get_settings()
    
    if not settings.log_channel_id:
        logger.warning("No log channel configured for startup notification")
        return
        
    channel = bot.get_channel(settings.log_channel_id)
    if not channel:
        logger.warning(f"Could not find log channel {settings.log_channel_id}")
        return
        
    embed = discord.Embed(
        title="🚀 SecurePath Agent - System Status",
        description="Agent successfully initialized and ready for operations",
        color=0x1D82B6,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Add status fields
    db_status = "🟢 Connected" if db_manager.pool else "🔴 Disconnected"
    embed.add_field(name="Database", value=db_status, inline=True)
    embed.add_field(name="Active Guilds", value=len(bot.guilds), inline=True)
    embed.add_field(name="Latency", value=f"{bot.latency*1000:.1f}ms", inline=True)
    
    # Add usage stats if database is connected
    if db_manager.pool:
        try:
            stats = await db_manager.get_global_stats()
            if stats and stats.get('overall'):
                overall = stats['overall']
                embed.add_field(
                    name="📊 Total Usage",
                    value=f"**Requests:** {overall['total_requests']:,}\n"
                          f"**Users:** {overall['unique_users']:,}\n"
                          f"**Cost:** ${overall['total_cost']:.4f}",
                    inline=True
                )
        except Exception as e:
            logger.error(f"Failed to get startup stats: {e}")
            
    embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1 & Perplexity Sonar-Pro")
    
    try:
        await channel.send(embed=embed)
        logger.info("Startup notification sent")
    except discord.HTTPException as e:
        logger.error(f"Failed to send startup notification: {e}")


async def handle_dm_conversation(bot, message: discord.Message) -> None:
    """Handle DM conversations with context management."""
    # Get or create context manager for user
    context_manager = ContextManager.get_instance()
    
    # Preload conversation history if new conversation
    if not context_manager.has_context(message.author.id):
        await preload_conversation_history(bot, message.author.id, message.channel)
        
    # This will be handled by the AI command handler
    # For now, just log that we received a DM
    logger.info(f"Received DM from {message.author}: {message.content[:50]}...")


async def preload_conversation_history(bot, user_id: int, channel: discord.DMChannel) -> None:
    """Preload conversation history for context."""
    context_manager = ContextManager.get_instance()
    messages = []
    
    try:
        async for msg in channel.history(limit=100, oldest_first=True):
            if msg.author.id == user_id:
                role = 'user'
            elif msg.author.id == bot.user.id:
                role = 'assistant'
            else:
                continue
                
            messages.append({
                'role': role,
                'content': msg.content,
                'timestamp': msg.created_at.timestamp()
            })
            
        # Initialize context with history
        for msg in messages:
            context_manager.update_context(user_id, msg['content'], msg['role'])
            
        logger.info(f"Preloaded {len(messages)} messages for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error preloading conversation history: {e}")