"""Discord-specific utility functions."""
import asyncio
import logging
import random
from typing import Optional, Union

import discord
from discord import Activity, ActivityType

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


async def reset_status(bot) -> None:
    """Reset bot status to a random default status."""
    try:
        status = random.choice(STATUS_MESSAGES)
        name, state, activity_type = status
        activity = Activity(type=activity_type, name=f"{name} • {state}")
        await bot.change_presence(activity=activity)
        logger.debug(f"Reset status to: {name} • {state}")
    except Exception as e:
        logger.error(f"Error resetting status: {e}")


async def send_long_message(channel, content: str, max_length: int = 2000) -> None:
    """
    Send a long message by splitting it into chunks.
    
    Args:
        channel: Discord channel to send to
        content: Message content
        max_length: Maximum length per message
    """
    if len(content) <= max_length:
        await channel.send(content)
        return
        
    # Split content into chunks
    chunks = []
    current_chunk = ""
    
    for line in content.split('\n'):
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # Send chunks
    for chunk in chunks:
        await channel.send(chunk)
        await asyncio.sleep(0.5)  # Small delay to avoid rate limits


async def send_long_embed(
    channel,
    content: str,
    color: int = 0x1D82B6,
    title: str = None,
    max_description_length: int = 4096
) -> None:
    """
    Send content as embeds, splitting if necessary.
    
    Args:
        channel: Discord channel to send to
        content: Content to send
        color: Embed color
        title: Embed title
        max_description_length: Maximum description length per embed
    """
    if len(content) <= max_description_length:
        embed = discord.Embed(
            title=title,
            description=content,
            color=color
        )
        await channel.send(embed=embed)
        return
        
    # Split content into chunks
    chunks = []
    current_chunk = ""
    
    for paragraph in content.split('\n\n'):
        if len(current_chunk) + len(paragraph) + 2 <= max_description_length:
            current_chunk += paragraph + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n\n'
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # Send chunks as embeds
    for i, chunk in enumerate(chunks):
        embed_title = title if i == 0 else f"{title} (continued)"
        embed = discord.Embed(
            title=embed_title,
            description=chunk,
            color=color
        )
        await channel.send(embed=embed)
        await asyncio.sleep(0.5)


async def send_structured_analysis_embed(
    channel,
    text: str,
    color: int = 0x1D82B6,
    title: str = "Analysis",
    image_url: Optional[str] = None,
    user_mention: Optional[str] = None
) -> None:
    """
    Send a structured analysis embed with proper formatting.
    
    Args:
        channel: Discord channel to send to
        text: Analysis content
        color: Embed color
        title: Embed title
        image_url: Optional image URL to include
        user_mention: Optional user mention
    """
    try:
        # Create main embed
        embed = discord.Embed(
            title=title,
            color=color,
            timestamp=discord.utils.utcnow()
        )
        
        # Add image if provided
        if image_url:
            embed.set_image(url=image_url)
            
        # Add user mention if provided
        if user_mention:
            embed.description = f"Analysis requested by {user_mention}"
            
        # Try to fit content in embed description
        if len(text) <= 4096:
            embed.description = (embed.description or "") + f"\n\n{text}"
            await channel.send(embed=embed)
        else:
            # Send title embed first, then use long embed for content
            await channel.send(embed=embed)
            await send_long_embed(
                channel=channel,
                content=text,
                color=color,
                title="📊 Detailed Analysis"
            )
            
    except discord.HTTPException as e:
        logger.error(f"Failed to send analysis embed: {e}")
        # Fallback to text message
        fallback_text = f"**{title}**\n\n{text[:1800]}{'...' if len(text) > 1800 else ''}"
        await channel.send(fallback_text)


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a percentage value with proper sign and color."""
    formatted = f"{value:+.{decimals}f}%"
    return formatted


def format_price(value: float, currency: str = "USD") -> str:
    """Format a price value with currency symbol."""
    if currency.upper() == "USD":
        return f"${value:,.2f}"
    else:
        return f"{value:,.4f} {currency}"


def format_large_number(value: int) -> str:
    """Format large numbers with K, M, B suffixes."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return str(value)


def create_progress_embed(
    title: str,
    description: str = None,
    status: str = "Initializing...",
    color: int = 0x1D82B6
) -> discord.Embed:
    """Create a standard progress embed."""
    embed = discord.Embed(
        title=title,
        description=description,
        color=color
    )
    embed.add_field(name="Status", value=status, inline=False)
    embed.set_footer(text="SecurePath Agent")
    return embed


def is_admin_user(user: discord.User, owner_id: int) -> bool:
    """Check if user is an admin."""
    return user.id == owner_id


def get_user_display_name(user: discord.User) -> str:
    """Get user's display name for database storage."""
    if user.discriminator != "0":
        return f"{user.name}#{user.discriminator}"
    else:
        return user.name


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_command_args(content: str, prefix: str) -> tuple[str, str]:
    """
    Extract command and arguments from message content.
    
    Args:
        content: Message content
        prefix: Bot prefix
        
    Returns:
        Tuple of (command, arguments)
    """
    if not content.startswith(prefix):
        return "", ""
        
    parts = content[len(prefix):].split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    
    return command, args