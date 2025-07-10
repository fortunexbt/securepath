"""Summary commands for the SecurePath bot."""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import discord
from discord.ext import commands
from discord.ext.commands import Context, Cog

from ...ai import AIManager
from ...database import db_manager
from ...config.settings import get_settings
from ...utils.discord_helpers import send_long_embed, reset_status

logger = logging.getLogger(__name__)


class SummaryCommands(Cog):
    """Cog for channel summary commands."""
    
    def __init__(self, bot: commands.Bot):
        """Initialize summary commands cog."""
        self.bot = bot
        self.settings = get_settings()
        self.ai_manager: Optional[AIManager] = None
        
    async def cog_load(self) -> None:
        """Set up the cog when loaded."""
        # Get AI manager from bot
        if hasattr(self.bot, 'ai_manager'):
            self.ai_manager = self.bot.ai_manager
        else:
            logger.warning("AI manager not found on bot instance")
            
    @commands.command(name='summary')
    async def summary(self, ctx: Context, channel: Optional[discord.TextChannel] = None) -> None:
        """Generate an alpha-focused summary of channel activity."""
        # Default to current channel if none specified
        if not channel:
            channel = ctx.channel
            
        # Validate permissions
        if not channel.permissions_for(ctx.guild.me).read_message_history:
            await ctx.send(f"❌ I don't have permission to read message history in {channel.mention}")
            return
            
        # Log the summary command
        await self._log_summary_command(ctx, channel)
        
        # Create status embed
        status_embed = discord.Embed(
            title="📄 SecurePath Agent Channel Analysis",
            description=f"**Channel:** {channel.mention}\n**Timeframe:** Last 72 hours",
            color=0x1D82B6,
            timestamp=datetime.now(timezone.utc)
        )
        status_embed.add_field(name="Status", value="🔄 Gathering messages...", inline=False)
        status_embed.set_footer(text="SecurePath Agent • Alpha Extraction")
        
        status_msg = await ctx.send(embed=status_embed)
        
        try:
            # Gather messages from the last 72 hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=72)
            messages = await self._gather_channel_messages(channel, cutoff_time)
            
            if len(messages) < 10:
                error_embed = discord.Embed(
                    title="❌ Insufficient Data",
                    description=f"Only found {len(messages)} messages in {channel.mention} from the last 72 hours.",
                    color=0xFF0000
                )
                error_embed.add_field(
                    name="Minimum Required", 
                    value="At least 10 messages needed for meaningful analysis.", 
                    inline=False
                )
                await status_msg.edit(embed=error_embed)
                return
                
            # Update status
            status_embed.set_field_at(
                0, 
                name="Status", 
                value=f"📊 Processing {len(messages):,} messages...", 
                inline=False
            )
            await status_msg.edit(embed=status_embed)
            
            # Generate summary using AI manager
            if not self.ai_manager:
                raise Exception("AI manager not available")
                
            summary = await self.ai_manager.summarize_messages(
                messages=messages,
                channel_name=channel.name
            )
            
            # Update status: finalizing
            status_embed.set_field_at(
                0, 
                name="Status", 
                value="✨ Finalizing intelligence report...", 
                inline=False
            )
            await status_msg.edit(embed=status_embed)
            
            # Brief pause for UX
            await asyncio.sleep(1)
            
            # Delete status message
            await status_msg.delete()
            
            # Send final summary
            await self._send_summary_result(ctx, channel, summary, len(messages))
            
            # Log to database
            await self._log_summary_usage(ctx, channel, summary, len(messages))
            
        except Exception as e:
            logger.error(f"Error in summary command: {e}")
            error_embed = discord.Embed(
                title="❌ Processing Failed",
                description=f"An error occurred while processing {channel.mention}.",
                color=0xFF0000
            )
            error_embed.add_field(name="Error", value=str(e)[:1000], inline=False)
            await status_msg.edit(embed=error_embed)
            
        finally:
            await reset_status(self.bot)
            
    async def _gather_channel_messages(
        self, 
        channel: discord.TextChannel, 
        cutoff_time: datetime
    ) -> List[str]:
        """Gather and filter messages from a channel."""
        messages = []
        
        try:
            async for message in channel.history(limit=None, after=cutoff_time):
                # Skip bot messages and system messages
                if message.author.bot or message.type != discord.MessageType.default:
                    continue
                    
                # Skip very short messages
                if len(message.content.strip()) < 10:
                    continue
                    
                # Skip messages that are just links
                if self._is_mostly_links(message.content):
                    continue
                    
                # Format message with metadata
                formatted_msg = self._format_message_for_analysis(message)
                messages.append(formatted_msg)
                
        except discord.HTTPException as e:
            logger.error(f"Error gathering messages from {channel.name}: {e}")
            raise Exception("Failed to gather channel messages")
            
        # Sort by timestamp (oldest first for context)
        messages.reverse()
        return messages
        
    def _is_mostly_links(self, content: str) -> bool:
        """Check if message is mostly links."""
        words = content.split()
        if not words:
            return False
            
        link_count = sum(1 for word in words if word.startswith(('http://', 'https://', 'www.')))
        return link_count / len(words) > 0.5
        
    def _format_message_for_analysis(self, message: discord.Message) -> str:
        """Format a message for AI analysis."""
        timestamp = message.created_at.strftime("%H:%M")
        username = message.author.display_name[:20]  # Truncate long usernames
        content = message.content[:500]  # Truncate long messages
        
        return f"[{timestamp}] {username}: {content}"
        
    async def _send_summary_result(
        self, 
        ctx: Context, 
        channel: discord.TextChannel, 
        summary: str, 
        message_count: int
    ) -> None:
        """Send the summary result to the user."""
        # Create title embed
        title_embed = discord.Embed(
            title=f"📄 {channel.name.title()} Intelligence Report",
            description=f"**Timeframe:** Last 72 hours | **Messages Analyzed:** {message_count:,}",
            color=0x1D82B6,
            timestamp=datetime.now(timezone.utc)
        )
        title_embed.set_footer(text="SecurePath Agent • Alpha Extraction Engine")
        
        # Send title embed first
        await ctx.send(embed=title_embed)
        
        # Send summary content
        if len(summary) <= 3800:  # Fits in single embed
            summary_embed = discord.Embed(
                description=summary,
                color=0x1D82B6
            )
            await ctx.send(embed=summary_embed)
        else:
            # Use long embed for detailed summaries
            await send_long_embed(
                channel=ctx.channel,
                content=summary,
                color=0x1D82B6,
                title="📈 Detailed Analysis"
            )
            
    async def _log_summary_command(self, ctx: Context, channel: discord.TextChannel) -> None:
        """Log summary command to database."""
        if db_manager.pool:
            username = f"{ctx.author.name}#{ctx.author.discriminator}" if ctx.author.discriminator != "0" else ctx.author.name
            query_text = f"Summary for #{channel.name}"
            
            await db_manager.log_user_query(
                user_id=ctx.author.id,
                username=username,
                command="summary",
                query_text=query_text,
                channel_id=ctx.channel.id,
                guild_id=ctx.guild.id if ctx.guild else None,
                response_generated=False
            )
            
    async def _log_summary_usage(
        self, 
        ctx: Context, 
        channel: discord.TextChannel, 
        summary: str, 
        message_count: int
    ) -> None:
        """Log summary usage to database."""
        if db_manager.pool:
            # Calculate estimated cost and tokens
            estimated_input_tokens = message_count * 50  # Rough estimate
            estimated_output_tokens = len(summary.split()) * 1.3  # Rough estimate
            estimated_cost = (estimated_input_tokens * 0.40 + estimated_output_tokens * 1.60) / 1_000_000
            
            try:
                await db_manager.log_usage(
                    user_id=ctx.author.id,
                    username=f"{ctx.author.name}#{ctx.author.discriminator}" if ctx.author.discriminator != "0" else ctx.author.name,
                    command="summary",
                    model="gpt-4.1",
                    input_tokens=int(estimated_input_tokens),
                    output_tokens=int(estimated_output_tokens),
                    cost=estimated_cost,
                    guild_id=ctx.guild.id if ctx.guild else None,
                    channel_id=ctx.channel.id
                )
                logger.info(f"Summary usage logged - Cost: ${estimated_cost:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to log summary usage: {e}")


async def setup(bot: commands.Bot) -> None:
    """Set up the summary commands cog."""
    await bot.add_cog(SummaryCommands(bot))