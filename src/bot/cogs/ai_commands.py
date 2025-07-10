"""AI-powered commands for the SecurePath bot."""
import asyncio
import logging
from typing import Optional

import discord
from discord import Activity, ActivityType
from discord.ext import commands
from discord.ext.commands import Context, Cog

from ...ai import AIManager
from ...database import db_manager
from ...utils.discord_helpers import send_structured_analysis_embed, reset_status
from ...config.settings import get_settings

logger = logging.getLogger(__name__)


class AICommands(Cog):
    """Cog for AI-powered commands."""
    
    def __init__(self, bot: commands.Bot):
        """Initialize AI commands cog."""
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
            
    @commands.command(name='ask')
    async def ask(self, ctx: Context, *, question: Optional[str] = None) -> None:
        """Get real-time crypto market insights with AI-powered research."""
        await self.bot.change_presence(activity=Activity(type=ActivityType.playing, name="researching..."))
        
        # Show help if no question provided
        if not question:
            await self._show_ask_help(ctx)
            await reset_status(self.bot)
            return
            
        # Validate input
        if len(question) < 5:
            await ctx.send("⚠️ Please provide a more detailed question (at least 5 characters).")
            await reset_status(self.bot)
            return
            
        if len(question) > 500:
            await ctx.send("⚠️ Question is too long. Please keep it under 500 characters.")
            await reset_status(self.bot)
            return
            
        # Log query to database
        await self._log_user_query(ctx, "ask", question)
        
        # Create progress embed
        progress_embed = discord.Embed(
            title="🔍 SecurePath Agent Research",
            description=f"**Query:** {question[:100]}{'...' if len(question) > 100 else ''}",
            color=0x1D82B6
        )
        progress_embed.add_field(name="Status", value="🔄 Initializing research...", inline=False)
        progress_embed.set_footer(text="SecurePath Agent • Real-time Intelligence")
        
        status_msg = await ctx.send(embed=progress_embed)
        
        try:
            # Update progress
            progress_embed.set_field_at(0, name="Status", value="🌐 Searching elite sources...", inline=False)
            await status_msg.edit(embed=progress_embed)
            
            # Process query with AI manager
            if not self.ai_manager:
                raise Exception("AI manager not available")
                
            result = await self.ai_manager.process_query(
                user_id=ctx.author.id,
                query=question,
                use_context=True
            )
            
            # Update progress
            progress_embed.set_field_at(0, name="Status", value="✨ Synthesizing insights...", inline=False)
            await status_msg.edit(embed=progress_embed)
            
            # Brief pause for UX
            await asyncio.sleep(1)
            
            # Delete progress and send result
            await status_msg.delete()
            
            # Send response
            response_embed = discord.Embed(
                title="🔍 Research Results",
                description=result['content'],
                color=0x1D82B6,
                timestamp=discord.utils.utcnow()
            )
            response_embed.set_footer(text="SecurePath Agent • Powered by Perplexity Sonar-Pro")
            
            await ctx.send(embed=response_embed)
            
            # Log interaction
            await self._log_interaction(ctx, 'ask', question, result['content'])
            
        except Exception as e:
            logger.error(f"Error in ask command: {e}")
            error_embed = discord.Embed(
                title="❌ Research Failed",
                description="An error occurred while processing your query.",
                color=0xFF0000
            )
            error_embed.add_field(name="Error", value=str(e)[:1000], inline=False)
            await status_msg.edit(embed=error_embed)
            
        finally:
            await reset_status(self.bot)
            
    @commands.command(name='analyze')
    async def analyze(self, ctx: Context, *, user_prompt: str = '') -> None:
        """Analyze charts and images with AI-powered technical analysis."""
        await self.bot.change_presence(activity=Activity(type=ActivityType.watching, name="image analysis..."))
        
        # Log query to database
        query_text = f"Image analysis request" + (f" with prompt: {user_prompt}" if user_prompt else " (no additional prompt)")
        await self._log_user_query(ctx, "analyze", query_text)
        
        # Find image to analyze
        attachment = None
        
        # Check for direct attachment
        if ctx.message.attachments:
            for att in ctx.message.attachments:
                if att.content_type and att.content_type.startswith('image/'):
                    attachment = att
                    break
                    
        # If no attachment, look for recent images in channel
        if not attachment:
            if isinstance(ctx.channel, discord.DMChannel):
                await self._request_image_in_dm(ctx)
                await reset_status(self.bot)
                return
            else:
                attachment = await self._find_recent_image(ctx.channel)
                
        if attachment:
            await self._analyze_image_attachment(ctx, attachment, user_prompt)
        else:
            await self._show_analyze_help(ctx)
            
        await reset_status(self.bot)
        
    async def _show_ask_help(self, ctx: Context) -> None:
        """Show help for ask command."""
        help_embed = discord.Embed(
            title="🤔 Ask Command Help",
            description="Get real-time crypto market insights with AI-powered research.",
            color=0x1D82B6
        )
        help_embed.add_field(
            name="Usage", 
            value="`!ask [your question]`", 
            inline=False
        )
        help_embed.add_field(
            name="Examples", 
            value="• `!ask What's the latest news on Bitcoin?`\n"
                  "• `!ask Ethereum price prediction trends`\n"
                  "• `!ask What's happening with DeFi protocols?`", 
            inline=False
        )
        help_embed.set_footer(text="SecurePath Agent • Powered by Perplexity Sonar-Pro")
        await ctx.send(embed=help_embed)
        
    async def _show_analyze_help(self, ctx: Context) -> None:
        """Show help for analyze command."""
        help_embed = discord.Embed(
            title="🖼️ Analyze Command Help",
            description="Upload or attach a chart/image for AI-powered technical analysis.",
            color=0x1D82B6
        )
        help_embed.add_field(
            name="Usage", 
            value="1. Attach an image to your `!analyze` command\n2. Or use `!analyze` in a channel with recent images", 
            inline=False
        )
        help_embed.add_field(
            name="Optional Prompt", 
            value="`!analyze Look for support and resistance levels`", 
            inline=False
        )
        help_embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1 Vision")
        await ctx.send(embed=help_embed)
        
    async def _request_image_in_dm(self, ctx: Context) -> None:
        """Request image upload in DM."""
        await ctx.send("Please post the image you'd like to analyze.")
        
        def check(msg):
            return msg.author == ctx.author and msg.channel == ctx.channel and msg.attachments
            
        try:
            chart_message = await self.bot.wait_for('message', check=check, timeout=60.0)
            attachment = chart_message.attachments[0]
            await self._analyze_image_attachment(ctx, attachment, "")
        except asyncio.TimeoutError:
            await ctx.send("You took too long to post an image. Please try again.")
            
    async def _find_recent_image(self, channel) -> Optional[discord.Attachment]:
        """Find recent image in channel."""
        async for message in channel.history(limit=20):
            for attachment in message.attachments:
                if attachment.content_type and attachment.content_type.startswith('image/'):
                    return attachment
        return None
        
    async def _analyze_image_attachment(self, ctx: Context, attachment: discord.Attachment, user_prompt: str) -> None:
        """Analyze a Discord image attachment."""
        # Create progress embed
        progress_embed = discord.Embed(
            title="📈 SecurePath Agent Analysis",
            description=f"**Image:** [Chart Analysis]({attachment.url})\n**Prompt:** {user_prompt or 'Standard technical analysis'}",
            color=0x1D82B6
        )
        progress_embed.add_field(name="Status", value="🔄 Initializing image analysis...", inline=False)
        progress_embed.set_thumbnail(url=attachment.url)
        progress_embed.set_footer(text="SecurePath Agent • Real-time Analysis")
        
        status_msg = await ctx.send(embed=progress_embed)
        
        try:
            # Update progress
            progress_embed.set_field_at(0, name="Status", value="🖼️ Processing image with GPT-4.1 Vision...", inline=False)
            await status_msg.edit(embed=progress_embed)
            
            # Analyze image with AI manager
            if not self.ai_manager:
                raise Exception("AI manager not available")
                
            result = await self.ai_manager.analyze_image(
                user_id=ctx.author.id,
                attachment=attachment,
                user_query=user_prompt
            )
            
            # Update progress
            progress_embed.set_field_at(0, name="Status", value="✨ Finalizing technical analysis...", inline=False)
            await status_msg.edit(embed=progress_embed)
            
            # Brief pause for UX
            await asyncio.sleep(1)
            
            # Delete progress and send result
            await status_msg.delete()
            
            await send_structured_analysis_embed(
                ctx.channel,
                text=result['content'],
                color=0x1D82B6,
                title="📈 Chart Analysis",
                image_url=attachment.url,
                user_mention=ctx.author.mention
            )
            
            # Log interaction
            await self._log_interaction(ctx, 'analyze', user_prompt or 'No additional prompt', result['content'])
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            error_embed = discord.Embed(
                title="❌ Analysis Failed",
                description="An error occurred during image analysis.",
                color=0xFF0000
            )
            error_embed.add_field(name="Error", value=str(e)[:1000], inline=False)
            await status_msg.edit(embed=error_embed)
            
    async def _log_user_query(self, ctx: Context, command: str, query_text: str) -> None:
        """Log user query to database."""
        if db_manager.pool:
            username = f"{ctx.author.name}#{ctx.author.discriminator}" if ctx.author.discriminator != "0" else ctx.author.name
            await db_manager.log_user_query(
                user_id=ctx.author.id,
                username=username,
                command=command,
                query_text=query_text,
                channel_id=ctx.channel.id,
                guild_id=ctx.guild.id if ctx.guild else None,
                response_generated=False
            )
            
    async def _log_interaction(self, ctx: Context, command: str, user_input: str, bot_response: str) -> None:
        """Log interaction to database."""
        # This would typically call a database logging function
        logger.info(f"Interaction logged: {command} - {len(bot_response)} chars")


async def setup(bot: commands.Bot) -> None:
    """Set up the AI commands cog."""
    await bot.add_cog(AICommands(bot))