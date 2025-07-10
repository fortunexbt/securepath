"""Administrative commands for the SecurePath bot."""
import logging
from datetime import datetime, timezone
from typing import Optional

import discord
from discord.ext import commands
from discord.ext.commands import Context, Cog

from ...ai import AIManager
from ...database import db_manager
from ...config.settings import get_settings
from ...utils.discord_helpers import is_admin_user

logger = logging.getLogger(__name__)


class AdminCommands(Cog):
    """Cog for administrative commands."""
    
    def __init__(self, bot: commands.Bot):
        """Initialize admin commands cog."""
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
            
    @commands.command(name='ping')
    async def ping(self, ctx: Context) -> None:
        """Check SecurePath Agent latency and database status."""
        start_time = discord.utils.utcnow()
        message = await ctx.send("🏓 Pinging...")
        end_time = discord.utils.utcnow()
        
        latency = round(self.bot.latency * 1000)
        response_time = round((end_time - start_time).total_seconds() * 1000)
        
        # Check database status
        db_status = "🟢 Connected" if db_manager.pool else "🔴 Disconnected"
        
        # Get AI manager stats if available
        ai_stats = {}
        if self.ai_manager:
            ai_stats = self.ai_manager.get_usage_stats()
            
        embed = discord.Embed(
            title="🏓 Agent Status Check",
            description="All systems operational",
            color=0x1D82B6,
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.add_field(name="Discord Latency", value=f"{latency}ms", inline=True)
        embed.add_field(name="Response Time", value=f"{response_time}ms", inline=True)
        embed.add_field(name="Database", value=db_status, inline=True)
        embed.add_field(name="Requests Today", value=f"{ai_stats.get('daily_requests', 0)}", inline=True)
        embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1 & Perplexity Sonar-Pro")
        
        await message.edit(content="", embed=embed)
        
    @commands.command(name='commands')
    async def commands_help(self, ctx: Context) -> None:
        """Show SecurePath Agent help and available commands."""
        embed = discord.Embed(
            title="⚡ SecurePath Agent",
            description="*mario's crypto agent • show me the docs, show me the code*",
            color=0x00D4AA,  # SecurePath green
            timestamp=datetime.now(timezone.utc)
        )
        
        # Main Commands Section
        embed.add_field(
            name="",
            value="**🔍 `!ask [question]`**\n"
                  "▸ real-time market insights via perplexity sonar-pro\n"
                  "▸ sources: github, defi data, news, official docs\n"
                  "▸ *example:* `!ask solana vs ethereum fees`\n\n"
                  
                  "**📊 `!analyze [image]`**\n"
                  "▸ advanced chart analysis with gpt-4.1 vision\n"
                  "▸ sentiment, key levels, patterns, trade setups\n"
                  "▸ *attach image or use recent chart in channel*\n\n"
                  
                  "**📄 `!summary #channel`**\n"
                  "▸ alpha-focused channel activity digest\n"
                  "▸ extracts sentiment, events, key movements\n"
                  "▸ *example:* `!summary #crypto-news`",
            inline=False
        )
        
        # Utilities & Info
        embed.add_field(
            name="",
            value="**📈 `!stats`** • usage analytics\n"
                  "**🏓 `!ping`** • latency check\n"
                  "**⚙️ `!cache_stats`** • performance metrics",
            inline=True
        )
        
        # Key Features
        embed.add_field(
            name="",
            value="**✨ features**\n"
                  "▸ elite source filtering\n"
                  "▸ context-aware conversations\n"
                  "▸ real-time progress tracking\n"
                  "▸ no-fluff alpha extraction",
            inline=True
        )
        
        # Bottom spacing
        embed.add_field(name="", value="", inline=False)
        
        embed.set_footer(
            text="SecurePath Agent • Powered by Perplexity Sonar-Pro & GPT-4.1 Vision"
        )
        
        await ctx.send(embed=embed)
        
    @commands.command(name='stats')
    @commands.has_permissions(administrator=True)
    async def unified_stats(self, ctx: Context) -> None:
        """Show comprehensive SecurePath Agent analytics (admin only)."""
        if not is_admin_user(ctx.author, self.settings.owner_id):
            await ctx.send("You do not have permission to use this command.")
            return
            
        if not db_manager.pool:
            await ctx.send("Database not available. Stats tracking is currently offline.")
            return
            
        try:
            # Get all data in parallel
            stats_data = await db_manager.get_global_stats()
            costs_data = await db_manager.get_costs_by_model()
            query_data = await db_manager.get_query_analytics()
            
            if not stats_data:
                await ctx.send("Failed to retrieve statistics.")
                return
                
            overall = stats_data['overall']
            top_users = stats_data['top_users']
            top_commands = stats_data['top_commands']
            
            embed = discord.Embed(
                title="📊 SecurePath Agent Analytics",
                description="Comprehensive usage analytics and performance metrics",
                color=0x1D82B6,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Overall Usage Statistics
            embed.add_field(
                name="📈 Overall Performance",
                value=f"**Total Requests:** {overall['total_requests']:,}\n"
                      f"**Active Users:** {overall['unique_users']:,}\n"
                      f"**Total Tokens:** {overall['total_tokens']:,}\n"
                      f"**Total Cost:** ${overall['total_cost']:.4f}\n"
                      f"**Avg Tokens/Request:** {overall['avg_tokens_per_request']:.1f}",
                inline=True
            )
            
            # Model Cost Breakdown
            if costs_data and costs_data['model_costs']:
                cost_text = ""
                for model in costs_data['model_costs'][:3]:
                    cost_text += f"**{model['model']}:** {model['requests']:,} req, ${model['total_cost']:.4f}\n"
                embed.add_field(name="💰 Model Costs", value=cost_text or "No data", inline=True)
                
            # Top Active Users
            if top_users:
                users_text = "\n".join([
                    f"**{user['username'][:15]}:** {user['total_requests']} req, ${user['total_cost']:.3f}"
                    for user in top_users[:6]
                ])
                embed.add_field(name="👑 Top Users", value=users_text, inline=True)
                
            # Popular Commands (filter out background commands)
            if top_commands:
                filtered_commands = [
                    cmd for cmd in top_commands 
                    if cmd['command'] not in ['summary_chunk', 'summary_final']
                ]
                commands_text = "\n".join([
                    f"**{cmd['command']}:** {cmd['usage_count']} uses, ${cmd['total_cost']:.3f}"
                    for cmd in filtered_commands[:6]
                ])
                embed.add_field(name="🎯 Popular Commands", value=commands_text, inline=False)
                
            # Query Analytics
            if query_data and query_data['command_patterns']:
                query_text = "\n".join([
                    f"**{cmd['command']}:** {cmd['total_queries']} queries, {cmd['unique_users']} users"
                    for cmd in query_data['command_patterns'][:4]
                ])
                embed.add_field(name="🔍 Query Patterns", value=query_text, inline=True)
                
            # Peak Usage Hours
            if query_data and query_data['hourly_activity']:
                hours_text = "\n".join([
                    f"**{int(hour['hour'])}:00:** {hour['query_count']} queries"
                    for hour in query_data['hourly_activity'][:4]
                ])
                embed.add_field(name="⏰ Peak Hours", value=hours_text, inline=True)
                
            # System Performance
            cache_hit_rate = 0.0
            ai_requests = 0
            if self.ai_manager:
                ai_stats = self.ai_manager.get_usage_stats()
                cache_hit_rate = ai_stats.get('cache_hit_rate', 0.0)
                ai_requests = ai_stats.get('daily_requests', 0)
                
            embed.add_field(
                name="⚡ System Performance",
                value=f"**Cache Hit Rate:** {cache_hit_rate:.1f}%\n"
                      f"**AI Requests:** {ai_requests:,}\n"
                      f"**Active Guilds:** {len(self.bot.guilds)}",
                inline=True
            )
            
            embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1 & Perplexity Sonar-Pro")
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in stats command: {e}")
            await ctx.send(f"Error retrieving stats: {str(e)}")
            
    @commands.command(name='token_usage')
    @commands.has_permissions(administrator=True)
    async def token_usage(self, ctx: Context) -> None:
        """Show token usage and costs (admin only)."""
        if not is_admin_user(ctx.author, self.settings.owner_id):
            await ctx.send("You do not have permission to use this command.")
            return
            
        if not self.ai_manager:
            await ctx.send("AI manager not available.")
            return
            
        embed = discord.Embed(
            title="📊 Token Usage and Costs", 
            color=0x1D82B6, 
            timestamp=datetime.now(timezone.utc)
        )
        
        # Get usage stats from AI manager
        stats = self.ai_manager.get_usage_stats()
        
        # OpenAI stats
        openai_stats = stats.get('openai', {})
        openai_text = "\n".join([
            f"**{k.replace('_', ' ').title()}:** {v}"
            for k, v in openai_stats.items()
        ])
        embed.add_field(name="OpenAI GPT-4.1", value=openai_text or "No data", inline=False)
        
        # Perplexity stats
        perplexity_stats = stats.get('perplexity', {})
        perplexity_text = "\n".join([
            f"**{k.replace('_', ' ').title()}:** {v}"
            for k, v in perplexity_stats.items()
        ])
        embed.add_field(name="Perplexity Sonar-Pro", value=perplexity_text or "No data", inline=False)
        
        await ctx.send(embed=embed)
        
    @commands.command(name='cache_stats')
    @commands.has_permissions(administrator=True)
    async def cache_stats(self, ctx: Context) -> None:
        """Show cache hit rate (admin only)."""
        if not is_admin_user(ctx.author, self.settings.owner_id):
            await ctx.send("You do not have permission to use this command.")
            return
            
        hit_rate = 0.0
        if self.ai_manager:
            stats = self.ai_manager.get_usage_stats()
            hit_rate = stats.get('cache_hit_rate', 0.0)
            
        embed = discord.Embed(
            title="📊 Cache Hit Rate", 
            description=f"OpenAI GPT-4.1 Cache Hit Rate: **{hit_rate:.2f}%**", 
            color=0x1D82B6
        )
        await ctx.send(embed=embed)


async def setup(bot: commands.Bot) -> None:
    """Set up the admin commands cog."""
    await bot.add_cog(AdminCommands(bot))