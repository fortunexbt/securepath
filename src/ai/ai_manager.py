"""AI service manager for coordinating all AI operations."""
import asyncio
import logging
from typing import Dict, List, Optional, Any

import aiohttp
import discord
from aiohttp import ClientSession, TCPConnector

from ..config.settings import get_settings
from ..services.context_manager import ContextManager
from ..services.rate_limiter import RateLimiter
from .openai_service import OpenAIService
from .perplexity_service import PerplexityService
from .vision_service import VisionService

logger = logging.getLogger(__name__)


class AIManager:
    """Manager for all AI services and operations."""
    
    def __init__(self, session: Optional[ClientSession] = None, rate_limiter: Optional[RateLimiter] = None):
        """Initialize AI manager."""
        self.settings = get_settings()
        self.session = session
        self.rate_limiter = rate_limiter
        
        # Initialize services
        self.openai_service = OpenAIService()
        self.perplexity_service = PerplexityService(session=session)
        self.vision_service = VisionService(self.openai_service)
        self.context_manager = ContextManager.get_instance()
        
        # Usage tracking
        self.total_requests = 0
        self.daily_requests = 0
        
    async def process_query(
        self, 
        user_id: int, 
        query: str, 
        use_context: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query using the appropriate AI service.
        
        Args:
            user_id: Discord user ID
            query: User's query
            use_context: Whether to use conversation context
            
        Returns:
            Response dict with content and metadata
        """
        # Check rate limits
        if self.rate_limiter:
            can_call, error_msg = self.rate_limiter.check_rate_limit(user_id)
            if not can_call:
                raise Exception(error_msg)
                
        # Update context
        if use_context:
            self.context_manager.update_context(user_id, query, 'user')
            
        # Choose AI service based on settings
        if self.settings.use_perplexity_api:
            result = await self._process_with_perplexity(user_id, query, use_context)
        else:
            result = await self._process_with_openai(user_id, query, use_context)
            
        # Update context with response
        if use_context and result.get('content'):
            self.context_manager.update_context(user_id, result['content'], 'assistant')
            
        # Track usage
        self.total_requests += 1
        self.daily_requests += 1
        
        return result
        
    async def analyze_image(
        self, 
        user_id: int, 
        image_data: bytes = None,
        attachment: discord.Attachment = None,
        prompt: str = None,
        user_query: str = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using vision models.
        
        Args:
            user_id: Discord user ID
            image_data: Raw image bytes
            attachment: Discord attachment
            prompt: Custom analysis prompt
            user_query: User's specific question
            
        Returns:
            Analysis result
        """
        # Check rate limits
        if self.rate_limiter:
            can_call, error_msg = self.rate_limiter.check_rate_limit(user_id)
            if not can_call:
                raise Exception(error_msg)
                
        # Analyze image
        if attachment:
            result = await self.vision_service.analyze_discord_image(
                attachment=attachment,
                prompt=prompt,
                user_query=user_query
            )
        elif image_data:
            result = await self.vision_service.analyze_image(
                image_data=image_data,
                prompt=prompt,
                user_query=user_query
            )
        else:
            raise ValueError("Either image_data or attachment must be provided")
            
        # Track usage
        self.total_requests += 1
        self.daily_requests += 1
        
        return result
        
    async def find_and_analyze_recent_image(
        self, 
        user_id: int, 
        channel: discord.TextChannel,
        user_query: str = None
    ) -> Dict[str, Any]:
        """
        Find and analyze the most recent image in a channel.
        
        Args:
            user_id: Discord user ID
            channel: Discord channel to search
            user_query: User's specific question
            
        Returns:
            Analysis result or error
        """
        # Find recent image
        attachment = await self.vision_service.find_recent_image(channel)
        if not attachment:
            raise ValueError("No recent images found in this channel")
            
        # Analyze the image
        return await self.analyze_image(
            user_id=user_id,
            attachment=attachment,
            user_query=user_query
        )
        
    async def _process_with_perplexity(
        self, 
        user_id: int, 
        query: str, 
        use_context: bool
    ) -> Dict[str, Any]:
        """Process query using Perplexity API."""
        if not self.session:
            raise ValueError("HTTP session not initialized")
            
        # Get messages for API call
        if use_context:
            messages = self.context_manager.get_context_messages(user_id)
        else:
            messages = [
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": query}
            ]
            
        # Make API call
        result = await self.perplexity_service.search_completion(
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        # Format citations for Discord
        if result.get('citations'):
            citations_text = self.perplexity_service.format_citations_for_discord(
                result['citations']
            )
            result['content'] += citations_text
            
        return result
        
    async def _process_with_openai(
        self, 
        user_id: int, 
        query: str, 
        use_context: bool
    ) -> Dict[str, Any]:
        """Process query using OpenAI API."""
        # Get messages for API call
        if use_context:
            messages = self.context_manager.get_context_messages(user_id)
        else:
            messages = [
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": query}
            ]
            
        # Make API call
        return await self.openai_service.chat_completion(
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
    async def summarize_messages(
        self, 
        messages: List[str], 
        channel_name: str,
        chunk_size: int = 50
    ) -> str:
        """
        Summarize a list of messages using chunked processing.
        
        Args:
            messages: List of message strings
            channel_name: Name of the channel
            chunk_size: Messages per chunk
            
        Returns:
            Final summary
        """
        if not messages:
            return "No messages to summarize"
            
        # Split messages into chunks
        chunks = [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]
        
        # Process chunks in parallel
        chunk_tasks = []
        for i, chunk in enumerate(chunks):
            task = self._summarize_chunk(chunk, i, len(chunks))
            chunk_tasks.append(task)
            
        # Wait for all chunks to complete
        chunk_summaries = await asyncio.gather(*chunk_tasks, return_exceptions=True)
        
        # Filter successful results
        valid_summaries = [
            summary for summary in chunk_summaries 
            if isinstance(summary, str) and len(summary.strip()) > 50
        ]
        
        if not valid_summaries:
            raise Exception("Failed to process message chunks")
            
        # Create final summary
        return await self._create_final_summary(valid_summaries, channel_name)
        
    async def _summarize_chunk(self, messages: List[str], chunk_index: int, total_chunks: int) -> str:
        """Summarize a chunk of messages."""
        chunk_text = "\n".join(messages)
        
        prompt = f"""Extract key crypto/trading insights from this Discord discussion (chunk {chunk_index + 1}/{total_chunks}):

Focus on:
• Price movements and market sentiment
• Technical analysis and trading signals
• News, events, and alpha opportunities
• DeFi protocols and yield strategies
• Risk factors and warnings

Ignore: casual chat, memes, off-topic discussions

Messages:
{chunk_text}

Provide a concise summary of actionable insights only."""

        result = await self.openai_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2
        )
        
        return result['content']
        
    async def _create_final_summary(self, chunk_summaries: List[str], channel_name: str) -> str:
        """Create final summary from chunk summaries."""
        combined_summaries = "\n\n".join(chunk_summaries)
        
        prompt = f"""Synthesize these {channel_name} channel summaries into actionable intelligence for crypto traders/investors.

**structure your response:**

**📈 market sentiment**
[overall sentiment: bullish/bearish/neutral with confidence %]

**🚨 key events**
• [most significant developments]

**💰 price action** 
• [notable price movements and levels]

**🔍 technical analysis**
• [key levels, patterns, indicators mentioned]

**🏦 regulatory/news**
• [regulatory updates, partnerships, announcements]

**🐋 whale activity**
• [large transactions, institutional moves]

**⚡ actionable insights**
• [trading opportunities and risk factors]

**no tables, no verbose explanations. pure alpha extraction with technical precision.**

CHUNK SUMMARIES:
{combined_summaries}"""

        result = await self.openai_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0.2
        )
        
        return result['content']
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get combined usage statistics from all services."""
        return {
            'total_requests': self.total_requests,
            'daily_requests': self.daily_requests,
            'openai': self.openai_service.get_usage_stats(),
            'perplexity': self.perplexity_service.get_usage_stats(),
            'cache_hit_rate': self.openai_service.calculate_cache_hit_rate(),
        }
        
    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_requests = 0
        self.openai_service.reset_usage_stats()
        self.perplexity_service.reset_usage_stats()
        
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Close HTTP session if we own it
        if hasattr(self, '_owned_session') and self._owned_session:
            await self.session.close()
            
        logger.info("AI manager cleanup completed")