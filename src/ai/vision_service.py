"""Vision analysis service for image processing."""
import base64
import io
import logging
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
import discord
from PIL import Image

from ..config.settings import get_settings
from ..config.constants import (
    MAX_IMAGE_SIZE_MB,
    SUPPORTED_IMAGE_FORMATS,
    OPENAI_VISION_MODEL,
)
from .openai_service import OpenAIService

logger = logging.getLogger(__name__)


class VisionService:
    """Service for image analysis using vision models."""
    
    def __init__(self, openai_service: OpenAIService):
        """Initialize vision service."""
        self.settings = get_settings()
        self.openai_service = openai_service
        
    async def analyze_image(
        self, 
        image_data: bytes, 
        prompt: str = None,
        user_query: str = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using GPT-4 Vision.
        
        Args:
            image_data: Raw image bytes
            prompt: Custom analysis prompt
            user_query: User's specific question about the image
            
        Returns:
            Analysis result with content and usage info
        """
        # Validate image
        validation_result = self._validate_image(image_data)
        if not validation_result['valid']:
            raise ValueError(validation_result['error'])
            
        # Convert to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        # Create analysis prompt
        analysis_prompt = self._create_analysis_prompt(prompt, user_query)
        
        # Analyze with OpenAI Vision
        try:
            result = await self.openai_service.vision_completion(
                prompt=analysis_prompt,
                image_url=image_url,
                max_tokens=1500,
                temperature=0.3  # Lower temperature for more focused analysis
            )
            
            logger.info(f"Vision analysis completed - {len(result['content'])} chars")
            return result
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise
            
    async def analyze_discord_image(
        self, 
        attachment: discord.Attachment,
        prompt: str = None,
        user_query: str = None
    ) -> Dict[str, Any]:
        """
        Analyze a Discord image attachment.
        
        Args:
            attachment: Discord attachment object
            prompt: Custom analysis prompt
            user_query: User's specific question about the image
            
        Returns:
            Analysis result with content and usage info
        """
        # Validate attachment
        if not self._is_supported_image(attachment.filename):
            raise ValueError(f"Unsupported image format. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}")
            
        if attachment.size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Image too large. Max size: {MAX_IMAGE_SIZE_MB}MB")
            
        # Download image
        try:
            image_data = await attachment.read()
            return await self.analyze_image(image_data, prompt, user_query)
            
        except Exception as e:
            logger.error(f"Failed to download Discord image: {e}")
            raise ValueError("Failed to download image")
            
    async def find_recent_image(
        self, 
        channel: discord.TextChannel,
        limit: int = 50
    ) -> Optional[discord.Attachment]:
        """
        Find the most recent image in a channel.
        
        Args:
            channel: Discord channel to search
            limit: Maximum messages to check
            
        Returns:
            Most recent image attachment or None
        """
        try:
            async for message in channel.history(limit=limit):
                for attachment in message.attachments:
                    if self._is_supported_image(attachment.filename):
                        logger.info(f"Found recent image: {attachment.filename}")
                        return attachment
                        
            return None
            
        except Exception as e:
            logger.error(f"Error finding recent image: {e}")
            return None
            
    def _validate_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Validate image data.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Validation result dict
        """
        try:
            # Check size
            if len(image_data) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                return {
                    'valid': False,
                    'error': f"Image too large. Max size: {MAX_IMAGE_SIZE_MB}MB"
                }
                
            # Try to open with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Check format
            if image.format.lower() not in [fmt.upper() for fmt in SUPPORTED_IMAGE_FORMATS]:
                return {
                    'valid': False,
                    'error': f"Unsupported format: {image.format}"
                }
                
            return {
                'valid': True,
                'format': image.format,
                'size': image.size,
                'mode': image.mode
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Invalid image: {str(e)}"
            }
            
    def _is_supported_image(self, filename: str) -> bool:
        """Check if filename has supported image extension."""
        if not filename:
            return False
            
        extension = filename.lower().split('.')[-1]
        return extension in SUPPORTED_IMAGE_FORMATS
        
    def _create_analysis_prompt(self, custom_prompt: str = None, user_query: str = None) -> str:
        """Create analysis prompt for vision model."""
        if custom_prompt:
            return custom_prompt
            
        base_prompt = """You are an expert crypto chart analyst. Analyze this image and provide:

**📊 Chart Analysis:**
• Asset and timeframe identification
• Current price action and trend direction
• Key support and resistance levels
• Volume patterns and significance

**📈 Technical Indicators:**
• Moving averages and their signals
• RSI, MACD, and momentum indicators
• Bollinger Bands or other volatility measures
• Any visible chart patterns

**💡 Trading Insights:**
• Potential entry/exit points
• Risk/reward considerations
• Bullish or bearish signals
• Short-term vs long-term outlook

**🚨 Key Observations:**
• Critical levels to watch
• Potential breakout scenarios
• Market structure analysis

Be specific about levels, percentages, and actionable insights. Focus on what matters for trading decisions."""

        if user_query:
            base_prompt += f"\n\n**User Question:** {user_query}"
            
        return base_prompt
        
    def estimate_tokens(self, image_data: bytes) -> int:
        """Estimate tokens for image analysis."""
        # Vision models use ~85 tokens per image plus text tokens
        base_tokens = 85
        
        # Add estimated tokens based on image size
        size_multiplier = len(image_data) / (1024 * 1024)  # MB
        additional_tokens = int(size_multiplier * 50)  # Rough estimate
        
        return base_tokens + additional_tokens