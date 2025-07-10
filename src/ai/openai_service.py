"""OpenAI API integration service."""
import logging
from typing import Dict, List, Optional, Any

from openai import AsyncOpenAI

from ..config.settings import get_settings
from ..config.constants import (
    OPENAI_MODEL,
    OPENAI_VISION_MODEL,
    MAX_TOKENS_RESPONSE,
)

logger = logging.getLogger(__name__)


class OpenAIService:
    """Service for OpenAI API interactions."""
    
    def __init__(self):
        """Initialize OpenAI service."""
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.usage_data = {
            'input_tokens': 0,
            'cached_input_tokens': 0,
            'output_tokens': 0,
            'total_cost': 0.0,
        }
        
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = OPENAI_MODEL,
        max_tokens: int = MAX_TOKENS_RESPONSE,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion.
        
        Args:
            messages: List of message dicts with role and content
            model: Model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters for the API
            
        Returns:
            Response dict with content and usage info
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Track usage
            if hasattr(response, 'usage') and response.usage:
                await self._track_usage(response.usage, model)
                
            return {
                'content': response.choices[0].message.content,
                'usage': self._format_usage(response.usage) if hasattr(response, 'usage') else None,
                'model': model,
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
            
    async def vision_completion(
        self,
        prompt: str,
        image_url: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Create a vision completion for image analysis.
        
        Args:
            prompt: Text prompt for the analysis
            image_url: URL of the image to analyze
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Response dict with content and usage info
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
        
        return await self.chat_completion(
            messages=messages,
            model=OPENAI_VISION_MODEL,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
    async def _track_usage(self, usage: Any, model: str) -> None:
        """Track token usage and costs."""
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        cached_tokens = getattr(usage, 'prompt_tokens_details', {}).get('cached_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        
        # Calculate costs based on model
        if model == OPENAI_MODEL:
            # GPT-4 pricing (example rates, adjust as needed)
            input_cost = (input_tokens - cached_tokens) * 0.40 / 1_000_000
            cached_cost = cached_tokens * 0.20 / 1_000_000  # Cached tokens are cheaper
            output_cost = output_tokens * 1.60 / 1_000_000
        else:
            # Vision model pricing
            input_cost = input_tokens * 0.50 / 1_000_000
            cached_cost = 0
            output_cost = output_tokens * 1.50 / 1_000_000
            
        total_cost = input_cost + cached_cost + output_cost
        
        # Update usage data
        self.usage_data['input_tokens'] += input_tokens
        self.usage_data['cached_input_tokens'] += cached_tokens
        self.usage_data['output_tokens'] += output_tokens
        self.usage_data['total_cost'] += total_cost
        
        logger.info(
            f"OpenAI usage - Model: {model}, "
            f"Input: {input_tokens} (cached: {cached_tokens}), "
            f"Output: {output_tokens}, Cost: ${total_cost:.4f}"
        )
        
    def _format_usage(self, usage: Any) -> Dict[str, Any]:
        """Format usage data for response."""
        if not usage:
            return {}
            
        return {
            'input_tokens': getattr(usage, 'prompt_tokens', 0),
            'cached_tokens': getattr(usage, 'prompt_tokens_details', {}).get('cached_tokens', 0),
            'output_tokens': getattr(usage, 'completion_tokens', 0),
            'total_tokens': getattr(usage, 'total_tokens', 0),
        }
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_data.copy()
        
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        for key in self.usage_data:
            self.usage_data[key] = 0.0 if key == 'total_cost' else 0
            
    def calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_input = self.usage_data['input_tokens']
        cached_input = self.usage_data['cached_input_tokens']
        
        if total_input == 0:
            return 0.0
            
        return (cached_input / total_input) * 100