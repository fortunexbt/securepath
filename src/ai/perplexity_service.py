"""Perplexity API integration service."""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from ..config.settings import get_settings
from ..config.constants import PERPLEXITY_MODEL

logger = logging.getLogger(__name__)


class PerplexityService:
    """Service for Perplexity API interactions."""
    
    # Elite sources for crypto/DeFi research
    DOMAIN_FILTER = [
        "ethereum.org",        # Official Ethereum docs
        "github.com",          # Source code & repos
        "defillama.com",       # DeFi analytics
        "etherscan.io",        # On-chain data
        "coinmarketcap.com",   # Market data
        "coingecko.com",       # Market data
        "docs.uniswap.org",    # Protocol docs
        "coindesk.com",        # Reputable news
        "-reddit.com",         # Exclusion: Forum noise
        "-pinterest.com"       # Exclusion: Irrelevant
    ]
    
    def __init__(self, session: Optional[ClientSession] = None):
        """Initialize Perplexity service."""
        self.settings = get_settings()
        self.session = session
        self.usage_data = {
            'requests': 0,
            'tokens': 0,
            'cost': 0.0,
        }
        
    async def search_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
        temperature: float = 0.7,
        search_recency_days: int = 90,
        return_citations: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a search-based completion using Perplexity.
        
        Args:
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            search_recency_days: Days to look back for search results
            return_citations: Whether to return citations
            
        Returns:
            Response dict with content, citations, and usage info
        """
        if not self.session:
            raise ValueError("Session not initialized")
            
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.settings.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        # Calculate date filter
        date_filter = (datetime.now() - timedelta(days=search_recency_days)).strftime("%m/%d/%Y")
        
        # Enhance system prompt with current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        enhanced_messages = self._enhance_messages_with_date(messages, current_date)
        
        # Prepare request data
        data = {
            "model": PERPLEXITY_MODEL,
            "messages": enhanced_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "search_after_date_filter": date_filter,
            "search_domain_filter": self.DOMAIN_FILTER,
            "search_context_size": "high",
            "return_citations": return_citations,
            "return_images": False,
        }
        
        # Track request
        self.usage_data['requests'] += 1
        start_time = time.time()
        
        try:
            timeout = ClientTimeout(total=self.settings.perplexity_timeout)
            async with self.session.post(
                self.settings.perplexity_api_url,
                json=data,
                headers=headers,
                timeout=timeout
            ) as response:
                elapsed_time = time.time() - start_time
                logger.info(f"Perplexity API request completed in {elapsed_time:.2f}s")
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Perplexity API error {response.status}: {error_text}")
                    raise Exception(f"API error {response.status}: {error_text}")
                    
                resp_json = await response.json()
                
                # Extract response content
                answer = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Process citations
                citations = self._process_citations(resp_json)
                
                # Track usage
                usage = resp_json.get('usage', {})
                await self._track_usage(usage)
                
                return {
                    'content': answer,
                    'citations': citations,
                    'usage': usage,
                    'model': PERPLEXITY_MODEL,
                    'elapsed_time': elapsed_time,
                }
                
        except asyncio.TimeoutError:
            logger.error(f"Perplexity API timeout after {self.settings.perplexity_timeout}s")
            raise Exception("⏱️ Request timed out. Please try again.")
        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            raise
            
    def _enhance_messages_with_date(
        self, 
        messages: List[Dict[str, str]], 
        current_date: str
    ) -> List[Dict[str, str]]:
        """Enhance messages with current date context."""
        enhanced = messages.copy()
        
        # Update system message with date
        if enhanced and enhanced[0]['role'] == 'system':
            enhanced[0]['content'] = (
                f"Today is {current_date}. All information must be accurate up to this date. "
                f"{enhanced[0]['content']}"
            )
            
        return enhanced
        
    def _process_citations(self, response_data: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Process and format citations from response."""
        citations = []
        
        # Extract from extras.citations
        extras_citations = (
            response_data.get('choices', [{}])[0]
            .get('extras', {})
            .get('citations', [])
        )
        
        for cite in extras_citations:
            title = cite.get('title', 'Source')
            url = cite.get('url', '#')
            if url != '#' and title != 'Source':
                citations.append((title, url))
                
        # Also check search_results for additional sources
        search_results = response_data.get('search_results', [])
        for result in search_results:
            title = result.get('title', '')
            url = result.get('url', '')
            if url and title and (title, url) not in citations:
                citations.append((title, url))
                
        logger.debug(f"Processed {len(citations)} citations")
        return citations[:6]  # Limit to top 6 citations
        
    async def _track_usage(self, usage: Dict[str, Any]) -> None:
        """Track token usage and costs."""
        tokens = usage.get('total_tokens', 0)
        
        # Estimate cost (adjust based on actual Perplexity pricing)
        cost = tokens * 0.0002  # Example rate
        
        self.usage_data['tokens'] += tokens
        self.usage_data['cost'] += cost
        
        logger.info(f"Perplexity usage - Tokens: {tokens}, Cost: ${cost:.4f}")
        
    def format_citations_for_discord(self, citations: List[Tuple[str, str]]) -> str:
        """Format citations for Discord message."""
        if not citations:
            return ""
            
        formatted = "\n\n**Sources:**\n"
        for i, (title, url) in enumerate(citations, 1):
            # Truncate title if too long
            if len(title) > 60:
                title = title[:57] + "..."
            formatted += f"[{i}] [{title}]({url})\n"
            
        return formatted
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_data.copy()
        
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        for key in self.usage_data:
            self.usage_data[key] = 0.0 if key == 'cost' else 0