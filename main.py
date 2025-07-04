# main.py
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional

import aiohttp
import discord
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from discord import Embed, Activity, ActivityType
from discord.ext import commands, tasks
from discord.ext.commands import Bot, Context
from openai import AsyncOpenAI
from rich.console import Console
from rich.logging import RichHandler
from tiktoken import encoding_for_model
import random
from PIL import Image
from io import BytesIO

# Local imports
import config
from database import db_manager

logger = logging.getLogger('SecurePathAgent')
console = Console()

def setup_logging() -> logging.Logger:
    logger.setLevel(getattr(logging, config.LOG_LEVEL, 'INFO'))
    console_handler = RichHandler(rich_tracebacks=True, console=console)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL, 'INFO'))
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    logger.addHandler(console_handler)
    for module in ['discord', 'discord.http', 'discord.gateway']:
        logging.getLogger(module).setLevel(logging.WARNING)
    return logger

logger = setup_logging()

aclient = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
bot = Bot(command_prefix=config.BOT_PREFIX, intents=intents)

conn: Optional[TCPConnector] = None
session: Optional[ClientSession] = None
user_contexts: Dict[int, Deque[Dict[str, Any]]] = {}
message_counter = Counter()
command_counter = Counter()
api_call_counter = 0
total_token_cost = 0.0  # Added for tracking total token cost

# Usage data tracking
usage_data = {
    'perplexity': {
        'requests': 0,
        'tokens': 0,
        'cost': 0.0,
    },
    'openai_gpt41': {
        'input_tokens': 0,
        'cached_input_tokens': 0,  # New field for cached tokens
        'cost': 0.0,
    },
    'openai_gpt41_mini_vision': {
        'requests': 0,
        'tokens': 0,
        'cost': 0.0,
        'average_tokens_per_request': 0.0,  # New field for tracking average tokens
    }
}

class RateLimiter:
    def __init__(self, max_calls: int, interval: int):
        self.max_calls = max_calls
        self.interval = interval
        self.calls: Dict[int, List[float]] = {}

    def is_rate_limited(self, user_id: int) -> bool:
        current_time = time.time()
        self.calls.setdefault(user_id, [])
        self.calls[user_id] = [t for t in self.calls[user_id] if current_time - t <= self.interval]
        if len(self.calls[user_id]) >= self.max_calls:
            logger.debug(f"User {user_id} is rate limited. {len(self.calls[user_id])} calls in the last {self.interval} seconds.")
            return True
        self.calls[user_id].append(current_time)
        logger.debug(f"User {user_id} made an API call. Total calls in the last {self.interval} seconds: {len(self.calls[user_id])}")
        return False

api_rate_limiter = RateLimiter(config.API_RATE_LIMIT_MAX, config.API_RATE_LIMIT_INTERVAL)

# No rate limiting - small server optimization

def get_user_context(user_id: int) -> Deque[Dict[str, Any]]:
    return user_contexts.setdefault(user_id, deque(maxlen=config.MAX_CONTEXT_MESSAGES))

def update_user_context(user_id: int, message_content: str, role: str) -> None:
    context = get_user_context(user_id)
    current_time = time.time()

    # Initialize context with system message if it's empty
    if not context:
        context.append({
            'role': 'system',
            'content': config.SYSTEM_PROMPT.strip(),
            'timestamp': current_time,
        })
        logger.debug(f"Initialized context with system prompt for user {user_id}.")

    # Add user message directly after system message if context length is 1
    if len(context) == 1 and role == 'user':
        context.append({
            'role': 'user',
            'content': message_content.strip(),
            'timestamp': current_time,
        })
        logger.debug(f"Appended first user message for user {user_id}: {message_content[:50]}...")
        return

    # Ensure roles alternate correctly for subsequent messages
    last_role = context[-1]['role']
    if last_role == 'user' and role == 'assistant':
        context.append({
            'role': role,
            'content': message_content.strip(),
            'timestamp': current_time,
        })
        logger.debug(f"Appended assistant message for user {user_id}: {message_content[:50]}...")
    elif last_role == 'assistant' and role == 'user':
        context.append({
            'role': role,
            'content': message_content.strip(),
            'timestamp': current_time,
        })
        logger.debug(f"Appended user message for user {user_id}: {message_content[:50]}...")
    else:
        logger.warning(f"Role mismatch for user {user_id}: Expected alternate role, got {role}. Message skipped.")

    # Remove old messages beyond the context's max age
    cutoff_time = current_time - config.MAX_CONTEXT_AGE
    old_length = len(context)
    user_contexts[user_id] = deque(
        [msg for msg in context if msg['timestamp'] >= cutoff_time],
        maxlen=config.MAX_CONTEXT_MESSAGES
    )
    new_length = len(user_contexts[user_id])
    if new_length < old_length:
        logger.debug(f"Removed {old_length - new_length} old messages from user {user_id}'s context.")

def get_context_messages(user_id: int) -> List[Dict[str, str]]:
    context = get_user_context(user_id)
    messages = [{"role": msg['role'], "content": msg['content']} for msg in context]

    # Ensure the first message is a system message
    if not messages or messages[0]['role'] != 'system':
        messages.insert(0, {
            "role": "system",
            "content": config.SYSTEM_PROMPT.strip(),
        })
        logger.debug("Inserted system prompt at the beginning of messages.")

    # Ensure alternating roles correctly
    cleaned_messages = [messages[0]]  # Start with system message
    for i in range(1, len(messages)):
        last_role = cleaned_messages[-1]['role']

        # Determine the expected role based on the last role
        if last_role in ['system', 'assistant']:
            expected_role = 'user'
        elif last_role == 'user':
            expected_role = 'assistant'
        else:
            logger.warning(f"Unknown last role '{last_role}' in context.")
            continue  # Skip unknown roles

        if messages[i]['role'] == expected_role:
            cleaned_messages.append(messages[i])
        else:
            logger.warning(f"Role mismatch at message {i}: Expected {expected_role}, got {messages[i]['role']}")

    logger.debug("Final cleaned context messages:")
    for idx, msg in enumerate(cleaned_messages):
        content_preview = msg['content'][:50] + '...' if len(msg['content']) > 50 else msg['content']
        logger.debug(f"Message {idx}: Role={msg['role']}, Content={content_preview}")

    return cleaned_messages

def truncate_prompt(prompt: str, max_tokens: int, model: str = 'gpt-4o-mini') -> str:
    encoding = encoding_for_model(model)
    tokens = encoding.encode(prompt)
    if len(tokens) > max_tokens:
        truncated = encoding.decode(tokens[-max_tokens:])
        logger.debug(f"Truncated prompt from {len(tokens)} to {max_tokens} tokens.")
        return truncated
    return prompt

def log_instance_info() -> None:
    hostname = os.uname().nodename
    pid = os.getpid()
    logger.info(f"Bot instance started - Hostname: {hostname}, PID: {pid}")

def increment_api_call_counter():
    global api_call_counter
    api_call_counter += 1
    logger.info(f"API call counter: {api_call_counter}")

def increment_token_cost(cost: float):
    global total_token_cost
    total_token_cost += cost
    logger.info(f"Total token cost: ${total_token_cost:.6f}")

async def log_usage_to_db(user: discord.User, command: str, model: str, 
                         input_tokens: int = 0, output_tokens: int = 0, 
                         cached_tokens: int = 0, cost: float = 0.0,
                         guild_id: Optional[int] = None, channel_id: Optional[int] = None):
    """Log usage to database if connected"""
    if db_manager.pool:
        try:
            await db_manager.log_usage(
                user_id=user.id,
                username=f"{user.name}#{user.discriminator}" if user.discriminator != "0" else user.name,
                command=command,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cost=cost,
                guild_id=guild_id,
                channel_id=channel_id
            )
        except Exception as e:
            logger.error(f"Failed to log usage to database: {e}")

def can_make_api_call(user_id: Optional[int] = None) -> tuple[bool, Optional[str]]:
    """Check if API call can be made - no rate limiting for small server"""
    return True, None

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB limit

def estimate_tokens(image_size_bytes: int) -> int:
    TOKENS_PER_BYTE = 1 / 100
    estimated = int(image_size_bytes * TOKENS_PER_BYTE)
    logger.debug(f"Estimated tokens based on image size ({image_size_bytes} bytes): {estimated} tokens")
    return estimated

async def fetch_perplexity_response(user_id: int, new_message: str) -> Optional[str]:
    if session is None:
        logger.error("Session is not initialized")
        raise Exception("🚫 Network session not available. Please try again.")
    
    can_call, error_msg = can_make_api_call(user_id)
    if not can_call:
        logger.warning(f"Rate limit reached for user {user_id}")
        raise Exception(error_msg)

    headers = {
        "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    current_date = datetime.now().strftime("%Y-%m-%d")
    dynamic_system_prompt = f"Today is {current_date}. All information must be accurate up to this date. {config.SYSTEM_PROMPT}"

    context_messages = get_context_messages(user_id)

    if context_messages and context_messages[-1]['role'] != 'user':
        context_messages.append({"role": "user", "content": new_message.strip()})

    messages = [{"role": "system", "content": dynamic_system_prompt}] + context_messages

    # --- Start of Corrected Search Optimization ---

    # 1. Set a dynamic date filter for the last 90 days.
    ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime("%m/%d/%Y")
    
    # 2. Optimized domain filter for crypto/DeFi research with authoritative sources
    domain_filter = [
        "ethereum.org",        # Primary: Official Ethereum docs
        "github.com",          # Primary: Source code & repos
        "defillama.com",       # Primary: DeFi analytics
        "etherscan.io",        # Primary: On-chain data
        "coinmarketcap.com",   # Primary: Market data
        "coingecko.com",       # Primary: Market data
        "docs.uniswap.org",    # Primary: Protocol docs
        "coindesk.com",        # Secondary: Reputable news
        "-reddit.com",         # Exclusion: Forum noise
        "-pinterest.com"       # Exclusion: Irrelevant
    ]
    logger.debug(f"Using Perplexity domain filter with {len(domain_filter)} elite sources.")

    # 3. Define the full data payload with optimized search parameters for better citations
    data = {
        "model": "sonar-pro",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.2,  # Lower temperature for more factual responses
        "search_after_date_filter": ninety_days_ago,  # Use date filter OR recency filter, not both
        "search_domain_filter": domain_filter,
        "search_context_size": "high",  # Maximum citation coverage
        "return_citations": True,  # Ensure citations are returned
        "return_images": False  # Disable images for faster response
    }
    # --- End of Corrected Search Optimization ---


    logger.info(f"Sending query to Perplexity API for user {user_id}")
    usage_data['perplexity']['requests'] += 1
    increment_api_call_counter()
    start_time = time.time()

    try:
        timeout = ClientTimeout(total=config.PERPLEXITY_TIMEOUT)
        async with session.post(config.PERPLEXITY_API_URL, json=data, headers=headers, timeout=timeout) as response:
            elapsed_time = time.time() - start_time
            logger.info(f"Perplexity API request completed in {elapsed_time:.2f} seconds")

            if response.status == 200:
                resp_json = await response.json()
                answer = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Enhanced citation processing for better source verification
                citations = resp_json.get('choices', [{}])[0].get('extras', {}).get('citations', [])
                search_results = resp_json.get('search_results', [])
                
                # Combine both citation sources for comprehensive referencing
                all_sources = []
                
                # Process traditional citations
                for cite in citations:
                    title = cite.get('title', 'Source')
                    url = cite.get('url', '#')
                    if url != '#' and title != 'Source':
                        all_sources.append((title, url))
                
                # Process search_results for additional sources
                for result in search_results:
                    title = result.get('title', '')
                    url = result.get('url', '')
                    if url and title and (title, url) not in all_sources:
                        all_sources.append((title, url))
                
                if all_sources:
                    formatted_citations = "\n\n**📚 Sources & Citations:**\n"
                    for i, (title, url) in enumerate(all_sources[:8], 1):  # Limit to 8 sources
                        # Truncate long titles for better formatting
                        display_title = title[:80] + "..." if len(title) > 80 else title
                        formatted_citations += f"{i}. [{display_title}]({url})\n"
                    
                    formatted_citations += "\n*DYOR: Always verify information from original sources*"
                    answer += formatted_citations

                usage = resp_json.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                usage_data['perplexity']['tokens'] += total_tokens

                cost = (usage_data['perplexity']['requests'] * 5 / 1000) + (usage_data['perplexity']['tokens'] / 1_000_000 * 1)
                usage_data['perplexity']['cost'] = round(cost, 6)
                increment_token_cost(cost)

                logger.info(f"Perplexity API usage: Prompt Tokens={prompt_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")
                logger.info(f"Estimated Perplexity API call cost: ${cost:.6f}")
                return answer
            else:
                response_text = await response.text()
                logger.error(f"Perplexity API request failed with status {response.status}. Response: {response_text}")
    except asyncio.TimeoutError:
        logger.error(f"Request to Perplexity API timed out after {config.PERPLEXITY_TIMEOUT} seconds")
    except Exception as e:
        logger.error(f"Error in fetch_perplexity_response: {str(e)}")
        logger.error(traceback.format_exc())
    return None

async def fetch_openai_response(user_id: int, new_message: str, user: Optional[discord.User] = None, 
                               command: str = "ask", guild_id: Optional[int] = None, 
                               channel_id: Optional[int] = None) -> Optional[str]:
    if not can_make_api_call():
        logger.warning("Daily API call limit reached. Skipping API call.")
        return None

    context_messages = get_context_messages(user_id)
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + context_messages

    if not messages or messages[-1]['role'] != 'user' or messages[-1]['content'] != new_message:
        messages.append({"role": "user", "content": new_message})

    try:
        response = await aclient.chat.completions.create(
            model='gpt-4.1',
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()

        if hasattr(response, 'usage'):
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
            cached_tokens = usage.prompt_tokens_details.cached_tokens if hasattr(usage.prompt_tokens_details, 'cached_tokens') else 0
            completion_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
            total_tokens = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
        else:
            prompt_tokens, cached_tokens, completion_tokens, total_tokens = 0, 0, 0, 0

        is_cached = cached_tokens >= 1024

        if is_cached:
            usage_data['openai_gpt41']['cached_input_tokens'] += cached_tokens
            cost = (cached_tokens / 1_000_000 * 0.30) + (completion_tokens / 1_000_000 * 1.20)  # GPT-4.1 cached pricing
            logger.debug(f"Cache hit detected. Cached Tokens: {cached_tokens}, Completion Tokens: {completion_tokens}, Cost: ${cost:.6f}")
        else:
            usage_data['openai_gpt41']['input_tokens'] += prompt_tokens
            cost = (prompt_tokens / 1_000_000 * 0.60) + (completion_tokens / 1_000_000 * 2.40)  # GPT-4.1 pricing
            logger.debug(f"No cache hit. Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Cost: ${cost:.6f}")

        usage_data['openai_gpt41']['cost'] += cost
        increment_token_cost(cost)
        
        # Log to database if user provided
        if user:
            await log_usage_to_db(
                user=user,
                command=command,
                model="gpt-4.1",
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                cost=cost,
                guild_id=guild_id,
                channel_id=channel_id
            )

        logger.info(f"OpenAI GPT-4.1 usage: Prompt Tokens={prompt_tokens}, Cached Tokens={cached_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")
        logger.info(f"Estimated OpenAI GPT-4.1 API call cost: ${cost:.6f}")
        return answer
    except Exception as e:
        logger.error(f"Error fetching response from OpenAI: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def send_structured_analysis_embed(
    channel: discord.abc.Messageable,
    text: str,
    color: int,
    title: str,
    image_url: Optional[str] = None,
    user_mention: Optional[str] = None
) -> None:
    """Send analysis with structured field-based formatting for better readability"""
    try:
        # Create structured embed with better formatting
        embed = discord.Embed(
            title=title,
            description="AI-powered technical analysis with actionable insights",
            color=color
        )
        
        if image_url:
            embed.set_image(url=image_url)
        
        # Parse text looking for markdown headers and natural sections
        parsed_sections = []
        lines = text.split('\n')
        current_header = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            # Look for markdown headers (# or ##)
            if line.startswith('##') or line.startswith('#'):
                # Save previous section
                if current_header and current_content:
                    parsed_sections.append((current_header, '\n'.join(current_content)))
                
                # Start new section with clean header
                current_header = line.strip('#').strip()
                current_content = []
            # Look for bold headers (**Header**)
            elif line.startswith('**') and line.endswith('**') and len(line.strip('*').strip()) < 80:
                # Save previous section
                if current_header and current_content:
                    parsed_sections.append((current_header, '\n'.join(current_content)))
                
                # Start new section
                current_header = line.strip('*').strip()
                current_content = []
            elif line:
                current_content.append(line)
        
        # Save final section
        if current_header and current_content:
            parsed_sections.append((current_header, '\n'.join(current_content)))
        
        # If no structured sections found, try double newline split
        if not parsed_sections:
            sections = text.split('\n\n')
            for i, section in enumerate(sections[:6]):
                if section.strip():
                    lines = section.strip().split('\n')
                    # Try to extract a meaningful title from first line
                    first_line = lines[0].strip()
                    if len(first_line) < 80 and any(word in first_line.lower() for word in ['sentiment', 'analysis', 'trend', 'support', 'resistance', 'recommendation', 'outlook', 'summary', 'technical', 'price', 'volume']):
                        header = first_line
                        content = '\n'.join(lines[1:]) if len(lines) > 1 else section.strip()
                    else:
                        header = "Market Analysis"
                        content = section.strip()
                    parsed_sections.append((header, content))
        
        # Add sections as fields
        for header, content in parsed_sections[:8]:  # Limit to 8 fields
            if content.strip():
                # Ensure content fits in field (1024 char limit)
                if len(content) > 1000:
                    content = content[:997] + "..."
                
                embed.add_field(
                    name=f"📈 {header[:250]}",
                    value=content or "No specific insights",
                    inline=False
                )
        
        # If no sections were parsed, use the full text as a single field
        if not parsed_sections or not embed.fields:
            content = text[:1000] + "..." if len(text) > 1000 else text
            embed.add_field(name="📈 Technical Analysis", value=content, inline=False)
        
        embed.set_author(name="SecurePath Agent", icon_url=bot.user.avatar.url if bot.user.avatar else None)
        embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1 Vision")
        
        content = user_mention if user_mention else None
        await channel.send(content=content, embed=embed)
        
    except Exception as e:
        logger.error(f"Failed to send structured analysis embed: {e}")
        # Fallback to regular embed
        await send_long_embed(channel, text, color, title, image_url)

async def send_long_embed(
    channel: discord.abc.Messageable,
    text: str,
    color: int = 0x1D82B6,
    title: Optional[str] = None,
    image_url: Optional[str] = None,
    user_mention: Optional[str] = None
) -> None:
    if not text:
        return

    embed_max_length = 4096
    parts = [text[i:i + embed_max_length] for i in range(0, len(text), embed_max_length)]

    for i, part in enumerate(parts):
        if i == 0:
            embed = Embed(title=title, description=part, color=color)
            if image_url:
                embed.set_image(url=image_url)
        else:
            embed = Embed(description=part, color=color)

        embed.set_author(name="SecurePath Agent", icon_url=bot.user.avatar.url if bot.user.avatar else None)
        
        if len(parts) > 1:
            embed.set_footer(text=f"SecurePath Agent • Part {i + 1}/{len(parts)} • Powered by GPT-4.1")
        else:
            embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1")

        try:
            # Add user mention to first part if provided
            content = user_mention if i == 0 and user_mention else None
            await channel.send(content=content, embed=embed)
            channel_name = getattr(channel, 'name', "Direct Message")
            logger.debug(f"Sent embed part {i + 1}/{len(parts)} to {channel_name}")
        except discord.errors.HTTPException as e:
            logger.error(f"Failed to send embed part {i + 1}/{len(parts)}: {str(e)}")
            break

async def log_interaction(user: discord.User, channel: discord.abc.Messageable, command: Optional[str], user_input: str, bot_response: str) -> None:
    log_channel = bot.get_channel(config.LOG_CHANNEL_ID)
    if not log_channel:
        logger.warning(f"Log channel with ID {config.LOG_CHANNEL_ID} not found.")
        return

    truncated_user_input = (user_input[:1024] + '...') if len(user_input) > 1024 else user_input
    truncated_bot_response = (bot_response[:1024] + '...') if len(bot_response) > 1024 else bot_response

    embed = Embed(
        title="📝 User Interaction",
        color=0x1D82B6,
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_author(name=f"{user}", icon_url=user.avatar.url if user.avatar else None)
    embed.add_field(name="Command", value=command if command else "N/A", inline=True)
    embed.add_field(name="User ID", value=str(user.id), inline=True)
    embed.add_field(name="Channel", value=channel.mention if isinstance(channel, discord.TextChannel) else "Direct Message", inline=True)
    embed.add_field(name="User Input", value=truncated_user_input, inline=False)
    embed.add_field(name="Bot Response Preview", value=truncated_bot_response, inline=False)
    embed.set_footer(text=f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    try:
        await log_channel.send(embed=embed)
        logger.debug(f"Logged interaction for user {user.id}")
    except discord.errors.HTTPException as e:
        logger.error(f"Failed to send interaction log embed: {str(e)}")

async def process_message_with_streaming(message: discord.Message, status_msg: discord.Message, *, question: Optional[str] = None, command: str = 'ask') -> None:
    """Enhanced message processing with streaming-like progress updates"""
    try:
        user_id = message.author.id
        is_dm = isinstance(message.channel, discord.DMChannel)
        logger.debug(f"Processing message from user {user_id} in {'DM' if is_dm else f'channel {message.channel.id}'}")

        # No rate limiting for small server
        logger.debug(f"Processing !ask for user {user_id}")
        
        # Update progress: Searching
        progress_embed = status_msg.embeds[0]
        progress_embed.set_field_at(0, name="Status", value="🔍 Searching Perplexity Sonar-Pro...", inline=False)
        await status_msg.edit(embed=progress_embed)

        if is_dm:
            await preload_user_messages(user_id, message.channel)

        try:
            perplexity_response = await fetch_perplexity_response(user_id, question or message.content)
            logger.info(f"Perplexity response generated for user {user_id}")
            update_user_context(user_id, question or message.content, 'user')
            
            # Update progress: Analyzing
            progress_embed.set_field_at(0, name="Status", value="🧑‍💻 Analyzing with GPT-4.1...", inline=False)
            await status_msg.edit(embed=progress_embed)
            
            try:
                openai_response = await fetch_openai_response(
                    user_id, 
                    f"Analyze this information and provide insights in a casual, conversational tone:\n\n{perplexity_response}", 
                    user=message.author,
                    command=command,
                    guild_id=message.guild.id if message.guild else None,
                    channel_id=message.channel.id
                )
                
                # Update progress: Finalizing
                progress_embed.set_field_at(0, name="Status", value="✨ Finalizing response...", inline=False)
                await status_msg.edit(embed=progress_embed)
                
                update_user_context(user_id, openai_response, 'assistant')
                
                # Delete progress message and send final response (with error handling)
                try:
                    await status_msg.delete()
                except discord.NotFound:
                    # Status message was already deleted
                    pass
                
                await send_long_embed(
                    message.channel, 
                    openai_response, 
                    user_mention=message.author.mention,
                    title="🔍 Research Results"
                )
                
                await log_interaction(user=message.author, channel=message.channel, command=command, user_input=question or message.content, bot_response=openai_response[:1024])
                logger.info(f"Successfully sent response to user {user_id}")
                
            except Exception as openai_error:
                # If OpenAI fails, send Perplexity response directly
                logger.warning(f"OpenAI failed, sending Perplexity response directly: {openai_error}")
                
                # Update to show fallback mode (with error handling)
                try:
                    progress_embed.set_field_at(0, name="Status", value="🔄 Using fallback mode...", inline=False)
                    await status_msg.edit(embed=progress_embed)
                    await asyncio.sleep(1)  # Brief pause for UX
                    await status_msg.delete()
                except discord.NotFound:
                    # Status message was already deleted or not found
                    pass
                
                await send_long_embed(
                    message.channel, 
                    perplexity_response, 
                    user_mention=message.author.mention, 
                    title="🔍 Research Results (Direct)"
                )
                
        except Exception as perplexity_error:
            # Both services failed
            raise perplexity_error

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        logger.error(traceback.format_exc())
        raise e

async def process_message(message: discord.Message, question: Optional[str] = None, command: Optional[str] = None) -> None:
    # Enhanced rate limiting with per-user tracking
    can_call, error_msg = can_make_api_call(message.author.id)
    if not can_call:
        await message.channel.send(f"{message.author.mention} {error_msg}")
        return

    if question is None:
        question = message.content.strip()
        if not isinstance(message.channel, discord.DMChannel):
            question = question[len(config.BOT_PREFIX):].strip()

    if not question:
        error_message = "Invalid question input."
        logger.error(error_message)
        await message.channel.send(error_message)
        return

    logger.info(f"Processing message from {message.author} (ID: {message.author.id}): {question}")
    
    # Log user query to database for analytics
    guild_id = message.guild.id if message.guild else None
    channel_id = message.channel.id
    username = f"{message.author.name}#{message.author.discriminator}" if message.author.discriminator != "0" else message.author.name
    
    if db_manager.pool:
        await db_manager.log_user_query(
            user_id=message.author.id,
            username=username,
            command=command or "dm_chat",
            query_text=question,
            channel_id=channel_id,
            guild_id=guild_id,
            response_generated=False,  # Will update this later
            error_occurred=False
        )

    if len(question) < 5:
        await message.channel.send("Please provide a more detailed question (at least 5 characters).")
        return
    if len(question) > 1000:
        await message.channel.send("Your question is too long. Please limit it to 1000 characters.")
        return

    async with message.channel.typing():
        try:
            if isinstance(message.channel, discord.DMChannel):
                await preload_user_messages(message.author.id, message.channel)

            update_user_context(message.author.id, question, role='user')

            guild_id = message.guild.id if message.guild else None
            channel_id = message.channel.id
            
            if config.USE_PERPLEXITY_API:
                answer = await fetch_perplexity_response(message.author.id, question)
                # Log perplexity usage to DB
                if answer and db_manager.pool:
                    await log_usage_to_db(
                        user=message.author,
                        command=command or "dm_chat",
                        model="perplexity-sonar-pro",
                        cost=0.001,  # Rough estimate for perplexity
                        guild_id=guild_id,
                        channel_id=channel_id
                    )
            else:
                answer = await fetch_openai_response(
                    message.author.id, 
                    question, 
                    user=message.author,
                    command=command or "dm_chat",
                    guild_id=guild_id,
                    channel_id=channel_id
                )

            if answer:
                update_user_context(message.author.id, answer, role='assistant')
                await send_long_embed(message.channel, answer, color=0x004200)
                await log_interaction(
                    user=message.author,
                    channel=message.channel,
                    command=command,
                    user_input=question,
                    bot_response=answer[:1024]
                )
            else:
                error_message = "I'm sorry, I couldn't get a response. Please try again later."
                embed = Embed(description=error_message, color=0xff0000)
                await message.channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Error processing message from user {message.author.id}: {str(e)}")
            logger.error(traceback.format_exc())
            error_message = "An unexpected error occurred. Please try again later."
            embed = Embed(description=error_message, color=0xff0000)
            await message.channel.send(embed=embed)

status_messages = [
    "the BTC chart 📊",
    "DeFi trends 📈",
    "questions ❓",
    "SecurePath 🛡️",
    "your commands... 👀"
]

@tasks.loop(seconds=15)
async def change_status():
    current_status = random.choice(status_messages)
    await bot.change_presence(activity=Activity(type=ActivityType.watching, name=current_status))
    logger.debug(f"Changed status to: {current_status}")

async def reset_status():
    if change_status.is_running():
        change_status.cancel()
        await asyncio.sleep(0.1)
    change_status.start()
    logger.debug("Status rotation restarted.")

@bot.event
async def on_ready() -> None:
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'SecurePath Agent is active in {len(bot.guilds)} guild(s)')
    log_instance_info()
    
    # Initialize database connection
    db_connected = await db_manager.connect()
    if db_connected:
        logger.info("Database connection established successfully")
    else:
        logger.error("Failed to connect to database - usage tracking will be limited")
    
    await send_initial_stats()
    if not change_status.is_running():
        change_status.start()
        logger.info("Started rotating status messages.")
    if not send_periodic_stats.is_running():
        send_periodic_stats.start()
        logger.info("Started periodic stats task.")
    if not reset_api_call_counter.is_running():
        reset_api_call_counter.start()
        logger.info("Started API call counter reset task.")

@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    await bot.process_commands(message)

    if isinstance(message.channel, discord.DMChannel) and not message.content.startswith(config.BOT_PREFIX):
        await preload_user_messages(message.author.id, message.channel)
        await process_message(message)

async def preload_user_messages(user_id: int, channel: discord.DMChannel) -> None:
    if user_id not in user_contexts:
        messages = []
        bot_user_id = bot.user.id
        async for msg in channel.history(limit=100, oldest_first=True):
            role = 'user' if msg.author.id == user_id else 'assistant' if msg.author.id == bot_user_id else None
            if role:
                messages.append({
                    'role': role,
                    'content': msg.content.strip(),
                    'timestamp': msg.created_at.timestamp(),
                })
            if len(messages) >= 20:
                break
        
        user_contexts[user_id] = deque(reversed(messages), maxlen=config.MAX_CONTEXT_MESSAGES)
        logger.info(f"Preloaded {len(user_contexts[user_id])} messages for user {user_id} in DMs.")

startup_stats_sent = False

async def send_initial_stats() -> None:
    global startup_stats_sent
    if startup_stats_sent:
        logger.debug("Startup stats already sent, skipping")
        return
    
    await asyncio.sleep(5)  # Wait for bot to fully initialize
    await send_stats()
    startup_stats_sent = True
    logger.info("Initial startup stats completed")

@bot.command(name='analyze')
async def analyze(ctx: Context, *, user_prompt: str = '') -> None:
    await bot.change_presence(activity=Activity(type=ActivityType.watching, name="image analysis..."))
    logger.debug("Status updated to: [watching] image analysis...")

    # Log the analyze command query
    if db_manager.pool:
        username = f"{ctx.author.name}#{ctx.author.discriminator}" if ctx.author.discriminator != "0" else ctx.author.name
        query_text = f"Image analysis request" + (f" with prompt: {user_prompt}" if user_prompt else " (no additional prompt)")
        await db_manager.log_user_query(
            user_id=ctx.author.id,
            username=username,
            command="analyze",
            query_text=query_text,
            channel_id=ctx.channel.id,
            guild_id=ctx.guild.id if ctx.guild else None,
            response_generated=False
        )

    chart_url = None
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if attachment.content_type and attachment.content_type.startswith('image/'):
            chart_url = attachment.url
            logger.debug(f"Image attachment detected: {chart_url}")
    else:
        if isinstance(ctx.channel, discord.DMChannel):
            await ctx.send("Please post the image you'd like to analyze.")
            def check(msg):
                return msg.author == ctx.author and msg.channel == ctx.channel and msg.attachments
            try:
                chart_message = await bot.wait_for('message', check=check, timeout=60.0)
                chart_url = chart_message.attachments[0].url
                logger.debug(f"Image uploaded in DM: {chart_url}")
            except asyncio.TimeoutError:
                await ctx.send("You took too long to post an image. Please try again.")
                await reset_status()
                return
        else:
            async for last_message in ctx.channel.history(limit=2):
                if last_message.id != ctx.message.id and last_message.attachments:
                    attachment = last_message.attachments[0]
                    if attachment.content_type and attachment.content_type.startswith('image/'):
                        chart_url = attachment.url
                        logger.debug(f"Image found in recent channel messages: {chart_url}")
                        break
            
            # If no recent images, look further back
            if not chart_url:
                async for older_message in ctx.channel.history(limit=20):
                    if older_message.attachments:
                        attachment = older_message.attachments[0]
                        if attachment.content_type and attachment.content_type.startswith('image/'):
                            chart_url = attachment.url
                            logger.debug(f"Image found in older channel messages: {chart_url}")
                            break
    
    if chart_url:
        # Enhanced self-editing progress for analyze command
        progress_embed = discord.Embed(
            title="📈 SecurePath Agent Analysis",
            description=f"**Image:** [Chart Analysis]({chart_url})\n**Prompt:** {user_prompt or 'Standard technical analysis'}",
            color=0x1D82B6
        )
        progress_embed.add_field(name="Status", value="🔄 Initializing image analysis...", inline=False)
        progress_embed.set_thumbnail(url=chart_url)
        progress_embed.set_footer(text="SecurePath Agent • Real-time Analysis")
        
        status_msg = await ctx.send(embed=progress_embed)
        logger.info(f"Chart URL detected: {chart_url}")

        try:
            # Update progress: Processing image
            progress_embed.set_field_at(0, name="Status", value="🖼️ Processing image with GPT-4.1 Vision...", inline=False)
            await status_msg.edit(embed=progress_embed)
            
            guild_id = ctx.guild.id if ctx.guild else None
            image_analysis = await analyze_chart_image(
                chart_url, 
                user_prompt, 
                user=ctx.author,
                guild_id=guild_id,
                channel_id=ctx.channel.id
            )

            if image_analysis:
                # Update progress: Finalizing
                progress_embed.set_field_at(0, name="Status", value="✨ Finalizing technical analysis...", inline=False)
                await status_msg.edit(embed=progress_embed)
                
                # Brief pause for UX
                await asyncio.sleep(1)
                
                # Delete progress and send final result
                await status_msg.delete()
                
                await send_structured_analysis_embed(
                    ctx.channel,
                    text=image_analysis,
                    color=0x1D82B6,
                    title="📈 Chart Analysis",
                    image_url=chart_url,
                    user_mention=ctx.author.mention
                )
                logger.info(f"Sent image analysis to channel {ctx.channel.name}")
                await log_interaction(
                    user=ctx.author,
                    channel=ctx.channel,
                    command='analyze',
                    user_input=user_prompt or 'No additional prompt provided',
                    bot_response=image_analysis[:1024]
                )
            else:
                # Update with error state
                error_embed = discord.Embed(
                    title="❌ Analysis Failed",
                    description="Sorry, I couldn't analyze the image. Please try again with a clearer chart image.",
                    color=0xFF0000
                )
                error_embed.add_field(name="Suggestion", value="Make sure the image is a clear chart or technical analysis diagram.", inline=False)
                await status_msg.edit(embed=error_embed)
                logger.warning("Image analysis failed to return a response.")
                
        except Exception as e:
            # Update with error information
            error_embed = discord.Embed(
                title="❌ Analysis Error",
                description="An error occurred during image analysis.",
                color=0xFF0000
            )
            error_embed.add_field(name="Error", value=str(e)[:1000], inline=False)
            await status_msg.edit(embed=error_embed)
            logger.error(f"Error in analyze command: {e}")
    else:
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
        logger.warning("No image URL detected for analysis.")

    await reset_status()

async def analyze_chart_image(chart_url: str, user_prompt: str = "", user: Optional[discord.User] = None, 
                             guild_id: Optional[int] = None, channel_id: Optional[int] = None) -> Optional[str]:
    try:
        async with session.get(chart_url) as resp:
            if resp.status != 200:
                logger.error(f"Failed to fetch image from URL: {chart_url} with status {resp.status}")
                return "Failed to fetch the image. Please ensure the URL is correct and accessible."
            image_bytes = await resp.read()

        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            logger.warning(f"Image size {len(image_bytes)} bytes exceeds the maximum allowed size.")
            return "The submitted image is too large to analyze. Please provide an image smaller than 5 MB."

        # Analysis based on the full image now, as gpt-4o handles it better
        base_prompt = (
            "you're a schizo defi agent. analyze this chart like you're on 5 cups of coffee and need to extract alpha:\n\n"
            "**sentiment:** [bullish/bearish/neutral + confidence %]\n"
            "**key levels:** [support/resistance with exact prices]\n"
            "**pattern:** [what you see + timeframe]\n"
            "**volume:** [unusual activity + implications]\n"
            "**risk/reward:** [entry/exit/stop levels]\n"
            "**timeframe:** [best trade horizon]\n"
            "**catalysts:** [what could move price]\n\n"
            "no tables, no narratives, no #### formatting. just raw alpha in bullet points. speak like crypto twitter. show me the charts and i'll show you the news."
        )
        full_prompt = f"{base_prompt} {user_prompt}" if user_prompt else base_prompt

        response = await aclient.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": chart_url}},
                    ],
                }
            ],
            max_tokens=1000,
        )
        
        analysis = response.choices[0].message.content.strip().replace("####", "###")
        logger.debug(f"Received analysis from Vision API: {analysis[:100]}...")

        # Update usage data - a simplified estimation as token count is complex
        # A more accurate method would parse the usage from the response if available
        estimated_tokens = 1000  # A rough estimate for a complex image
        cost = (estimated_tokens / 1_000_000) * 0.60  # GPT-4.1 input pricing
        
        usage_data['openai_gpt41_mini_vision']['requests'] += 1
        usage_data['openai_gpt41_mini_vision']['tokens'] += estimated_tokens
        usage_data['openai_gpt41_mini_vision']['cost'] += cost
        increment_token_cost(cost)
        
        # Log to database if user provided
        if user:
            await log_usage_to_db(
                user=user,
                command="analyze",
                model="gpt-4.1-vision",
                input_tokens=estimated_tokens,
                output_tokens=500,  # Rough estimate
                cost=cost,
                guild_id=guild_id,
                channel_id=channel_id
            )
        
        logger.info(f"Estimated OpenAI GPT-4.1 Vision usage: Tokens={estimated_tokens}, Cost=${cost:.6f}")
        return analysis

    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return "An error occurred during image analysis. Please try again later."

@bot.command(name='ask')
async def ask(ctx: Context, *, question: Optional[str] = None) -> None:
    await bot.change_presence(activity=Activity(type=ActivityType.playing, name="researching..."))
    logger.debug("Status updated to: [playing] researching...")

    if not question:
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
        await reset_status()
        return

    # Enhanced input validation
    if len(question) < 5:
        await ctx.send("⚠️ Please provide a more detailed question (at least 5 characters).")
        await reset_status()
        return
    
    if len(question) > 500:
        await ctx.send("⚠️ Question is too long. Please keep it under 500 characters.")
        await reset_status()
        return
    
    # Log the ask command query
    if db_manager.pool:
        username = f"{ctx.author.name}#{ctx.author.discriminator}" if ctx.author.discriminator != "0" else ctx.author.name
        await db_manager.log_user_query(
            user_id=ctx.author.id,
            username=username,
            command="ask",
            query_text=question,
            channel_id=ctx.channel.id,
            guild_id=ctx.guild.id if ctx.guild else None,
            response_generated=False
        )
    
    message_counter[ctx.author.id] += 1
    command_counter['ask'] += 1
    
    # Enhanced ask with streaming-like progress updates
    try:
        # Send initial "thinking" message that we'll edit for progress
        progress_embed = discord.Embed(
            title="🔍 SecurePath Agent Research",
            description=f"**Query:** {question[:100]}{'...' if len(question) > 100 else ''}",
            color=0x1D82B6
        )
        progress_embed.add_field(name="Status", value="🔄 Initializing research...", inline=False)
        progress_embed.set_footer(text="SecurePath Agent • Real-time Intelligence")
        
        status_msg = await ctx.send(embed=progress_embed)
        
        await process_message_with_streaming(ctx.message, status_msg, question=question, command='ask')
        logger.info(f"Successfully completed ask command for user {ctx.author.id}")
        
    except Exception as e:
        logger.error(f"Error in ask command for user {ctx.author.id}: {e}")
        logger.error(traceback.format_exc())
        
        # Enhanced error handling with user-friendly messages
        error_msg = str(e) if any(emoji in str(e) for emoji in ['🚫', '⏱️', '🔑', '🌐', '⚠️', '🤷']) else f"🚫 Error: {str(e)[:100]}"
        
        error_embed = discord.Embed(
            title="❌ Research Failed",
            description=error_msg,
            color=0xFF0000
        )
        error_embed.add_field(name="Debug Info", value=f"Error type: {type(e).__name__}", inline=False)
        error_embed.add_field(name="Suggestion", value="Try rephrasing your question or wait a moment before trying again.", inline=False)
        error_embed.set_footer(text="If this persists, please contact support")
        
        # Try to edit the status message if it exists
        try:
            if 'status_msg' in locals():
                await status_msg.edit(embed=error_embed)
            else:
                await ctx.send(embed=error_embed)
        except Exception as edit_error:
            logger.error(f"Failed to edit status message: {edit_error}")
            await ctx.send(f"{ctx.author.mention} {error_msg}")
    finally:
        await reset_status()

@bot.command(name='summary')
async def summary(ctx: Context, channel: discord.TextChannel = None) -> None:
    await bot.change_presence(activity=Activity(type=ActivityType.playing, name="channel summary..."))
    logger.debug("Status updated to: [playing] channel summary...")

    if channel is None:
        await ctx.send("Please specify a channel to summarize. Example: !summary #market-analysis")
        await reset_status()
        return

    if not channel.permissions_for(channel.guild.me).read_messages:
        await ctx.send(f"I don't have permission to read messages in {channel.mention}.")
        logger.warning(f"Missing permissions to read messages in channel {channel.name}")
        await reset_status()
        return

    # Log the summary command query
    if db_manager.pool:
        username = f"{ctx.author.name}#{ctx.author.discriminator}" if ctx.author.discriminator != "0" else ctx.author.name
        await db_manager.log_user_query(
            user_id=ctx.author.id,
            username=username,
            command="summary",
            query_text=f"Summary request for #{channel.name}",
            channel_id=ctx.channel.id,
            guild_id=ctx.guild.id if ctx.guild else None,
            response_generated=False
        )

    command_counter['summary'] += 1
    await perform_channel_summary(ctx, channel, command='summary')
    await reset_status()

async def perform_channel_summary(ctx: Context, channel: discord.TextChannel, command: Optional[str] = None) -> None:
    logger.info(f"Starting summary for channel: {channel.name} (ID: {channel.id})")
    
    # Send enhanced status message with progress tracking
    status_embed = discord.Embed(
        title="🔍 Analyzing Channel Activity",
        description=f"Processing messages from {channel.mention} (last 72 hours)...",
        color=0x1D82B6
    )
    status_embed.add_field(name="Status", value="🔄 Fetching messages...", inline=False)
    status_msg = await ctx.send(embed=status_embed)
    
    try:
        time_limit = datetime.now(timezone.utc) - timedelta(hours=72)  # 72 hours max for context length limits
        messages = []
        
        # Enhanced message filtering - INCLUDE bot/webhook messages for crypto channels
        message_count = 0
        async for msg in channel.history(after=time_limit, limit=3000, oldest_first=True):  # Reduced limit for context
            message_count += 1
            content = msg.content.strip()
            # Very inclusive filtering - crypto channels often have valuable bot data
            if (content and 
                len(content) > 5 and  # Reasonable minimum length
                not content.startswith(('!ping', '!help', '!commands', '!stats', '!test'))):  # Skip only basic bot commands
                # Include ALL users including bots, webhooks, and automated feeds
                author_name = msg.author.display_name if not msg.author.bot else f"🤖{msg.author.display_name}"
                messages.append(f"[{author_name}]: {content}")
        
        logger.info(f"Scanned {message_count} messages from {channel.name}, found {len(messages)} valid messages")
        
        # If still no content, try with absolute maximum inclusivity
        if not messages and message_count > 0:
            logger.warning(f"No messages passed filtering in {channel.name}, trying absolute maximum inclusivity")
            async for msg in channel.history(after=time_limit, limit=5000, oldest_first=True):
                content = msg.content.strip()
                if content:  # Include absolutely everything with any content
                    author_name = msg.author.display_name if not msg.author.bot else f"🤖{msg.author.display_name}"
                    messages.append(f"[{author_name}]: {content}")

        logger.info(f"Found {len(messages)} quality messages to summarize in channel {channel.name}")

        if not messages:
            # Debug: check what messages actually exist
            debug_count = 0
            bot_count = 0
            recent_count = 0
            
            # Check recent messages (last 24 hours)
            recent_limit = datetime.now(timezone.utc) - timedelta(hours=24)
            async for msg in channel.history(after=recent_limit, limit=1000):
                if msg.content.strip():
                    recent_count += 1
            
            # Check all messages in timeframe
            async for msg in channel.history(after=time_limit, limit=2000):
                if msg.content.strip():
                    debug_count += 1
                    if msg.author.bot:
                        bot_count += 1
            
            error_embed = discord.Embed(
                title="⚠️ No Content Found",
                description=f"No substantial messages found in {channel.mention} from the last 72 hours.",
                color=0xFF6B35
            )
            error_embed.add_field(
                name="Debug Info", 
                value=f"Messages (72h): {debug_count}\nMessages (24h): {recent_count}\nBot messages: {bot_count}\nFiltered messages: 0", 
                inline=False
            )
            error_embed.add_field(
                name="Suggestion", 
                value="Channel may be inactive or bot lacks message history permissions.", 
                inline=False
            )
            await status_msg.edit(embed=error_embed)
            return
        
        # Update status
        status_embed.set_field_at(0, name="Status", value=f"🧠 Processing {len(messages)} messages...", inline=False)
        await status_msg.edit(embed=status_embed)

        full_text = "\n".join(messages)
        # Optimized chunking for better performance and context retention
        chunk_size = 15000  # Increased for better context
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

        logger.info(f"Processing {len(chunks)} chunks for summary (enhanced processing)")
        
        # Enhanced concurrent processing with better error handling
        async def process_chunk(i, chunk):
            # Enhanced prompt for higher quality output
            prompt = f"""you're a schizo defi agent analyzing {channel.name} for alpha. extract actionable intelligence like your bags depend on it:
            
**focus areas:**
• market sentiment & crowd psychology
• price movements & volume patterns  
• breaking news & catalyst events
• whale activity & large transactions
• technical analysis & key levels
• regulatory developments
• project updates & partnerships

**output format:**
- bullet points only, no tables
- include specific numbers/percentages when mentioned
- flag high-impact info with 🚨
- keep insights concise and actionable
- speak like crypto twitter, lowercase preferred
- show me the docs, show me the code mentality

MESSAGES:
{chunk}"""
            
            for attempt in range(2):  # Retry logic
                try:
                    response = await aclient.chat.completions.create(
                        model='gpt-4.1', 
                        messages=[{"role": "user", "content": prompt}], 
                        max_tokens=1500,  # Increased for better quality
                        temperature=0.3  # Lower temperature for more focused output
                    )
                    result = response.choices[0].message.content.strip()
                    increment_api_call_counter()
                    
                    # Track processing cost for summary command
                    if hasattr(response, 'usage') and response.usage:
                        usage = response.usage
                        input_tokens = getattr(usage, 'prompt_tokens', 0)
                        output_tokens = getattr(usage, 'completion_tokens', 0)
                        cost = (input_tokens * 0.40 + output_tokens * 1.60) / 1_000_000
                        
                        # Accumulate cost for final logging
                        if not hasattr(process_chunk, 'total_cost'):
                            process_chunk.total_cost = 0
                            process_chunk.total_input_tokens = 0
                            process_chunk.total_output_tokens = 0
                        process_chunk.total_cost += cost
                        process_chunk.total_input_tokens += input_tokens
                        process_chunk.total_output_tokens += output_tokens
                    
                    logger.info(f"Successfully processed chunk {i+1}/{len(chunks)}")
                    return result
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed for chunk {i+1}: {e}")
                    if attempt == 1:  # Final attempt failed
                        logger.error(f"Failed to process chunk {i+1} after retries")
                        return None
                    await asyncio.sleep(1)  # Brief delay before retry
            return None

        # Update status with processing info
        status_embed.set_field_at(0, name="Status", value=f"⚙️ Processing {len(chunks)} chunks concurrently...", inline=False)
        await status_msg.edit(embed=status_embed)
        
        # Process all chunks concurrently with progress tracking
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter and validate results
        chunk_summaries = []
        for r in results:
            if r and not isinstance(r, Exception) and len(r.strip()) > 50:
                chunk_summaries.append(r)
            elif isinstance(r, Exception):
                logger.error(f"Chunk processing exception: {r}")

        if not chunk_summaries:
            error_embed = discord.Embed(
                title="❌ Processing Failed",
                description=f"Unable to process messages from {channel.mention}. Please try again later.",
                color=0xFF0000
            )
            error_embed.add_field(name="Tip", value="Make sure the channel has substantial discussion in the last 72 hours.", inline=False)
            await status_msg.edit(embed=error_embed)
            return

        # Update status for final synthesis
        status_embed.set_field_at(0, name="Status", value=f"🧑‍💻 Synthesizing {len(chunk_summaries)} summaries...", inline=False)
        await status_msg.edit(embed=status_embed)
        
        # Enhanced final synthesis prompt
        current_date = datetime.now().strftime("%Y-%m-%d")
        final_prompt = f"""Synthesize these {channel.name} channel summaries into actionable intelligence for crypto traders/investors.

DATE: {current_date}
CHANNEL: #{channel.name}
TIMEFRAME: Last 72 hours

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

**no tables, no verbose explanations. pure alpha extraction. speak like crypto twitter - casual, lowercase, confident. show me the data, not the narrative.**

CHUNK SUMMARIES:
{chr(10).join(chunk_summaries)}"""
        
        try:
            response = await aclient.chat.completions.create(
                model='gpt-4.1', 
                messages=[{"role": "user", "content": final_prompt}], 
                max_tokens=2500,  # Increased for comprehensive output
                temperature=0.2   # Lower for more focused synthesis
            )
            final_summary = response.choices[0].message.content.strip()
            increment_api_call_counter()
            
            # Log complete summary usage to database
            total_cost = getattr(process_chunk, 'total_cost', 0)
            total_input = getattr(process_chunk, 'total_input_tokens', 0)
            total_output = getattr(process_chunk, 'total_output_tokens', 0)
            
            # Add final synthesis cost
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                final_input = getattr(usage, 'prompt_tokens', 0)
                final_output = getattr(usage, 'completion_tokens', 0)
                final_cost = (final_input * 0.40 + final_output * 1.60) / 1_000_000
                total_cost += final_cost
                total_input += final_input
                total_output += final_output
            
            # Log to database
            await log_usage_to_db(
                user=ctx.author,
                command="summary",
                model="gpt-4.1",
                input_tokens=total_input,
                output_tokens=total_output,
                cost=total_cost,
                guild_id=ctx.guild.id if ctx.guild else None,
                channel_id=ctx.channel.id
            )
            
            # Delete status message and send final result
            await status_msg.delete()
            
            # Enhanced summary embed with metadata
            summary_embed = discord.Embed(
                title=f"📄 {channel.name.title()} Intelligence Report",
                description=f"**Timeframe:** Last 72 hours | **Messages Analyzed:** {len(messages):,}",
                color=0x1D82B6,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Simple, reliable embed creation to avoid breaking
            summary_embed.set_footer(text=f"SecurePath Agent • Cost: ${total_cost:.4f} | Processed {len(chunks)} chunks")
            
            # Use reliable content delivery
            try:
                # If summary is short enough, use single embed
                if len(final_summary) <= 3800:  # Leave room for description
                    summary_embed.description += f"\n\n{final_summary}"
                    await ctx.send(embed=summary_embed)
                else:
                    # For long summaries, send title embed first, then use send_long_embed
                    await ctx.send(embed=summary_embed)
                    await send_long_embed(
                        ctx.channel,
                        final_summary,
                        color=0x1D82B6,
                        title="📈 Detailed Analysis"
                    )
            except discord.HTTPException as e:
                logger.error(f"Failed to send summary embed: {e}")
                # Fallback to simple text message
                fallback_text = f"**{channel.name.title()} Summary (72h) - {len(messages):,} messages**\n\n{final_summary[:1800]}{'...' if len(final_summary) > 1800 else ''}"
                await ctx.send(fallback_text)
            logger.info(f"Successfully sent enhanced summary for channel {channel.name} (Cost: ${total_cost:.4f})")

            await log_interaction(user=ctx.author, channel=ctx.channel, command=command, user_input=f"Summary for #{channel.name}", bot_response=final_summary[:1024])
            
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            logger.error(traceback.format_exc())
            error_embed = discord.Embed(
                title="❌ Synthesis Failed",
                description="An error occurred while generating the final summary.",
                color=0xFF0000
            )
            error_embed.add_field(name="Error", value=str(e)[:1000], inline=False)
            await status_msg.edit(embed=error_embed)
    except Exception as e:
        logger.error(f"Error in perform_channel_summary: {e}")
        logger.error(traceback.format_exc())
        await ctx.send(f"An error occurred while processing the summary for channel {channel.mention}.")

def ensure_single_instance():
    lock_file = '/tmp/securepath_bot.lock'
    try:
        import fcntl
        fp = open(lock_file, 'w')
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.debug(f"Acquired lock on {lock_file}")
        return fp
    except (IOError, ImportError):
        logger.warning("Could not acquire lock. This may be due to running on a non-Unix OS or another instance is running.")
        class DummyLock:
            def close(self): pass
        return DummyLock()

async def force_shutdown() -> None:
    logger.info("Shutting down SecurePath AI Bot.")
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop=loop):
        if task is not asyncio.current_task(): task.cancel()
    await asyncio.sleep(0.1)
    
    # Close database connection
    if db_manager.pool:
        await db_manager.disconnect()
        logger.info("Database connection closed")
    
    if session: await session.close()
    if conn: await conn.close()
    if bot: await bot.close()
    loop.stop()

def handle_exit() -> None:
    asyncio.create_task(force_shutdown())
    asyncio.get_event_loop().call_later(2, quiet_exit)

def quiet_exit() -> None:
    console.print("Bot has been shut down.")
    logging.shutdown()
    sys.exit(0)

async def health_check(request):
    return web.Response(text="Healthy")

async def start_bot() -> None:
    global conn, session
    import signal
    try:
        logger.info("Setting up signal handlers")
        if sys.platform != "win32":
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_event_loop().add_signal_handler(sig, handle_exit)

        conn = TCPConnector(limit=10)
        session = ClientSession(connector=conn)

        if not config.DISCORD_TOKEN:
            logger.error("DISCORD_TOKEN is not set. Cannot start the bot.")
            return

        app = web.Application()
        app.router.add_get('/', health_check)
        port = int(os.environ.get("PORT", 5050))
        web_runner = web.AppRunner(app)
        await web_runner.setup()
        site = web.TCPSite(web_runner, '0.0.0.0', port)
        await site.start()
        logger.info(f"Health check endpoint running at http://0.0.0.0:{port}")

        await bot.start(config.DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"Error during bot startup: {type(e).__name__}: {str(e)}", exc_info=True)
    finally:
        if 'web_runner' in locals(): await web_runner.cleanup()
        if session: await session.close()
        if conn: await conn.close()

async def send_stats() -> None:
    channel = bot.get_channel(config.LOG_CHANNEL_ID)
    if not channel: 
        logger.warning("Admin channel not found for startup stats")
        return

    embed = discord.Embed(
        title="🚀 SecurePath Agent - System Status", 
        description="Agent successfully initialized and ready for operations",
        color=0x1D82B6,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Database connectivity status
    db_status = "🟢 Connected" if db_manager.pool else "🔴 Disconnected"
    embed.add_field(name="Database", value=db_status, inline=True)
    embed.add_field(name="Active Guilds", value=len(bot.guilds), inline=True)
    embed.add_field(name="Latency", value=f"{bot.latency*1000:.1f}ms", inline=True)
    
    # Get database stats if available
    if db_manager.pool:
        try:
            stats = await db_manager.get_global_stats()
            if stats and stats['overall']:
                overall = stats['overall']
                embed.add_field(
                    name="📊 Total Usage",
                    value=f"**Requests:** {overall['total_requests']:,}\n"
                          f"**Users:** {overall['unique_users']:,}\n"
                          f"**Cost:** ${overall['total_cost']:.4f}",
                    inline=True
                )
                
                # Top command if available
                if stats['top_commands']:
                    # Filter out background commands
                    top_cmd = next(
                        (cmd for cmd in stats['top_commands'] 
                         if cmd['command'] not in ['summary_chunk', 'summary_final']), 
                        None
                    )
                    if top_cmd:
                        embed.add_field(
                            name="🎯 Most Popular", 
                            value=f"**{top_cmd['command']}** ({top_cmd['usage_count']} uses)", 
                            inline=True
                        )
        except Exception as e:
            logger.error(f"Failed to get database stats for startup: {e}")
            embed.add_field(name="📊 Usage Stats", value="Initializing...", inline=True)
    else:
        embed.add_field(name="📊 Usage Stats", value="Database offline", inline=True)
    
    embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1 & Perplexity Sonar-Pro")
    
    try:
        await channel.send(embed=embed)
        logger.info("Startup stats sent to admin channel")
    except discord.HTTPException as e:
        logger.error(f"Failed to send startup stats embed: {e}")

@tasks.loop(hours=12)
async def send_periodic_stats() -> None:
    await send_stats()

@tasks.loop(hours=24)
async def reset_api_call_counter():
    global api_call_counter, total_token_cost
    api_call_counter = 0
    total_token_cost = 0.0 # Also reset cost daily
    for key in usage_data: # Reset all usage data
        for sub_key in usage_data[key]:
            usage_data[key][sub_key] = 0 if isinstance(usage_data[key][sub_key], (int, float)) else {}
    logger.info("API call counter and usage data reset")

@bot.command(name='token_usage')
@commands.has_permissions(administrator=True)
async def token_usage(ctx: Context) -> None:
    if ctx.author.id != config.OWNER_ID:
        await ctx.send("You do not have permission to use this command.")
        return

    embed = discord.Embed(title="📊 Token Usage and Costs", color=0x1D82B6, timestamp=datetime.now(timezone.utc))
    for model, data in usage_data.items():
        value = "\n".join([f"**{k.replace('_', ' ').title()}:** {v}" for k, v in data.items()])
        embed.add_field(name=model.replace('_', ' ').title(), value=value or "No data", inline=False)
    await ctx.send(embed=embed)

@bot.command(name='cache_stats')
@commands.has_permissions(administrator=True)
async def cache_stats(ctx: Context) -> None:
    if ctx.author.id != config.OWNER_ID:
        await ctx.send("You do not have permission to use this command.")
        return
    hit_rate = calculate_cache_hit_rate()
    embed = discord.Embed(title="📊 Cache Hit Rate", description=f"OpenAI GPT-4.1 Cache Hit Rate: **{hit_rate:.2f}%**", color=0x1D82B6)
    await ctx.send(embed=embed)

def calculate_cache_hit_rate() -> float:
    total_cached = usage_data['openai_gpt41_mini']['cached_input_tokens']
    total_input = usage_data['openai_gpt41_mini']['input_tokens'] + total_cached
    return (total_cached / total_input * 100) if total_input > 0 else 0.0

@bot.command(name='stats')
@commands.has_permissions(administrator=True)
async def unified_stats(ctx: Context) -> None:
    """Show comprehensive SecurePath Agent analytics (admin only)"""
    if ctx.author.id != config.OWNER_ID:
        await ctx.send("You do not have permission to use this command.")
        return
    
    if not db_manager.pool:
        await ctx.send("Database not available. Stats tracking is currently offline.")
        return
    
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
    hit_rate = calculate_cache_hit_rate()
    embed.add_field(
        name="⚡ System Performance",
        value=f"**Cache Hit Rate:** {hit_rate:.1f}%\n"
              f"**API Calls:** {api_call_counter:,}\n"
              f"**Active Guilds:** {len(bot.guilds)}",
        inline=True
    )
    
    embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1 & Perplexity Sonar-Pro")
    await ctx.send(embed=embed)




@bot.command(name='commands')
async def commands_help(ctx: Context) -> None:
    """Show SecurePath Agent help and available commands"""
    embed = discord.Embed(
        title="🤖 SecurePath Agent - Commands",
        description="Advanced crypto analysis powered by AI",
        color=0x1D82B6,
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(
        name="🔍 !ask [question]",
        value="Get real-time market insights using Perplexity AI\n"
              "Sources from GitHub, news, DeFi data, and more\n"
              "Example: `!ask What's happening with Bitcoin?`",
        inline=False
    )
    
    embed.add_field(
        name="📊 !analyze [image]",
        value="Advanced chart analysis using GPT-4.1 Vision\n"
              "Attach an image or use recent chart in channel\n"
              "Gets sentiment, key levels, patterns, and trade setups",
        inline=False
    )
    
    embed.add_field(
        name="📄 !summary #channel",
        value="Generate alpha-focused summary of channel activity\n"
              "Extracts market sentiment, events, and key movements\n"
              "Example: `!summary #crypto-news`",
        inline=False
    )
    
    embed.add_field(
        name="📈 !stats",
        value="View server-wide usage statistics\n"
              "See popular commands and active users",
        inline=True
    )
    
    embed.add_field(
        name="🎯 Features",
        value="• Real-time data from elite sources\n"
              "• Alpha-focused, no fluff responses\n"
              "• Persistent usage tracking\n"
              "• Context-aware conversations in DMs",
        inline=True
    )
    
    embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1 & Perplexity Sonar-Pro")
    await ctx.send(embed=embed)

@bot.command(name='ping')
async def ping(ctx: Context) -> None:
    """Check SecurePath Agent latency and database status"""
    start_time = time.time()
    message = await ctx.send("🏓 Pinging...")
    end_time = time.time()
    
    latency = round(bot.latency * 1000)
    response_time = round((end_time - start_time) * 1000)
    
    # Check database status
    db_status = "🟢 Connected" if db_manager.pool else "🔴 Disconnected"
    
    embed = discord.Embed(
        title="🏓 Agent Status Check",
        description="All systems operational",
        color=0x1D82B6,
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(name="Discord Latency", value=f"{latency}ms", inline=True)
    embed.add_field(name="Response Time", value=f"{response_time}ms", inline=True)
    embed.add_field(name="Database", value=db_status, inline=True)
    embed.add_field(name="API Calls Today", value=f"{api_call_counter}", inline=True)
    embed.set_footer(text="SecurePath Agent • Powered by GPT-4.1 & Perplexity Sonar-Pro")
    
    await message.edit(content="", embed=embed)

if __name__ == "__main__":
    lock_file_handle = ensure_single_instance()
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user")
        handle_exit()
    finally:
        if lock_file_handle: lock_file_handle.close()
        quiet_exit()