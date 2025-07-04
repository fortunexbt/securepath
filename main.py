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

logger = logging.getLogger('SecurePathBot')
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
    'openai_gpt41_mini': {
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

def can_make_api_call() -> bool:
    # Removed API call limits for maximum performance with small user base
    return True

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB limit

def estimate_tokens(image_size_bytes: int) -> int:
    TOKENS_PER_BYTE = 1 / 100
    estimated = int(image_size_bytes * TOKENS_PER_BYTE)
    logger.debug(f"Estimated tokens based on image size ({image_size_bytes} bytes): {estimated} tokens")
    return estimated

async def fetch_perplexity_response(user_id: int, new_message: str) -> Optional[str]:
    if session is None:
        logger.error("Session is not initialized")
        return None
    if not can_make_api_call():
        logger.warning("Daily API call limit reached. Skipping API call.")
        return None

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
    
    # 2. Curate an elite, TOP-10 search source list to respect the API limit.
    domain_filter = [
        "github.com",          # Primary Source: Code
        "ethereum.org",        # Primary Source: Docs
        "solana.com",          # Primary Source: Docs
        "defillama.com",       # Primary Source: Data
        "etherscan.io",        # Primary Source: On-chain activity
        "medium.com",          # Secondary Source: Project blogs/updates
        "coindesk.com",        # Secondary Source: Reputable News
        "thedefiant.io",       # Secondary Source: Reputable DeFi News
        "-reddit.com",         # Exclusion: High noise
        "-pinterest.com"       # Exclusion: Irrelevant
    ]
    logger.debug(f"Using Perplexity domain filter with {len(domain_filter)} elite sources.")

    # 3. Define the full data payload with optimized search parameters.
    data = {
        "model": "sonar-pro",
        "messages": messages,
        "max_tokens": 1000,
        "search_after_date_filter": ninety_days_ago,
        "search_domain_filter": domain_filter,
        "web_search_options": {
            "search_context_size": "high"
        }
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
                
                citations = resp_json.get('choices', [{}])[0].get('extras', {}).get('citations', [])

                if citations:
                    formatted_citations = "\n\n**DYOR (do your own research):**\n"
                    for cite in citations:
                        title = cite.get('title', 'Source')
                        url = cite.get('url', '#')
                        formatted_citations += f"- [{title}]({url})\n"
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
            model='gpt-4.1-mini',
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
            usage_data['openai_gpt41_mini']['cached_input_tokens'] += cached_tokens
            cost = (cached_tokens / 1_000_000 * 0.20) + (completion_tokens / 1_000_000 * 0.80)  # GPT-4.1-mini cached pricing
            logger.debug(f"Cache hit detected. Cached Tokens: {cached_tokens}, Completion Tokens: {completion_tokens}, Cost: ${cost:.6f}")
        else:
            usage_data['openai_gpt41_mini']['input_tokens'] += prompt_tokens
            cost = (prompt_tokens / 1_000_000 * 0.40) + (completion_tokens / 1_000_000 * 1.60)  # GPT-4.1-mini pricing
            logger.debug(f"No cache hit. Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Cost: ${cost:.6f}")

        usage_data['openai_gpt41_mini']['cost'] += cost
        increment_token_cost(cost)
        
        # Log to database if user provided
        if user:
            await log_usage_to_db(
                user=user,
                command=command,
                model="gpt-4.1-mini",
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                cost=cost,
                guild_id=guild_id,
                channel_id=channel_id
            )

        logger.info(f"OpenAI GPT-4.1-mini usage: Prompt Tokens={prompt_tokens}, Cached Tokens={cached_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")
        logger.info(f"Estimated OpenAI GPT-4.1-mini API call cost: ${cost:.6f}")
        return answer
    except Exception as e:
        logger.error(f"Error fetching response from OpenAI: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def send_long_embed(
    channel: discord.abc.Messageable,
    text: str,
    color: int = 0x004200,
    title: Optional[str] = None,
    image_url: Optional[str] = None
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

        embed.set_author(name=bot.user.name, icon_url=bot.user.avatar.url if bot.user.avatar else None)
        
        if len(parts) > 1:
            embed.set_footer(text=f"Part {i + 1}/{len(parts)}")

        try:
            await channel.send(embed=embed)
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
        color=0xFF9900,
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

async def process_message(message: discord.Message, question: Optional[str] = None, command: Optional[str] = None) -> None:
    # Removed rate limiting for maximum performance with small user base

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
    logger.info(f'Bot is active in {len(bot.guilds)} guild(s)')
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

async def send_initial_stats() -> None:
    await asyncio.sleep(5)
    await send_stats()

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
    
    if chart_url:
        await ctx.send("Detected a chart, analyzing it...")
        logger.info(f"Chart URL detected: {chart_url}")

        guild_id = ctx.guild.id if ctx.guild else None
        image_analysis = await analyze_chart_image(
            chart_url, 
            user_prompt, 
            user=ctx.author,
            guild_id=guild_id,
            channel_id=ctx.channel.id
        )

        if image_analysis:
            await send_long_embed(
                ctx.channel,
                text=image_analysis,
                color=0x00ff00,
                title="Image Analysis",
                image_url=chart_url
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
            await ctx.send("Sorry, I couldn't analyze the image. Please try again.")
            logger.warning("Image analysis failed to return a response.")
    else:
        await ctx.send("No chart detected. Please attach an image to analyze.")
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
            "Analyze this chart like a top-tier quant. Extract pure alpha:\n\n"
            "**Sentiment:** [Bullish/Bearish/Neutral + confidence %]\n"
            "**Key Levels:** [Support/Resistance with exact prices]\n"
            "**Pattern:** [What you see + timeframe]\n"
            "**Volume:** [Unusual activity + implications]\n"
            "**Risk/Reward:** [Entry/Exit/Stop levels]\n"
            "**Timeframe:** [Best trade horizon]\n"
            "**Catalysts:** [What could move price]\n\n"
            "No narratives. Just actionable data. Don't use #### formatting."
        )
        full_prompt = f"{base_prompt} {user_prompt}" if user_prompt else base_prompt

        response = await aclient.chat.completions.create(
            model="gpt-4.1-mini",
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
        cost = (estimated_tokens / 1_000_000) * 0.40  # GPT-4.1-mini input pricing
        
        usage_data['openai_gpt41_mini_vision']['requests'] += 1
        usage_data['openai_gpt41_mini_vision']['tokens'] += estimated_tokens
        usage_data['openai_gpt41_mini_vision']['cost'] += cost
        increment_token_cost(cost)
        
        # Log to database if user provided
        if user:
            await log_usage_to_db(
                user=user,
                command="analyze",
                model="gpt-4.1-mini-vision",
                input_tokens=estimated_tokens,
                output_tokens=500,  # Rough estimate
                cost=cost,
                guild_id=guild_id,
                channel_id=channel_id
            )
        
        logger.info(f"Estimated OpenAI GPT-4.1-mini Vision usage: Tokens={estimated_tokens}, Cost=${cost:.6f}")
        return analysis

    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return "An error occurred during image analysis. Please try again later."

@bot.command(name='ask')
async def ask(ctx: Context, *, question: Optional[str] = None) -> None:
    await bot.change_presence(activity=Activity(type=ActivityType.watching, name="a question..."))
    logger.debug("Status updated to: [watching] a question...")

    if not question:
        await ctx.send("Please provide a question after the !ask command. Example: !ask What is yield farming?")
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
    await process_message(ctx.message, question=question, command='ask')
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
    await ctx.send(f"Generating summary for {channel.mention}... This may take a moment.")
    
    try:
        time_limit = datetime.now(timezone.utc) - timedelta(hours=48)
        messages = []
        async for msg in channel.history(after=time_limit, limit=None, oldest_first=True):
            if msg.content.strip():
                messages.append(msg.content)

        logger.info(f"Found {len(messages)} messages to summarize in channel {channel.name}")

        if not messages:
            await ctx.send(f"No messages to summarize in channel {channel.mention}.")
            return

        full_text = "\n".join(messages)
        # Optimize for speed: larger chunks = fewer API calls = faster processing
        chunk_size, chunk_summaries = 12000, []
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

        logger.info(f"Processing {len(chunks)} chunks for summary (optimized for speed)")
        
        # Process chunks concurrently for maximum speed
        async def process_chunk(i, chunk):
            prompt = f"Extract key alpha from these {channel.name} messages. Focus on:\n• Market sentiment signals\n• Price/volume anomalies\n• Breaking news impact\n• Whale movements\n• Technical patterns\n• Regulatory updates\n\nFormat as bullet points. Be concise, no fluff:\n\n{chunk}"
            try:
                response = await aclient.chat.completions.create(
                    model='gpt-4.1-mini', 
                    messages=[{"role": "user", "content": prompt}], 
                    max_tokens=1200  # Reduced for faster processing
                )
                result = response.choices[0].message.content.strip()
                increment_api_call_counter()
                
                # Log chunk processing to database
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    input_tokens = getattr(usage, 'prompt_tokens', 500)
                    output_tokens = getattr(usage, 'completion_tokens', 300)
                    cost = (input_tokens * 0.40 + output_tokens * 1.60) / 1_000_000
                    
                    await log_usage_to_db(
                        user=ctx.author,
                        command="summary_chunk",
                        model="gpt-4.1-mini",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                        guild_id=ctx.guild.id if ctx.guild else None,
                        channel_id=ctx.channel.id
                    )
                
                logger.info(f"Successfully processed chunk {i+1}/{len(chunks)}")
                return result
            except Exception as e:
                logger.error(f"Error summarizing chunk {i+1}: {e}")
                return None

        # Process all chunks concurrently for maximum speed
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        chunk_summaries = [r for r in results if r and not isinstance(r, Exception)]

        if not chunk_summaries:
            await ctx.send(f"Could not generate a summary for channel {channel.mention}.")
            return

        final_prompt = f"Synthesize these channel summaries into pure alpha. Structure as:\n\n**Market Sentiment:** [Bullish/Bearish/Neutral + %]\n**Key Events:** [Bullet points]\n**Price Action:** [Notable movements]\n**Technical:** [Important levels/patterns]\n**Regulatory:** [Updates if any]\n**Whale Activity:** [Large moves if any]\n\nBe direct, actionable, no narrative fluff:\n\n{' '.join(chunk_summaries)}"
        try:
            response = await aclient.chat.completions.create(model='gpt-4.1-mini', messages=[{"role": "user", "content": final_prompt}], max_tokens=2000)
            final_summary = response.choices[0].message.content.strip()
            increment_api_call_counter()
            
            # Log final summary to database
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                input_tokens = getattr(usage, 'prompt_tokens', 800)  # Estimate if not available
                output_tokens = getattr(usage, 'completion_tokens', 600)
                cost = (input_tokens * 0.40 + output_tokens * 1.60) / 1_000_000
                
                await log_usage_to_db(
                    user=ctx.author,
                    command="summary_final",
                    model="gpt-4.1-mini",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    guild_id=ctx.guild.id if ctx.guild else None,
                    channel_id=ctx.channel.id
                )
            
            await send_long_embed(ctx.channel, final_summary, color=0x1D82B6, title=f"📄 48-Hour Summary for #{channel.name}")
            logger.info(f"Successfully sent summary for channel {channel.name}")

            await log_interaction(user=ctx.author, channel=ctx.channel, command=command, user_input=f"Summary for #{channel.name}", bot_response=final_summary[:1024])
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            logger.error(traceback.format_exc())
            await ctx.send(f"An error occurred while generating the summary for channel {channel.mention}.")
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
    if not channel: return

    embed = discord.Embed(title="Bot Statistics", color=0x00ff00)
    embed.add_field(name="Total Messages", value=sum(message_counter.values()), inline=True)
    embed.add_field(name="Unique Users", value=len(message_counter), inline=True)
    embed.add_field(name="Commands Used", value=sum(command_counter.values()), inline=True)
    embed.add_field(name="API Calls", value=api_call_counter, inline=True)
    embed.add_field(name="Est. Cost", value=f"${total_token_cost:.6f}", inline=True)
    
    top_users = []
    for user_id, count in message_counter.most_common(5):
        try:
            user = await bot.fetch_user(user_id)
            top_users.append(f"{user.name}: {count}")
        except discord.NotFound:
            top_users.append(f"Unknown ({user_id}): {count}")

    embed.add_field(name="Top 5 Users", value="\n".join(top_users) or "No data", inline=False)
    embed.timestamp = datetime.now(timezone.utc)
    try:
        await channel.send(embed=embed)
    except discord.HTTPException as e:
        logger.error(f"Failed to send stats embed: {e}")

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
    embed = discord.Embed(title="📊 Cache Hit Rate", description=f"OpenAI GPT-4.1-mini Cache Hit Rate: **{hit_rate:.2f}%**", color=0x1D82B6)
    await ctx.send(embed=embed)

def calculate_cache_hit_rate() -> float:
    total_cached = usage_data['openai_gpt41_mini']['cached_input_tokens']
    total_input = usage_data['openai_gpt41_mini']['input_tokens'] + total_cached
    return (total_cached / total_input * 100) if total_input > 0 else 0.0

@bot.command(name='stats')
async def server_stats(ctx: Context) -> None:
    """Show server-wide bot usage statistics"""
    if not db_manager.pool:
        await ctx.send("Database not available. Stats tracking is currently offline.")
        return
    
    stats = await db_manager.get_global_stats()
    if not stats:
        await ctx.send("Failed to retrieve server statistics.")
        return
    
    overall = stats['overall']
    top_users = stats['top_users']
    top_commands = stats['top_commands']
    
    embed = discord.Embed(
        title="📊 Server Bot Statistics",
        description="Usage analytics for this server",
        color=0x00ff00,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Overall stats (without costs)
    embed.add_field(
        name="📈 Overall Usage",
        value=f"**Total Requests:** {overall['total_requests']:,}\n"
              f"**Active Users:** {overall['unique_users']:,}\n"
              f"**Total Tokens:** {overall['total_tokens']:,}\n"
              f"**Avg Tokens/Request:** {overall['avg_tokens_per_request']:.1f}",
        inline=True
    )
    
    # Top users (without costs)
    if top_users:
        top_users_text = "\n".join([
            f"**{user['username'][:20]}:** {user['total_requests']} requests"
            for user in top_users[:8]
        ])
        embed.add_field(name="👑 Most Active Users", value=top_users_text, inline=True)
    
    # Top commands
    if top_commands:
        top_commands_text = "\n".join([
            f"**{cmd['command']}:** {cmd['usage_count']} uses"
            for cmd in top_commands[:6]
        ])
        embed.add_field(name="🎯 Popular Commands", value=top_commands_text, inline=False)
    
    embed.set_footer(text="Powered by GPT-4.1-mini & Perplexity Sonar-Pro")
    await ctx.send(embed=embed)

@bot.command(name='global_stats')
@commands.has_permissions(administrator=True)
async def global_stats(ctx: Context) -> None:
    """Show global bot usage statistics (admin only)"""
    if ctx.author.id != config.OWNER_ID:
        await ctx.send("You do not have permission to use this command.")
        return
    
    if not db_manager.pool:
        await ctx.send("Database not available. Stats tracking is currently offline.")
        return
    
    stats = await db_manager.get_global_stats()
    if not stats:
        await ctx.send("Failed to retrieve global statistics.")
        return
    
    overall = stats['overall']
    top_users = stats['top_users']
    top_commands = stats['top_commands']
    
    embed = discord.Embed(
        title="🌍 Global Bot Statistics",
        color=0x00ff00,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Overall stats
    embed.add_field(
        name="📊 Overall",
        value=f"**Total Requests:** {overall['total_requests']:,}\n"
              f"**Unique Users:** {overall['unique_users']:,}\n"
              f"**Total Tokens:** {overall['total_tokens']:,}\n"
              f"**Total Cost:** ${overall['total_cost']:.4f}\n"
              f"**Avg Tokens/Request:** {overall['avg_tokens_per_request']:.1f}",
        inline=True
    )
    
    # Top users
    if top_users:
        top_users_text = "\n".join([
            f"**{user['username'][:20]}:** {user['total_requests']} requests"
            for user in top_users[:5]
        ])
        embed.add_field(name="👑 Top Users", value=top_users_text, inline=True)
    
    # Top commands
    if top_commands:
        top_commands_text = "\n".join([
            f"**{cmd['command']}:** {cmd['usage_count']} uses"
            for cmd in top_commands[:5]
        ])
        embed.add_field(name="🎯 Popular Commands", value=top_commands_text, inline=False)
    
    await ctx.send(embed=embed)

@bot.command(name='model_costs')
@commands.has_permissions(administrator=True)
async def model_costs(ctx: Context) -> None:
    """Show cost breakdown by AI model (admin only)"""
    if ctx.author.id != config.OWNER_ID:
        await ctx.send("You do not have permission to use this command.")
        return
    
    if not db_manager.pool:
        await ctx.send("Database not available. Stats tracking is currently offline.")
        return
    
    costs = await db_manager.get_costs_by_model()
    if not costs or not costs['model_costs']:
        await ctx.send("No model cost data available.")
        return
    
    embed = discord.Embed(
        title="💰 Model Cost Breakdown",
        color=0xFF6B35,
        timestamp=datetime.now(timezone.utc)
    )
    
    for model in costs['model_costs']:
        embed.add_field(
            name=f"🤖 {model['model']}",
            value=f"**Requests:** {model['requests']:,}\n"
                  f"**Input Tokens:** {model['input_tokens']:,}\n"
                  f"**Output Tokens:** {model['output_tokens']:,}\n"
                  f"**Total Cost:** ${model['total_cost']:.4f}\n"
                  f"**Avg Cost/Request:** ${model['avg_cost_per_request']:.4f}",
            inline=True
        )
    
    await ctx.send(embed=embed)

@bot.command(name='queries')
@commands.has_permissions(administrator=True)
async def query_analytics(ctx: Context) -> None:
    """Show user query analytics (admin only)"""
    if ctx.author.id != config.OWNER_ID:
        await ctx.send("You do not have permission to use this command.")
        return
    
    if not db_manager.pool:
        await ctx.send("Database not available. Query tracking is currently offline.")
        return
    
    analytics = await db_manager.get_query_analytics()
    if not analytics:
        await ctx.send("Failed to retrieve query analytics.")
        return
    
    embed = discord.Embed(
        title="🔍 User Query Analytics",
        description="What users are asking about (last 7 days)",
        color=0x9B59B6,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Popular queries
    if analytics['popular_queries']:
        popular_text = ""
        for query in analytics['popular_queries'][:8]:
            query_preview = query['query_text'][:50] + "..." if len(query['query_text']) > 50 else query['query_text']
            popular_text += f"**{query['command']}:** {query_preview} ({query['frequency']}x)\n"
        embed.add_field(name="🔥 Popular Queries", value=popular_text, inline=False)
    
    # Command patterns
    if analytics['command_patterns']:
        patterns_text = "\n".join([
            f"**{cmd['command']}:** {cmd['total_queries']} queries, {cmd['unique_users']} users"
            for cmd in analytics['command_patterns'][:5]
        ])
        embed.add_field(name="📊 Command Usage", value=patterns_text, inline=True)
    
    # Most active hours
    if analytics['hourly_activity']:
        active_hours = analytics['hourly_activity'][:5]
        hours_text = "\n".join([
            f"**{int(hour['hour'])}:00:** {hour['query_count']} queries"
            for hour in active_hours
        ])
        embed.add_field(name="⏰ Peak Hours", value=hours_text, inline=True)
    
    embed.set_footer(text="All user queries and commands are tracked for analytics")
    await ctx.send(embed=embed)

@bot.command(name='commands')
async def commands_help(ctx: Context) -> None:
    """Show bot help and available commands"""
    embed = discord.Embed(
        title="🤖 SecurePath AI Bot - Commands",
        description="Advanced crypto analysis powered by AI",
        color=0x004200,
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
        value="Advanced chart analysis using GPT-4.1-mini Vision\n"
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
    
    embed.set_footer(text="Powered by GPT-4.1-mini & Perplexity Sonar-Pro")
    await ctx.send(embed=embed)

@bot.command(name='ping')
async def ping(ctx: Context) -> None:
    """Check bot latency and database status"""
    start_time = time.time()
    message = await ctx.send("🏓 Pinging...")
    end_time = time.time()
    
    latency = round(bot.latency * 1000)
    response_time = round((end_time - start_time) * 1000)
    
    # Check database status
    db_status = "🟢 Connected" if db_manager.pool else "🔴 Disconnected"
    
    embed = discord.Embed(
        title="🏓 Pong!",
        color=0x00ff00,
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(name="Discord Latency", value=f"{latency}ms", inline=True)
    embed.add_field(name="Response Time", value=f"{response_time}ms", inline=True)
    embed.add_field(name="Database", value=db_status, inline=True)
    embed.add_field(name="API Calls Today", value=f"{api_call_counter}/{config.DAILY_API_CALL_LIMIT}", inline=True)
    
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