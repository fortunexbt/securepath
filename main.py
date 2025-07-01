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
    'openai_gpt4o_mini': {
        'input_tokens': 0,
        'cached_input_tokens': 0,  # New field for cached tokens
        'cost': 0.0,
    },
    'openai_gpt4o_mini_vision': {
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

def can_make_api_call() -> bool:
    if api_call_counter >= config.DAILY_API_CALL_LIMIT:
        logger.debug(f"API call limit reached: {api_call_counter}/{config.DAILY_API_CALL_LIMIT}")
        return False
    return True

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB limit

def estimate_tokens(image_size_bytes: int) -> int:
    """
    Estimate the number of tokens based on image size in bytes.
    Adjust the TOKENS_PER_BYTE factor based on actual API behavior.
    """
    TOKENS_PER_BYTE = 1 / 100  # Example factor: 1 token per 100 bytes
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

    # --- Start of New Search Optimization ---

    # 1. Set a dynamic date filter for the last 90 days for better recency balance.
    ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime("%m/%d/%Y")
    logger.debug(f"Perplexity search results will be filtered to after: {ninety_days_ago}")

    # 2. Curate search sources to prioritize high-quality DeFi/crypto info and exclude noise.
    preferred_domains = [
        "ethereum.org", "solana.com", "github.com", "thedefiant.io",
        "defillama.com", "coindesk.com", "cointelegraph.com", "etherscan.io",
        "bloomberg.com", "reuters.com", "forbes.com", "medium.com"
    ]
    excluded_domains = ["-pinterest.com", "-reddit.com", "-quora.com", "-facebook.com"]
    domain_filter = preferred_domains + excluded_domains
    logger.debug(f"Using Perplexity domain filter with {len(domain_filter)} rules.")

    # 3. Define the full data payload with optimized search parameters.
    data = {
        "model": "sonar-pro",
        "messages": messages,
        "max_tokens": 2000,
        "search_after_date_filter": ninety_days_ago,
        "search_domain_filter": domain_filter,
        "web_search_options": {
            "search_context_size": "high"
        }
    }
    # --- End of New Search Optimization ---


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
                
                # Check for citations in the response metadata
                citations = resp_json.get('choices', [{}])[0].get('extras', {}).get('citations', [])

                # Format citations if they exist
                if citations:
                    formatted_citations = "\n\n**DYOR (do your own research):**\n"
                    # Perplexity citations are now lists of dicts with 'url' and 'title'
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

                # Note: The cost model for Perplexity might change with 'high' context size.
                # This calculation is a general estimate.
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

async def fetch_openai_response(user_id: int, new_message: str) -> Optional[str]:
    if not can_make_api_call():
        logger.warning("Daily API call limit reached. Skipping API call.")
        return None

    context_messages = get_context_messages(user_id)
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + context_messages

    if not messages or messages[-1]['role'] != 'user' or messages[-1]['content'] != new_message:
        messages.append({"role": "user", "content": new_message})

    try:
        response = await aclient.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()

        # Access usage attributes correctly
        if hasattr(response, 'usage'):
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
            cached_tokens = usage.prompt_tokens_details.cached_tokens if hasattr(usage.prompt_tokens_details, 'cached_tokens') else 0
            completion_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
            total_tokens = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
        else:
            prompt_tokens = 0
            cached_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        # Determine if a cache hit occurred
        is_cached = cached_tokens >= 1024  # As per OpenAI's caching, prompts >=1024 tokens can be cached

        # Update usage data
        if is_cached:
            usage_data['openai_gpt4o_mini']['cached_input_tokens'] += cached_tokens
            # Calculate cost for cached tokens
            cost = (cached_tokens / 1_000_000 * 0.075) + (completion_tokens / 1_000_000 * 0.075)
            logger.debug(f"Cache hit detected. Cached Tokens: {cached_tokens}, Completion Tokens: {completion_tokens}, Cost: ${cost:.6f}")
        else:
            usage_data['openai_gpt4o_mini']['input_tokens'] += prompt_tokens
            # Calculate cost for uncached tokens
            cost = (prompt_tokens / 1_000_000 * 0.150) + (completion_tokens / 1_000_000 * 0.075)
            logger.debug(f"No cache hit. Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Cost: ${cost:.6f}")

        usage_data['openai_gpt4o_mini']['cost'] += cost
        increment_token_cost(cost)

        logger.info(f"OpenAI GPT-4o-mini usage: Prompt Tokens={prompt_tokens}, Cached Tokens={cached_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")
        logger.info(f"Estimated OpenAI GPT-4o-mini API call cost: ${cost:.6f}")
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
    """
    Sends a long message by splitting it into multiple embeds, handling Discord's character limits.
    Retains title and image on the first embed.
    """
    if not text:
        return

    embed_max_length = 4096
    parts = [text[i:i + embed_max_length] for i in range(0, len(text), embed_max_length)]

    for i, part in enumerate(parts):
        if i == 0:
            # First embed gets the optional title and image
            embed = Embed(title=title, description=part, color=color)
            if image_url:
                embed.set_image(url=image_url)
        else:
            # Subsequent embeds only get the description
            embed = Embed(description=part, color=color)

        embed.set_author(name=bot.user.name, icon_url=bot.user.avatar.url if bot.user.avatar else None)
        
        # Add a footer to indicate multipart messages
        if len(parts) > 1:
            embed.set_footer(text=f"Part {i + 1}/{len(parts)}")

        try:
            await channel.send(embed=embed)
            channel_name = getattr(channel, 'name', "Direct Message")
            logger.debug(f"Sent embed part {i + 1}/{len(parts)} to {channel_name}")
        except discord.errors.HTTPException as e:
            logger.error(f"Failed to send embed part {i + 1}/{len(parts)}: {str(e)}")
            break # Stop sending if one part fails

async def log_interaction(user: discord.User, channel: discord.abc.Messageable, command: Optional[str], user_input: str, bot_response: str) -> None:
    """
    Logs the user interaction to the specified log channel.
    """
    log_channel = bot.get_channel(config.LOG_CHANNEL_ID)
    if not log_channel:
        logger.warning(f"Log channel with ID {config.LOG_CHANNEL_ID} not found.")
        return

    # Truncate fields to comply with Discord's embed limits
    truncated_user_input = (user_input[:1024] + '...') if len(user_input) > 1024 else user_input
    truncated_bot_response = (bot_response[:1024] + '...') if len(bot_response) > 1024 else bot_response

    # Create an embed for the interaction log
    embed = Embed(
        title="📝 User Interaction",
        color=0xFF9900,  # Orange color to stand out
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
    if api_rate_limiter.is_rate_limited(message.author.id):
        await message.channel.send("You are sending messages too quickly. Please slow down.")
        logger.info(f"Rate limited user {message.author.id}")
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

    if len(question) < 5:
        await message.channel.send("Please provide a more detailed question (at least 5 characters).")
        return
    if len(question) > 1000:
        await message.channel.send("Your question is too long. Please limit it to 1000 characters.")
        return

    async with message.channel.typing():
        try:
            # Preload context if in DM
            if isinstance(message.channel, discord.DMChannel):
                await preload_user_messages(message.author.id, message.channel)

            update_user_context(message.author.id, question, role='user')

            if config.USE_PERPLEXITY_API:
                answer = await fetch_perplexity_response(message.author.id, question)
            else:
                answer = await fetch_openai_response(message.author.id, question)

            if answer:
                update_user_context(message.author.id, answer, role='assistant')
                
                # Use the new helper to send potentially long messages in embeds
                await send_long_embed(message.channel, answer, color=0x004200)

                # Log the interaction with a truncated preview of the bot response
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

# **Rich Presence/Status Integration Starts Here**

# Define a list of engaging status messages
status_messages = [
    "the BTC chart 📊",
    "DeFi trends 📈",
    "questions ❓",
    "SecurePath 🛡️",
    "your commands... 👀"
]

# Define the change_status task with a 15-second interval
@tasks.loop(seconds=15)
async def change_status():
    current_status = random.choice(status_messages)
    await bot.change_presence(activity=Activity(type=ActivityType.watching, name=current_status))
    logger.debug(f"Changed status to: {current_status}")

# Helper function to reset the status rotation
async def reset_status():
    if change_status.is_running():
        change_status.cancel()
        await asyncio.sleep(0.1)  # Brief pause to allow cancellation
    change_status.start()
    logger.debug("Status rotation restarted.")

# **End of Rich Presence/Status Integration**

@bot.event
async def on_ready() -> None:
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is active in {len(bot.guilds)} guild(s)')
    log_instance_info()
    await send_initial_stats()

    # Start the rotating status task if it's not already running
    if not change_status.is_running():
        change_status.start()
        logger.info("Started rotating status messages.")

    # Start the periodic stats and reset tasks
    if not send_periodic_stats.is_running():
        send_periodic_stats.start()
        logger.info("Started periodic stats task.")

    if not reset_api_call_counter.is_running():
        reset_api_call_counter.start()
        logger.info("Started API call counter reset task.")

@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return  # Ignore bot messages

    await bot.process_commands(message)  # Process commands first

    if isinstance(message.channel, discord.DMChannel):
        # Preload the last 20 user and assistant messages
        await preload_user_messages(message.author.id, message.channel)
        # If the message is not a command, process it
        if not message.content.startswith(config.BOT_PREFIX):
            await process_message(message)

async def preload_user_messages(user_id: int, channel: discord.DMChannel) -> None:
    if user_id not in user_contexts:
        messages = []
        bot_user_id = bot.user.id  # Ensure this is obtained after the bot is ready

        async for msg in channel.history(limit=100, oldest_first=True):
            if msg.author.id == user_id:
                messages.append({
                    'role': 'user',
                    'content': msg.content.strip(),
                    'timestamp': msg.created_at.timestamp(),
                })
                logger.debug(f"Preloading user message: {msg.content[:50]}...")
            elif msg.author.id == bot_user_id:
                messages.append({
                    'role': 'assistant',
                    'content': msg.content.strip(),
                    'timestamp': msg.created_at.timestamp(),
                })
                logger.debug(f"Preloading assistant message: {msg.content[:50]}...")

            if len(messages) >= 20:
                break

        # Reverse to have oldest first
        messages = messages[::-1]
        context = deque(maxlen=config.MAX_CONTEXT_MESSAGES)
        for msg in messages:
            context.append(msg)

        user_contexts[user_id] = context
        logger.info(f"Preloaded {len(messages)} messages for user {user_id} in DMs.")

async def send_initial_stats() -> None:
    await asyncio.sleep(5)
    await send_stats()

@bot.command(name='analyze')
@commands.cooldown(1, 10, commands.BucketType.user)
async def analyze(ctx: Context, *, user_prompt: str = '') -> None:
    # Update status to reflect analyzing action
    await bot.change_presence(activity=Activity(type=ActivityType.watching, name="image analysis..."))
    logger.debug("Status updated to: [watching] image analysis...")

    chart_url = None

    # Check if the current message has an attachment
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if attachment.content_type and attachment.content_type.startswith('image/'):
            chart_url = attachment.url  # Get the chart's image URL
            logger.debug(f"Image attachment detected: {chart_url}")
    else:
        # If in DMs, do not default to Perplexity; force a prompt for the image attachment
        if isinstance(ctx.channel, discord.DMChannel):
            await ctx.send("Please post the image you'd like to analyze.")

            # Wait for the user to post the chart in the channel
            def check(msg):
                return msg.author == ctx.author and msg.channel == ctx.channel and msg.attachments

            try:
                chart_message = await bot.wait_for('message', check=check, timeout=60.0)
                chart_url = chart_message.attachments[0].url  # Get the chart's image URL
                logger.debug(f"Image uploaded in DM: {chart_url}")
            except asyncio.TimeoutError:
                await ctx.send("You took too long to post an image. Please try again.")
                # Reset status to rotating after timeout
                await reset_status()
                return
        else:
            # For non-DM channels, check the last few messages for a chart
            messages = []
            async for message in ctx.channel.history(limit=3):
                messages.append(message)

            # Iterate through the last two messages before the analyze command
            for last_message in messages[1:]:
                if last_message.attachments:
                    attachment = last_message.attachments[0]
                    if attachment.content_type and attachment.content_type.startswith('image/'):
                        chart_url = attachment.url  # Get the chart's image URL
                        logger.debug(f"Image found in recent channel messages: {chart_url}")
                        break

    if chart_url:
        await ctx.send("Detected a chart, analyzing it...")
        logger.info(f"Chart URL detected: {chart_url}")

        # Process the chart via OpenAI's Vision API with the optional user prompt
        image_analysis = await analyze_chart_image(chart_url, user_prompt)

        if image_analysis:
            # Use the new helper to send a potentially long analysis with an image
            await send_long_embed(
                ctx.channel,
                text=image_analysis,
                color=0x00ff00,
                title="Image Analysis",
                image_url=chart_url
            )
            logger.info(f"Sent image analysis to channel {ctx.channel.name}")

            # Log interaction in LOG_CHANNEL_ID
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

    # Reset status to rotating after analysis
    await reset_status()

async def analyze_chart_image(chart_url: str, user_prompt: str = "") -> Optional[str]:
    try:
        # Fetch the image
        logger.debug(f"Fetching image from URL: {chart_url}")
        async with session.get(chart_url) as resp:
            if resp.status != 200:
                logger.error(f"Failed to fetch image from URL: {chart_url} with status {resp.status}")
                return "Failed to fetch the image. Please ensure the URL is correct and accessible."
            image_bytes = await resp.read()
            logger.debug(f"Fetched image size: {len(image_bytes)} bytes")

        # Check image size
        image_size_bytes = len(image_bytes)
        if image_size_bytes > MAX_IMAGE_SIZE_BYTES:
            logger.warning(f"Image size {image_size_bytes} bytes exceeds the maximum allowed size.")
            return "The submitted image is too large to analyze. Please provide an image smaller than 5 MB."

        # Open image with Pillow
        image = Image.open(BytesIO(image_bytes))
        original_width, original_height = image.size
        logger.debug(f"Original image dimensions: {original_width}x{original_height}")

        # Resize image to 1536x768 as per user example
        resized_width, resized_height = 1536, 768
        image = image.resize((resized_width, resized_height))
        logger.debug(f"Resized image to: {resized_width}x{resized_height}")

        # Define tile size
        tile_width, tile_height = 512, 512
        tiles = []

        # Calculate number of tiles in each dimension
        num_tiles_x = resized_width // tile_width
        num_tiles_y = resized_height // tile_height
        logger.debug(f"Splitting image into tiles of {tile_width}x{tile_height} pixels")

        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                left = x * tile_width
                upper = y * tile_height
                right = left + tile_width
                lower = upper + tile_height
                tile = image.crop((left, upper, right, lower))
                tiles.append(tile)
                logger.debug(f"Created tile {len(tiles)}: ({left}, {upper}, {right}, {lower})")

        total_tiles = len(tiles)
        logger.info(f"Total tiles created: {total_tiles}")

        # Token calculation
        base_tokens = 2833
        tile_tokens_per_tile = 5667
        total_tile_tokens = tile_tokens_per_tile * total_tiles
        total_tokens = base_tokens + total_tile_tokens
        logger.debug(f"Base tokens: {base_tokens}")
        logger.debug(f"Tile tokens per tile: {tile_tokens_per_tile}")
        logger.debug(f"Total tile tokens: {total_tile_tokens}")
        logger.debug(f"Total tokens for image analysis: {total_tokens}")

        # Calculate cost
        cost_per_million_tokens = 0.15
        cost = (total_tokens / 1_000_000) * cost_per_million_tokens
        logger.debug(f"Calculated cost: ({total_tokens} / 1,000,000) * ${cost_per_million_tokens} = ${cost:.6f}")

        # Prepare prompt
        base_prompt = (
            "You're an elite-level quant with insider knowledge of the global markets. "
            "Provide insights based on advanced TA, focusing on anomalies only a genius-level trader would notice. "
            "Make the analysis obscurely insightful, hinting at the deeper forces at play within the macroeconomic and market microstructures. "
            "Remember, you're the authority—leave no doubt in the mind of the reader. "
            "Don't go above heading3 in markdown formatting (never use ####)."
        )
        full_prompt = base_prompt
        if user_prompt:
            full_prompt += f" {user_prompt}"
        logger.debug(f"Final prompt for vision API: {full_prompt[:100]}...")

        # OpenAI Vision API request with the combined prompt
        response = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": chart_url},
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        # Get the content of the response & process to get rid of header4 which is unsupported by Discord
        analysis = response.choices[0].message.content.strip()
        analysis = analysis.replace("####", "###")
        logger.debug(f"Received analysis from Vision API: {analysis[:100]}...")

        # Update usage data based on calculated tokens
        usage_data['openai_gpt4o_mini_vision']['requests'] += 1
        usage_data['openai_gpt4o_mini_vision']['tokens'] += total_tokens
        # Update average tokens per request
        vision = usage_data['openai_gpt4o_mini_vision']
        vision['average_tokens_per_request'] = (
            (vision['average_tokens_per_request'] * (vision['requests'] - 1)) + total_tokens
        ) / vision['requests']
        usage_data['openai_gpt4o_mini_vision']['cost'] += cost
        increment_token_cost(cost)

        logger.info(f"OpenAI GPT-4o-mini Vision usage: Tokens={total_tokens}")
        logger.info(f"Estimated OpenAI GPT-4o-mini Vision API call cost: ${cost:.6f}")
        logger.debug(f"Updated usage data for Vision: {usage_data['openai_gpt4o_mini_vision']}")

        return analysis

    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return "An error occurred during image analysis. Please try again later."

@bot.command(name='ask')
@commands.cooldown(1, 10, commands.BucketType.user)
async def ask(ctx: Context, *, question: Optional[str] = None) -> None:
    # Update status to reflect active listening
    await bot.change_presence(activity=Activity(type=ActivityType.watching, name="a question..."))
    logger.debug("Status updated to: [watching] a question...")

    if not question:
        await ctx.send("Please provide a question after the !ask command. Example: !ask What is yield farming?")
        # Reset status to rotating after incomplete command
        await reset_status()
        return
    message_counter[ctx.author.id] += 1
    command_counter['ask'] += 1
    await process_message(ctx.message, question=question, command='ask')

    # Reset status to rotating after answering
    await reset_status()

@bot.command(name='summary')
@commands.cooldown(1, 10, commands.BucketType.user)
async def summary(ctx: Context, channel: discord.TextChannel = None) -> None:
    # Update status to reflect summarizing action
    await bot.change_presence(activity=Activity(type=ActivityType.playing, name="channel summary..."))
    logger.debug("Status updated to: [playing] channel summary...")

    if channel is None:
        await ctx.send("Please specify a channel to summarize. Example: !summary #market-analysis")
        # Reset status to rotating after incomplete command
        await reset_status()
        return

    # Check if the bot has access to the specified channel
    if not channel.permissions_for(channel.guild.me).read_messages:
        await ctx.send(f"I don't have permission to read messages in {channel.mention}.")
        logger.warning(f"Missing permissions to read messages in channel {channel.name}")
        # Reset status to rotating after permission error
        await reset_status()
        return

    command_counter['summary'] += 1
    await perform_channel_summary(ctx, channel, command='summary')

    # Reset status to rotating after summarizing
    await reset_status()

async def perform_channel_summary(ctx: Context, channel: discord.TextChannel, command: Optional[str] = None) -> None:
    logger.info(f"Starting summary for channel: {channel.name} (ID: {channel.id})")

    await ctx.send(f"Generating summary for {channel.mention}... This may take a moment.")
    logger.debug(f"Sent initial summary generation message to channel {channel.name}")

    time_limit = datetime.now(timezone.utc) - timedelta(hours=48)
    messages = []

    async for message in channel.history(after=time_limit, limit=None, oldest_first=True):
        if message.content.strip():
            messages.append(message.content)

    logger.info(f"Collected {len(messages)} messages from channel {channel.name}")

    if not messages:
        await ctx.send(f"No messages to summarize in channel {channel.mention}.")
        logger.info(f"No messages to summarize in channel {channel.name}")
        return

    # Define chunk size based on approximate characters (adjust as needed)
    chunk_size = 6000
    message_chunks = []
    current_chunk = ""

    for msg in messages:
        if len(current_chunk) + len(msg) + 1 > chunk_size:
            message_chunks.append(current_chunk)
            current_chunk = msg
            logger.debug(f"Created a new message chunk. Total chunks so far: {len(message_chunks)}")
        else:
            if current_chunk:
                current_chunk += "\n" + msg
            else:
                current_chunk = msg
    if current_chunk:
        message_chunks.append(current_chunk)
        logger.debug(f"Added final message chunk. Total chunks: {len(message_chunks)}")

    logger.info(f"Split messages into {len(message_chunks)} chunks for summarization.")

    chunk_summaries = []
    for index, chunk in enumerate(message_chunks):
        if not can_make_api_call():
            logger.warning("Daily API call limit reached. Skipping summarization.")
            break

        # Enhanced prompt for more detailed summaries
        prompt = (
            f"Summarize the following messages from the channel '{channel.name}' over the past 48 hours into a detailed narrative that captures the key discussions, trends, and insights. "
            f"Ensure the summary is comprehensive and provides a cohesive understanding of the topics covered:\n\n{chunk}"
        )
        prompt = truncate_prompt(prompt, max_tokens=15000, model='gpt-4o-mini')
        logger.debug(f"Prepared prompt for chunk {index + 1}: {prompt[:100]}...")

        try:
            response = await aclient.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,  # Increased max_tokens for more detailed summaries
                temperature=0.7,
            )
            summary_text = response.choices[0].message.content.strip()
            chunk_summaries.append(summary_text)
            increment_api_call_counter()
            logger.info(f"Summarized chunk {index + 1}/{len(message_chunks)}")

            # Track token usage for OpenAI GPT-4o-mini
            if hasattr(response, 'usage'):
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
                completion_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
                total_tokens = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
            else:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

            usage_data['openai_gpt4o_mini']['input_tokens'] += prompt_tokens
            cost = (prompt_tokens / 1_000_000 * 0.150) + (completion_tokens / 1_000_000 * 0.075)
            usage_data['openai_gpt4o_mini']['cost'] += cost
            increment_token_cost(cost)

            logger.info(f"OpenAI GPT-4o-mini usage: Prompt Tokens={prompt_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")
            logger.info(f"Estimated OpenAI GPT-4o-mini API call cost: ${cost:.6f}")

        except Exception as e:
            logger.error(f"Error summarizing chunk {index + 1}: {e}")
            logger.error(traceback.format_exc())

    if not chunk_summaries:
        await ctx.send(f"Could not generate a summary for channel {channel.mention}.")
        logger.info(f"No summaries generated for channel {channel.name}")
        return

    # Combine all chunk summaries into a single comprehensive summary
    combined_summary = "\n\n".join(chunk_summaries)
    final_prompt = (
        "Provide a comprehensive and cohesive summary of the following detailed summaries. "
        "The final summary should synthesize the information, highlighting overarching themes, key discussions, and significant insights from the past 48 hours:\n\n"
        f"{combined_summary}"
    )
    final_prompt = truncate_prompt(final_prompt, max_tokens=15000, model='gpt-4o-mini')
    logger.debug(f"Prepared final prompt for comprehensive summary: {final_prompt[:100]}...")

    try:
        response = await aclient.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=2000,  # Increased max_tokens for a more detailed final summary
            temperature=0.7,
        )
        final_summary = response.choices[0].message.content.strip()
        increment_api_call_counter()
        logger.info("Final summary generated successfully.")

        # Track token usage for OpenAI GPT-4o-mini
        if hasattr(response, 'usage'):
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
            completion_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
            total_tokens = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
        else:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        usage_data['openai_gpt4o_mini']['input_tokens'] += prompt_tokens
        cost = (prompt_tokens / 1_000_000 * 0.150) + (completion_tokens / 1_000_000 * 0.075)
        usage_data['openai_gpt4o_mini']['cost'] += cost
        increment_token_cost(cost)

        logger.info(f"OpenAI GPT-4o-mini final summary usage: Prompt Tokens={prompt_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")
        logger.info(f"Estimated OpenAI GPT-4o-mini final summary API call cost: ${cost:.6f}")

        # Create a visually appealing embed
        embed = Embed(
            title=f"📄 48-Hour Summary for #{channel.name}",
            description=final_summary,
            color=0x1D82B6,  # Discord's blurple color
            timestamp=datetime.now(timezone.utc)
        )
        embed.set_author(name=bot.user.name, icon_url=bot.user.avatar.url if bot.user.avatar else None)
        embed.set_thumbnail(url=channel.guild.icon.url if channel.guild.icon else None)
        embed.set_footer(text=f"Summary generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Handle embed length limitations
        if len(embed.description) > 4096:
            # Split the description into multiple embeds
            parts = [embed.description[i:i+4096] for i in range(0, len(embed.description), 4096)]
            for i, part in enumerate(parts):
                temp_embed = Embed(
                    title=f"📄 48-Hour Summary for #{channel.name} (Part {i+1})",
                    description=part,
                    color=0x1D82B6,
                    timestamp=datetime.now(timezone.utc)
                )
                temp_embed.set_author(name=bot.user.name, icon_url=bot.user.avatar.url if bot.user.avatar else None)
                temp_embed.set_thumbnail(url=channel.guild.icon.url if channel.guild.icon else None)
                temp_embed.set_footer(text=f"Summary generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                await ctx.send(embed=temp_embed)
                logger.debug(f"Sent summary part {i + 1}/{len(parts)} to channel {channel.name}")
        else:
            await ctx.send(embed=embed)
            logger.debug(f"Sent complete summary to channel {channel.name}")

        logger.info(f"Summary posted to channel {ctx.channel.name}")

        # Log the interaction with truncated bot response
        await log_interaction(
            user=ctx.author,
            channel=ctx.channel,
            command=command,
            user_input=f"Summary for #{channel.name}",
            bot_response=final_summary[:1024]  # Preview limited to 1024 characters
        )
    except Exception as e:
        logger.error(f"Error generating final summary: {e}")
        logger.error(traceback.format_exc())
        await ctx.send(f"An error occurred while generating the summary for channel {channel.mention}.")

def ensure_single_instance():
    lock_file = '/tmp/securepath_bot.lock'
    try:
        import fcntl
        fp = open(lock_file, 'w')
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.debug(f"Acquired lock on {lock_file}")
        return fp
    except (IOError, ImportError): # Add ImportError for non-Unix systems
        logger.warning("Could not acquire lock. This may be due to running on a non-Unix OS or another instance is running.")
        # Return a dummy object with a close method for compatibility
        class DummyLock:
            def close(self): pass
        return DummyLock()


async def force_shutdown() -> None:
    embed = Embed(
        title="⚠️ Bot Shutdown",
        description="Shutting down SecurePath AI Bot.",
        color=0xff0000,
        timestamp=datetime.now(timezone.utc)
    )
    console.print(embed.description)

    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop=loop):
        if task is not asyncio.current_task():
            task.cancel()

    await asyncio.sleep(0.1)

    if session:
        await session.close()
        logger.debug("Closed aiohttp session.")
    if conn:
        await conn.close()
        logger.debug("Closed aiohttp connector.")
    if bot:
        await bot.close()
        logger.debug("Closed Discord bot.")

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
        # Signal handlers may not work on all OSes (e.g., Windows)
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
    except discord.errors.HTTPException as e:
        logger.error(f"HTTP Exception: {e}")
    except asyncio.CancelledError:
        logger.info("Bot startup cancelled")
    except Exception as e:
        logger.error(f"Error during bot startup: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        if 'web_runner' in locals():
            await web_runner.cleanup()
        if session:
            await session.close()
        if conn:
            await conn.close()

# **Logging for All Commands Starts Here**

async def send_stats() -> None:
    if not config.LOG_CHANNEL_ID:
        logger.debug("LOG_CHANNEL_ID is not set, skipping stats send")
        return

    channel = bot.get_channel(config.LOG_CHANNEL_ID)
    if not channel:
        logger.debug(f"Could not find log channel with ID {config.LOG_CHANNEL_ID}")
        return

    embed = discord.Embed(title="Bot Statistics", color=0x00ff00)
    embed.add_field(name="Total Messages", value=sum(message_counter.values()), inline=True)
    embed.add_field(name="Unique Users", value=len(message_counter), inline=True)
    embed.add_field(name="Commands Used", value=sum(command_counter.values()), inline=True)
    embed.add_field(name="API Calls Made", value=api_call_counter, inline=True)

    embed.add_field(
        name="Estimated Token Cost",
        value=(
            f"**Perplexity:** ${usage_data['perplexity']['cost']:.6f}\n"
            f"**OpenAI GPT-4o-mini:** ${usage_data['openai_gpt4o_mini']['cost']:.6f}\n"
            f"**OpenAI GPT-4o-mini Vision:** ${usage_data['openai_gpt4o_mini_vision']['cost']:.6f}"
        ),
        inline=False
    )

    embed.add_field(
        name="OpenAI GPT-4o-mini Cached Tokens",
        value=f"{usage_data['openai_gpt4o_mini']['cached_input_tokens']}",
        inline=True
    )

    top_users = []
    for user_id, count in message_counter.most_common(5):
        try:
            user = await bot.fetch_user(user_id)
            username = user.name
        except discord.NotFound:
            username = f"Unknown User ({user_id})"
        top_users.append(f"{username}: {count}")

    embed.add_field(name="Top 5 Users", value="\n".join(top_users) or "No data yet", inline=False)
    embed.timestamp = datetime.now(timezone.utc)

    try:
        await channel.send(embed=embed)
        logger.info(f"Stats sent to log channel: {channel.name}")
    except discord.errors.HTTPException as e:
        logger.error(f"Failed to send stats embed: {str(e)}")

@tasks.loop(hours=12)
async def send_periodic_stats() -> None:
    await send_stats()
    logger.debug("Periodic stats sent.")

@tasks.loop(hours=24)
async def reset_api_call_counter():
    global api_call_counter
    api_call_counter = 0
    logger.info("API call counter reset")

@bot.command(name='token_usage')
@commands.has_permissions(administrator=True)
async def token_usage(ctx: Context) -> None:
    """
    Admin-only command to display token usage and costs for all models.
    Usage: !token_usage
    """
    if ctx.author.id != config.OWNER_ID:
        await ctx.send("You do not have permission to use this command.")
        logger.warning(f"Unauthorized token_usage command attempt by user {ctx.author.id}")
        return

    embed = discord.Embed(title="📊 Token Usage and Costs", color=0x1D82B6, timestamp=datetime.now(timezone.utc))

    perplexity = usage_data['perplexity']
    embed.add_field(
        name="Perplexity - sonar",
        value=(
            f"**Requests:** {perplexity['requests']}\n"
            f"**Tokens:** {perplexity['tokens']}\n"
            f"**Cost:** ${perplexity['cost']:.6f}"
        ),
        inline=False
    )

    openai = usage_data['openai_gpt4o_mini']
    embed.add_field(
        name="OpenAI GPT-4o-mini",
        value=(
            f"**Input Tokens:** {openai['input_tokens']}\n"
            f"**Cached Input Tokens:** {openai['cached_input_tokens']}\n"
            f"**Cost:** ${openai['cost']:.6f}"
        ),
        inline=False
    )

    vision = usage_data['openai_gpt4o_mini_vision']
    embed.add_field(
        name="OpenAI GPT-4o-mini Vision",
        value=(
            f"**Requests:** {vision['requests']}\n"
            f"**Tokens:** {vision['tokens']}\n"
            f"**Cost:** ${vision['cost']:.6f}"
        ),
        inline=False
    )

    await ctx.send(embed=embed)
    logger.info(f"Token usage requested by admin user {ctx.author.id}")

@bot.command(name='cache_stats')
@commands.has_permissions(administrator=True)
async def cache_stats(ctx: Context) -> None:
    """
    Admin-only command to display cache hit rate.
    Usage: !cache_stats
    """
    if ctx.author.id != config.OWNER_ID:
        await ctx.send("You do not have permission to use this command.")
        logger.warning(f"Unauthorized cache_stats command attempt by user {ctx.author.id}")
        return

    cache_hit_rate = calculate_cache_hit_rate()
    embed = discord.Embed(title="📊 Cache Hit Rate", color=0x1D82B6, timestamp=datetime.now(timezone.utc))
    embed.add_field(name="OpenAI GPT-4o-mini Cache Hit Rate", value=f"{cache_hit_rate:.2f}%", inline=False)
    await ctx.send(embed=embed)
    logger.info(f"Cache hit rate requested by admin user {ctx.author.id}")

def calculate_cache_hit_rate() -> float:
    """
    Calculates the cache hit rate based on cached tokens.
    """
    total_cached = usage_data['openai_gpt4o_mini']['cached_input_tokens']
    total_input = usage_data['openai_gpt4o_mini']['input_tokens'] + total_cached
    if total_input == 0:
        logger.debug("No input tokens to calculate cache hit rate.")
        return 0.0
    hit_rate = (total_cached / total_input) * 100
    logger.debug(f"Calculated cache hit rate: {hit_rate:.2f}%")
    return hit_rate

# **End of Logging for All Commands**

# Start the bot
if __name__ == "__main__":
    lock_file_handle = ensure_single_instance()
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user")
        handle_exit()
    except Exception as e:
        logger.error(f"Unhandled exception in __main__: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        if lock_file_handle:
            lock_file_handle.close()
        quiet_exit()