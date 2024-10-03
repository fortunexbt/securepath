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
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from discord import Embed
from discord.ext import commands, tasks
from discord.ext.commands import Bot, Context
from openai import AsyncOpenAI
from rich.console import Console
from rich.logging import RichHandler
from tiktoken import encoding_for_model

# Local imports
import config

# Initialize logging
logger = logging.getLogger('SecurePathBot')
console = Console()

def setup_logging() -> logging.Logger:
    logger.setLevel(getattr(logging, config.LOG_LEVEL, 'INFO'))

    console_handler = RichHandler(rich_tracebacks=True, console=console)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL, 'INFO'))
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    logger.addHandler(console_handler)

    # Suppress verbose logging from discord library
    for module in ['discord', 'discord.http', 'discord.gateway']:
        logging.getLogger(module).setLevel(logging.WARNING)

    return logger

logger = setup_logging()

# Initialize AsyncOpenAI client
aclient = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = Bot(command_prefix=config.BOT_PREFIX, intents=intents)

# Global variables
conn: Optional[TCPConnector] = None
session: Optional[ClientSession] = None
user_contexts: Dict[int, Deque[Dict[str, Any]]] = {}
message_counter = Counter()
command_counter = Counter()
api_call_counter = 0  # Initialize global API call counter

# Rate Limiter Class
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
            return True

        self.calls[user_id].append(current_time)
        return False

api_rate_limiter = RateLimiter(config.API_RATE_LIMIT_MAX, config.API_RATE_LIMIT_INTERVAL)

# Context Management Functions
def get_user_context(user_id: int) -> Deque[Dict[str, Any]]:
    return user_contexts.setdefault(user_id, deque(maxlen=config.MAX_CONTEXT_MESSAGES))

def update_user_context(user_id: int, message_content: str, role: str) -> None:
    context = get_user_context(user_id)
    current_time = time.time()
    context.append({
        'role': role,
        'content': message_content,
        'timestamp': current_time,
    })

    # Remove old messages beyond MAX_CONTEXT_AGE
    cutoff_time = current_time - config.MAX_CONTEXT_AGE
    user_contexts[user_id] = deque(
        [msg for msg in context if msg['timestamp'] >= cutoff_time],
        maxlen=config.MAX_CONTEXT_MESSAGES
    )

def get_context_messages(user_id: int) -> List[Dict[str, str]]:
    context = get_user_context(user_id)
    return [{"role": msg['role'], "content": msg['content']} for msg in context]

def truncate_prompt(prompt: str, max_tokens: int, model: str = 'gpt-4') -> str:
    encoding = encoding_for_model(model)
    tokens = encoding.encode(prompt)
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    return encoding.decode(tokens)

def log_instance_info() -> None:
    hostname = os.uname().nodename
    pid = os.getpid()
    logger.info(f"Bot instance started - Hostname: {hostname}, PID: {pid}")

def increment_api_call_counter():
    global api_call_counter
    api_call_counter += 1
    logger.info(f"API call counter: {api_call_counter}")

def can_make_api_call() -> bool:
    return api_call_counter < config.DAILY_API_CALL_LIMIT

# Fetch Response from Perplexity API
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
    messages = [{"role": "system", "content": dynamic_system_prompt}] + context_messages

    if not messages or messages[-1]['role'] != 'user' or messages[-1]['content'] != new_message:
        messages.append({"role": "user", "content": new_message})

    data = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": messages,
        "max_tokens": 1000,
        "search_recency_filter": "day"
    }

    logger.info(f"Sending query to Perplexity API for user {user_id}")

    increment_api_call_counter()

    start_time = time.time()
    try:
        timeout = ClientTimeout(total=config.PERPLEXITY_TIMEOUT)
        async with session.post(config.PERPLEXITY_API_URL, json=data, headers=headers, timeout=timeout) as response:
            elapsed_time = time.time() - start_time
            logger.info(f"API request completed in {elapsed_time:.2f} seconds")
            if response.status == 200:
                resp_json = await response.json()
                answer = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                # Log token usage if available
                usage = resp_json.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                logger.info(f"Perplexity API usage: Prompt Tokens={prompt_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")
                # Estimate cost if applicable
                cost = (prompt_tokens * 0.00015 + completion_tokens * 0.0006) / 1000
                logger.info(f"Estimated Perplexity API call cost: ${cost:.6f}")
                return answer
            else:
                response_text = await response.text()
                logger.error(f"API request failed with status {response.status}. Response: {response_text}")
    except asyncio.TimeoutError:
        logger.error(f"Request to Perplexity API timed out after {config.PERPLEXITY_TIMEOUT} seconds")
    except Exception as e:
        logger.error(f"Error in fetch_perplexity_response: {str(e)}")
        logger.error(traceback.format_exc())
    return None

# Fetch Response from OpenAI API (if needed)
async def fetch_openai_response(user_id: int, new_message: str) -> Optional[str]:
    if not can_make_api_call():
        logger.warning("Daily API call limit reached. Skipping API call.")
        return None

    context_messages = get_context_messages(user_id)
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + context_messages

    if not messages or messages[-1]['role'] != 'user' or messages[-1]['content'] != new_message:
        messages.append({"role": "user", "content": new_message})

    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    try:
        response = await aclient.chat.completions.create(
            model='gpt-4',
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        logger.error(f"Error fetching response from OpenAI: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Send Long Messages (handling Discord's character limit with Rich Embeds)
async def send_long_message(channel: discord.abc.Messageable, message: str) -> None:
    max_length = 2000  # Discord's character limit per message
    embed_max_length = 4096  # Discord's Embed description limit

    if not message:
        return

    # Split message into chunks that fit within Discord's embed description limit
    message_parts = [message[i:i + embed_max_length] for i in range(0, len(message), embed_max_length)]

    for i, part in enumerate(message_parts):
        embed = Embed(description=part, color=0x004200)
        embed.set_author(name=bot.user.name, icon_url=bot.user.avatar.url if bot.user.avatar else None)
        if len(message_parts) > 1:
            embed.set_footer(text=f"Part {i + 1}/{len(message_parts)}")

        try:
            await channel.send(embed=embed)
        except discord.errors.HTTPException as e:
            logger.error(f"Failed to send message part {i + 1}/{len(message_parts)}: {str(e)}")
            break

# Process Incoming Messages
async def process_message(message: discord.Message, question: Optional[str] = None) -> None:
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
            update_user_context(message.author.id, question, role='user')

            # Choose which API to use based on configuration
            if config.USE_PERPLEXITY_API:
                answer = await fetch_perplexity_response(message.author.id, question)
            else:
                answer = await fetch_openai_response(message.author.id, question)

            if answer:
                update_user_context(message.author.id, answer, role='assistant')
                await send_long_message(message.channel, answer)
                logger.info(f"Sent response to user {message.author.id}")
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

# Bot Events and Commands
@bot.event
async def on_ready() -> None:
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is active in {len(bot.guilds)} guild(s)')
    log_instance_info()
    await send_initial_stats()

async def send_initial_stats() -> None:
    await asyncio.sleep(5)
    await send_stats()

@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author == bot.user:
        return

    if isinstance(message.channel, discord.DMChannel):
        message_counter[message.author.id] += 1
        await process_message(message)
    elif message.content.startswith(config.BOT_PREFIX):
        message_counter[message.author.id] += 1
        command_counter[message.content.split()[0]] += 1
        await bot.process_commands(message)  # Ensure commands are processed

@bot.command(name='defi')
@commands.cooldown(1, 10, commands.BucketType.user)
async def defi(ctx: Context, *, question: Optional[str] = None) -> None:
    if not question:
        await ctx.send("Please provide a question after the !defi command. Example: `!defi What is yield farming?`")
        return

    message_counter[ctx.author.id] += 1
    command_counter['defi'] += 1
    await process_message(ctx.message, question=question)

@bot.event
async def on_command_error(ctx: Context, error: commands.CommandError) -> None:
    if isinstance(error, commands.CommandOnCooldown):
        if ctx.author.id != config.OWNER_ID:
            await ctx.send(f"This command is on cooldown. Please try again in {error.retry_after:.2f} seconds.")
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Command invoke error: {str(error.original)}")
        await ctx.send("An error occurred while processing your command. Please try again.")
    else:
        logger.error(f"Unhandled error: {str(error)}")
        await ctx.send("An unexpected error occurred. Please try again.")

# Statistics and Summaries
async def send_stats() -> None:
    if not config.LOG_CHANNEL_ID:
        logger.debug("LOG_CHANNEL_ID is not set, skipping stats send")
        return

    channel = bot.get_channel(config.LOG_CHANNEL_ID)
    if not channel:
        logger.debug(f"Could not find log channel with ID {config.LOG_CHANNEL_ID}")
        return

    embed = Embed(title="Bot Statistics", color=0x00ff00)
    embed.add_field(name="Total Messages", value=sum(message_counter.values()), inline=True)
    embed.add_field(name="Unique Users", value=len(message_counter), inline=True)
    embed.add_field(name="Commands Used", value=sum(command_counter.values()), inline=True)
    embed.add_field(name="API Calls Made", value=api_call_counter, inline=True)

    top_users = []
    for user_id, count in message_counter.most_common(5):
        user = bot.get_user(user_id)
        username = user.name if user else f"Unknown User ({user_id})"
        top_users.append(f"{username}: {count}")

    embed.add_field(name="Top 5 Users", value="\n".join(top_users) or "No data yet", inline=False)
    embed.timestamp = datetime.now(timezone.utc)

    await channel.send(embed=embed)
    logger.info(f"Stats sent to log channel: {channel.name}")

@tasks.loop(hours=24)
async def send_periodic_stats() -> None:
    await send_stats()

@tasks.loop(hours=24)
async def reset_api_call_counter():
    global api_call_counter
    api_call_counter = 0
    logger.info("API call counter reset")

async def perform_daily_summary():
    logger.info("Starting daily summary generation")
    try:
        await asyncio.wait_for(_perform_daily_summary(), timeout=3600)  # Timeout after 1 hour
    except asyncio.TimeoutError:
        logger.error("Daily summary generation timed out")
    except Exception as e:
        logger.error(f"Unexpected error during daily summary generation: {e}")
        logger.exception(e)
    else:
        logger.info("Completed daily summary generation")

async def _perform_daily_summary():
    if not can_make_api_call():
        logger.warning("Daily API call limit reached. Skipping daily summary.")
        return

    if not config.SUMMARY_CHANNEL_ID:
        logger.error("SUMMARY_CHANNEL_ID is not set. Cannot post the daily summary.")
        return

    summary_channel = bot.get_channel(config.SUMMARY_CHANNEL_ID)
    if not summary_channel:
        logger.error(f"Could not find the summary channel with ID {config.SUMMARY_CHANNEL_ID}")
        return

    if not config.NEWS_CHANNEL_ID:
        logger.error("NEWS_CHANNEL_ID is not set. Cannot generate the summary.")
        return

    news_channel = bot.get_channel(config.NEWS_CHANNEL_ID)
    if not news_channel:
        logger.error(f"Could not find the news channel with ID {config.NEWS_CHANNEL_ID}")
        return

    time_limit = datetime.now(timezone.utc) - timedelta(hours=24)
    messages = []

    permissions = news_channel.permissions_for(news_channel.guild.me)
    if not permissions.read_messages or not permissions.read_message_history:
        logger.warning(f"Missing permissions for channel: {news_channel.name}")
        return

    async for message in news_channel.history(after=time_limit, limit=config.MAX_MESSAGES_PER_CHANNEL, oldest_first=True):
        if (message.author.bot and message.author.id != config.NEWS_BOT_USER_ID) or not message.content.strip():
            continue
        messages.append(f"{message.author.display_name}: {message.content}")

    logger.info(f"Collected {len(messages)} messages from channel {news_channel.name}")

    if not messages:
        logger.info(f"No messages to summarize in channel {news_channel.name}")
        return

    # Handle large amount of data by chunking
    chunk_size = 5000  # Adjust based on tokens per message
    message_chunks = [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]
    logger.info(f"Splitting messages into {len(message_chunks)} chunks for channel {news_channel.name}")
    chunk_summaries = []

    for index, chunk in enumerate(message_chunks):
        logger.info(f"Processing chunk {index + 1}/{len(message_chunks)} for channel {news_channel.name}")
        chunk_prompt = (
            f"Summarize the following messages from the Discord channel '{news_channel.name}' "
            f"in a clear and concise manner, using bullet points where appropriate:\n\n"
            + "\n".join(chunk)
        )
        chunk_prompt = truncate_prompt(chunk_prompt, max_tokens=4096, model='gpt-4')

        if not can_make_api_call():
            logger.warning("Daily API call limit reached. Skipping API call for chunk summary.")
            break

        increment_api_call_counter()

        try:
            # Start timing the API call
            api_start_time = time.time()

            response = await aclient.chat.completions.create(
                model='gpt-4',
                messages=[{"role": "user", "content": chunk_prompt}],
                max_tokens=500,
                temperature=0.7,
                timeout=config.PERPLEXITY_TIMEOUT
            )

            # Calculate API call duration
            api_duration = time.time() - api_start_time
            logger.info(f"OpenAI API call completed in {api_duration:.2f} seconds")

            summary = response.choices[0].message.content.strip()
            chunk_summaries.append(summary)

            # Log token usage and estimated cost
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            logger.info(f"OpenAI API usage: Prompt Tokens={prompt_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")

            # Estimate cost based on your pricing
            cost = (total_tokens * 0.00015)  # Adjust based on your actual pricing
            logger.info(f"Estimated API call cost: ${cost:.6f}")

        except Exception as e:
            logger.error(f"Error summarizing chunk in channel {news_channel.name}: {e}")
            logger.exception(e)

    # Combine chunk summaries
    if not chunk_summaries:
        logger.info("No chunk summaries generated.")
        return

    combined_channel_summary = "\n".join(chunk_summaries)

    combined_prompt = (
        "Provide an overall summary of the following messages from the news channel, highlighting key points:\n\n"
        + combined_channel_summary
    )

    combined_prompt = truncate_prompt(combined_prompt, max_tokens=4096, model='gpt-4')

    if not can_make_api_call():
        logger.warning("Daily API call limit reached. Skipping API call for final summary.")
        return

    increment_api_call_counter()

    try:
        api_start_time = time.time()

        response = await aclient.chat.completions.create(
            model='gpt-4',
            messages=[{"role": "user", "content": combined_prompt}],
            max_tokens=1000,
            temperature=0.7,
            timeout=config.PERPLEXITY_TIMEOUT
        )

        api_duration = time.time() - api_start_time
        logger.info(f"Final summary OpenAI API call completed in {api_duration:.2f} seconds")

        final_summary = response.choices[0].message.content.strip()

        # Log token usage and estimated cost
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        logger.info(f"Final summary OpenAI API usage: Prompt Tokens={prompt_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")

        # Estimate cost based on your pricing
        cost = (total_tokens * 0.00015)  # Adjust based on your actual pricing
        logger.info(f"Estimated final summary API call cost: ${cost:.6f}")

        # Send the summary with rich formatting
        embed = Embed(
            title="News Channel Summary",
            description=final_summary,
            color=0x00ff00,
            timestamp=datetime.now(timezone.utc)
        )
        embed.set_author(name=bot.user.name, icon_url=bot.user.avatar.url if bot.user.avatar else None)
        await summary_channel.send(embed=embed)
        logger.info("Summary posted successfully.")

    except Exception as e:
        logger.error(f"Error generating final summary: {e}")
        logger.exception(e)

# Command to trigger the summary
@bot.command(name='summary')
@commands.cooldown(1, 10, commands.BucketType.user)
async def summary(ctx: Context) -> None:
    await perform_daily_summary()

# Ensure Single Instance
def ensure_single_instance():
    lock_file = '/tmp/securepath_bot.lock'
    try:
        import fcntl
        fp = open(lock_file, 'w')
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fp
    except IOError:
        print("Another instance of the bot is already running. Exiting.")
        sys.exit(1)

# Graceful Shutdown
async def force_shutdown() -> None:
    """Force shutdown of all tasks."""
    embed = Embed(
        title="Bot Shutdown",
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
    if conn:
        await conn.close()
    if bot:
        await bot.close()

    loop.stop()

def handle_exit() -> None:
    asyncio.create_task(force_shutdown())
    asyncio.get_event_loop().call_later(2, quiet_exit)

def quiet_exit() -> None:
    """Exit the program quietly without showing any tracebacks."""
    console.print("Bot has been shut down.")
    logging.shutdown()
    sys.exit(0)

# Start the Bot
async def start_bot() -> None:
    global conn, session
    import signal  # Fixing diagnostics by importing signal module
    try:
        logger.info("Setting up signal handlers")
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, handle_exit)

        conn = TCPConnector(limit=10)
        session = ClientSession(connector=conn)

        send_periodic_stats.start()
        reset_api_call_counter.start()

        if not config.DISCORD_TOKEN:
            logger.error("DISCORD_TOKEN is not set. Cannot start the bot.")
            return

        await bot.start(config.DISCORD_TOKEN)
    except discord.errors.HTTPException as e:
        logger.error(f"HTTP Exception: {e}")
    except asyncio.CancelledError:
        logger.info("Bot startup cancelled")
    except Exception as e:
        logger.error(f"Error during bot startup: {type(e).__name__}: {str(e)}")
    finally:
        if session:
            await session.close()
        if conn:
            await conn.close()

if __name__ == "__main__":
    lock_file_handle = ensure_single_instance()
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        lock_file_handle.close()
        quiet_exit()
