import asyncio
import json
import os
import logging
import time
from datetime import datetime, timezone
import signal
import sys
import traceback
from collections import deque, Counter, defaultdict
from typing import Optional, Dict, Any, List, Tuple

import aiohttp
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
import discord
from discord.ext import commands, tasks
from discord import Embed
from dotenv import load_dotenv

# Load configuration
from config import (
    DISCORD_TOKEN, PERPLEXITY_API_KEY, LOG_CHANNEL_ID, PERPLEXITY_API_URL,
    PERPLEXITY_TIMEOUT, MAX_RETRIES, RETRY_DELAY, MAX_CONTEXT_MESSAGES,
    MAX_CONTEXT_AGE, STATS_INTERVAL, API_RATE_LIMIT_INTERVAL, API_RATE_LIMIT_MAX,
    SYSTEM_PROMPT, LOG_FORMAT, LOG_LEVEL
)

def setup_logging() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler with RichHandler
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # Set Discord-related loggers to WARNING
    for module in ['discord', 'discord.http', 'discord.gateway']:
        logging.getLogger(module).setLevel(logging.WARNING)

    return logger

logger = setup_logging()

console = Console()
conn: Optional[TCPConnector] = None
session: Optional[ClientSession] = None
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
user_contexts: Dict[int, deque] = {}
message_counter = Counter()
command_counter = Counter()
api_rate_limits = defaultdict(lambda: 0)

def get_user_context(user_id: int) -> deque:
    return user_contexts.setdefault(user_id, deque(maxlen=MAX_CONTEXT_MESSAGES))

def update_user_context(user_id: int, message_content: str, is_bot_response: bool = False) -> None:
    context = get_user_context(user_id)
    current_time = time.time()
    
    if context and context[-1]['role'] == ("assistant" if is_bot_response else "user"):
        context[-1]['content'] += f"\n{message_content}"
        context[-1]['timestamp'] = current_time
    else:
        context.append({
            'role': "assistant" if is_bot_response else "user",
            'content': message_content,
            'timestamp': current_time,
        })
    
    context = deque([msg for msg in context if current_time - msg['timestamp'] <= MAX_CONTEXT_AGE], 
                    maxlen=MAX_CONTEXT_MESSAGES)
    user_contexts[user_id] = context

def get_context_messages(user_id: int) -> List[Dict[str, str]]:
    context = get_user_context(user_id)
    messages = []
    last_role = None
    for msg in context:
        if msg['role'] != last_role:
            messages.append({"role": msg['role'], "content": msg['content']})
            last_role = msg['role']
        else:
            messages[-1]['content'] += f"\n{msg['content']}"
    return messages

async def fetch_perplexity_response(user_id: int, new_message: str) -> Optional[Dict[str, Any]]:
    if session is None:
        logger.error("Session is not initialized")
        return None

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    dynamic_system_prompt = f"today is {current_date}. all info must be accurate up to this date. {SYSTEM_PROMPT}"
    
    context_messages = get_context_messages(user_id)
    
    messages = [{"role": "system", "content": dynamic_system_prompt}]
    messages.extend(context_messages)
    
    if not messages or messages[-1]['role'] != 'user' or messages[-1]['content'] != new_message:
        messages.append({"role": "user", "content": new_message})
    
    data = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": messages,
        "max_tokens": 1000,
        "search_recency_filter": "day"
    }
    
    logger.info(f"Sending query to Perplexity:\n{json.dumps(data, indent=2)}")
    
    start_time = time.time()
    try:
        timeout = ClientTimeout(total=PERPLEXITY_TIMEOUT)
        async with session.post(PERPLEXITY_API_URL, json=data, headers=headers, timeout=timeout) as response:
            elapsed_time = time.time() - start_time
            logger.info(f"API request completed in {elapsed_time:.2f} seconds")
            if response.status == 200:
                return await response.json()
            else:
                response_text = await response.text()
                logger.error(f"API request failed with status {response.status}. Full Response: {response_text}")
                raise Exception(f"API request failed with status {response.status}: {response_text}")
    except asyncio.TimeoutError:
        logger.error(f"Request to Perplexity API timed out after {PERPLEXITY_TIMEOUT} seconds")
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"API request failed after {elapsed_time:.2f} seconds: {str(e)}", exc_info=True)
        raise

async def send_long_message(ctx: commands.Context, message: str) -> Optional[discord.Message]:
    max_length = 4096

    if isinstance(message, dict):
        message = message.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    if not message:
        return None

    message_parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]

    last_sent_message = None
    for i, part in enumerate(message_parts):
        embed = discord.Embed(description=part, color=0x004200)
        embed.set_author(name=bot.user.name, icon_url=bot.user.avatar.url if bot.user.avatar else None)
        embed.set_footer(text=f"Part {i+1}/{len(message_parts)}" if len(message_parts) > 1 else "")
        
        try:
            sent_message = await ctx.send(embed=embed)
            if i == len(message_parts) - 1:
                last_sent_message = sent_message
        except discord.errors.HTTPException as e:
            logger.error(f"Failed to send message: {str(e)}")
            break
    
    return last_sent_message

async def send_stats() -> None:
    if not LOG_CHANNEL_ID:
        logger.debug("LOG_CHANNEL_ID is not set, skipping stats send")
        return

    try:
        log_channel_id = int(LOG_CHANNEL_ID)
        channel = bot.get_channel(log_channel_id)
        if not channel:
            logger.debug(f"Could not find log channel with ID {log_channel_id}")
            return

        embed = Embed(title="Bot Statistics", color=0xff0000)
        embed.add_field(name="Total Messages", value=sum(message_counter.values()))
        embed.add_field(name="Unique Users", value=len(message_counter))
        
        top_users = []
        for user_id, count in message_counter.most_common(5):
            user = bot.get_user(user_id)
            username = user.name if user else f"Unknown User ({user_id})"
            top_users.append(f"{username}: {count}")
        
        embed.add_field(name="Top 5 Users", value="\n".join(top_users) or "No data yet")
        embed.add_field(name="Command Usage", value="\n".join(f"{cmd}: {count}" for cmd, count in command_counter.most_common()) or "No commands used yet")
        embed.timestamp = datetime.now(timezone.utc)

        await channel.send(embed=embed)
        logger.info(f"Stats sent to log channel: {channel.name}")
    except ValueError:
        logger.debug(f"Invalid LOG_CHANNEL_ID: {LOG_CHANNEL_ID}")
    except Exception as e:
        logger.debug(f"Error in send_stats: {str(e)}")

@tasks.loop(seconds=STATS_INTERVAL)
async def send_periodic_stats() -> None:
    await send_stats()

@bot.event
async def on_ready() -> None:
    try:
        logger.info(f'{bot.user} has connected to Discord!')
        logger.info(f'Bot is active in {len(bot.guilds)} guilds')
        logger.info("Bot is ready to receive DMs")
        
        asyncio.create_task(send_initial_stats())
    except Exception as e:
        logger.error(f"Error in on_ready event: {type(e).__name__}: {str(e)}")

async def send_initial_stats() -> None:
    await asyncio.sleep(5)
    
    if LOG_CHANNEL_ID:
        try:
            log_channel_id = int(LOG_CHANNEL_ID)
            log_channel = bot.get_channel(log_channel_id)
            if log_channel:
                logger.info(f"Log channel found: {log_channel.name}")
            else:
                logger.debug(f"Could not find log channel with ID {log_channel_id}")
        except ValueError:
            logger.debug(f"Invalid LOG_CHANNEL_ID: {LOG_CHANNEL_ID}")
    else:
        logger.debug("LOG_CHANNEL_ID is not set")
    
    await send_stats()

async def process_message(message: discord.Message, question: Optional[str] = None) -> None:
    current_time = time.time()
    if api_rate_limits[message.author.id] >= API_RATE_LIMIT_MAX:
        await message.channel.send("You are sending messages too quickly. Please slow down.")
        return

    api_rate_limits[message.author.id] += 1
    asyncio.create_task(reset_api_rate_limit(message.author.id))

    if not question:
        question = message.content.strip()
        if not isinstance(message.channel, discord.DMChannel):
            question = question[6:].strip()

    logger.info(f"Processing message from {message.author} (ID: {message.author.id}): {question}")

    if len(question) < 5:
        await message.channel.send("Please provide a more detailed question (at least 5 characters).")
        return
    if len(question) > 500:
        await message.channel.send("Your question is too long. Please limit it to 500 characters.")
        return

    async with message.channel.typing():
        try:
            update_user_context(message.author.id, question)
            
            response = await fetch_perplexity_response(message.author.id, question)
            if response and 'choices' in response:
                answer = response['choices'][0]['message']['content']
                update_user_context(message.author.id, answer, is_bot_response=True)
                await send_long_message(message.channel, answer)
                
                log_channel = bot.get_channel(LOG_CHANNEL_ID)
                if log_channel:
                    await log_channel.send(f"Received message from {message.author} (ID: {message.author.id}): {question}")
                    await log_channel.send(f"Sent response to {message.author} (ID: {message.author.id}): {answer}")
            else:
                error_message = "I'm sorry, I couldn't get a response. Please try again later."
                await message.channel.send(embed=discord.Embed(description=error_message, color=0xff0000))
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            error_message = "An unexpected error occurred. Please try again later."
            await message.channel.send(embed=discord.Embed(description=error_message, color=0xff0000))

async def reset_api_rate_limit(user_id: int) -> None:
    await asyncio.sleep(API_RATE_LIMIT_INTERVAL)
    api_rate_limits[user_id] = max(0, api_rate_limits[user_id] - 1)

@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author == bot.user:
        return

    if isinstance(message.channel, discord.DMChannel):
        message_counter[message.author.id] += 1
        await process_message(message)
    elif message.content.startswith('!defi'):
        message_counter[message.author.id] += 1
        command_counter['!defi'] += 1
        await process_message(message)

@bot.command(name='defi')
@commands.cooldown(1, 10, commands.BucketType.user)
async def defi(ctx: commands.Context, *, question: Optional[str] = None) -> None:
    if not question:
        await ctx.send("Please provide a question after the !defi command. Example: `!defi What is yield farming?`")
        return

    await process_message(ctx, question=question)

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
    if isinstance(error, commands.CommandOnCooldown):
        if ctx.author.id != 804823236222779413:
            await ctx.send(f"This command is on cooldown. Please try again in {error.retry_after:.2f} seconds.")
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Command invoke error for {ctx.author} (ID: {ctx.author.id})", exc_info=error.original)
        await ctx.send("An error occurred while processing your command. Please try again.")
    else:
        logger.error(f"Unhandled error for {ctx.author} (ID: {ctx.author.id}): {type(error).__name__}: {str(error)}")
        await ctx.send("An unexpected error occurred. Please try again.")

async def force_shutdown() -> None:
    """Force shutdown of all tasks."""
    console.print(Panel.fit("Shutting down SecurePath AI Bot", border_style="red"))
    
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

async def start_bot() -> None:
    try:
        logger.info("Setting up signal handlers")
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, handle_exit)

        logger.info("Calling startup function")
        await startup()
        
        logger.info("Starting bot")
        await bot.start(DISCORD_TOKEN)
        
    except discord.errors.HTTPException as e:
        logger.error(f"HTTP Exception: {e}")
    except asyncio.CancelledError:
        logger.info("Bot startup cancelled")
    except Exception as e:
        logger.error(f"Error during bot startup: {type(e).__name__}: {str(e)}")
    finally:
        logger.info("Bot startup process completed")
        if session:
            await session.close()
        if conn:
            await conn.close()

async def startup() -> None:
    global conn, session
    console.print(Panel.fit("Starting SecurePath AI Bot", border_style="green"))
    logger.info("Bot startup initiated")
    
    conn = TCPConnector(limit=10)
    session = ClientSession(connector=conn)
    
    asyncio.create_task(delayed_start_periodic_stats())
    
    logger.info("Bot startup completed")

async def delayed_start_periodic_stats() -> None:
    await asyncio.sleep(10)
    send_periodic_stats.start()
    logger.info("Periodic stats sending started")

if __name__ == "__main__":
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Bot shutting down")
        quiet_exit()