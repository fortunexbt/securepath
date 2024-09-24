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
import fcntl
import socket

import aiohttp
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
import discord
from discord.ext import commands, tasks
from discord import Embed
from dotenv import load_dotenv

from config import (
    DISCORD_TOKEN, PERPLEXITY_API_KEY, LOG_CHANNEL_ID, PERPLEXITY_API_URL,
    PERPLEXITY_TIMEOUT, MAX_RETRIES, RETRY_DELAY, MAX_CONTEXT_MESSAGES,
    MAX_CONTEXT_AGE, STATS_INTERVAL, API_RATE_LIMIT_INTERVAL, API_RATE_LIMIT_MAX,
    SYSTEM_PROMPT, LOG_FORMAT, LOG_LEVEL
)

def setup_logging() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

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

class RateLimiter:
    def __init__(self, max_calls, interval):
        self.max_calls = max_calls
        self.interval = interval
        self.calls = {}

    def is_rate_limited(self, user_id):
        current_time = time.time()
        if user_id not in self.calls:
            self.calls[user_id] = []
        
        self.calls[user_id] = [call_time for call_time in self.calls[user_id] if current_time - call_time <= self.interval]
        
        if len(self.calls[user_id]) >= self.max_calls:
            return True
        
        self.calls[user_id].append(current_time)
        return False

api_rate_limiter = RateLimiter(API_RATE_LIMIT_MAX, API_RATE_LIMIT_INTERVAL)

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
                return None
    except asyncio.TimeoutError:
        logger.error(f"Request to Perplexity API timed out after {PERPLEXITY_TIMEOUT} seconds")
    except Exception as e:
        logger.error(f"Error in fetch_perplexity_response: {str(e)}")
        logger.error(traceback.format_exc())
    return None

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

        async for message in channel.history(limit=1):
            if message.author == bot.user and message.embeds and message.embeds[0].title == "Bot Statistics":
                if (datetime.now(timezone.utc) - message.created_at).total_seconds() < STATS_INTERVAL / 2:
                    logger.debug("Recent stats message found, skipping")
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

def log_instance_info():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    pid = os.getpid()
    logger.info(f"Bot instance started - Hostname: {hostname}, IP: {ip_address}, PID: {pid}")

@bot.event
async def on_ready() -> None:
    try:
        logger.info(f'{bot.user} has connected to Discord!')
        logger.info(f'Bot is active in {len(bot.guilds)} guilds')
        logger.info("Bot is ready to receive DMs")
        
        log_instance_info()
        
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
    process_id = f"{message.id}_{int(time.time())}"
    logger.info(f"Starting process_message {process_id}")

    if api_rate_limiter.is_rate_limited(message.author.id):
        await message.channel.send("You are sending messages too quickly. Please slow down.")
        logger.info(f"Rate limited message {process_id}")
        return

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
                
                async for msg in message.channel.history(limit=5):
                    if msg.author == bot.user and msg.content.startswith(answer[:100]):
                        logger.info(f"Duplicate response detected for {process_id}, skipping")
                        return
                
                await send_long_message(message.channel, answer)
                
                log_channel = bot.get_channel(LOG_CHANNEL_ID)
                if log_channel:
                    # Create an embed for the incoming message
                    incoming_embed = discord.Embed(
                        title="Incoming Message",
                        description=question,
                        color=0x00ff00,  # Green color
                        timestamp=message.created_at
                    )
                    incoming_embed.set_author(name=message.author.name, icon_url=message.author.avatar.url if message.author.avatar else None)
                    incoming_embed.add_field(name="User ID", value=message.author.id, inline=True)
                    incoming_embed.add_field(name="Channel Type", value="DM" if isinstance(message.channel, discord.DMChannel) else "Server", inline=True)
                    if not isinstance(message.channel, discord.DMChannel):
                        incoming_embed.add_field(name="Server", value=message.guild.name, inline=True)
                        incoming_embed.add_field(name="Channel", value=message.channel.name, inline=True)
                    incoming_embed.add_field(name="Total Messages", value=message_counter[message.author.id], inline=True)
                    incoming_embed.add_field(name="Command Used", value="DM" if isinstance(message.channel, discord.DMChannel) else "!defi", inline=True)
                    
                    await log_channel.send(embed=incoming_embed)

                    # Create an embed for the outgoing message
                    outgoing_embed = discord.Embed(
                        title="Outgoing Message",
                        description=answer[:1024],  # Discord has a 1024 character limit for embed description
                        color=0x0000ff,  # Blue color
                        timestamp=datetime.now(timezone.utc)
                    )
                    outgoing_embed.set_author(name=bot.user.name, icon_url=bot.user.avatar.url if bot.user.avatar else None)
                    outgoing_embed.add_field(name="Recipient", value=f"{message.author.name} (ID: {message.author.id})", inline=False)
                    
                    await log_channel.send(embed=outgoing_embed)

                    # Send full response in chunks if it's longer than 2000 characters
                    if len(answer) > 2000:
                        chunks = [answer[i:i+1990] for i in range(0, len(answer), 1990)]
                        for i, chunk in enumerate(chunks):
                            await log_channel.send(f"Full response part {i+1}/{len(chunks)}:\n\n{chunk}")
                    else:
                        await log_channel.send(f"Full response to {message.author.name} (ID: {message.author.id}):\n\n{answer}")
            else:
                error_message = "I'm sorry, I couldn't get a response. Please try again later."
                await message.channel.send(embed=discord.Embed(description=error_message, color=0xff0000))
        except Exception as e:
            logger.error(f"Error in process_message {process_id}: {str(e)}")
            logger.error(traceback.format_exc())
            error_message = "An unexpected error occurred. Please try again later."
            await message.channel.send(embed=discord.Embed(description=error_message, color=0xff0000))
    
    logger.info(f"Completed process_message {process_id}")

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
    
    log_instance_info()
    
    conn = TCPConnector(limit=10)
    session = ClientSession(connector=conn)
    
    asyncio.create_task(delayed_start_periodic_stats())
    
    logger.info("Bot startup completed")

async def delayed_start_periodic_stats() -> None:
    await asyncio.sleep(10)
    send_periodic_stats.start()
    logger.info("Periodic stats sending started")

def ensure_single_instance():
    lock_file = '/tmp/securepath_bot.lock'
    fp = open(lock_file, 'w')
    try:
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Another instance of the bot is already running. Exiting.")
        sys.exit(1)
    return fp

lock_file_handle = ensure_single_instance()

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
        lock_file_handle.close()
        quiet_exit()