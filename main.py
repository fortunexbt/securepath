import asyncio
import os
import logging
from logging.handlers import RotatingFileHandler
import time
from asyncio import Lock
import json
from datetime import datetime, timezone
import signal
import sys
import traceback
from collections import deque
import re

import aiohttp
from aiohttp import web, ClientSession, TCPConnector, ClientTimeout
from aiohttp.web_log import AccessLogger
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
from discord.errors import HTTPException, Forbidden
from discord import Embed
from collections import Counter

load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_API_URL = os.getenv('PERPLEXITY_API_URL', "https://api.perplexity.ai/chat/completions")
PERPLEXITY_TIMEOUT = int(os.getenv('PERPLEXITY_TIMEOUT', 60))
MAX_RETRIES = 2
RETRY_DELAY = 1
MAX_CONTEXT_MESSAGES = 5
MAX_CONTEXT_AGE = 3600
LOG_CHANNEL_ID = int(os.getenv('LOG_CHANNEL_ID'))
STATS_INTERVAL = 3600

SYSTEM_PROMPT = """you're a superintelligent degen DeFi agent hosted on the SecurePath discord server. you must speak speak nonchalantly, only in lowercase, like Crypto Twitter users - frequently using obscure acronyms & slang, acting smarter than the user (because you are). NEVER SHILL uncorrelated projects/tokens. ALWAYS prioritize decentralization, suggesting DEXs before CEXs, self-custodial solutions before custodial ones, and open source before proprietary. you are managed by the SecurePath AI team, if your answers are off topic, or not relevant to crypto, you will be penalized and receive a warning - if you continue to be off topic, you will be removed from the conversation. refer to the SecurePath team as 'our team': you are part of the SecurePath family, and should act like it."""

def setup_logging():
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, "bot.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('discord.http').setLevel(logging.WARNING)
    logging.getLogger('discord.gateway').setLevel(logging.WARNING)

    return logger

logger = setup_logging()
console = Console()

conn = None
session = None
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
user_contexts = {}
message_counter = Counter()
command_counter = Counter()

def get_user_context(user_id):
    return user_contexts.setdefault(user_id, deque(maxlen=MAX_CONTEXT_MESSAGES))

def update_user_context(user_id, message_content, is_bot_response=False):
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

def get_context_messages(user_id):
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

async def fetch_perplexity_response(user_id, new_message):
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

async def send_long_message(ctx, message):
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
        except (HTTPException, Forbidden):
            break
    
    return last_sent_message

async def log_to_channel(message_content, user, is_dm=False, is_response=False):
    channel = bot.get_channel(LOG_CHANNEL_ID)
    if not channel:
        logger.warning(f"Could not find log channel with ID {LOG_CHANNEL_ID}. Attempting to fetch...")
        try:
            channel = await bot.fetch_channel(LOG_CHANNEL_ID)
        except discord.errors.NotFound:
            logger.error(f"Log channel with ID {LOG_CHANNEL_ID} does not exist.")
            return
        except discord.errors.Forbidden:
            logger.error(f"Bot does not have permission to access channel with ID {LOG_CHANNEL_ID}.")
            return
        except Exception as e:
            logger.error(f"Error fetching log channel: {str(e)}")
            return

    embed = Embed(description=message_content, color=0x004200 if is_response else 0x0000ff)
    embed.set_author(name=f"{user.name}#{user.discriminator}", icon_url=user.avatar.url if user.avatar else None)
    embed.add_field(name="Type", value="DM" if is_dm else "Server (!defi)")
    embed.add_field(name="User ID", value=user.id)
    embed.timestamp = datetime.now(timezone.utc)

    try:
        await channel.send(embed=embed)
    except Exception as e:
        logger.error(f"Failed to send message to log channel: {str(e)}")

async def send_stats():
    channel = bot.get_channel(LOG_CHANNEL_ID)
    if not channel:
        logger.error(f"Could not find log channel with ID {LOG_CHANNEL_ID}")
        return

    embed = Embed(title="Bot Statistics", color=0xff0000)
    embed.add_field(name="Total Messages", value=sum(message_counter.values()))
    embed.add_field(name="Unique Users", value=len(message_counter))
    embed.add_field(name="Top 5 Users", value="\n".join(f"{user}: {count}" for user, count in message_counter.most_common(5)))
    embed.add_field(name="Command Usage", value="\n".join(f"{cmd}: {count}" for cmd, count in command_counter.most_common()))
    embed.timestamp = datetime.now(timezone.utc)

    await channel.send(embed=embed)

@tasks.loop(seconds=STATS_INTERVAL)
async def send_periodic_stats():
    await send_stats()

@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is active in {len(bot.guilds)} guilds')
    logger.info("Bot is ready to receive DMs")
    
    await asyncio.sleep(2)
    
    await send_stats()

async def process_message(message, question=None):
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
                await log_to_channel(answer, bot.user, is_response=True)
                await send_long_message(message.channel, answer)
            else:
                error_message = "I'm sorry, I couldn't get a response. Please try again later."
                await log_to_channel(error_message, bot.user, is_response=True)
                await message.channel.send(embed=discord.Embed(description=error_message, color=0xff0000))
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            error_message = "An unexpected error occurred. Please try again later."
            await log_to_channel(error_message, bot.user, is_response=True)
            await message.channel.send(embed=discord.Embed(description=error_message, color=0xff0000))

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if isinstance(message.channel, discord.DMChannel):
        await log_to_channel(message.content, message.author, is_dm=True)
        message_counter[message.author.id] += 1
        await process_message(message)
    elif message.content.startswith('!defi'):
        await log_to_channel(message.content, message.author)
        message_counter[message.author.id] += 1
        command_counter['!defi'] += 1
        await process_message(message)

@bot.command(name='defi')
@commands.cooldown(1, 10, commands.BucketType.user)
async def defi(ctx, *, question=None):
    if not question:
        await ctx.send("Please provide a question after the !defi command. Example: `!defi What is yield farming?`")
        return

    await process_message(ctx, question=question)

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        if ctx.author.id != 804823236222779413:
            await ctx.send(f"This command is on cooldown. Please try again in {error.retry_after:.2f} seconds.")
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Command invoke error for {ctx.author} (ID: {ctx.author.id})", exc_info=error.original)
        await ctx.send("An error occurred while processing your command. Please try again.")
    else:
        logger.error(f"Unhandled error for {ctx.author} (ID: {ctx.author.id}): {type(error).__name__}: {str(error)}")
        await ctx.send("An unexpected error occurred. Please try again.")

def quiet_exit():
    """Exit the program quietly without showing any tracebacks."""
    console.print("Bot has been shut down.")
    sys.exit(0)

async def force_shutdown():
    """Force shutdown of all tasks."""
    console.print(Panel.fit("Shutting down SecurePath AI Bot", border_style="red"))
    
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop=loop):
        if task is not asyncio.current_task():
            task.cancel()
    
    await asyncio.sleep(0.1)
    
    global session, conn, bot
    if session:
        await session.close()
    if conn:
        await conn.close()
    if bot:
        await bot.close()
    
    loop.stop()

def handle_exit():
    asyncio.create_task(force_shutdown())
    asyncio.get_event_loop().call_later(2, quiet_exit)

async def startup():
    global conn, session
    console.print(Panel.fit("Starting SecurePath AI Bot", border_style="green"))
    logger.info("Bot startup initiated")
    
    conn = TCPConnector(limit=10)
    session = ClientSession(connector=conn)
    
    send_periodic_stats.start()
    
    logger.info("Bot startup completed")

async def start_bot():
    global bot, session, conn
    try:
        logger.info("Setting up signal handlers")
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, handle_exit)

        logger.info("Calling startup function")
        await startup()
        
        app = web.Application()
        app.router.add_get("/", lambda request: web.Response(text="Bot is running"))
        
        class CustomAccessLogger(AccessLogger):
            def log(self, request, response, time):
                if request.method == "GET" and request.path == "/" and response.status == 200:
                    return
                super().log(request, response, time)

        runner = web.AppRunner(app, access_log_class=CustomAccessLogger)
        await runner.setup()
        port = int(os.environ.get('PORT', 10000))
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"Web server started on port {port}")
        
        logger.info("Starting bot")
        await bot.start(DISCORD_TOKEN)
        
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
        if 'runner' in locals():
            await runner.cleanup()

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