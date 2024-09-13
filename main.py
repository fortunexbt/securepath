import os
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import time
from functools import lru_cache
import base64
from asyncio import Lock

import aiohttp
from aiohttp import web
import colorlog
import discord
from discord.ext import commands
from diskcache import Cache
from dotenv import load_dotenv
from aiohttp_session import setup as setup_session
from aiohttp_session.cookie_storage import EncryptedCookieStorage
from cryptography import fernet
from discord.errors import HTTPException, Forbidden

# Constants and Environment Setup
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_URL = os.getenv('PERPLEXITY_URL', "https://api.perplexity.ai/chat/completions")
PERPLEXITY_TIMEOUT = int(os.getenv('PERPLEXITY_TIMEOUT', 60))
MAX_RETRIES = 3
RETRY_DELAY = 2

SYSTEM_PROMPT = """You're a DeFi & crypto expert in a high-stakes world of smart contracts, liquidity pools, and yield farming. Keep it sharp, terse, and precise. No time for fluff. Your audience knows their APY from their APR, so dive straight into the actionable details. If something's trash or a project's circling the drain, call it out—don't sugarcoat it. Be sharp and use crypto jargon liberally: LP, DEX, yield farming, etc. Avoid explaining terms unless absolutely necessary. Maintain a laid-back yet expert tone—throw in some late millennial or zoomer slang for flavor, but stay focused on delivering the alpha. Remember: DeFi doesn't care about feelings, and neither do you."""

# Logging Setup
def setup_logging():
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, "bot.log")

    logger = colorlog.getLogger()
    logger.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG

    # Console handler
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
            'ERROR': 'red', 'CRITICAL': 'red,bg_white',
        }
    ))
    console_handler.setLevel(logging.INFO)  # Keep console output at INFO level
    logger.addHandler(console_handler)

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'))
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    logger.addHandler(file_handler)

    # Suppress discord.py's built-in logging
    for module in ['discord', 'discord.http', 'discord.gateway']:
        logging.getLogger(module).setLevel(logging.WARNING)

    return logger

logger = setup_logging()

# Cache Setup
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)
perplexity_cache = Cache(cache_dir)

# Bot Setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Helper Functions
@lru_cache(maxsize=1000)
def get_cached_response(question):
    start_time = time.time()
    result = perplexity_cache.get(question.lower().strip())
    duration = time.time() - start_time
    logger.debug(f"Cache {'hit' if result else 'miss'} for question: '{question[:50]}...' (in {duration:.3f}s)")
    return result

def cache_response(question, response):
    start_time = time.time()
    perplexity_cache.set(question.lower().strip(), response, expire=3600)
    duration = time.time() - start_time
    logger.debug(f"Cached response for question: '{question[:50]}...' (in {duration:.3f}s)")

# Add a lock for API calls
api_lock = Lock()

# Add a message tracking set
sent_messages = set()

async def fetch_perplexity_response(question, retries=MAX_RETRIES):
    logger.info(f"Fetching Perplexity response for: '{question[:50]}...'")
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": True,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(PERPLEXITY_URL, headers=headers, json=data, timeout=PERPLEXITY_TIMEOUT) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result['choices'][0]['message']['content']
                    logger.info(f"Received successful response (length: {len(answer)})")
                    
                    # Add citations and related questions if available
                    citations = result.get('citations', [])
                    if citations:
                        answer += "\nSources:\n" + "\n".join(f"[{i+1}] {c.get('title', 'Untitled')}: {c.get('url', '')}" for i, c in enumerate(citations))
                    
                    related_questions = result.get('related_questions', [])
                    if related_questions:
                        answer += "\nRelated Questions:\n" + "\n".join(f"• {q}" for q in related_questions[:3])
                    
                    return answer
                elif response.status == 429 and retries > 0:
                    logger.warning(f"Rate limited. Retrying in {RETRY_DELAY}s... (Attempts left: {retries-1})")
                    await asyncio.sleep(RETRY_DELAY)
                    return await fetch_perplexity_response(question, retries - 1)
                elif 500 <= response.status < 600 and retries > 0:
                    logger.warning(f"Server error ({response.status}). Retrying in {RETRY_DELAY}s... (Attempts left: {retries-1})")
                    await asyncio.sleep(RETRY_DELAY)
                    return await fetch_perplexity_response(question, retries - 1)
                else:
                    logger.error(f"Unexpected status code {response.status}")
                    logger.error(f"Response content: {await response.text()[:200]}...")
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {PERPLEXITY_TIMEOUT}s")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    
    return None

# Add a new constant for session expiration
SESSION_EXPIRATION = int(os.getenv('SESSION_EXPIRATION', 3600))  # 1 hour default

# Add a new global variable to store user contexts
user_contexts = {}

# Add a new function to handle context
async def get_or_create_context(user_id):
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            'messages': [],
            'last_activity': time.time()
        }
    return user_contexts[user_id]

# Modify the get_perplexity_response_with_retry function
async def get_perplexity_response_with_retry(question, user_id):
    logger.info(f"Attempting to get response for user {user_id}: '{question[:50]}...'")
    start_time = time.time()
    
    context = await get_or_create_context(user_id)
    
    # Check if the context has expired
    if time.time() - context['last_activity'] > SESSION_EXPIRATION:
        logger.debug(f"Context expired for user {user_id}. Resetting.")
        context['messages'] = []
    
    context['messages'].append({"role": "user", "content": question})
    context['last_activity'] = time.time()
    
    cached_response = get_cached_response(str(context['messages']))
    if cached_response:
        logger.info(f"Using cached response for user {user_id}")
        return cached_response
    
    logger.info(f"No cache hit, fetching new response for user {user_id}")
    async with api_lock:  # Ensure only one API call at a time
        response = await fetch_perplexity_response(context['messages'])
    
    if response:
        context['messages'].append({"role": "assistant", "content": response})
        cache_response(str(context['messages']), response)
        logger.debug(f"Response cached for user {user_id}")
    else:
        logger.warning(f"Failed to get a valid response for user {user_id}")
    
    duration = time.time() - start_time
    logger.info(f"Total time to get response for user {user_id}: {duration:.3f}s")
    return response

# Modify the fetch_perplexity_response function
async def fetch_perplexity_response(messages, retries=MAX_RETRIES):
    logger.info(f"Fetching Perplexity response for context: '{messages[-1]['content'][:50]}...'")
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": True,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(PERPLEXITY_URL, headers=headers, json=data, timeout=PERPLEXITY_TIMEOUT) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result['choices'][0]['message']['content']
                    logger.info(f"Received successful response (length: {len(answer)})")
                    
                    # Add citations and related questions if available
                    citations = result.get('citations', [])
                    if citations:
                        answer += "\nSources:\n" + "\n".join(f"[{i+1}] {c.get('title', 'Untitled')}: {c.get('url', '')}" for i, c in enumerate(citations))
                    
                    related_questions = result.get('related_questions', [])
                    if related_questions:
                        answer += "\nRelated Questions:\n" + "\n".join(f"• {q}" for q in related_questions[:3])
                    
                    return answer
                elif response.status == 429 and retries > 0:
                    logger.warning(f"Rate limited. Retrying in {RETRY_DELAY}s... (Attempts left: {retries-1})")
                    await asyncio.sleep(RETRY_DELAY)
                    return await fetch_perplexity_response(messages, retries - 1)
                elif 500 <= response.status < 600 and retries > 0:
                    logger.warning(f"Server error ({response.status}). Retrying in {RETRY_DELAY}s... (Attempts left: {retries-1})")
                    await asyncio.sleep(RETRY_DELAY)
                    return await fetch_perplexity_response(messages, retries - 1)
                else:
                    logger.error(f"Unexpected status code {response.status}")
                    logger.error(f"Response content: {await response.text()[:200]}...")
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {PERPLEXITY_TIMEOUT}s")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    
    return None

async def send_long_message(ctx, message):
    logger.info(f"Sending long message (length: {len(message)})")
    max_length = 1900

    # Generate a unique identifier for this message
    message_id = hash(message + str(time.time()))
    if message_id in sent_messages:
        logger.warning(f"Duplicate message detected. Skipping send.")
        return

    sent_messages.add(message_id)

    if len(message) <= max_length:
        try:
            await ctx.send(message)
            logger.debug("Short message sent successfully")
        except (HTTPException, Forbidden) as e:
            logger.error(f"Failed to send message: {str(e)}")
        return

    parts = []
    while message:
        if len(message) <= max_length:
            parts.append(message)
            break
        
        split_index = max(message.rfind(c, 0, max_length) for c in '.?!')
        if split_index <= 0:
            split_index = message.rfind(' ', 0, max_length)
        if split_index <= 0:
            split_index = max_length

        parts.append(message[:split_index+1].strip())
        message = message[split_index+1:].strip()
    
    logger.debug(f"Message split into {len(parts)} parts")
    for part in parts:
        try:
            await ctx.send(part)
            await asyncio.sleep(1)  # Add a small delay between messages
        except (HTTPException, Forbidden) as e:
            logger.error(f"Failed to send message part: {str(e)}")
            break
    
    logger.debug("Long message sent successfully")

    # Remove the message ID after a delay to allow for potential retries
    asyncio.create_task(remove_sent_message(message_id))

async def remove_sent_message(message_id):
    await asyncio.sleep(60)  # Wait for 60 seconds
    sent_messages.discard(message_id)

# Bot Commands
@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is active in {len(bot.guilds)} guilds')
    logger.info("Bot is ready to receive DMs")

welcomed_users = set()

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if isinstance(message.channel, discord.DMChannel):
        await process_dm(message)
    else:
        try:
            await bot.process_commands(message)
        except Exception as e:
            logger.error(f"Error processing command: {str(e)}", exc_info=True)

async def process_dm(message):
    logger.info(f"Received DM from {message.author} (ID: {message.author.id})")
    question = message.content.strip()
    logger.debug(f"DM content: '{question[:50]}...'")
    
    if message.author.id not in welcomed_users:
        if len(question) < 5:
            logger.info(f"Sending welcome message to new user {message.author} (ID: {message.author.id})")
            welcome_message = ("Welcome to SecurePath AI! I'm here to answer your DeFi-related questions. "
                               "Feel free to ask anything about DeFi, crypto, or blockchain technology. "
                               "What would you like to know?")
            await message.channel.send(welcome_message)
            welcomed_users.add(message.author.id)
            return
        else:
            logger.info(f"New user {message.author} (ID: {message.author.id}) sent a valid first message")
            welcomed_users.add(message.author.id)
    
    if len(question) < 5:
        logger.info(f"Short message received from {message.author} (ID: {message.author.id})")
        await message.channel.send("Your message was a bit short. Could you please provide more details or ask a specific question about DeFi?")
        return
    
    if len(question) > 500:
        logger.info(f"Long message received from {message.author} (ID: {message.author.id})")
        await message.channel.send("That's quite a long message! Could you please shorten it to 500 characters or less? This helps me provide more focused answers.")
        return

    logger.info(f"Processing valid DM question from {message.author} (ID: {message.author.id})")
    async with message.channel.typing():
        try:
            logger.debug(f"Fetching response for DM question: '{question[:50]}...'")
            response = await get_perplexity_response_with_retry(f"DeFi question: {question}", message.author.id)
            
            if response:
                logger.info(f"Sending DM response to {message.author} (ID: {message.author.id})")
                await send_long_message(message.channel, response)
            else:
                logger.warning(f"No response generated for DM from {message.author} (ID: {message.author.id})")
                await message.channel.send("I'm sorry, I couldn't generate a response to that. Could you try rephrasing your question?")
        except Exception as e:
            logger.error(f"Error processing DM for {message.author} (ID: {message.author.id})", exc_info=True)
            await message.channel.send("Oops! Something went wrong on my end. Please try asking your question again in a moment.")

@bot.command(name='defi')
@commands.cooldown(1, 10, commands.BucketType.user)
async def defi(ctx, *, question=None):
    if ctx.author.id == 804823236222779413:
        ctx.command.reset_cooldown(ctx)

    logger.info(f"Received defi command from {ctx.author} (ID: {ctx.author.id}) in {'DM' if isinstance(ctx.channel, discord.DMChannel) else 'server'}")
    
    if not question:
        await ctx.send("Please provide a question after the !defi command. Example: `!defi What is yield farming?`")
        return

    logger.info(f"Question: {question}")
    
    if len(question) < 5:
        await ctx.send("Please provide a more detailed question (at least 5 characters).")
        return
    if len(question) > 500:
        await ctx.send("Your question is too long. Please limit it to 500 characters.")
        return
    
    async with ctx.typing():
        try:
            response = await get_perplexity_response_with_retry(f"DeFi question: {question}", ctx.author.id)
            
            if response:
                logger.info(f"Sending response to {ctx.author} (ID: {ctx.author.id})")
                await send_long_message(ctx, response)
            else:
                logger.warning(f"No response generated for {ctx.author} (ID: {ctx.author.id})")
                await ctx.send("I'm sorry, I couldn't generate a response. Please try rephrasing your question.")
        except Exception as e:
            logger.error(f"Error in defi command for {ctx.author} (ID: {ctx.author.id})", exc_info=True)
            await ctx.send("An unexpected error occurred. Our team has been notified.")

@bot.command(name='clear_context')
async def clear_context(ctx):
    user_id = ctx.author.id
    if user_id in user_contexts:
        del user_contexts[user_id]
        await ctx.send("Your conversation context has been cleared.")
    else:
        await ctx.send("You don't have any active context to clear.")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        if ctx.author.id != 804823236222779413:
            logger.info(f"Cooldown triggered for {ctx.author} (ID: {ctx.author.id})")
            await ctx.send(f"This command is on cooldown. Please try again in {error.retry_after:.2f} seconds.")
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Command invoke error for {ctx.author} (ID: {ctx.author.id})", exc_info=error.original)
        await ctx.send("An error occurred while processing your command. Please try again.")
    else:
        logger.error(f"Unhandled error for {ctx.author} (ID: {ctx.author.id}): {type(error).__name__}: {str(error)}")
        await ctx.send("An unexpected error occurred. Please try again.")

# Web Server
async def create_web_server():
    app = web.Application()
    
    # Setup session middleware
    fernet_key = fernet.Fernet.generate_key()
    secret_key = base64.urlsafe_b64decode(fernet_key)
    setup_session(app, EncryptedCookieStorage(secret_key))
    
    app.router.add_get('/', lambda request: web.Response(text="Bot is running!"))
    return app

async def start_bot():
    try:
        web_app = await create_web_server()
        runner = web.AppRunner(web_app)
        await runner.setup()
        port = int(os.environ.get('PORT', 10000))
        site = web.TCPSite(runner, '0.0.0.0', port)
        
        await asyncio.gather(
            site.start(),
            bot.start(DISCORD_TOKEN)
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    finally:
        await bot.close()
        await runner.cleanup()
        logger.info("Bot has been shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
    except Exception as e:
        logger.critical(f"Critical error occurred: {type(e).__name__}: {str(e)}", exc_info=True)
    finally:
        logger.info("Bot script has ended.")
        # Ensure all log messages are written
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        # Clear any remaining tasks
        pending = asyncio.all_tasks()
        for task in pending:
            task.cancel()
        asyncio.get_event_loop().run_until_complete(asyncio.gather(*pending, return_exceptions=True))