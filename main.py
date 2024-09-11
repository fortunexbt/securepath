import os
import asyncio
import logging
import colorlog
from dotenv import load_dotenv
import discord
from discord.ext import commands
import aiohttp
from logging.handlers import RotatingFileHandler
from diskcache import Cache
from functools import lru_cache
import time

# Move this block to the top of the file, right after imports
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file = os.path.join(log_directory, "bot.log")

# Configure color logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Add file handler for logging to file
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)

# Suppress discord.py's built-in logging
logging.getLogger('discord').setLevel(logging.WARNING)
logging.getLogger('discord.http').setLevel(logging.WARNING)
logging.getLogger('discord.gateway').setLevel(logging.WARNING)

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

# Initialize the cache
cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
perplexity_cache = Cache(cache_dir)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Perplexity API setup
PERPLEXITY_URL = os.getenv('PERPLEXITY_URL', "https://api.perplexity.ai/chat/completions")
PERPLEXITY_TIMEOUT = int(os.getenv('PERPLEXITY_TIMEOUT', 60))
MAX_RETRIES = 3
RETRY_DELAY = 2

SYSTEM_PROMPT = """you’re a defi & crypto expert. give solid, up-to-date info—no fluff. focus on actionable advice. decentralized solutions FIRST, centralized ONLY if you have to. be sharp, precise. use citations [1], [2] when needed. highlight time-sensitive stuff. when telling people how to buy/interact with protocols, hit DEXs & cross-chain bridges. centralized? last resort. sarcasm welcome, but stay sharp. no easy-money seekers or bedtime stories here."""

# Helper functions
@lru_cache(maxsize=1000)
def get_cached_response(question):
    start_time = time.time()
    result = perplexity_cache.get(question.lower().strip())
    duration = time.time() - start_time
    if result:
        logger.debug(f"Cache hit for question: '{question[:50]}...' (retrieved in {duration:.3f}s)")
    else:
        logger.debug(f"Cache miss for question: '{question[:50]}...' (checked in {duration:.3f}s)")
    return result

def cache_response(question, response):
    start_time = time.time()
    perplexity_cache.set(question.lower().strip(), response, expire=3600)
    duration = time.time() - start_time
    logger.debug(f"Cached response for question: '{question[:50]}...' (stored in {duration:.3f}s)")

async def fetch_perplexity_response(question, retries=MAX_RETRIES):
    logger.info(f"Fetching Perplexity response for question: '{question[:50]}...'")
    url = PERPLEXITY_URL
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
    
    try:
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.post(url, headers=headers, json=data, timeout=PERPLEXITY_TIMEOUT) as response:
                duration = time.time() - start_time
                logger.debug(f"Perplexity API request completed in {duration:.3f}s with status {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    answer = result['choices'][0]['message']['content']
                    logger.info(f"Received successful response from Perplexity API (length: {len(answer)})")
                    
                    # Extract citations if available
                    citations = result.get('citations', [])
                    if citations:
                        formatted_citations = []
                        for i, citation in enumerate(citations, 1):
                            title = citation.get('title', 'Untitled')
                            url = citation.get('url', '')
                            formatted_citations.append(f"[{i}] {title}: {url}")
                        
                        answer += "\nSources:\n" + "\n".join(formatted_citations)
                    
                    # Extract related questions if available
                    related_questions = result.get('related_questions', [])
                    if related_questions:
                        answer += "\nRelated Questions:\n" + "\n".join(f"• {q}" for q in related_questions[:3])
                    
                    return answer
                elif response.status == 429 and retries > 0:
                    logger.warning(f"Rate limited (status 429). Retrying in {RETRY_DELAY} seconds... (Attempts left: {retries-1})")
                    await asyncio.sleep(RETRY_DELAY)
                    return await fetch_perplexity_response(question, retries - 1)
                elif 500 <= response.status < 600 and retries > 0:
                    logger.warning(f"Server error (status {response.status}). Retrying in {RETRY_DELAY} seconds... (Attempts left: {retries-1})")
                    await asyncio.sleep(RETRY_DELAY)
                    return await fetch_perplexity_response(question, retries - 1)
                else:
                    logger.error(f"Perplexity API returned unexpected status code {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response content: {response_text[:200]}...")
    except aiohttp.ClientError as e:
        logger.error(f"Network error when calling Perplexity API: {str(e)}")
    except asyncio.TimeoutError:
        logger.error(f"Request to Perplexity API timed out after {PERPLEXITY_TIMEOUT}s")
    except Exception as e:
        logger.error(f"Unexpected error calling Perplexity API: {str(e)}", exc_info=True)
    
    return None

async def get_perplexity_response_with_retry(question):
    logger.info(f"Attempting to get response for question: '{question[:50]}...'")
    start_time = time.time()
    cached_response = get_cached_response(question)
    if cached_response:
        logger.info("Using cached Perplexity response")
        return cached_response
    
    logger.info("No cache hit, fetching new Perplexity response")
    response = await fetch_perplexity_response(question)
    
    if response:
        cache_response(question, response)
    else:
        logger.warning("Failed to get a valid response from Perplexity API")
    
    duration = time.time() - start_time
    logger.info(f"Total time to get response: {duration:.3f}s")
    return response

async def send_long_message(ctx, message):
    logger.info(f"Sending long message (length: {len(message)})")
    max_length = 1900  # Reduced from 2000 to provide some buffer

    if len(message) <= max_length:
        await ctx.send(message)
    else:
        parts = []
        while message:
            if len(message) <= max_length:
                parts.append(message)
                break
            
            # Find the last sentence end within the max_length
            last_period = message.rfind('.', 0, max_length)
            last_question = message.rfind('?', 0, max_length)
            last_exclamation = message.rfind('!', 0, max_length)
            split_index = max(last_period, last_question, last_exclamation)
            
            if split_index == -1 or split_index == 0:
                # If no sentence end found, split at the last space
                split_index = message.rfind(' ', 0, max_length)
                if split_index == -1:
                    # If no space found, force split at max_length
                    split_index = max_length

            parts.append(message[:split_index+1].strip())
            message = message[split_index+1:].strip()
        
        logger.debug(f"Message split into {len(parts)} parts")
        for part in parts:
            await ctx.send(part)
    
    logger.debug("Long message sent successfully")

# Bot commands
@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is active in {len(bot.guilds)} guilds')

@bot.command(name='defi')
@commands.cooldown(1, 10, commands.BucketType.user)
async def defi(ctx, *, question):
    logger.info(f"Received defi command from user {ctx.author} (ID: {ctx.author.id})")
    logger.info(f"Question: {question}")
    
    if len(question) < 5:
        logger.warning(f"Rejected short question from user {ctx.author} (ID: {ctx.author.id})")
        await ctx.send("Please provide a more detailed question (at least 5 characters).")
        return
    if len(question) > 500:
        logger.warning(f"Rejected long question from user {ctx.author} (ID: {ctx.author.id})")
        await ctx.send("Your question is too long. Please limit it to 500 characters.")
        return
    
    async with ctx.typing():
        start_time = time.time()
        try:
            response = await get_perplexity_response_with_retry(f"DeFi question: {question}")
            
            if response:
                logger.info(f"Sending response to user {ctx.author} (ID: {ctx.author.id})")
                await send_long_message(ctx, response)
            else:
                logger.warning(f"No response generated for user {ctx.author} (ID: {ctx.author.id})")
                await ctx.send("I'm sorry, I couldn't generate a response. Please try rephrasing your question.")
        except aiohttp.ClientError as e:
            logger.error(f"Network error for user {ctx.author} (ID: {ctx.author.id}): {str(e)}")
            await ctx.send("I'm having trouble connecting to the AI service. Please try again later.")
        except Exception as e:
            logger.error(f"Unexpected error in defi command for user {ctx.author} (ID: {ctx.author.id})", exc_info=True)
            await ctx.send("An unexpected error occurred. Our team has been notified.")
        finally:
            duration = time.time() - start_time
            logger.info(f"Total command processing time: {duration:.3f}s")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        logger.info(f"Cooldown triggered for user {ctx.author} (ID: {ctx.author.id})")
        await ctx.send(f"This command is on cooldown. Please try again in {error.retry_after:.2f} seconds.")
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Command invoke error for user {ctx.author} (ID: {ctx.author.id})", exc_info=error.original)
        await ctx.send("An error occurred while processing your command. Please try again.")
    else:
        logger.error(f"Unhandled error for user {ctx.author} (ID: {ctx.author.id})", exc_info=error)
        await ctx.send("An unexpected error occurred. Please try again.")

# Add this function to send a request to your bot's URL
async def keep_alive():
    url = "https://securepath-bot.onrender.com"  # Replace with your actual bot URL
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(url) as response:
                    print(f"Keep-alive request sent. Status: {response.status}")
            except Exception as e:
                print(f"Error sending keep-alive request: {e}")
            await asyncio.sleep(600)  # Wait for 10 minutes before the next request

# Modify your bot.run() call to include the keep_alive coroutine
async def start_bot():
    await asyncio.gather(
        bot.start("YOUR_BOT_TOKEN"),
        keep_alive()
    )

# Replace the existing bot.run() call with this
if __name__ == "__main__":
    asyncio.run(start_bot())