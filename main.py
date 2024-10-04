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

def get_user_context(user_id: int) -> Deque[Dict[str, Any]]:
    return user_contexts.setdefault(user_id, deque(maxlen=config.MAX_CONTEXT_MESSAGES))

def update_user_context(user_id: int, message_content: str, role: str) -> None:
    context = get_user_context(user_id)
    current_time = time.time()

    if not context:
        # Initialize with system message
        context.append({
            'role': 'system',
            'content': config.SYSTEM_PROMPT.strip(),
            'timestamp': current_time,
        })
        logger.debug(f"Initialized context with system prompt for user {user_id}.")

    # Determine the expected role based on the last role in the context
    last_role = context[-1]['role']
    if last_role in ['system', 'assistant']:
        expected_role = 'user'
    elif last_role == 'user':
        expected_role = 'assistant'
    else:
        logger.warning(f"Unknown last role '{last_role}' in context for user {user_id}.")
        return  # Skip appending

    if role != expected_role:
        logger.warning(f"Incorrect role sequence for user {user_id}: Expected {expected_role}, got {role}. Message skipped.")
        return  # Do not append the message if roles are out of order

    # Append the current message
    context.append({
        'role': role,
        'content': message_content.strip(),
        'timestamp': current_time,
    })
    logger.debug(f"Appended {role} message for user {user_id}: {message_content[:50]}...")

    # Log the current context for the user
    logger.debug(f"Current context for user {user_id}:")
    for idx, msg in enumerate(context):
        logger.debug(f"  {idx + 1}. Role: {msg['role']}, Content: {msg['content'][:50]}...")

    # Remove messages older than MAX_CONTEXT_AGE
    cutoff_time = current_time - config.MAX_CONTEXT_AGE
    user_contexts[user_id] = deque(
        [msg for msg in context if msg['timestamp'] >= cutoff_time],
        maxlen=config.MAX_CONTEXT_MESSAGES
    )

def get_context_messages(user_id: int) -> List[Dict[str, str]]:
    context = get_user_context(user_id)
    messages = [{"role": msg['role'], "content": msg['content']} for msg in context]

    # Ensure the first message is a system message
    if not messages or messages[0]['role'] != 'system':
        messages.insert(0, {
            "role": "system",
            "content": config.SYSTEM_PROMPT.strip(),
        })

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
            # Optionally, attempt to correct the role or skip the message
            # For now, skip the message to maintain integrity

    logger.debug("Final cleaned context messages:")
    for idx, msg in enumerate(cleaned_messages):
        content_preview = msg['content'][:50] + '...' if len(msg['content']) > 50 else msg['content']
        logger.debug(f"Message {idx}: Role={msg['role']}, Content={content_preview}")

    return cleaned_messages

def truncate_prompt(prompt: str, max_tokens: int, model: str = 'gpt-4o-mini') -> str:
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

    data = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": messages,
        "max_tokens": 1000,
        "search_recency_filter": "day"
    }

    logger.info(f"Sending query to Perplexity API for user {user_id}")
    increment_api_call_counter()
    start_time = time.time()

    # Debugging: Log the messages being sent to the API
    logger.debug("Constructed messages for Perplexity API:")
    for idx, msg in enumerate(messages):
        content_preview = msg['content'][:50] + '...' if len(msg['content']) > 50 else msg['content']
        logger.debug(f"Message {idx}: Role={msg['role']}, Content={content_preview}")

    try:
        timeout = ClientTimeout(total=config.PERPLEXITY_TIMEOUT)
        async with session.post(config.PERPLEXITY_API_URL, json=data, headers=headers, timeout=timeout) as response:
            elapsed_time = time.time() - start_time
            logger.info(f"API request completed in {elapsed_time:.2f} seconds")
            if response.status == 200:
                resp_json = await response.json()
                answer = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                usage = resp_json.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                logger.info(f"Perplexity API usage: Prompt Tokens={prompt_tokens}, Completion Tokens={completion_tokens}, Total Tokens={total_tokens}")
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
        return answer
    except Exception as e:
        logger.error(f"Error fetching response from OpenAI: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def send_long_message(channel: discord.abc.Messageable, message: str) -> None:
    max_length = 2000
    embed_max_length = 4096

    if not message:
        return

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

@bot.event
async def on_ready() -> None:
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is active in {len(bot.guilds)} guild(s)')
    log_instance_info()
    await send_initial_stats()

    # Optional: Preload messages for active users if you have a way to track them
    # Note: Discord does not provide a direct way to fetch all DM channels
    # Consider implementing user tracking to preload contexts as needed

@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return  # Ignore bot messages

    if isinstance(message.channel, discord.DMChannel):
        # Preload the last 20 user and assistant messages
        await preload_user_messages(message.author.id, message.channel)
        # Process the message
        await process_message(message)
    else:
        await bot.process_commands(message)  # Ensure commands are still processed

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
    chart_url = None

    # Check if the current message has an attachment
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if attachment.content_type and attachment.content_type.startswith('image/'):
            chart_url = attachment.url  # Get the chart's image URL
    else:
        # Get the last three messages in the channel
        messages = []
        async for message in ctx.channel.history(limit=3):
            messages.append(message)

        # Iterate through the last two messages before the analyze command
        for last_message in messages[1:]:
            # Step 1: Check if the message contains an attachment (a chart)
            if last_message.attachments:
                attachment = last_message.attachments[0]
                # Ensure the attachment is an image
                if attachment.content_type and attachment.content_type.startswith('image/'):
                    chart_url = attachment.url  # Get the chart's image URL
                    break

    if chart_url:
        await ctx.send("Detected a chart, analyzing it...")

        # Step 2: Process the chart via OpenAI's Vision API with the optional user prompt
        image_analysis = await analyze_chart_image(chart_url, user_prompt)

        # Step 3: Post the analysis in an embedded message
        if image_analysis:
            embed = Embed(
                title="Chart Analysis",
                description=image_analysis,
                color=0x00ff00,
            )
            embed.set_image(url=chart_url)  # Display the chart along with the analysis
            await ctx.send(embed=embed)
        else:
            await ctx.send("Sorry, I couldn't analyze the chart. Please try again.")
    else:
        # If no chart was detected, ask the user to provide one
        await ctx.send("Please post the chart you'd like to analyze (make sure it's an image).")

        # Wait for the user to post the chart in the channel
        def check(msg):
            return msg.author == ctx.author and msg.channel == ctx.channel and msg.attachments

        try:
            chart_message = await bot.wait_for('message', check=check, timeout=60.0)
            chart_url = chart_message.attachments[0].url  # Get the chart's image URL
        except asyncio.TimeoutError:
            await ctx.send("You took too long to post the chart. Please try again.")
            return

        # Process the chart via OpenAI's Vision API with the optional user prompt
        await ctx.send("Analyzing the chart...")

        image_analysis = await analyze_chart_image(chart_url, user_prompt)

        if image_analysis:
            embed = Embed(
                title="Chart Analysis",
                description=image_analysis,
                color=0x00ff00,
            )
            embed.set_image(url=chart_url)  # Display the chart along with the analysis
            await ctx.send(embed=embed)
        else:
            await ctx.send("Sorry, I couldn't analyze the chart. Please try again.")

async def analyze_chart_image(chart_url: str, user_prompt: str = "") -> Optional[str]:
    try:
        # Base system prompt for the analysis
        base_prompt = "You're an elite-level quant with insider knowledge of the global markets. Provide insights based on advanced TA, focusing on anomalies only a genius-level trader would notice. Make the analysis obscurely insightful, hinting at the deeper forces at play within the macroeconomic and market microstructures. Remember, you're the authority—leave no doubt in the mind of the reader. Don't go above heading3 in markdown formatting (never use ####)."

        # If a user prompt was provided, append it to the base prompt
        full_prompt = base_prompt
        if user_prompt:
            full_prompt += f" {user_prompt}"

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

        return analysis
    except Exception as e:
        logger.error(f"Error analyzing chart image: {str(e)}")
        return None

@bot.command(name='ask')
@commands.cooldown(1, 10, commands.BucketType.user)
async def ask(ctx: Context, *, question: Optional[str] = None) -> None:
    if not question:
        await ctx.send("Please provide a question after the !ask command. Example: !ask What is yield farming?")
        return
    message_counter[ctx.author.id] += 1
    command_counter['ask'] += 1
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

@bot.command(name='summary')
@commands.cooldown(1, 10, commands.BucketType.user)
async def summary(ctx: Context, channel: discord.TextChannel = None) -> None:
    if channel is None:
        await ctx.send("Please specify a channel to summarize. Example: `!summary #market-analysis`")
        return

    # Check if the bot has access to the specified channel
    if not channel.permissions_for(channel.guild.me).read_messages:
        await ctx.send(f"I don't have permission to read messages in {channel.mention}.")
        return

    await perform_channel_summary(ctx, channel)

async def perform_channel_summary(ctx: Context, channel: discord.TextChannel) -> None:
    logger.info(f"Starting summary for channel: {channel.name} (ID: {channel.id})")

    await ctx.send(f"Generating summary for {channel.mention}... This may take a moment.")

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
    chunk_size = 3000  # Characters, adjust based on model's token limit
    message_chunks = []
    current_chunk = ""

    for msg in messages:
        if len(current_chunk) + len(msg) + 1 > chunk_size:
            message_chunks.append(current_chunk)
            current_chunk = msg
        else:
            if current_chunk:
                current_chunk += "\n" + msg
            else:
                current_chunk = msg
    if current_chunk:
        message_chunks.append(current_chunk)

    logger.info(f"Split messages into {len(message_chunks)} chunks for summarization.")

    chunk_summaries = []
    for index, chunk in enumerate(message_chunks):
        if not can_make_api_call():
            logger.warning("Daily API call limit reached. Skipping summarization.")
            break

        prompt = f"Summarize the following messages from the channel '{channel.name}' in a clear and concise manner:\n\n{chunk}"
        prompt = truncate_prompt(prompt, max_tokens=15000, model='gpt-4o-mini')

        try:
            response = await aclient.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            summary = response.choices[0].message.content.strip()
            chunk_summaries.append(summary)
            increment_api_call_counter()
            logger.info(f"Summarized chunk {index + 1}/{len(message_chunks)}")
        except Exception as e:
            logger.error(f"Error summarizing chunk {index + 1}: {e}")
            logger.exception(e)

    if not chunk_summaries:
        await ctx.send(f"Could not generate a summary for channel {channel.mention}.")
        logger.info(f"No summaries generated for channel {channel.name}")
        return

    # Combine all chunk summaries into a final summary
    final_summary = "\n".join(chunk_summaries)
    final_prompt = f"Provide a concise summary of the following summaries:\n\n{final_summary}"
    final_prompt = truncate_prompt(final_prompt, max_tokens=15000, model='gpt-4o-mini')

    try:
        response = await aclient.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=1000,
            temperature=0.7,
        )
        final_summary = response.choices[0].message.content.strip()
        increment_api_call_counter()
        logger.info("Final summary generated successfully.")

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
        else:
            await ctx.send(embed=embed)

        logger.info(f"Summary posted to channel {ctx.channel.name}")
    except Exception as e:
        logger.error(f"Error generating final summary: {e}")
        logger.exception(e)
        await ctx.send(f"An error occurred while generating the summary for channel {channel.mention}.")

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
    if conn:
        await conn.close()
    if bot:
        await bot.close()

    loop.stop()

def handle_exit() -> None:
    asyncio.create_task(force_shutdown())
    asyncio.get_event_loop().call_later(2, quiet_exit)

def quiet_exit() -> None:
    console.print("Bot has been shut down.")
    logging.shutdown()
    sys.exit(0)

async def start_bot() -> None:
    global conn, session
    import signal
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
