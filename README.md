# SecurePath AI Discord Bot

SecurePath AI is a high-performance Discord bot engineered for the crypto and DeFi world. It integrates with AI models to deliver real-time insights, advanced chart analysis, and blockchain intelligence, all within Discord. Designed to scale, SecurePath AI leverages efficient caching, dynamic logging, and API handling to ensure it provides top-tier information with minimal delays.

## Key Features

- **Expert Crypto Insights**: Responds to user queries with advanced DeFi and blockchain information.
- **Image and Chart Analysis**: Processes charts through the Vision API and provides quant-level technical analysis.
- **Contextual Conversation Flow**: Maintains awareness across user interactions, making conversations coherent and dynamic.
- **Rich Logging with `rich`**: Provides highly detailed, colorful logs to make debugging and monitoring seamless.
- **API Rate Management**: Ensures graceful API handling with rate limiting, retry mechanisms, and automatic error recovery.

---

## Installation Guide

### Prerequisites

- **Python 3.9+**
- **`pip`** (Python package manager)
- **Git**
- **Discord Bot Token**: Setup required in the [Discord Developer Portal](https://discord.com/developers/applications).
- **API Key**: Required for using OpenAI or Perplexity.

### Step 1: Clone the Repository

```bash
git clone https://github.com/fortunexbt/securepath.git
cd securepath
```

### Step 2: Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the root directory with your configuration:

#### **Essential Configuration:**

```
DISCORD_TOKEN=your_discord_bot_token
OWNER_ID=your_discord_user_id

# If using OpenAI
OPENAI_API_KEY=your_openai_api_key

# If using Perplexity
PERPLEXITY_API_KEY=your_perplexity_api_key
PERPLEXITY_API_URL=https://api.perplexity.ai/chat/completions
PERPLEXITY_TIMEOUT=60

# Set to True if using Perplexity, otherwise it will default to OpenAI.
USE_PERPLEXITY_API=True
```

- **`DISCORD_TOKEN`**: (Required) Your bot's authentication token from Discord.
- **`OWNER_ID`**: (Required) Your Discord User ID, allowing you to manage privileged commands.
- **`OPENAI_API_KEY`**: (Required if not using Perplexity) API key to use OpenAI's GPT models.
- **`PERPLEXITY_API_KEY`**: (Required if using Perplexity) API key for Perplexity.
- **`USE_PERPLEXITY_API`**: (Optional) Whether to use Perplexity or OpenAI APIs.

#### **Optional Configuration:**

```
LOG_CHANNEL_ID=your_discord_log_channel_id
SUMMARY_CHANNEL_ID=your_discord_summary_channel_id
NEWS_CHANNEL_ID=your_discord_news_channel_id
CHARTIST_CHANNEL_ID=your_discord_chartist_channel_id
NEWS_BOT_USER_ID=your_news_bot_user_id

API_RATE_LIMIT_MAX=100
API_RATE_LIMIT_INTERVAL=60
DAILY_API_CALL_LIMIT=1000

MAX_CONTEXT_MESSAGES=50
MAX_CONTEXT_AGE=3600
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

- **`LOG_CHANNEL_ID`**: (Optional) Discord channel ID for logging bot activity. Defaults to no logging if not provided.
- **`SUMMARY_CHANNEL_ID`**: (Optional) Used if generating summaries in specific channels.
- **`NEWS_CHANNEL_ID`**: (Optional) ID of the news feed channel the bot can post summaries to.
- **`CHARTIST_CHANNEL_ID`**: (Optional) Channel ID to track market charts and trends.
- **`NEWS_BOT_USER_ID`**: (Optional) Used if monitoring or interacting with a bot that posts news updates.

---

### Step 5: Bot Configuration in Discord Developer Portal

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Select your bot application, navigate to **Bot**, and enable the following:
   - **Message Content Intent**
3. Save and generate the OAuth2 URL to invite your bot to your server.

### Step 6: Running the Bot

Once your `.env` is set up, run the bot:

```bash
python main.py
```

You should see real-time logs displayed in your terminal confirming the bot is running.

---

## Advanced Features

### Caching and Rate Limiting

SecurePath AI uses advanced caching to avoid redundant API calls and enforces rate limits to prevent overuse. You can configure API call limits and intervals in the `.env`:

```env
API_RATE_LIMIT_MAX=100
API_RATE_LIMIT_INTERVAL=60
DAILY_API_CALL_LIMIT=1000
```

### Custom Context and Message Limits

Fine-tune how much historical context the bot retains by adjusting these optional environment variables:

```env
MAX_CONTEXT_MESSAGES=50  # Number of messages stored in conversation history
MAX_CONTEXT_AGE=3600     # Maximum age of messages in seconds
```

### Logging and Debugging

Use the `LOG_CHANNEL_ID` and `LOG_LEVEL` to control logging. Logs will be sent to your specified Discord channel or can be viewed directly in the console. For example:

```env
LOG_CHANNEL_ID=1234567890
LOG_LEVEL=DEBUG  # Can be INFO, DEBUG, WARNING, ERROR
```

### Dynamic Status and Presence

The bot periodically updates its Discord presence, indicating its current task (e.g., analyzing charts or fetching market insights). The statuses rotate automatically during operation.

---

## Troubleshooting

- **Module Not Found**: Ensure the virtual environment is activated and dependencies installed via `pip install -r requirements.txt`.
- **Bot Not Responding**: Check if the bot token and API key(s) are correctly set in your `.env`. Verify bot permissions on Discord.
- **Rate Limiting**: If you hit the API limit, adjust the `API_RATE_LIMIT_MAX` and `DAILY_API_CALL_LIMIT` as needed.

---

## License

This project is licensed under the MIT License.

---

## Disclaimer

SecurePath AI provides information for educational purposes only and should not be considered financial advice. Always conduct your own research before making investment decisions.
