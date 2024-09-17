# SecurePath AI Discord Bot

SecurePath AI is a high-performance Discord bot built to deliver expert-level insights on DeFi protocols, cryptocurrencies, and blockchain networks. Designed for real-time engagement, it provides comprehensive, accurate responses with a future-proof caching and logging system.

## Key Features

- **In-depth DeFi, Crypto, and Blockchain Knowledge**: Responds to user queries with expert insights.
- **Efficient Caching**: Reduces API calls by caching responses for improved performance.
- **Contextual Awareness**: Maintains conversation context to provide coherent, flowing interactions.
- **Rich Logging**: Uses `rich` for detailed and colorful logs, making it easier to monitor and debug.
- **Real-time Updates**: Ensures up-to-date responses by dynamically including the current date in API queries.
- **Graceful API Handling**: Manages rate limits, timeouts, and errors effectively.

## Installation Guide

### Prerequisites

- Python 3.9+
- `pip` (Python package manager)
- `git`
- Discord bot setup: [Discord Developer Portal](https://discord.com/developers/applications)
- Optional: API key for Perplexity or another knowledge API

### Step 1: Clone the Repository

In your terminal, run the following commands:
```bash
git clone https://github.com/fortunexbt/securepath.git
cd securepath
```

### Step 2: Set Up a Virtual Environment

Create a virtual environment to isolate dependencies:
```bash
python -m venv venv
```

Activate the virtual environment:
- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root with the following details:
```
DISCORD_TOKEN=your_discord_bot_token
PERPLEXITY_API_KEY=your_perplexity_api_key (if using)
PERPLEXITY_API_URL=https://api.perplexity.ai/chat/completions
PERPLEXITY_TIMEOUT=60
```

Replace `your_discord_bot_token` with your actual bot token from the Discord Developer Portal.

### Step 5: Configure Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Select or create your bot application.
3. Navigate to **Bot** settings.
4. Enable the following under **Privileged Gateway Intents**:
   - Presence Intent
   - Server Members Intent
   - Message Content Intent
5. Save the changes.

### Step 6: Invite the Bot to Your Server

1. In the **OAuth2** section of the Developer Portal, select **bot** under Scopes.
2. Choose the necessary bot permissions (at minimum: Read Messages, Send Messages, Read Message History).
3. Copy the generated URL and use it to invite the bot to your server.

### Step 7: Running the Bot

Run the bot with:
```bash
python main.py
```

You should see the Rich console output confirming the bot is running and connected to Discord.

## Usage

### DM Functionality

Send your questions directly in a DM to the bot—no need for prefixes:

Example:
```
What's the latest with Uniswap liquidity?
```

### Channel Commands

In public channels, use the following commands:

- `!defi <question>`: Ask about DeFi, crypto, or blockchain topics.
- `!clear_context`: Reset the conversation context.

Example:
```
!defi What are the advantages of layer 2 solutions?
```

## Troubleshooting

- **Module Not Found**: Ensure the virtual environment is activated and dependencies installed with `pip install -r requirements.txt`.
- **Bot Not Responding**: Verify environment variables, bot token, and API key. Ensure the bot has the right permissions in Discord.
- **API Issues**: Ensure the API key is valid, and you haven’t hit rate limits.

## Advanced Features

- **Contextual Conversations**: The bot remembers recent conversations for more natural, fluid responses.
- **Real-time Accuracy**: API queries dynamically account for the current date, ensuring up-to-date information.
- **Error Handling**: Built-in mechanisms to handle API rate limits, timeouts, and other errors gracefully.
- **Enhanced Logging**: The `rich` library provides colorful, detailed logs for easier monitoring and debugging.

## Contributing

Contributions are welcome. Please submit a Pull Request with a detailed description of your changes.

## License

This project is licensed under the MIT License.

## Disclaimer

SecurePath AI provides information for educational purposes only and should not be considered financial advice. Always conduct your own research (DYOR) before making investment decisions.

---

This version simplifies the instructions, removes any redundant or unverified information (such as the use of APIs not confirmed to be in use), and streamlines the key features to match the core functionality of your bot.