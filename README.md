# SecurePath AI Discord Bot

SecurePath AI is an advanced Discord bot designed to provide expert insights and up-to-date information about DeFi protocols, cryptocurrencies, and blockchain networks.

## Key Features

- Responds to user queries with in-depth knowledge of DeFi, crypto, and blockchain
- Utilizes Perplexity API for generating accurate, real-time responses
- Implements efficient caching for improved response times
- Maintains conversation context for more coherent interactions
- Provides detailed, colorful logging using Rich for enhanced monitoring and debugging
- Ensures up-to-date information by including the current date in API requests
- Handles API rate limits and timeouts gracefully

## Comprehensive Installation Guide

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git
- A Discord account and a registered Discord application/bot: https://discord.com/developers/applications
- A Perplexity API key: https://perplexity.ai/

### Step 1: Clone the Repository

1. Open your terminal or command prompt.
2. Navigate to the directory where you want to install the bot.
3. Run the following command:
   ```
   git clone https://github.com/fortunexbt/securepath.git
   cd securepath
   ```

### Step 2: Set Up a Virtual Environment

Creating a virtual environment is recommended to avoid conflicts with other Python projects.

1. Create a virtual environment:
   ```
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

### Step 3: Install Dependencies

Install the required packages using pip:
```
pip install -r requirements.txt
```


### Step 4: Set Up Environment Variables

1. Create a new file named `.env` in the root directory of the project.
2. Add the following content to the `.env` file:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   PERPLEXITY_API_KEY=your_perplexity_api_key
   PERPLEXITY_API_URL=https://api.perplexity.ai/chat/completions
   PERPLEXITY_TIMEOUT=60
   ```
3. Replace `your_discord_bot_token` with your actual Discord bot token.
4. Replace `your_perplexity_api_key` with your Perplexity API key.

### Step 5: Configure Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Click on your application (or create a new one if you haven't already).
3. Navigate to the "Bot" section.
4. Under "Privileged Gateway Intents", enable:
   - Presence Intent
   - Server Members Intent
   - Message Content Intent
5. Save your changes.

### Step 6: Invite the Bot to Your Server

1. In the Discord Developer Portal, go to the "OAuth2" section.
2. In the "Scopes" section, select "bot".
3. In the "Bot Permissions" section, select the permissions your bot needs (at minimum: Read Messages/View Channels, Send Messages, Read Message History).
4. Copy the generated URL and open it in a new browser tab.
5. Select the server you want to add the bot to and follow the prompts.

### Step 7: Run the Bot

1. Ensure your virtual environment is activated.
2. Run the following command:
   ```
   python main.py
   ```
3. You should see Rich console output indicating that the bot has started and connected to Discord.

## Usage

### DM Functionality

The bot responds to direct messages without needing a prefix. Simply send your question directly:

Example:
```
What's the current state of Uniswap's liquidity?
```

### Chat Functionality

To use the chat functionality, you need to prefix your message with a specific command. For example:

```

### Channel Commands

In channels, use these commands:

- `!defi <question>`: Ask about DeFi, crypto, or blockchain topics
- `!clear_context`: Reset your conversation context

Example:
```
!defi What are the advantages of layer 2 solutions?
```

## Troubleshooting

- If you encounter any "Module not found" errors, ensure you've activated the virtual environment and installed all dependencies.
- If the bot doesn't respond, check that you've correctly set up the environment variables and that your Discord bot token is valid.
- For any API-related issues, verify that your Perplexity API key is correct and that you haven't exceeded your API rate limits.

## Advanced Features

- **Context Awareness**: The bot maintains conversation context, allowing for more natural, flowing conversations.
- **Up-to-date Information**: Each query includes the current date, ensuring responses are based on the most recent data available.
- **Efficient Caching**: Responses are cached to improve performance and reduce API calls.
- **Error Handling**: The bot gracefully handles API rate limits, timeouts, and other potential errors.
- **Rich Logging**: Detailed, colorful logs are generated for easy monitoring and debugging.

## Contributing

We welcome contributions! Please submit a Pull Request with your improvements.

## License

This project is licensed under the MIT License.

## Disclaimer

SecurePath AI provides information for educational purposes only. It should not be considered financial advice. Always conduct your own research (DYOR) before making investment decisions.