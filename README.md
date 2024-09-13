# SecurePath AI Discord Bot

SecurePath AI is an advanced Discord bot designed to provide expert insights and up-to-date information about DeFi protocols, cryptocurrencies, and blockchain networks.

## Key Features

- Responds to user queries with in-depth knowledge of DeFi, crypto, and blockchain
- Utilizes Perplexity API for generating accurate, real-time responses
- Implements efficient caching for improved response times
- Maintains conversation context for more coherent interactions
- Provides detailed logging for monitoring and debugging

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/fortunexbt/securepath
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Create a `.env` file in the root directory with:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```

4. Launch the bot:
   ```
   python main.py
   ```

## Usage

Once the bot is active in your Discord server, use these commands:

- `!defi <question>`: Ask about DeFi, crypto, or blockchain topics
- `!clear_context`: Reset your conversation context

Example:
```
!defi What's the current state of Uniswap's liquidity?
```

## DM Functionality

The bot also responds to direct messages, providing a more personalized experience.

## Contributing

We welcome contributions! Please submit a Pull Request with your improvements.

## License

This project is licensed under the MIT License.

## Disclaimer

SecurePath AI provides information for educational purposes only. It should not be considered financial advice. Always DYOR before making investment decisions.