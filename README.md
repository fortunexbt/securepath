# SecurePath Discord Bot

SecurePath is a Discord bot designed to provide up-to-date information and insights about decentralized finance (DeFi) protocols, cryptocurrencies, and blockchain networks.

## Features

- Responds to user queries about DeFi protocols, tokens, and blockchain networks
- Utilizes the Perplexity API for generating informative responses
- Implements caching to improve response times for repeated queries
- Provides detailed logging for monitoring and debugging

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/securepath.git
   cd securepath
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```

4. Run the bot:
   ```
   python main.py
   ```

## Usage

Once the bot is running and added to your Discord server, you can use the following command:

- `!defi <question>`: Ask a question about DeFi, cryptocurrencies, or blockchain networks.

Example:

- `!defi What are the top protocols on Solana?`


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Disclaimer

This bot provides information for educational purposes only. It should not be considered financial advice. Always do your own research before making any investment decisions.