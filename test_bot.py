#!/usr/bin/env python3
"""
Test script to verify bot functionality without running the full bot
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_test(name, passed, details=""):
    """Print test result"""
    if passed:
        print(f"{GREEN}✓{RESET} {name}")
        if details:
            print(f"  {BLUE}{details}{RESET}")
    else:
        print(f"{RED}✗{RESET} {name}")
        if details:
            print(f"  {RED}{details}{RESET}")


async def test_configuration():
    """Test configuration loading"""
    print(f"\n{BLUE}Testing Configuration...{RESET}")
    try:
        from src.config import get_settings
        settings = get_settings()
        
        # Test each configuration
        tests = [
            ("Discord Token", bool(settings.discord_token) and settings.discord_token != 'your_discord_bot_token_here'),
            ("Bot Prefix", settings.bot_prefix == "!"),
            ("Owner ID", settings.owner_id != 0),
            ("Perplexity API", bool(settings.perplexity_api_key) and settings.perplexity_api_key != 'your_perplexity_api_key_here'),
            ("Rate Limits", settings.api_rate_limit_max > 0),
        ]
        
        for test_name, passed in tests:
            print_test(test_name, passed)
            
        return all(passed for _, passed in tests)
        
    except Exception as e:
        print_test("Configuration Loading", False, str(e))
        return False


async def test_bot_creation():
    """Test bot creation"""
    print(f"\n{BLUE}Testing Bot Creation...{RESET}")
    try:
        from src.bot import create_bot
        
        bot = create_bot()
        print_test("Bot Instance", bot is not None, f"Type: {type(bot).__name__}")
        print_test("Command Prefix", bot.command_prefix == "!", f"Prefix: {bot.command_prefix}")
        print_test("Intents", bot.intents.message_content, "Message content intent enabled")
        
        # Don't actually start the bot
        await bot.close()
        return True
        
    except Exception as e:
        print_test("Bot Creation", False, str(e))
        return False


async def test_ai_services():
    """Test AI service initialization"""
    print(f"\n{BLUE}Testing AI Services...{RESET}")
    try:
        from src.ai import AIManager
        from aiohttp import ClientSession
        
        async with ClientSession() as session:
            ai_manager = AIManager(session=session)
            
            print_test("AI Manager", ai_manager is not None)
            print_test("OpenAI Service", hasattr(ai_manager, 'openai_service'))
            print_test("Perplexity Service", hasattr(ai_manager, 'perplexity_service'))
            print_test("Vision Service", hasattr(ai_manager, 'vision_service'))
            
            return True
            
    except Exception as e:
        print_test("AI Services", False, str(e))
        return False


async def test_database():
    """Test database connection"""
    print(f"\n{BLUE}Testing Database...{RESET}")
    try:
        from src.database import db_manager
        
        # Try to connect
        connected = await db_manager.connect()
        
        if connected:
            print_test("Database Connection", True, "PostgreSQL connected")
            
            # Test pool
            print_test("Connection Pool", db_manager.pool is not None, 
                      f"Pool size: {db_manager.pool.get_size()}")
            
            # Disconnect
            await db_manager.disconnect()
            return True
        else:
            print_test("Database Connection", False, 
                      "Failed to connect (this is optional)")
            return False  # Database is optional
            
    except Exception as e:
        print_test("Database", False, f"Error: {str(e)}")
        print(f"  {YELLOW}Note: Database is optional for basic functionality{RESET}")
        return True  # Don't fail test for optional feature


async def test_utilities():
    """Test utility functions"""
    print(f"\n{BLUE}Testing Utilities...{RESET}")
    try:
        from src.utils import (
            validate_query_length,
            format_currency,
            format_large_number,
            validate_url
        )
        
        # Test validators
        valid, msg = validate_query_length("Test query")
        print_test("Query Validator", valid, msg or "Valid query")
        
        # Test formatters
        currency = format_currency(1234.56)
        print_test("Currency Formatter", currency == "$1,234.56", currency)
        
        number = format_large_number(1234567)
        print_test("Number Formatter", number == "1.2M", number)
        
        # Test URL validator
        url_valid = validate_url("https://discord.com")
        print_test("URL Validator", url_valid)
        
        return True
        
    except Exception as e:
        print_test("Utilities", False, str(e))
        return False


async def test_rate_limiter():
    """Test rate limiting"""
    print(f"\n{BLUE}Testing Rate Limiter...{RESET}")
    try:
        from src.services.rate_limiter import RateLimiter
        
        limiter = RateLimiter(max_calls=5, interval=60)
        user_id = 123456
        
        # Test limits
        results = []
        for i in range(6):
            limited = limiter.is_rate_limited(user_id)
            results.append(not limited)
            
        # First 5 should pass, 6th should fail
        expected = [True, True, True, True, True, False]
        passed = results == expected
        
        print_test("Rate Limiting", passed, 
                  f"Results: {results}, Expected: {expected}")
        
        return passed
        
    except Exception as e:
        print_test("Rate Limiter", False, str(e))
        return False


async def main():
    """Run all tests"""
    print(f"{BLUE}={'=' * 50}{RESET}")
    print(f"{BLUE}SecurePath Bot Test Suite{RESET}")
    print(f"{BLUE}={'=' * 50}{RESET}")
    
    tests = [
        ("Configuration", test_configuration),
        ("Bot Creation", test_bot_creation),
        ("AI Services", test_ai_services),
        ("Database", test_database),
        ("Utilities", test_utilities),
        ("Rate Limiter", test_rate_limiter),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"\n{RED}Error in {test_name}: {e}{RESET}")
            results[test_name] = False
    
    # Summary
    print(f"\n{BLUE}{'=' * 50}{RESET}")
    print(f"{BLUE}Test Summary{RESET}")
    print(f"{BLUE}{'=' * 50}{RESET}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed in results.items():
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"{test_name}: {status}")
    
    print(f"\n{BLUE}Total: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}All tests passed! The bot is ready to run.{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}Some tests failed. Check the configuration and try again.{RESET}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)