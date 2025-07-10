#!/usr/bin/env python3
"""
Simple test script to verify that all refactored modules can be imported successfully.
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config_imports():
    """Test configuration module imports."""
    print("Testing configuration imports...")
    try:
        from src.config.settings import get_settings
        from src.config.constants import DISCORD_MESSAGE_LIMIT
        
        settings = get_settings()
        print(f"✅ Config loaded - Bot prefix: {settings.bot_prefix}")
        print(f"✅ Constants loaded - Message limit: {DISCORD_MESSAGE_LIMIT}")
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False


def test_bot_imports():
    """Test bot module imports."""
    print("\nTesting bot imports...")
    try:
        from src.bot.client import create_bot, SecurePathBot
        from src.bot.events import setup_background_tasks
        
        # Test bot creation (don't actually start it)
        bot = create_bot()
        print(f"✅ Bot created successfully - Type: {type(bot).__name__}")
        print(f"✅ Bot prefix configured: {bot.command_prefix}")
        return True
    except Exception as e:
        print(f"❌ Bot import failed: {e}")
        return False


def test_ai_imports():
    """Test AI module imports."""
    print("\nTesting AI imports...")
    try:
        from src.ai import AIManager, OpenAIService, PerplexityService, VisionService
        
        # Test service creation
        openai_service = OpenAIService()
        print(f"✅ OpenAI service created")
        print(f"✅ Usage data initialized: {bool(openai_service.usage_data)}")
        return True
    except Exception as e:
        print(f"❌ AI import failed: {e}")
        return False


def test_database_imports():
    """Test database module imports."""
    print("\nTesting database imports...")
    try:
        from src.database import db_manager, UsageRepository, AnalyticsRepository
        from src.database.models import UsageRecord, UserAnalytics
        
        print(f"✅ Database manager created: {type(db_manager).__name__}")
        print(f"✅ Models imported successfully")
        
        # Test model creation
        usage_record = UsageRecord(
            user_id=123456789,
            username="testuser", 
            command="test",
            model="gpt-4.1"
        )
        print(f"✅ UsageRecord model works: {usage_record.user_id}")
        return True
    except Exception as e:
        print(f"❌ Database import failed: {e}")
        return False


def test_utils_imports():
    """Test utility module imports."""
    print("\nTesting utils imports...")
    try:
        from src.utils import (
            validate_query_length, 
            format_currency, 
            send_long_message,
            reset_status
        )
        
        # Test utility functions
        is_valid, error = validate_query_length("This is a test query")
        print(f"✅ Validator works: valid={is_valid}")
        
        formatted = format_currency(123.456)
        print(f"✅ Formatter works: {formatted}")
        return True
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False


def test_cogs_imports():
    """Test bot cogs imports."""
    print("\nTesting cogs imports...")
    try:
        from src.bot.cogs import AICommands, AdminCommands, SummaryCommands
        
        print(f"✅ Cogs imported successfully")
        print(f"  - AICommands: {AICommands.__name__}")
        print(f"  - AdminCommands: {AdminCommands.__name__}")
        print(f"  - SummaryCommands: {SummaryCommands.__name__}")
        return True
    except Exception as e:
        print(f"❌ Cogs import failed: {e}")
        return False


def test_services_imports():
    """Test services module imports."""
    print("\nTesting services imports...")
    try:
        from src.services.rate_limiter import RateLimiter
        from src.services.context_manager import ContextManager
        
        # Test service creation
        rate_limiter = RateLimiter(max_calls=100, interval=60)
        context_manager = ContextManager.get_instance()
        
        print(f"✅ RateLimiter created: {rate_limiter.max_calls} calls per {rate_limiter.interval}s")
        print(f"✅ ContextManager singleton: {type(context_manager).__name__}")
        return True
    except Exception as e:
        print(f"❌ Services import failed: {e}")
        return False


def main():
    """Run all import tests."""
    print("🧪 Testing SecurePath Refactored Module Imports")
    print("=" * 50)
    
    tests = [
        test_config_imports,
        test_bot_imports,
        test_ai_imports,
        test_database_imports,
        test_utils_imports,
        test_cogs_imports,
        test_services_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All imports successful! The refactoring is working correctly.")
        return 0
    else:
        print("⚠️ Some imports failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)