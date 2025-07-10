#!/usr/bin/env python3
"""
Basic import test that doesn't require discord.py
"""
import sys
from pathlib import Path

# Add parent directory to path for direct imports
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test configuration imports."""
    print("Testing configuration...")
    try:
        from src.config.settings_simple import Settings, get_settings
        from src.config.constants import DISCORD_MESSAGE_LIMIT
        
        # Test settings creation
        settings = Settings()
        print(f"✅ Settings created with bot prefix: {settings.bot_prefix}")
        print(f"✅ Constants loaded - Message limit: {DISCORD_MESSAGE_LIMIT}")
        return True
    except Exception as e:
        print(f"❌ Config failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_models():
    """Test database models."""
    print("\nTesting database models...")
    try:
        from src.database.models_simple import (
            UsageRecord, UserAnalytics, UserQuery,
            dict_to_model, model_to_dict
        )
        
        # Test model creation
        record = UsageRecord(
            user_id=123,
            username="test",
            command="test",
            model="gpt-4"
        )
        print(f"✅ UsageRecord created: {record.user_id}")
        
        # Test conversion
        data = model_to_dict(record)
        print(f"✅ Model to dict conversion works: {bool(data)}")
        return True
    except Exception as e:
        print(f"❌ Database models failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_services():
    """Test service imports."""
    print("\nTesting services...")
    try:
        from src.services.rate_limiter import RateLimiter
        from src.services.context_manager import ContextManager
        
        # Test rate limiter
        limiter = RateLimiter(max_calls=10, interval=60)
        print(f"✅ RateLimiter created: {limiter.max_calls} calls")
        
        # Test context manager
        ctx_mgr = ContextManager()
        print(f"✅ ContextManager created")
        return True
    except Exception as e:
        print(f"❌ Services failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utilities...")
    try:
        from src.utils.validators import validate_query_length
        from src.utils.formatting import format_currency
        
        # Test validator
        valid, msg = validate_query_length("test query")
        print(f"✅ Validator works: {valid}")
        
        # Test formatter
        formatted = format_currency(123.45)
        print(f"✅ Formatter works: {formatted}")
        return True
    except Exception as e:
        print(f"❌ Utils failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run basic import tests."""
    print("🧪 Basic Import Tests (No Discord Required)")
    print("=" * 50)
    
    tests = [
        test_config,
        test_database_models,
        test_services,
        test_utils,
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
        print("🎉 Basic imports working correctly!")
        return 0
    else:
        print("⚠️ Some imports failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)