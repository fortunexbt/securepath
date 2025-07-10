#!/usr/bin/env python3
"""
Direct import test that avoids __init__.py files with dependencies
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_models_only():
    """Test database models directly."""
    print("Testing database models only...")
    try:
        # Import models using direct file access
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "models_simple", 
            "src/database/models_simple.py"
        )
        models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        
        UsageRecord = models.UsageRecord
        dict_to_model = models.dict_to_model
        model_to_dict = models.model_to_dict
        
        # Test model creation
        record = UsageRecord(
            user_id=123,
            username="test",
            command="test",
            model="gpt-4"
        )
        print(f"✅ UsageRecord created: user_id={record.user_id}")
        
        # Test conversion functions
        data = model_to_dict(record)
        print(f"✅ model_to_dict works: {len(data)} fields")
        
        # Test dict to model
        new_record = dict_to_model(data, UsageRecord)
        print(f"✅ dict_to_model works: {new_record.username}")
        
        return True
    except Exception as e:
        print(f"❌ Models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validators_only():
    """Test validators directly."""
    print("\nTesting validators only...")
    try:
        # Import validators using direct file access
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "validators", 
            "src/utils/validators.py"
        )
        validators = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validators)
        
        validate_query_length = validators.validate_query_length
        validate_url = validators.validate_url
        validate_username = validators.validate_username
        validate_model_name = validators.validate_model_name
        
        # Test query validation
        valid, msg = validate_query_length("test query")
        print(f"✅ Query validation: {valid}")
        
        # Test URL validation
        url_valid = validate_url("https://example.com")
        print(f"✅ URL validation: {url_valid}")
        
        # Test username validation
        user_valid = validate_username("testuser123")
        print(f"✅ Username validation: {user_valid}")
        
        # Test model validation
        model_valid = validate_model_name("gpt-4.1")
        print(f"✅ Model validation: {model_valid}")
        
        return True
    except Exception as e:
        print(f"❌ Validators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_formatting_only():
    """Test formatters directly."""
    print("\nTesting formatters only...")
    try:
        # Import formatters using direct file access
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "formatting", 
            "src/utils/formatting.py"
        )
        formatting = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(formatting)
        
        format_currency = formatting.format_currency
        format_large_number = formatting.format_large_number
        format_percentage = formatting.format_percentage
        truncate_with_ellipsis = formatting.truncate_with_ellipsis
        
        # Test currency formatting
        currency = format_currency(1234.56)
        print(f"✅ Currency format: {currency}")
        
        # Test large number formatting
        large = format_large_number(1234567)
        print(f"✅ Large number format: {large}")
        
        # Test percentage formatting
        percent = format_percentage(12.34)
        print(f"✅ Percentage format: {percent}")
        
        # Test truncation
        truncated = truncate_with_ellipsis("This is a long text", 10)
        print(f"✅ Truncation: {truncated}")
        
        return True
    except Exception as e:
        print(f"❌ Formatters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rate_limiter_only():
    """Test rate limiter directly."""
    print("\nTesting rate limiter only...")
    try:
        from src.services.rate_limiter import RateLimiter
        
        # Create rate limiter
        limiter = RateLimiter(max_calls=5, interval=60)
        print(f"✅ RateLimiter created: {limiter.max_calls} calls per {limiter.interval}s")
        
        # Test rate limiting
        user_id = 123456
        for i in range(3):
            limited = limiter.is_rate_limited(user_id)
            print(f"✅ Call {i+1}: limited={limited}")
            
        # Test remaining calls
        remaining = limiter.get_remaining_calls(user_id)
        print(f"✅ Remaining calls: {remaining}")
        
        return True
    except Exception as e:
        print(f"❌ Rate limiter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_settings_only():
    """Test settings directly."""
    print("\nTesting settings only...")
    try:
        from src.config.settings_simple import Settings, get_settings
        from src.config.constants import OPENAI_MODEL, MAX_TOKENS_RESPONSE
        
        # Test settings creation
        settings = Settings()
        print(f"✅ Settings created: prefix={settings.bot_prefix}")
        
        # Test settings from env
        env_settings = Settings.from_env()
        print(f"✅ Settings from env: log_level={env_settings.log_level}")
        
        # Test singleton
        singleton = get_settings()
        print(f"✅ Settings singleton: timeout={singleton.perplexity_timeout}s")
        
        # Test constants
        print(f"✅ Constants: model={OPENAI_MODEL}, max_tokens={MAX_TOKENS_RESPONSE}")
        
        return True
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run direct import tests."""
    print("🧪 Direct Import Tests (No External Dependencies)")
    print("=" * 50)
    
    tests = [
        test_settings_only,
        test_models_only,
        test_validators_only,
        test_formatting_only,
        test_rate_limiter_only,
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
        print("🎉 All direct imports working correctly!")
        return 0
    else:
        print("⚠️ Some imports failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)