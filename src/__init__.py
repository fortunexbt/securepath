"""
SecurePath AI Discord Bot - Refactored Package

A modular, well-structured crypto-focused Discord bot with AI-powered analysis.
"""

__version__ = "2.0.0"
__author__ = "SecurePath Team"
__description__ = "AI-powered crypto analysis Discord bot"

# Lazy imports to avoid circular dependencies
def get_settings():
    """Get settings instance."""
    from .config import get_settings as _get_settings
    return _get_settings()

def create_bot():
    """Create bot instance."""
    from .bot import create_bot as _create_bot
    return _create_bot()

__all__ = [
    'get_settings',
    'create_bot',
]