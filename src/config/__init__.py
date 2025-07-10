"""Configuration module for SecurePath bot."""

from .settings import get_settings, Settings
from .constants import *

__all__ = [
    'get_settings',
    'Settings',
    'DISCORD_MESSAGE_LIMIT',
    'DISCORD_EMBED_LIMIT',
    'OPENAI_MODEL',
    'OPENAI_VISION_MODEL',
    'PERPLEXITY_MODEL',
    'MAX_TOKENS_RESPONSE',
    'MAX_IMAGE_SIZE_MB',
    'SUPPORTED_IMAGE_FORMATS',
]