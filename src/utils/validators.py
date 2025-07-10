"""Input validation utilities."""
import re
from typing import Optional, Tuple
from urllib.parse import urlparse


def validate_discord_id(discord_id: str) -> bool:
    """
    Validate Discord ID format.
    
    Args:
        discord_id: Discord ID as string
        
    Returns:
        True if valid Discord ID format
    """
    if not discord_id or not discord_id.isdigit():
        return False
        
    # Discord IDs are typically 17-19 digits
    return 15 <= len(discord_id) <= 20


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid URL format
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def validate_image_url(url: str) -> bool:
    """
    Validate image URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid image URL
    """
    if not validate_url(url):
        return False
        
    # Check for common image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
    return any(url.lower().endswith(ext) for ext in image_extensions)


def validate_query_length(query: str, min_length: int = 5, max_length: int = 500) -> Tuple[bool, Optional[str]]:
    """
    Validate query text length.
    
    Args:
        query: Query text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
        
    query_length = len(query.strip())
    
    if query_length < min_length:
        return False, f"Query too short. Minimum {min_length} characters required."
        
    if query_length > max_length:
        return False, f"Query too long. Maximum {max_length} characters allowed."
        
    return True, None


def validate_command_name(command: str) -> bool:
    """
    Validate command name format.
    
    Args:
        command: Command name to validate
        
    Returns:
        True if valid command name
    """
    if not command:
        return False
        
    # Command names should be alphanumeric with underscores/hyphens
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, command)) and len(command) <= 32


def validate_username(username: str) -> bool:
    """
    Validate username format.
    
    Args:
        username: Username to validate
        
    Returns:
        True if valid username
    """
    if not username:
        return False
        
    # Remove discriminator if present
    clean_username = username.split('#')[0]
    
    # Username length check
    if not (2 <= len(clean_username) <= 32):
        return False
        
    # Allow alphanumeric, underscores, dots, and hyphens
    pattern = r'^[a-zA-Z0-9._-]+$'
    return bool(re.match(pattern, clean_username))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed_file"
        
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure reasonable length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = 250 - len(ext) - 1 if ext else 255
        sanitized = name[:max_name_length] + ('.' + ext if ext else '')
        
    return sanitized or "unnamed_file"


def validate_model_name(model: str) -> bool:
    """
    Validate AI model name format.
    
    Args:
        model: Model name to validate
        
    Returns:
        True if valid model name
    """
    valid_models = [
        'gpt-4.1',
        'gpt-4-1106-preview', 
        'gpt-4-vision-preview',
        'gpt-4o',
        'gpt-4o-mini',
        'sonar-pro',
        'llama-3.1-sonar-large-128k-online'
    ]
    
    return model in valid_models


def validate_cost(cost: float) -> bool:
    """
    Validate cost value.
    
    Args:
        cost: Cost value to validate
        
    Returns:
        True if valid cost
    """
    return isinstance(cost, (int, float)) and cost >= 0 and cost < 1000


def validate_token_count(tokens: int) -> bool:
    """
    Validate token count.
    
    Args:
        tokens: Token count to validate
        
    Returns:
        True if valid token count
    """
    return isinstance(tokens, int) and 0 <= tokens <= 1000000


def extract_mentions(text: str) -> list[str]:
    """
    Extract Discord mentions from text.
    
    Args:
        text: Text to extract mentions from
        
    Returns:
        List of user IDs mentioned
    """
    # Discord user mention pattern: <@!?123456789>
    pattern = r'<@!?(\d+)>'
    matches = re.findall(pattern, text)
    return matches


def extract_channel_mentions(text: str) -> list[str]:
    """
    Extract Discord channel mentions from text.
    
    Args:
        text: Text to extract channel mentions from
        
    Returns:
        List of channel IDs mentioned
    """
    # Discord channel mention pattern: <#123456789>
    pattern = r'<#(\d+)>'
    matches = re.findall(pattern, text)
    return matches


def is_spam_like(text: str) -> bool:
    """
    Check if text appears to be spam.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears spam-like
    """
    if not text:
        return False
        
    # Check for excessive repetition
    words = text.lower().split()
    if len(words) > 5:
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
            return True
            
    # Check for excessive caps
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if caps_ratio > 0.7 and len(text) > 20:
        return True
        
    # Check for excessive special characters
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / len(text) > 0.5 and len(text) > 10:
        return True
        
    return False