"""Text formatting utilities for consistent output."""
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, Dict, Any


def format_currency(amount: float, currency: str = "USD", decimals: int = 4) -> str:
    """
    Format currency amount with appropriate symbol.
    
    Args:
        amount: Amount to format
        currency: Currency code
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    if currency.upper() == "USD":
        if amount >= 1:
            return f"${amount:,.{min(decimals, 2)}f}"
        else:
            return f"${amount:.{decimals}f}"
    else:
        return f"{amount:.{decimals}f} {currency.upper()}"


def format_percentage(value: float, decimals: int = 2, show_sign: bool = True) -> str:
    """
    Format percentage with optional sign.
    
    Args:
        value: Percentage value
        decimals: Number of decimal places
        show_sign: Whether to show + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    if show_sign:
        return f"{value:+.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}%"


def format_large_number(value: int, decimals: int = 1) -> str:
    """
    Format large numbers with K, M, B, T suffixes.
    
    Args:
        value: Number to format
        decimals: Number of decimal places for abbreviated values
        
    Returns:
        Formatted number string
    """
    if abs(value) >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.{decimals}f}T"
    elif abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:,}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def format_timestamp(dt: datetime, format_type: str = "relative") -> str:
    """
    Format timestamp in various formats.
    
    Args:
        dt: Datetime to format
        format_type: Type of formatting (relative, short, long, iso)
        
    Returns:
        Formatted timestamp string
    """
    if format_type == "iso":
        return dt.isoformat()
    elif format_type == "short":
        return dt.strftime("%Y-%m-%d %H:%M")
    elif format_type == "long":
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    elif format_type == "relative":
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
            
        diff = now - dt
        
        if diff.total_seconds() < 60:
            return "just now"
        elif diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        elif diff.days < 7:
            return f"{diff.days}d ago"
        else:
            return dt.strftime("%Y-%m-%d")
    else:
        return str(dt)


def clean_text_for_discord(text: str, max_length: int = 2000) -> str:
    """
    Clean text for Discord message format.
    
    Args:
        text: Text to clean
        max_length: Maximum length for Discord messages
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
        
    # Remove or replace problematic characters
    cleaned = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length - 3] + "..."
        
    return cleaned.strip()


def escape_markdown(text: str) -> str:
    """
    Escape Discord markdown characters.
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text
    """
    # Discord markdown characters that need escaping
    markdown_chars = ['*', '_', '`', '~', '\\', '|', '>', '#']
    
    for char in markdown_chars:
        text = text.replace(char, f'\\{char}')
        
    return text


def format_code_block(code: str, language: str = "") -> str:
    """
    Format code in Discord code block.
    
    Args:
        code: Code to format
        language: Programming language for syntax highlighting
        
    Returns:
        Formatted code block
    """
    return f"```{language}\n{code}\n```"


def format_inline_code(code: str) -> str:
    """
    Format inline code for Discord.
    
    Args:
        code: Code to format
        
    Returns:
        Formatted inline code
    """
    return f"`{code}`"


def format_usage_stats(stats: Dict[str, Any]) -> str:
    """
    Format usage statistics for display.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        Formatted statistics string
    """
    lines = []
    
    if 'total_requests' in stats:
        lines.append(f"**Requests:** {format_large_number(stats['total_requests'])}")
        
    if 'total_tokens' in stats:
        lines.append(f"**Tokens:** {format_large_number(stats['total_tokens'])}")
        
    if 'total_cost' in stats:
        cost = float(stats['total_cost'])
        lines.append(f"**Cost:** {format_currency(cost)}")
        
    if 'unique_users' in stats:
        lines.append(f"**Users:** {format_large_number(stats['unique_users'])}")
        
    return "\n".join(lines)


def format_model_name(model: str) -> str:
    """
    Format AI model name for display.
    
    Args:
        model: Model name to format
        
    Returns:
        Formatted model name
    """
    model_display_names = {
        'gpt-4.1': 'GPT-4.1',
        'gpt-4-1106-preview': 'GPT-4 Turbo',
        'gpt-4-vision-preview': 'GPT-4 Vision',
        'gpt-4o': 'GPT-4o',
        'gpt-4o-mini': 'GPT-4o Mini',
        'sonar-pro': 'Perplexity Sonar-Pro',
        'llama-3.1-sonar-large-128k-online': 'Perplexity Sonar-Pro',
    }
    
    return model_display_names.get(model, model.title())


def format_error_message(error: str, max_length: int = 1000) -> str:
    """
    Format error message for user display.
    
    Args:
        error: Error message
        max_length: Maximum length for error message
        
    Returns:
        Formatted error message
    """
    if not error:
        return "An unknown error occurred."
        
    # Clean up technical details
    cleaned = str(error)
    
    # Remove file paths and line numbers
    cleaned = re.sub(r'File ".*?", line \d+', '', cleaned)
    
    # Remove module paths
    cleaned = re.sub(r'\w+\.\w+\.\w+:', '', cleaned)
    
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length - 3] + "..."
        
    return cleaned.strip() or "An error occurred while processing your request."


def format_list_items(items: List[str], max_items: int = 10) -> str:
    """
    Format list of items for Discord display.
    
    Args:
        items: List of items to format
        max_items: Maximum number of items to show
        
    Returns:
        Formatted list string
    """
    if not items:
        return "No items to display."
        
    # Limit number of items
    display_items = items[:max_items]
    
    # Format as bullet points
    formatted = "\n".join(f"• {item}" for item in display_items)
    
    # Add "and X more" if truncated
    if len(items) > max_items:
        remaining = len(items) - max_items
        formatted += f"\n*...and {remaining} more*"
        
    return formatted


def format_embed_field_value(value: Any, max_length: int = 1024) -> str:
    """
    Format value for Discord embed field.
    
    Args:
        value: Value to format
        max_length: Maximum length for embed field value
        
    Returns:
        Formatted value string
    """
    if value is None:
        return "N/A"
        
    str_value = str(value)
    
    if len(str_value) > max_length:
        str_value = str_value[:max_length - 3] + "..."
        
    return str_value or "N/A"


def truncate_with_ellipsis(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text with ellipsis if too long.
    
    Args:
        text: Text to potentially truncate
        max_length: Maximum allowed length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text with suffix if needed
    """
    if not text or len(text) <= max_length:
        return text
        
    return text[:max_length - len(suffix)] + suffix