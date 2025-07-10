"""Simple database models using dataclasses."""
from datetime import datetime, date
from decimal import Decimal
from typing import Optional
from dataclasses import dataclass


@dataclass
class UsageRecord:
    """Model for usage tracking records."""
    user_id: int
    username: str
    command: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost: Decimal = Decimal('0')
    timestamp: Optional[datetime] = None
    guild_id: Optional[int] = None
    channel_id: Optional[int] = None
    id: Optional[int] = None


@dataclass
class UserAnalytics:
    """Model for user analytics."""
    user_id: int
    username: str
    first_interaction: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: Decimal = Decimal('0')
    favorite_command: Optional[str] = None
    avg_tokens_per_request: Decimal = Decimal('0')


@dataclass
class DailyUsageSummary:
    """Model for daily usage summaries."""
    date: date
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: Decimal = Decimal('0')
    unique_users: int = 0
    top_command: Optional[str] = None
    id: Optional[int] = None


@dataclass
class UserQuery:
    """Model for user queries."""
    user_id: int
    username: str
    command: str
    query_text: str
    channel_id: Optional[int] = None
    guild_id: Optional[int] = None
    timestamp: Optional[datetime] = None
    response_generated: bool = False
    error_occurred: bool = False
    id: Optional[int] = None


@dataclass
class GlobalStats:
    """Model for global statistics."""
    total_requests: int
    unique_users: int
    total_tokens: int
    total_cost: Decimal
    avg_tokens_per_request: Decimal


@dataclass
class CommandStats:
    """Model for command statistics."""
    command: str
    usage_count: int
    total_cost: Decimal


@dataclass
class ModelCosts:
    """Model for model cost breakdown."""
    model: str
    requests: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    total_cost: Decimal
    avg_cost_per_request: Decimal


@dataclass
class QueryPattern:
    """Model for query patterns."""
    command: str
    total_queries: int
    unique_users: int
    avg_query_length: Decimal


@dataclass
class HourlyActivity:
    """Model for hourly activity stats."""
    hour: int
    query_count: int


def dict_to_model(data_dict: dict, model_class):
    """Convert dictionary to dataclass model instance."""
    if not data_dict:
        return None
        
    # Filter dictionary to only include fields that exist in the model
    import inspect
    model_fields = set(inspect.signature(model_class).parameters.keys())
    filtered_dict = {k: v for k, v in data_dict.items() if k in model_fields}
    
    return model_class(**filtered_dict)


def model_to_dict(model_instance) -> dict:
    """Convert dataclass model instance to dictionary."""
    if hasattr(model_instance, '__dict__'):
        return model_instance.__dict__
    return {}


# Backward compatibility functions
def create_usage_record(**kwargs) -> UsageRecord:
    """Create usage record from keyword arguments."""
    return UsageRecord(**kwargs)


def create_user_analytics(**kwargs) -> UserAnalytics:
    """Create user analytics from keyword arguments."""
    return UserAnalytics(**kwargs)


def create_user_query(**kwargs) -> UserQuery:
    """Create user query from keyword arguments."""
    return UserQuery(**kwargs)