"""Database module with repository pattern."""
import logging
from decimal import Decimal
from typing import Optional, Dict, List, Any

from .connection import DatabaseManager, db_manager as _db_manager
from .models import UsageRecord, UserQuery, UserAnalytics
from .repositories import UsageRepository, AnalyticsRepository

logger = logging.getLogger(__name__)


class UnifiedDatabaseManager:
    """Unified database manager that provides backward compatibility."""
    
    def __init__(self):
        """Initialize unified database manager."""
        self.db_manager = _db_manager
        self.usage_repo: Optional[UsageRepository] = None
        self.analytics_repo: Optional[AnalyticsRepository] = None
        
    @property
    def pool(self):
        """Get database pool for backward compatibility."""
        return self.db_manager.pool
        
    async def connect(self) -> bool:
        """Connect to database and initialize repositories."""
        success = await self.db_manager.connect()
        
        if success:
            self.usage_repo = UsageRepository(self.db_manager)
            self.analytics_repo = AnalyticsRepository(self.db_manager)
            
        return success
        
    async def disconnect(self) -> None:
        """Disconnect from database."""
        await self.db_manager.disconnect()
        self.usage_repo = None
        self.analytics_repo = None
        
    # Backward compatibility methods
    async def log_usage(
        self, 
        user_id: int, 
        username: str, 
        command: str, 
        model: str, 
        input_tokens: int = 0, 
        output_tokens: int = 0, 
        cached_tokens: int = 0,
        cost: float = 0.0, 
        guild_id: Optional[int] = None, 
        channel_id: Optional[int] = None
    ) -> bool:
        """Log usage (backward compatibility method)."""
        if not self.usage_repo:
            return False
            
        record = UsageRecord(
            user_id=user_id,
            username=username,
            command=command,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=Decimal(str(cost)),
            guild_id=guild_id,
            channel_id=channel_id
        )
        
        # Create usage record
        success = await self.usage_repo.create_usage_record(record)
        
        if success:
            # Update user analytics
            await self.analytics_repo.create_or_update_user_analytics(
                user_id=user_id,
                username=username,
                tokens_used=record.total_tokens,
                cost=record.cost
            )
            
            # Update daily summary
            await self.usage_repo.update_daily_summary(record)
            
        return success
        
    async def log_user_query(
        self,
        user_id: int,
        username: str,
        command: str,
        query_text: str,
        channel_id: Optional[int] = None,
        guild_id: Optional[int] = None,
        response_generated: bool = False,
        error_occurred: bool = False
    ) -> bool:
        """Log user query (backward compatibility method)."""
        if not self.analytics_repo:
            return False
            
        query = UserQuery(
            user_id=user_id,
            username=username,
            command=command,
            query_text=query_text,
            channel_id=channel_id,
            guild_id=guild_id,
            response_generated=response_generated,
            error_occurred=error_occurred
        )
        
        return await self.analytics_repo.log_user_query(query)
        
    async def get_global_stats(self) -> Optional[Dict[str, Any]]:
        """Get global statistics (backward compatibility method)."""
        if not self.usage_repo or not self.analytics_repo:
            return None
            
        try:
            # Get overall stats
            overall_stats = await self.usage_repo.get_global_stats()
            if not overall_stats:
                return None
                
            # Get top users
            top_users = await self.analytics_repo.get_top_users(10)
            
            # Get top commands
            top_commands = await self.usage_repo.get_top_commands(10)
            
            # Get daily stats
            daily_stats = await self.usage_repo.get_daily_stats(7)
            
            from .models import model_to_dict
            
            return {
                'overall': model_to_dict(overall_stats),
                'top_users': [model_to_dict(user) for user in top_users],
                'top_commands': [model_to_dict(cmd) for cmd in top_commands],
                'daily_stats': daily_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get global stats: {e}")
            return None
            
    async def get_user_stats(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user statistics (backward compatibility method)."""
        if not self.usage_repo or not self.analytics_repo:
            return None
            
        try:
            # Get user analytics
            user_analytics = await self.analytics_repo.get_user_analytics(user_id)
            if not user_analytics:
                return None
                
            # Get usage stats
            usage_stats = await self.usage_repo.get_user_usage_stats(user_id)
            
            return {
                'user_data': model_to_dict(user_analytics),
                'command_stats': usage_stats.get('commands', []),
                'recent_activity': usage_stats.get('recent_activity', [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return None
            
    async def get_costs_by_model(self) -> Optional[Dict[str, Any]]:
        """Get cost breakdown by model (backward compatibility method)."""
        if not self.usage_repo:
            return None
            
        try:
            model_costs = await self.usage_repo.get_model_costs()
            return {
                'model_costs': [model_to_dict(cost) for cost in model_costs]
            }
            
        except Exception as e:
            logger.error(f"Failed to get model costs: {e}")
            return None
            
    async def get_query_analytics(self) -> Optional[Dict[str, Any]]:
        """Get query analytics (backward compatibility method)."""
        if not self.analytics_repo:
            return None
            
        try:
            # Get popular queries
            popular_queries = await self.analytics_repo.get_popular_queries(20, 7)
            
            # Get command patterns
            command_patterns = await self.analytics_repo.get_query_patterns(7)
            
            # Get hourly activity
            hourly_activity = await self.analytics_repo.get_hourly_activity(7)
            
            return {
                'popular_queries': popular_queries,
                'command_patterns': [model_to_dict(pattern) for pattern in command_patterns],
                'hourly_activity': [model_to_dict(activity) for activity in hourly_activity]
            }
            
        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return None


# Create global instance for backward compatibility
db_manager = UnifiedDatabaseManager()

# Export everything for easy imports
__all__ = [
    'db_manager',
    'DatabaseManager',
    'UsageRepository', 
    'AnalyticsRepository',
    'UsageRecord',
    'UserQuery',
    'UserAnalytics',
]