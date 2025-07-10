"""Repository for usage tracking data."""
import logging
from datetime import datetime, timezone, date
from decimal import Decimal
from typing import List, Optional, Dict, Any

from ..connection import DatabaseManager
from ..models import UsageRecord, GlobalStats, CommandStats, ModelCosts, dict_to_model

logger = logging.getLogger(__name__)


class UsageRepository:
    """Repository for managing usage tracking data."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize usage repository."""
        self.db = db_manager
        
    async def create_usage_record(self, record: UsageRecord) -> bool:
        """
        Create a new usage record.
        
        Args:
            record: Usage record to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate total tokens if not provided
            if record.total_tokens == 0:
                record.total_tokens = record.input_tokens + record.output_tokens + record.cached_tokens
                
            await self.db.execute('''
                INSERT INTO usage_tracking 
                (user_id, username, command, model, input_tokens, output_tokens, 
                 cached_tokens, total_tokens, cost, guild_id, channel_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ''', record.user_id, record.username, record.command, record.model,
                record.input_tokens, record.output_tokens, record.cached_tokens,
                record.total_tokens, record.cost, record.guild_id, record.channel_id)
                
            logger.debug(f"Created usage record for user {record.user_id}, command {record.command}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create usage record: {e}")
            return False
            
    async def get_global_stats(self) -> Optional[GlobalStats]:
        """
        Get global usage statistics.
        
        Returns:
            Global statistics or None if error
        """
        try:
            row = await self.db.fetch_one('''
                SELECT 
                    COUNT(*) as total_requests,
                    COUNT(DISTINCT user_id) as unique_users,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost), 0) as total_cost,
                    COALESCE(AVG(total_tokens), 0) as avg_tokens_per_request
                FROM usage_tracking
            ''')
            
            if row:
                return dict_to_model(dict(row), GlobalStats)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get global stats: {e}")
            return None
            
    async def get_top_commands(self, limit: int = 10) -> List[CommandStats]:
        """
        Get most used commands.
        
        Args:
            limit: Maximum number of commands to return
            
        Returns:
            List of command statistics
        """
        try:
            rows = await self.db.fetch_many('''
                SELECT 
                    command, 
                    COUNT(*) as usage_count, 
                    COALESCE(SUM(cost), 0) as total_cost
                FROM usage_tracking 
                GROUP BY command 
                ORDER BY usage_count DESC
                LIMIT $1
            ''', limit)
            
            return [dict_to_model(dict(row), CommandStats) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get top commands: {e}")
            return []
            
    async def get_model_costs(self) -> List[ModelCosts]:
        """
        Get cost breakdown by model.
        
        Returns:
            List of model cost statistics
        """
        try:
            rows = await self.db.fetch_many('''
                SELECT 
                    model,
                    COUNT(*) as requests,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    COALESCE(SUM(cached_tokens), 0) as cached_tokens,
                    COALESCE(SUM(cost), 0) as total_cost,
                    COALESCE(AVG(cost), 0) as avg_cost_per_request
                FROM usage_tracking 
                GROUP BY model 
                ORDER BY total_cost DESC
            ''')
            
            return [dict_to_model(dict(row), ModelCosts) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get model costs: {e}")
            return []
            
    async def get_user_usage_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Get usage statistics for a specific user.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            Dictionary with user usage statistics
        """
        try:
            # Get overall user stats
            overall = await self.db.fetch_one('''
                SELECT 
                    COUNT(*) as total_requests,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost), 0) as total_cost,
                    COALESCE(AVG(total_tokens), 0) as avg_tokens_per_request
                FROM usage_tracking 
                WHERE user_id = $1
            ''', user_id)
            
            # Get command breakdown
            commands = await self.db.fetch_many('''
                SELECT 
                    command, 
                    COUNT(*) as count, 
                    COALESCE(SUM(total_tokens), 0) as tokens, 
                    COALESCE(SUM(cost), 0) as cost
                FROM usage_tracking 
                WHERE user_id = $1 
                GROUP BY command 
                ORDER BY count DESC
            ''', user_id)
            
            # Get recent activity (last 7 days)
            recent = await self.db.fetch_many('''
                SELECT 
                    DATE(timestamp) as date, 
                    COUNT(*) as requests, 
                    COALESCE(SUM(cost), 0) as daily_cost
                FROM usage_tracking 
                WHERE user_id = $1 AND timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            ''', user_id)
            
            return {
                'overall': dict(overall) if overall else {},
                'commands': [dict(row) for row in commands],
                'recent_activity': [dict(row) for row in recent]
            }
            
        except Exception as e:
            logger.error(f"Failed to get user usage stats: {e}")
            return {}
            
    async def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get daily usage statistics.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of daily statistics
        """
        try:
            rows = await self.db.fetch_many('''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_requests,
                    COUNT(DISTINCT user_id) as unique_users,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost), 0) as total_cost
                FROM usage_tracking 
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            ''' % days)
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get daily stats: {e}")
            return []
            
    async def update_daily_summary(self, record: UsageRecord) -> bool:
        """
        Update daily usage summary for a usage record.
        
        Args:
            record: Usage record to process
            
        Returns:
            True if successful, False otherwise
        """
        try:
            today = datetime.now(timezone.utc).date()
            
            await self.db.execute('''
                INSERT INTO daily_usage_summary 
                (date, total_requests, total_tokens, total_cost, unique_users)
                VALUES ($1, 1, $2, $3, 1)
                ON CONFLICT (date)
                DO UPDATE SET 
                    total_requests = daily_usage_summary.total_requests + 1,
                    total_tokens = daily_usage_summary.total_tokens + EXCLUDED.total_tokens,
                    total_cost = daily_usage_summary.total_cost + EXCLUDED.total_cost
            ''', today, record.total_tokens, record.cost)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update daily summary: {e}")
            return False