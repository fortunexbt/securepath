"""Repository for user analytics data."""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any

from ..connection import DatabaseManager
from ..models import UserAnalytics, UserQuery, QueryPattern, HourlyActivity, dict_to_model

logger = logging.getLogger(__name__)


class AnalyticsRepository:
    """Repository for managing user analytics and query data."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize analytics repository."""
        self.db = db_manager
        
    async def create_or_update_user_analytics(
        self, 
        user_id: int, 
        username: str,
        tokens_used: int = 0,
        cost: Decimal = Decimal('0')
    ) -> bool:
        """
        Create or update user analytics record.
        
        Args:
            user_id: Discord user ID
            username: User's display name
            tokens_used: Number of tokens used in this interaction
            cost: Cost of this interaction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.db.execute('''
                INSERT INTO user_analytics 
                (user_id, username, last_interaction, total_requests, total_tokens, total_cost)
                VALUES ($1, $2, NOW(), 1, $3, $4)
                ON CONFLICT (user_id) 
                DO UPDATE SET 
                    username = EXCLUDED.username,
                    last_interaction = NOW(),
                    total_requests = user_analytics.total_requests + 1,
                    total_tokens = user_analytics.total_tokens + EXCLUDED.total_tokens,
                    total_cost = user_analytics.total_cost + EXCLUDED.total_cost,
                    avg_tokens_per_request = CASE 
                        WHEN user_analytics.total_requests > 0 
                        THEN (user_analytics.total_tokens + EXCLUDED.total_tokens) / (user_analytics.total_requests + 1)
                        ELSE EXCLUDED.total_tokens
                    END
            ''', user_id, username, tokens_used, cost)
            
            logger.debug(f"Updated analytics for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user analytics: {e}")
            return False
            
    async def get_user_analytics(self, user_id: int) -> Optional[UserAnalytics]:
        """
        Get analytics for a specific user.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            User analytics or None if not found
        """
        try:
            row = await self.db.fetch_one('''
                SELECT * FROM user_analytics WHERE user_id = $1
            ''', user_id)
            
            if row:
                return dict_to_model(dict(row), UserAnalytics)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user analytics: {e}")
            return None
            
    async def get_top_users(self, limit: int = 10) -> List[UserAnalytics]:
        """
        Get top users by total requests.
        
        Args:
            limit: Maximum number of users to return
            
        Returns:
            List of user analytics sorted by total requests
        """
        try:
            rows = await self.db.fetch_many('''
                SELECT * FROM user_analytics 
                ORDER BY total_requests DESC 
                LIMIT $1
            ''', limit)
            
            return [dict_to_model(dict(row), UserAnalytics) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get top users: {e}")
            return []
            
    async def log_user_query(self, query: UserQuery) -> bool:
        """
        Log a user query to the database.
        
        Args:
            query: User query to log
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.db.execute('''
                INSERT INTO user_queries 
                (user_id, username, command, query_text, channel_id, guild_id, 
                 response_generated, error_occurred)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', query.user_id, query.username, query.command, query.query_text,
                query.channel_id, query.guild_id, query.response_generated, 
                query.error_occurred)
                
            logger.debug(f"Logged query for user {query.user_id}, command {query.command}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log user query: {e}")
            return False
            
    async def get_query_patterns(self, days: int = 7) -> List[QueryPattern]:
        """
        Get query patterns by command for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of query patterns
        """
        try:
            rows = await self.db.fetch_many('''
                SELECT 
                    command,
                    COUNT(*) as total_queries,
                    COUNT(DISTINCT user_id) as unique_users,
                    COALESCE(AVG(LENGTH(query_text)), 0) as avg_query_length
                FROM user_queries
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY command
                ORDER BY total_queries DESC
            ''' % days)
            
            return [dict_to_model(dict(row), QueryPattern) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get query patterns: {e}")
            return []
            
    async def get_hourly_activity(self, days: int = 7) -> List[HourlyActivity]:
        """
        Get hourly activity patterns for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of hourly activity data
        """
        try:
            rows = await self.db.fetch_many('''
                SELECT 
                    EXTRACT(HOUR FROM timestamp) as hour,
                    COUNT(*) as query_count
                FROM user_queries
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY EXTRACT(HOUR FROM timestamp)
                ORDER BY query_count DESC
            ''' % days)
            
            return [HourlyActivity(hour=int(row['hour']), query_count=row['query_count']) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get hourly activity: {e}")
            return []
            
    async def get_popular_queries(self, limit: int = 20, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get most popular queries for the last N days.
        
        Args:
            limit: Maximum number of queries to return
            days: Number of days to look back
            
        Returns:
            List of popular queries with metadata
        """
        try:
            rows = await self.db.fetch_many('''
                SELECT 
                    query_text,
                    command,
                    COUNT(*) as frequency,
                    username,
                    MAX(timestamp) as last_used
                FROM user_queries 
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY query_text, command, username
                ORDER BY frequency DESC
                LIMIT $1
            ''' % days, limit)
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get popular queries: {e}")
            return []
            
    async def get_user_query_history(
        self, 
        user_id: int, 
        limit: int = 50
    ) -> List[UserQuery]:
        """
        Get query history for a specific user.
        
        Args:
            user_id: Discord user ID
            limit: Maximum number of queries to return
            
        Returns:
            List of user queries
        """
        try:
            rows = await self.db.fetch_many('''
                SELECT * FROM user_queries 
                WHERE user_id = $1 
                ORDER BY timestamp DESC 
                LIMIT $2
            ''', user_id, limit)
            
            return [dict_to_model(dict(row), UserQuery) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get user query history: {e}")
            return []
            
    async def update_favorite_command(self, user_id: int) -> bool:
        """
        Update a user's favorite command based on usage patterns.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get most used command for this user
            row = await self.db.fetch_one('''
                SELECT command, COUNT(*) as usage_count
                FROM user_queries 
                WHERE user_id = $1 
                GROUP BY command 
                ORDER BY usage_count DESC 
                LIMIT 1
            ''', user_id)
            
            if row:
                await self.db.execute('''
                    UPDATE user_analytics 
                    SET favorite_command = $1 
                    WHERE user_id = $2
                ''', row['command'], user_id)
                
                logger.debug(f"Updated favorite command for user {user_id}: {row['command']}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to update favorite command: {e}")
            return False
            
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive analytics summary.
        
        Returns:
            Dictionary with various analytics metrics
        """
        try:
            # Total users
            total_users = await self.db.fetch_value('''
                SELECT COUNT(*) FROM user_analytics
            ''')
            
            # Active users (last 7 days)
            active_users = await self.db.fetch_value('''
                SELECT COUNT(*) FROM user_analytics 
                WHERE last_interaction >= NOW() - INTERVAL '7 days'
            ''')
            
            # Most active user
            most_active = await self.db.fetch_one('''
                SELECT username, total_requests 
                FROM user_analytics 
                ORDER BY total_requests DESC 
                LIMIT 1
            ''')
            
            # Average requests per user
            avg_requests = await self.db.fetch_value('''
                SELECT COALESCE(AVG(total_requests), 0) 
                FROM user_analytics
            ''')
            
            return {
                'total_users': total_users or 0,
                'active_users_7d': active_users or 0,
                'most_active_user': dict(most_active) if most_active else None,
                'avg_requests_per_user': float(avg_requests or 0),
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}