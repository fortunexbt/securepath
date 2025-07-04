import asyncio
import asyncpg
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from urllib.parse import urlparse

logger = logging.getLogger('SecurePathBot.Database')

class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = os.environ.get('DATABASE_URL')
        
    async def connect(self):
        """Initialize database connection pool"""
        if not self.database_url:
            logger.error("DATABASE_URL not found in environment variables")
            return False
            
        try:
            # Parse the database URL for asyncpg
            parsed = urlparse(self.database_url)
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=parsed.hostname,
                port=parsed.port,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:],  # Remove leading slash
                ssl='require',
                min_size=1,
                max_size=3,
                command_timeout=60
            )
            
            logger.info("Database connection pool created successfully")
            await self.init_tables()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def init_tables(self):
        """Create database tables if they don't exist"""
        try:
            async with self.pool.acquire() as conn:
                # Usage tracking table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS usage_tracking (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        username VARCHAR(255),
                        command VARCHAR(50) NOT NULL,
                        model VARCHAR(50) NOT NULL,
                        input_tokens INTEGER DEFAULT 0,
                        output_tokens INTEGER DEFAULT 0,
                        cached_tokens INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        cost DECIMAL(10, 8) DEFAULT 0,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        guild_id BIGINT,
                        channel_id BIGINT
                    )
                ''')
                
                # Create indexes for better performance
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_usage_user_id ON usage_tracking(user_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_tracking(timestamp)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_usage_command ON usage_tracking(command)')
                
                # Daily usage summary table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS daily_usage_summary (
                        id SERIAL PRIMARY KEY,
                        date DATE NOT NULL,
                        total_requests INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        total_cost DECIMAL(10, 6) DEFAULT 0,
                        unique_users INTEGER DEFAULT 0,
                        top_command VARCHAR(50),
                        UNIQUE(date)
                    )
                ''')
                
                # User analytics table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_analytics (
                        user_id BIGINT PRIMARY KEY,
                        username VARCHAR(255),
                        first_interaction TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        last_interaction TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        total_requests INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        total_cost DECIMAL(10, 6) DEFAULT 0,
                        favorite_command VARCHAR(50),
                        avg_tokens_per_request DECIMAL(8, 2) DEFAULT 0
                    )
                ''')
                
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
            raise
    
    async def log_usage(self, user_id: int, username: str, command: str, model: str, 
                       input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0,
                       cost: float = 0.0, guild_id: Optional[int] = None, 
                       channel_id: Optional[int] = None) -> bool:
        """Log a usage event to the database"""
        try:
            async with self.pool.acquire() as conn:
                total_tokens = input_tokens + output_tokens + cached_tokens
                
                # Insert usage record
                await conn.execute('''
                    INSERT INTO usage_tracking 
                    (user_id, username, command, model, input_tokens, output_tokens, 
                     cached_tokens, total_tokens, cost, guild_id, channel_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ''', user_id, username, command, model, input_tokens, output_tokens,
                     cached_tokens, total_tokens, cost, guild_id, channel_id)
                
                # Update user analytics
                await conn.execute('''
                    INSERT INTO user_analytics 
                    (user_id, username, last_interaction, total_requests, total_tokens, total_cost)
                    VALUES ($1, $2, NOW(), 1, $3, $4)
                    ON CONFLICT (user_id) 
                    DO UPDATE SET 
                        username = EXCLUDED.username,
                        last_interaction = NOW(),
                        total_requests = user_analytics.total_requests + 1,
                        total_tokens = user_analytics.total_tokens + EXCLUDED.total_tokens,
                        total_cost = user_analytics.total_cost + EXCLUDED.total_cost
                ''', user_id, username, total_tokens, cost)
                
                # Update daily summary
                today = datetime.now(timezone.utc).date()
                await conn.execute('''
                    INSERT INTO daily_usage_summary (date, total_requests, total_tokens, total_cost, unique_users)
                    VALUES ($1, 1, $2, $3, 1)
                    ON CONFLICT (date)
                    DO UPDATE SET 
                        total_requests = daily_usage_summary.total_requests + 1,
                        total_tokens = daily_usage_summary.total_tokens + EXCLUDED.total_tokens,
                        total_cost = daily_usage_summary.total_cost + EXCLUDED.total_cost
                ''', today, total_tokens, cost)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to log usage: {e}")
            return False
    
    async def get_user_stats(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive stats for a specific user"""
        try:
            async with self.pool.acquire() as conn:
                # Get user analytics
                user_data = await conn.fetchrow('''
                    SELECT * FROM user_analytics WHERE user_id = $1
                ''', user_id)
                
                if not user_data:
                    return None
                
                # Get command usage breakdown
                command_stats = await conn.fetch('''
                    SELECT command, COUNT(*) as count, SUM(total_tokens) as tokens, SUM(cost) as cost
                    FROM usage_tracking 
                    WHERE user_id = $1 
                    GROUP BY command 
                    ORDER BY count DESC
                ''', user_id)
                
                # Get recent activity (last 7 days)
                recent_activity = await conn.fetch('''
                    SELECT DATE(timestamp) as date, COUNT(*) as requests, SUM(cost) as daily_cost
                    FROM usage_tracking 
                    WHERE user_id = $1 AND timestamp >= NOW() - INTERVAL '7 days'
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                ''', user_id)
                
                return {
                    'user_data': dict(user_data),
                    'command_stats': [dict(row) for row in command_stats],
                    'recent_activity': [dict(row) for row in recent_activity]
                }
                
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return None
    
    async def get_global_stats(self) -> Optional[Dict[str, Any]]:
        """Get global usage statistics"""
        try:
            async with self.pool.acquire() as conn:
                # Overall stats
                overall = await conn.fetchrow('''
                    SELECT 
                        COUNT(*) as total_requests,
                        COUNT(DISTINCT user_id) as unique_users,
                        SUM(total_tokens) as total_tokens,
                        SUM(cost) as total_cost,
                        AVG(total_tokens) as avg_tokens_per_request
                    FROM usage_tracking
                ''')
                
                # Top users
                top_users = await conn.fetch('''
                    SELECT username, total_requests, total_cost, total_tokens
                    FROM user_analytics 
                    ORDER BY total_requests DESC 
                    LIMIT 10
                ''')
                
                # Most used commands
                top_commands = await conn.fetch('''
                    SELECT command, COUNT(*) as usage_count, SUM(cost) as total_cost
                    FROM usage_tracking 
                    GROUP BY command 
                    ORDER BY usage_count DESC
                ''')
                
                # Daily stats for last 7 days
                daily_stats = await conn.fetch('''
                    SELECT date, total_requests, total_cost, unique_users
                    FROM daily_usage_summary 
                    WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                    ORDER BY date DESC
                ''')
                
                return {
                    'overall': dict(overall),
                    'top_users': [dict(row) for row in top_users],
                    'top_commands': [dict(row) for row in top_commands],
                    'daily_stats': [dict(row) for row in daily_stats]
                }
                
        except Exception as e:
            logger.error(f"Failed to get global stats: {e}")
            return None
    
    async def get_costs_by_model(self) -> Optional[Dict[str, Any]]:
        """Get cost breakdown by model"""
        try:
            async with self.pool.acquire() as conn:
                model_costs = await conn.fetch('''
                    SELECT 
                        model,
                        COUNT(*) as requests,
                        SUM(input_tokens) as input_tokens,
                        SUM(output_tokens) as output_tokens,
                        SUM(cached_tokens) as cached_tokens,
                        SUM(cost) as total_cost,
                        AVG(cost) as avg_cost_per_request
                    FROM usage_tracking 
                    GROUP BY model 
                    ORDER BY total_cost DESC
                ''')
                
                return {
                    'model_costs': [dict(row) for row in model_costs]
                }
                
        except Exception as e:
            logger.error(f"Failed to get model costs: {e}")
            return None

# Global database manager instance
db_manager = DatabaseManager()