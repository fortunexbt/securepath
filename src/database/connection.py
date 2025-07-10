"""Database connection management."""
import asyncio
import logging
from typing import Optional
from urllib.parse import urlparse

import asyncpg

from ..config.settings import get_settings
from ..config.constants import DB_CONNECTION_TIMEOUT, DB_POOL_MIN_SIZE, DB_POOL_MAX_SIZE

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and connection pooling."""
    
    def __init__(self):
        """Initialize database manager."""
        self.pool: Optional[asyncpg.Pool] = None
        self.settings = get_settings()
        self._connected = False
        
    async def connect(self) -> bool:
        """
        Initialize database connection pool.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self._connected:
            return True
            
        if not self.settings.database_url:
            logger.error("DATABASE_URL not configured")
            return False
            
        try:
            # Parse the database URL for asyncpg
            parsed = urlparse(self.settings.database_url)
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=parsed.hostname,
                port=parsed.port,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:],  # Remove leading slash
                ssl='require',
                min_size=DB_POOL_MIN_SIZE,
                max_size=DB_POOL_MAX_SIZE,
                command_timeout=DB_CONNECTION_TIMEOUT
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
                
            self._connected = True
            logger.info("Database connection pool created successfully")
            
            # Initialize tables
            await self._init_tables()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._connected = False
            return False
            
    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._connected = False
            logger.info("Database connection pool closed")
            
    async def get_connection(self):
        """
        Get a database connection from the pool.
        
        Returns:
            Database connection context manager
        """
        if not self._connected or not self.pool:
            raise RuntimeError("Database not connected")
        return self.pool.acquire()
        
    async def execute(self, query: str, *args) -> None:
        """
        Execute a query without returning results.
        
        Args:
            query: SQL query
            *args: Query parameters
        """
        async with self.get_connection() as conn:
            await conn.execute(query, *args)
            
    async def fetch_one(self, query: str, *args):
        """
        Fetch a single row.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            Single row or None
        """
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
            
    async def fetch_many(self, query: str, *args):
        """
        Fetch multiple rows.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            List of rows
        """
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
            
    async def fetch_value(self, query: str, *args):
        """
        Fetch a single value.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            Single value or None
        """
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
            
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self.pool is not None
        
    async def _init_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            async with self.get_connection() as conn:
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
                
                # User queries table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_queries (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        username VARCHAR(255),
                        command VARCHAR(50) NOT NULL,
                        query_text TEXT NOT NULL,
                        channel_id BIGINT,
                        guild_id BIGINT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        response_generated BOOLEAN DEFAULT FALSE,
                        error_occurred BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # Create indexes for queries table
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_queries_user_id ON user_queries(user_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON user_queries(timestamp)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_queries_command ON user_queries(command)')
                
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()