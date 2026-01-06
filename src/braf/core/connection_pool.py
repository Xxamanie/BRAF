"""
Connection pooling utilities for BRAF.

This module provides connection pool management for database and Redis connections
with automatic reconnection and health monitoring.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from braf.core.config import BRAFConfig
from braf.core.database import DatabaseManager, get_database

logger = logging.getLogger(__name__)


class RedisConnectionPool:
    """Redis connection pool manager with health monitoring."""
    
    def __init__(self, redis_url: str, max_connections: int = 20):
        """
        Initialize Redis connection pool.
        
        Args:
            redis_url: Redis connection URL
            max_connections: Maximum number of connections in pool
        """
        self.redis_url = redis_url
        self.max_connections = max_connections
        self._pool: Optional[redis.ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self._pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            
            self._redis = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._redis.ping()
            logger.info(f"Redis connection pool initialized: {self.redis_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            raise
    
    async def get_redis(self) -> redis.Redis:
        """
        Get Redis client from pool.
        
        Returns:
            Redis client instance
            
        Raises:
            RuntimeError: If pool is not initialized
        """
        if self._redis is None:
            raise RuntimeError("Redis pool not initialized. Call initialize() first.")
        return self._redis
    
    async def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if self._redis is None:
                return False
            
            await self._redis.ping()
            return True
            
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._redis:
            await self._redis.close()
            self._redis = None
        
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        
        logger.info("Redis connection pool closed")


class ConnectionManager:
    """Centralized connection manager for all BRAF services."""
    
    def __init__(self, config: BRAFConfig):
        """
        Initialize connection manager with configuration.
        
        Args:
            config: BRAF configuration
        """
        self.config = config
        self.db_manager: Optional[DatabaseManager] = None
        self.redis_pool: Optional[RedisConnectionPool] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all connection pools."""
        if self._initialized:
            logger.warning("Connection manager already initialized")
            return
        
        try:
            # Initialize database manager
            self.db_manager = DatabaseManager(
                database_url=self.config.database.url,
                echo=self.config.database.echo
            )
            
            # Initialize Redis connection pool
            self.redis_pool = RedisConnectionPool(
                redis_url=self.config.redis.url,
                max_connections=self.config.redis.max_connections
            )
            await self.redis_pool.initialize()
            
            self._initialized = True
            logger.info("Connection manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection manager: {e}")
            await self.close()
            raise
    
    @asynccontextmanager
    async def get_db_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session context manager.
        
        Yields:
            Database session
        """
        if not self._initialized or self.db_manager is None:
            raise RuntimeError("Connection manager not initialized")
        
        async with self.db_manager.async_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_redis(self) -> redis.Redis:
        """
        Get Redis client.
        
        Returns:
            Redis client instance
        """
        if not self._initialized or self.redis_pool is None:
            raise RuntimeError("Connection manager not initialized")
        
        return await self.redis_pool.get_redis()
    
    async def health_check(self) -> dict:
        """
        Perform health check on all connections.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            "database": False,
            "redis": False,
            "overall": False
        }
        
        # Check database health
        try:
            if self.db_manager:
                async with self.get_db_session() as session:
                    await session.execute("SELECT 1")
                health_status["database"] = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
        
        # Check Redis health
        if self.redis_pool:
            health_status["redis"] = await self.redis_pool.health_check()
        
        # Overall health
        health_status["overall"] = all([
            health_status["database"],
            health_status["redis"]
        ])
        
        return health_status
    
    async def close(self) -> None:
        """Close all connections."""
        if self.redis_pool:
            await self.redis_pool.close()
            self.redis_pool = None
        
        if self.db_manager:
            await self.db_manager.close()
            self.db_manager = None
        
        self._initialized = False
        logger.info("Connection manager closed")


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


async def init_connections(config: BRAFConfig) -> ConnectionManager:
    """
    Initialize global connection manager.
    
    Args:
        config: BRAF configuration
        
    Returns:
        Connection manager instance
    """
    global _connection_manager
    
    _connection_manager = ConnectionManager(config)
    await _connection_manager.initialize()
    
    return _connection_manager


def get_connection_manager() -> ConnectionManager:
    """
    Get global connection manager instance.
    
    Returns:
        Connection manager instance
        
    Raises:
        RuntimeError: If connection manager is not initialized
    """
    if _connection_manager is None:
        raise RuntimeError("Connection manager not initialized. Call init_connections() first.")
    return _connection_manager


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Convenience function to get database session.
    
    Yields:
        Database session
    """
    manager = get_connection_manager()
    async with manager.get_db_session() as session:
        yield session


async def get_redis() -> redis.Redis:
    """
    Convenience function to get Redis client.
    
    Returns:
        Redis client instance
    """
    manager = get_connection_manager()
    return await manager.get_redis()
