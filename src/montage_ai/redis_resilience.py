"""
Redis Connection Resilience Module

Provides automatic retry logic and connection health monitoring for
montage-ai Redis connections.

Usage:
    from montage_ai.redis_resilience import ResilientRedis, ResilientQueue
    
    redis_conn = ResilientRedis(host='localhost', port=6379)
    q = ResilientQueue(connection=redis_conn)
"""

import time
import logging
from typing import Optional, Any, Tuple
from redis import Redis
from redis.exceptions import ConnectionError, TimeoutError as RedisTimeoutError
from rq import Queue
from functools import wraps

logger = logging.getLogger(__name__)


class ResilientRedis(Redis):
    """Redis client with automatic retry and connection resilience."""
    
    def __init__(self, *args, **kwargs):
        """
        Initialize ResilientRedis with retry configuration.
        
        Kwargs:
            retry_backoff: exponential backoff multiplier (default: 1)
            max_retries: max retry attempts (default: 3)
            socket_keepalive: enable TCP keepalive (default: True)
        """
        self.retry_backoff = kwargs.pop('retry_backoff', 1)
        self.max_retries = kwargs.pop('max_retries', 3)
        self.socket_keepalive = kwargs.pop('socket_keepalive', True)
        
        # Set socket keepalive
        if self.socket_keepalive:
            kwargs.setdefault('socket_keepalive', True)
            kwargs.setdefault('socket_keepalive_options', {
                1: 1,  # TCP_KEEPIDLE: start after 1 second
                2: 1,  # TCP_KEEPINTVL: interval 1 second
                3: 3,  # TCP_KEEPCNT: 3 probes
            })
        
        super().__init__(*args, **kwargs)
    
    def with_retry(func):
        """Decorator for automatic retry on connection failure."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries):
                try:
                    return func(self, *args, **kwargs)
                except (ConnectionError, RedisTimeoutError) as e:
                    last_exception = e
                    
                    if attempt < self.max_retries - 1:
                        backoff_time = self.retry_backoff * (2 ** attempt)
                        logger.warning(
                            f"Redis {func.__name__} failed (attempt {attempt + 1}/{self.max_retries}), "
                            f"retrying in {backoff_time}s: {e}"
                        )
                        time.sleep(backoff_time)
                    else:
                        logger.error(f"Redis {func.__name__} failed after {self.max_retries} attempts: {e}")
            
            raise last_exception or ConnectionError("Failed to connect to Redis")
        
        return wrapper
    
    def ping(self) -> bool:
        """Ping Redis with retry."""
        try:
            return super().ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    def is_healthy(self) -> Tuple[bool, Optional[str]]:
        """
        Check if Redis is healthy.
        
        Returns:
            (is_healthy, error_message)
        """
        try:
            if not self.ping():
                return False, "Ping failed"
            
            # Check connectivity
            info = self.info()
            if info.get('role') not in ('master', 'slave'):
                return False, f"Unexpected role: {info.get('role')}"
            
            # Check memory usage
            memory_percent = info.get('used_memory', 0) / info.get('maxmemory', 1)
            if memory_percent > 0.9:
                return False, f"Memory usage > 90% ({memory_percent:.0%})"
            
            return True, None
            
        except Exception as e:
            return False, str(e)


class ResilientQueue(Queue):
    """RQ Queue with Redis connection resilience."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with ResilientRedis connection."""
        redis_conn = kwargs.get('connection')
        
        if not isinstance(redis_conn, ResilientRedis):
            logger.warning("ResilientQueue expects ResilientRedis connection")
        
        super().__init__(*args, **kwargs)
    
    def health_check(self) -> Tuple[bool, Optional[str]]:
        """Check if queue is healthy."""
        if hasattr(self.connection, 'is_healthy'):
            return self.connection.is_healthy()
        
        try:
            self.connection.ping()
            return True, None
        except Exception as e:
            return False, str(e)


def create_resilient_redis_connection(
    host: str = 'localhost',
    port: int = 6379,
    retry_backoff: int = 1,
    max_retries: int = 3,
    socket_keepalive: bool = True
) -> ResilientRedis:
    """
    Create a resilient Redis connection.
    
    Args:
        host: Redis host
        port: Redis port
        retry_backoff: Exponential backoff multiplier
        max_retries: Max retry attempts
        socket_keepalive: Enable TCP keepalive
    
    Returns:
        ResilientRedis connection instance
    """
    conn = ResilientRedis(
        host=host,
        port=port,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_backoff=retry_backoff,
        max_retries=max_retries,
        socket_keepalive=socket_keepalive
    )
    
    # Test connection
    is_healthy, error = conn.is_healthy()
    if not is_healthy:
        logger.warning(f"Redis connection may be unhealthy: {error}")
    else:
        logger.info("Redis connection established and healthy")
    
    return conn
