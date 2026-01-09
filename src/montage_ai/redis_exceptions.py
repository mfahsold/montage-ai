"""
Custom Redis exceptions with actionable error messages.

Usage:
    from montage_ai.redis_exceptions import RedisConnectionError, RedisTimeoutError
    
    try:
        redis_conn.ping()
    except RedisConnectionError as e:
        logger.error(e.user_message)
        # Trigger failover or alert ops
"""

from montage_ai.exceptions_custom import MontageException


class RedisConnectionError(MontageException):
    """Redis connection failed."""
    
    def __init__(self, host: str, port: int, original_error: str):
        user_message = (
            f"❌ Cannot connect to Redis at {host}:{port}\n"
            f"   {original_error}"
        )
        
        suggestion = (
            "1. Check Redis is running: `redis-cli ping`\n"
            "2. Verify REDIS_HOST and REDIS_PORT environment variables\n"
            "3. Check network connectivity: `telnet {host} {port}`\n"
            "4. Check Redis logs for errors"
        )
        
        super().__init__(
            user_message=user_message,
            technical_details=f"Connection error: {original_error}",
            suggestion=suggestion
        )


class RedisTimeoutError(MontageException):
    """Redis operation timed out."""
    
    def __init__(self, operation: str, timeout_seconds: float):
        user_message = (
            f"❌ Redis operation '{operation}' timed out after {timeout_seconds}s\n"
            f"   Redis server may be overloaded or network is slow"
        )
        
        suggestion = (
            "1. Increase socket timeout: REDIS_SOCKET_TIMEOUT=10\n"
            "2. Check Redis memory usage: `redis-cli info memory`\n"
            "3. Monitor network latency: `ping <redis-host>`\n"
            "4. Check if Redis is processing many commands"
        )
        
        super().__init__(
            user_message=user_message,
            technical_details=f"Timeout on operation: {operation}",
            suggestion=suggestion
        )


class RedisMemoryError(MontageException):
    """Redis memory threshold exceeded."""
    
    def __init__(self, used_percent: float):
        user_message = (
            f"❌ Redis memory usage critical: {used_percent:.0%} full\n"
            f"   Cannot accept new jobs until memory is freed"
        )
        
        suggestion = (
            "1. Clear old job entries: `rq empty`\n"
            "2. Increase Redis maxmemory: redis.conf maxmemory setting\n"
            "3. Monitor Redis: `redis-cli INFO memory`\n"
            "4. Reduce job retention period"
        )
        
        super().__init__(
            user_message=user_message,
            technical_details=f"Memory: {used_percent:.0%}",
            suggestion=suggestion
        )
