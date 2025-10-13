"""Redis caching layer for KisanCredit API."""

import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional, Union
from datetime import timedelta
import time
from functools import wraps

from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)


class RedisCache:
    """Redis cache manager with async support."""

    def __init__(self, redis_url: str = None):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL (uses config if None)
        """
        self.redis_url = redis_url or settings.redis_url
        self.client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self):
        """Connect to Redis."""
        if self._connected:
            logger.warning("Redis already connected")
            return

        try:
            logger.info("Connecting to Redis...")

            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                max_connections=50,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

            # Test connection
            await self.client.ping()

            self._connected = True
            logger.info("Redis connected", redis_url=self._mask_url(self.redis_url))

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Redis."""
        if not self._connected:
            return

        try:
            logger.info("Disconnecting from Redis...")

            if self.client:
                await self.client.close()

            self._connected = False
            logger.info("Redis disconnected")

        except Exception as e:
            logger.error(f"Redis disconnection failed: {e}")
            raise

    async def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key
            deserialize: Whether to deserialize the value (default: True)

        Returns:
            Cached value or None if not found
        """
        if not self._connected:
            await self.connect()

        try:
            start_time = time.time()

            value = await self.client.get(key)

            if value is None:
                logger.debug("Cache miss", key=key)
                return None

            # Deserialize if needed
            if deserialize:
                try:
                    value = pickle.loads(value)
                except:
                    # Fallback to JSON
                    value = json.loads(value)

            latency = (time.time() - start_time) * 1000
            logger.debug("Cache hit", key=key, latency_ms=round(latency, 2))

            return value

        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        serialize: bool = True
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 1 hour)
            serialize: Whether to serialize the value (default: True)

        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            await self.connect()

        try:
            # Serialize if needed
            if serialize:
                try:
                    value = pickle.dumps(value)
                except:
                    # Fallback to JSON
                    value = json.dumps(value)

            await self.client.setex(key, ttl, value)

            logger.debug("Cache set", key=key, ttl=ttl)
            return True

        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False otherwise
        """
        if not self._connected:
            await self.connect()

        try:
            result = await self.client.delete(key)
            logger.debug("Cache delete", key=key, deleted=result > 0)
            return result > 0

        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists, False otherwise
        """
        if not self._connected:
            await self.connect()

        try:
            result = await self.client.exists(key)
            return result > 0

        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {e}")
            return False

    async def clear(self, pattern: str = "*") -> int:
        """Clear cache keys matching pattern.

        Args:
            pattern: Key pattern (default: all keys)

        Returns:
            Number of keys deleted
        """
        if not self._connected:
            await self.connect()

        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Cleared {deleted} cache keys with pattern: {pattern}")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0

    async def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self._connected:
            return {"connected": False}

        try:
            info = await self.client.info("stats")
            memory = await self.client.info("memory")

            # Calculate hit rate
            hits = int(info.get("keyspace_hits", 0))
            misses = int(info.get("keyspace_misses", 0))
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0

            return {
                "connected": True,
                "total_keys": await self.client.dbsize(),
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hit_rate, 2),
                "memory_used_mb": round(int(memory.get("used_memory", 0)) / 1024 / 1024, 2),
                "memory_peak_mb": round(int(memory.get("used_memory_peak", 0)) / 1024 / 1024, 2),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"connected": True, "error": str(e)}

    async def health_check(self) -> dict:
        """Check Redis health.

        Returns:
            Dictionary with health status
        """
        if not self._connected:
            return {"is_healthy": False, "error": "Not connected"}

        try:
            start_time = time.time()
            await self.client.ping()
            latency = (time.time() - start_time) * 1000

            return {
                "is_healthy": True,
                "latency_ms": round(latency, 2),
                "connected": True
            }

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "is_healthy": False,
                "error": str(e)
            }

    def _mask_url(self, url: str) -> str:
        """Mask password in Redis URL for logging."""
        if '@' in url:
            parts = url.split('@')
            if ':' in parts[0]:
                protocol_user = parts[0].rsplit(':', 1)[0]
                return f"{protocol_user}:****@{parts[1]}"
        return url


# Global cache instance
cache = RedisCache()


# Cache decorator
def cached(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key

    Usage:
        @cached(ttl=300, key_prefix="user")
        async def get_user(user_id: str):
            return await fetch_user(user_id)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(filter(None, key_parts))

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            await cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# Cache warming utility
async def warm_cache(keys_and_loaders: dict):
    """Warm cache with precomputed values.

    Args:
        keys_and_loaders: Dict mapping cache keys to loader functions

    Usage:
        await warm_cache({
            "model:latest": load_model_metadata,
            "features:list": load_feature_names
        })
    """
    logger.info(f"Warming cache with {len(keys_and_loaders)} keys...")

    for key, loader in keys_and_loaders.items():
        try:
            value = await loader() if callable(loader) else loader
            await cache.set(key, value, ttl=86400)  # 24 hours
            logger.debug(f"Warmed cache key: {key}")
        except Exception as e:
            logger.error(f"Failed to warm cache key {key}: {e}")

    logger.info("Cache warming completed")
