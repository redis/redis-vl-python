"""Base cache interface for RedisVL.

This module defines the abstract base cache interface that is implemented by
specific cache types such as LLM caches and embedding caches.
"""

from typing import Any, Dict, Optional

from redis import Redis
from redis.asyncio import Redis as AsyncRedis


class BaseCache:
    """Base abstract cache interface for all RedisVL caches.

    This class defines common functionality shared by all cache implementations,
    including TTL management, connection handling, and basic cache operations.
    """

    def __init__(
        self,
        name: str,
        ttl: Optional[int] = None,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: Dict[str, Any] = {},
    ):
        """Initialize a base cache.

        Args:
            name (str): The name of the cache.
            ttl (Optional[int], optional): The time-to-live for records cached
                in Redis. Defaults to None.
            redis_client (Optional[Redis], optional): A redis client connection instance.
                Defaults to None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.
        """
        self.name = name
        self._ttl: Optional[int] = None
        self.set_ttl(ttl)

        self.redis_kwargs = {
            "redis_client": redis_client,
            "redis_url": redis_url,
            "connection_kwargs": connection_kwargs,
        }

        if redis_client:
            self._owns_redis_client = False
            self._redis_client = redis_client
        else:
            self._owns_redis_client = True
            self._redis_client = None

    def _get_prefix(self) -> str:
        """Get the key prefix for Redis keys.

        Returns:
            str: The prefix to use for Redis keys.
        """
        return f"{self.name}:"

    def _make_key(self, entry_id: str) -> str:
        """Generate a full Redis key for the given entry ID.

        Args:
            entry_id (str): The unique entry ID.

        Returns:
            str: The full Redis key including prefix.
        """
        return f"{self._get_prefix()}{entry_id}"

    @property
    def ttl(self) -> Optional[int]:
        """The default TTL, in seconds, for entries in the cache."""
        return self._ttl

    def set_ttl(self, ttl: Optional[int] = None) -> None:
        """Set the default TTL, in seconds, for entries in the cache.

        Args:
            ttl (Optional[int], optional): The optional time-to-live expiration
                for the cache, in seconds.

        Raises:
            ValueError: If the time-to-live value is not an integer.
        """
        if ttl:
            if not isinstance(ttl, int):
                raise ValueError(f"TTL must be an integer value, got {ttl}")
            self._ttl = int(ttl)
        else:
            self._ttl = None

    def _get_redis_client(self) -> Redis:
        """Get or create a Redis client.

        Returns:
            Redis: A Redis client instance.
        """
        if self._redis_client is None:
            self._redis_client = Redis.from_url(
                self.redis_kwargs["redis_url"], **self.redis_kwargs["connection_kwargs"]
            )
        return self._redis_client

    async def _get_async_redis_client(self) -> AsyncRedis:
        """Get or create an async Redis client.

        Returns:
            AsyncRedis: An async Redis client instance.
        """
        if not hasattr(self, "_async_redis_client") or self._async_redis_client is None:
            self._async_redis_client = AsyncRedis.from_url(
                self.redis_kwargs["redis_url"], **self.redis_kwargs["connection_kwargs"]
            )
        return self._async_redis_client

    def expire(self, key: str, ttl: Optional[int] = None) -> None:
        """Set or refresh the expiration time for a key in the cache.

        Args:
            key (str): The Redis key to set the expiration on.
            ttl (Optional[int], optional): The time-to-live in seconds. If None,
                uses the default TTL configured for this cache instance.
                Defaults to None.

        Note:
            If neither the provided TTL nor the default TTL is set (both are None),
            this method will have no effect.
        """
        _ttl = ttl if ttl is not None else self._ttl
        if _ttl:
            client = self._get_redis_client()
            client.expire(key, _ttl)

    async def aexpire(self, key: str, ttl: Optional[int] = None) -> None:
        """Asynchronously set or refresh the expiration time for a key in the cache.

        Args:
            key (str): The Redis key to set the expiration on.
            ttl (Optional[int], optional): The time-to-live in seconds. If None,
                uses the default TTL configured for this cache instance.
                Defaults to None.

        Note:
            If neither the provided TTL nor the default TTL is set (both are None),
            this method will have no effect.
        """
        _ttl = ttl if ttl is not None else self._ttl
        if _ttl:
            client = await self._get_async_redis_client()
            await client.expire(key, _ttl)

    def clear(self) -> None:
        """Clear the cache of all keys."""
        client = self._get_redis_client()
        prefix = self._get_prefix()

        # Scan for all keys with our prefix
        cursor = "0"
        while cursor != 0:
            cursor, keys = client.scan(cursor=cursor, match=f"{prefix}*", count=100)
            if keys:
                client.delete(*keys)

    async def aclear(self) -> None:
        """Async clear the cache of all keys."""
        client = await self._get_async_redis_client()
        prefix = self._get_prefix()

        # Scan for all keys with our prefix
        cursor = "0"
        while cursor != 0:
            cursor, keys = await client.scan(
                cursor=cursor, match=f"{prefix}*", count=100
            )
            if keys:
                await client.delete(*keys)

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._owns_redis_client is False:
            return

        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None

        if hasattr(self, "_async_redis_client") and self._async_redis_client:
            self._async_redis_client.close()
            self._async_redis_client = None

    async def adisconnect(self) -> None:
        """Async disconnect from Redis."""
        if self._owns_redis_client is False:
            return

        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None

        if hasattr(self, "_async_redis_client") and self._async_redis_client:
            await self._async_redis_client.aclose()
            self._async_redis_client = None
