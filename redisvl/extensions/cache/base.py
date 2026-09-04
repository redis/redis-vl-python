"""Base cache interface for RedisVL.

This module defines the abstract base cache interface that is implemented by
specific cache types such as LLM caches and embedding caches.
"""

from typing import Any, cast

from redis import Redis  # For backwards compatibility in type checking
from redis.cluster import RedisCluster

from redisvl.redis.connection import RedisConnectionFactory
from redisvl.types import AsyncRedisClient, SyncRedisClient

# Keys deleted per DEL when clearing. Also the SCAN count hint, so one page of
# keys maps to one delete round-trip.
CLEAR_BATCH_SIZE = 500


class BaseCache:
    """Base abstract cache interface for all RedisVL caches.

    This class defines common functionality shared by all cache implementations,
    including TTL management, connection handling, and basic cache operations.
    """

    _redis_client: SyncRedisClient | None
    _async_redis_client: AsyncRedisClient | None

    def __init__(
        self,
        name: str,
        ttl: int | None = None,
        redis_client: SyncRedisClient | None = None,
        async_redis_client: AsyncRedisClient | None = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize a base cache.

        Args:
            name (str): The name of the cache.
            ttl (Optional[int], optional): The time-to-live for records cached
                in Redis. Defaults to None.
            redis_client (Optional[SyncRedisClient], optional): A redis client connection instance.
                Defaults to None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.
        """
        connection_kwargs = connection_kwargs or {}
        self.name = name
        self._ttl: int | None = None
        self.set_ttl(ttl)

        self.redis_kwargs = {
            "redis_client": redis_client,
            "redis_url": redis_url,
            "connection_kwargs": connection_kwargs,
        }

        # Initialize Redis clients
        self._async_redis_client = async_redis_client
        self._redis_client = redis_client

        if redis_client or async_redis_client:
            self._owns_redis_client = False
        else:
            self._owns_redis_client = True

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
    def ttl(self) -> int | None:
        """The default TTL, in seconds, for entries in the cache."""
        return self._ttl

    def set_ttl(self, ttl: int | None = None) -> None:
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

    def _get_redis_client(self) -> SyncRedisClient:
        """Get or create a Redis client.

        Returns:
            SyncRedisClient: A Redis client instance.
        """
        if self._redis_client is None:
            # Create new Redis client
            url = cast(str | None, self.redis_kwargs["redis_url"])
            kwargs = cast(dict[str, Any], self.redis_kwargs["connection_kwargs"])
            self._redis_client = RedisConnectionFactory.get_redis_connection(
                redis_url=url,
                **kwargs,
            )
        return self._redis_client

    async def _get_async_redis_client(self) -> AsyncRedisClient:
        """Get or create an async Redis client.

        Returns:
            AsyncRedisClient: An async Redis client instance.
        """
        if not hasattr(self, "_async_redis_client") or self._async_redis_client is None:
            client = self.redis_kwargs.get("redis_client")

            if client and isinstance(client, (Redis, RedisCluster)):
                self._async_redis_client = RedisConnectionFactory.sync_to_async_redis(
                    client
                )
            else:
                url = cast(str | None, self.redis_kwargs["redis_url"])
                kwargs = cast(dict[str, Any], self.redis_kwargs["connection_kwargs"])
                self._async_redis_client = (
                    RedisConnectionFactory.get_async_redis_connection(
                        redis_url=url, **kwargs
                    )
                )
        return self._async_redis_client

    def expire(self, key: str, ttl: int | None = None) -> None:
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

    async def aexpire(self, key: str, ttl: int | None = None) -> None:
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
        """Clear the cache of all keys.

        Deletes every Redis key under the cache's prefix (``<name>:``) with
        ``SCAN`` + ``DEL``. The cache object itself stays usable for future
        writes.

        Note:
            ``SCAN`` is not a point-in-time snapshot, so this is a best-effort
            sweep rather than an atomic flush:

            - Keys written by other clients while the sweep is in progress may
              or may not be deleted, so the cache is not guaranteed to be empty
              when this returns. Quiesce writers first if you need that.
            - ``SCAN`` may return the same key on more than one page. ``DEL``
              on an already-deleted key is a no-op, so this is harmless.
            - Deletion is not atomic across keys. If the call raises partway
              through, some keys are already gone. The operation is idempotent,
              so retrying is safe and converges.
        """
        client = self._get_redis_client()
        # scan_iter, not a hand-rolled SCAN loop: on a cluster client SCAN is
        # broadcast to every primary and replies with a {node_name: cursor}
        # mapping, and those cursors are node-local -- they can neither be fed
        # back as a single cursor nor broadcast. redis-py's scan_iter already
        # drives each primary on its own cursor via target_nodes.
        batch: list[Any] = []
        for key in client.scan_iter(
            match=f"{self._get_prefix()}*", count=CLEAR_BATCH_SIZE
        ):
            batch.append(key)
            if len(batch) >= CLEAR_BATCH_SIZE:
                client.delete(*batch)
                batch.clear()
        if batch:
            client.delete(*batch)

    async def aclear(self) -> None:
        """Asynchronously clear the cache of all keys.

        Deletes every Redis key under the cache's prefix (``<name>:``) with
        ``SCAN`` + ``DEL``. The cache object itself stays usable for future
        writes.

        Note:
            ``SCAN`` is not a point-in-time snapshot, so this is a best-effort
            sweep rather than an atomic flush:

            - Keys written by other clients while the sweep is in progress may
              or may not be deleted, so the cache is not guaranteed to be empty
              when this returns. Quiesce writers first if you need that.
            - ``SCAN`` may return the same key on more than one page. ``DEL``
              on an already-deleted key is a no-op, so this is harmless.
            - Deletion is not atomic across keys. If the call raises partway
              through, some keys are already gone. The operation is idempotent,
              so retrying is safe and converges.
        """
        client = await self._get_async_redis_client()
        # See the note in clear() on why this delegates to scan_iter.
        batch: list[Any] = []
        async for key in client.scan_iter(
            match=f"{self._get_prefix()}*", count=CLEAR_BATCH_SIZE
        ):
            batch.append(key)
            if len(batch) >= CLEAR_BATCH_SIZE:
                await client.delete(*batch)
                batch.clear()
        if batch:
            await client.delete(*batch)

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._owns_redis_client is False:
            return

        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None
            # Async clients don't have a sync close method, so we just
            # zero them out to allow garbage collection.
            self._async_redis_client = None

    async def adisconnect(self) -> None:
        """Async disconnect from Redis."""
        if self._owns_redis_client is False:
            return

        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None

        if hasattr(self, "_async_redis_client") and self._async_redis_client:
            # Use proper async close method
            await self._async_redis_client.aclose()
            self._async_redis_client = None
