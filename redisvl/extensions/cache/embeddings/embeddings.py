"""Embeddings cache implementation for RedisVL.

This module provides a concrete implementation of the BaseEmbeddingsCache that
stores and retrieves embedding vectors with exact key matching.
"""

from typing import Any, Dict, List, Optional

from redis import Redis

from redisvl.extensions.cache.base import BaseCache
from redisvl.extensions.cache.embeddings.schema import CacheEntry
from redisvl.redis.utils import hashify


class EmbeddingsCache(BaseCache):
    """Embeddings Cache for storing embedding vectors with exact key matching."""

    def __init__(
        self,
        name: str = "embedcache",
        ttl: Optional[int] = None,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: Dict[str, Any] = {},
    ):
        """Initialize an embeddings cache.

        Args:
            name (str): The name of the cache. Defaults to "embedcache".
            ttl (Optional[int]): The time-to-live for cached embeddings. Defaults to None.
            redis_client (Optional[Redis]): Redis client instance. Defaults to None.
            redis_url (str): Redis URL for connection. Defaults to "redis://localhost:6379".
            connection_kwargs (Dict[str, Any]): Redis connection arguments. Defaults to {}.

        Raises:
            ValueError: If vector dimensions are invalid

        .. code-block:: python

            cache = EmbeddingsCache(
                name="my_embeddings_cache",
                ttl=3600,  # 1 hour
                redis_url="redis://localhost:6379"
            )
        """
        super().__init__(
            name=name,
            ttl=ttl,
            redis_client=redis_client,
            redis_url=redis_url,
            connection_kwargs=connection_kwargs,
        )

    def _make_entry_id(self, text: str, model_name: str) -> str:
        """Generate a deterministic entry ID for the given text and model name.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            str: A deterministic entry ID based on the text and model name.
        """
        return hashify(f"{text}:{model_name}")

    def _make_cache_key(self, text: str, model_name: str) -> str:
        """Generate a full Redis key for the given text and model name.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            str: The full Redis key.
        """
        entry_id = self._make_entry_id(text, model_name)
        return self._make_key(entry_id)

    def _prepare_entry_data(
        self,
        text: str,
        model_name: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare data for storage in Redis"""
        # Create cache entry with entry_id
        entry_id = self._make_entry_id(text, model_name)
        key = self._make_key(entry_id)
        entry = CacheEntry(
            entry_id=entry_id,
            text=text,
            model_name=model_name,
            embedding=embedding,
            metadata=metadata,
        )
        return key, entry.to_dict()

    def clear(self) -> None:
        """Clear the cache of all keys.

        Removes all entries from the cache that match the cache prefix.

        .. code-block:: python

            cache.clear()
        """
        client = self._get_redis_client()

        # Scan for all keys with our prefix
        cursor = "0"
        while cursor != 0:
            cursor, keys = client.scan(
                cursor=cursor, match=f"{self.prefix}*", count=100
            )
            if keys:
                client.delete(*keys)

    async def aclear(self) -> None:
        """Async clear the cache of all keys.

        Asynchronously removes all entries from the cache that match the cache prefix.

        .. code-block:: python

            await cache.aclear()
        """
        client = await self._get_async_redis_client()

        # Scan for all keys with our prefix
        cursor = "0"
        while cursor != 0:
            cursor, keys = await client.scan(
                cursor=cursor, match=f"{self.prefix}*", count=100
            )
            if keys:
                await client.delete(*keys)

    def get(
        self,
        text: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get embedding by text and model name.

        Retrieves a cached embedding for the given text and model name.
        If found, refreshes the TTL of the entry.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            Optional[Dict[str, Any]]: Embedding cache entry or None if not found.

        .. code-block:: python

            embedding_data = cache.get(
                text="What is machine learning?",
                model_name="text-embedding-ada-002"
            )
        """
        client = self._get_redis_client()
        key = self._make_cache_key(text, model_name)

        # Get all fields
        if data := client.hgetall(key):
            # Refresh TTL
            self.expire(key)
            cache_hit = CacheEntry(**data)
            response = cache_hit.model_dump(exclude_none=True)
            return response
        return None

    async def aget(
        self,
        text: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Async get embedding by text and model name.

        Asynchronously retrieves a cached embedding for the given text and model name.
        If found, refreshes the TTL of the entry.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            Optional[Dict[str, Any]]: Embedding cache entry or None if not found.

        .. code-block:: python

            embedding_data = await cache.aget(
                text="What is machine learning?",
                model_name="text-embedding-ada-002"
            )
        """
        client = await self._get_async_redis_client()
        key = self._make_cache_key(text, model_name)

        if data := await client.hgetall(key):
            # Refresh TTL
            await self.aexpire(key)
            cache_hit = CacheEntry(**data)
            response = cache_hit.model_dump(exclude_none=True)
            return response
        return None

    def exists(self, text: str, model_name: str) -> bool:
        """Check if an embedding exists for the given text and model.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            bool: True if the embedding exists in the cache, False otherwise.

        .. code-block:: python

            if cache.exists("What is machine learning?", "text-embedding-ada-002"):
                print("Embedding is in cache")
        """
        client = self._get_redis_client()
        key = self._make_cache_key(text, model_name)
        return bool(client.exists(key))

    async def aexists(self, text: str, model_name: str) -> bool:
        """Async check if an embedding exists.

        Asynchronously checks if an embedding exists for the given text and model.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            bool: True if the embedding exists in the cache, False otherwise.

        .. code-block:: python

            if await cache.aexists("What is machine learning?", "text-embedding-ada-002"):
                print("Embedding is in cache")
        """
        client = await self._get_async_redis_client()
        key = self._make_cache_key(text, model_name)
        return bool(await client.exists(key))

    def set(
        self,
        text: str,
        model_name: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store an embedding with its text and model name.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.
            embedding (List[float]): The embedding vector to store.
            metadata (Optional[Dict[str, Any]]): Optional metadata to store with the embedding.
            ttl (Optional[int]): Optional TTL override for this specific entry.

        Returns:
            str: The Redis key where the embedding was stored.

        .. code-block:: python

            key = cache.set(
                text="What is machine learning?",
                model_name="text-embedding-ada-002",
                embedding=[0.1, 0.2, 0.3, ...],
                metadata={"source": "user_query"}
            )
        """
        # Prepare data
        key, cache_entry = self._prepare_entry_data(
            text, model_name, embedding, metadata
        )

        # Store in Redis
        client = self._get_redis_client()
        client.hset(name=key, mapping=cache_entry)

        # Set TTL if specified
        self.expire(key, ttl)

        return key

    async def aset(
        self,
        text: str,
        model_name: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Async store an embedding with its text and model name.

        Asynchronously stores an embedding with its text and model name.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.
            embedding (List[float]): The embedding vector to store.
            metadata (Optional[Dict[str, Any]]): Optional metadata to store with the embedding.
            ttl (Optional[int]): Optional TTL override for this specific entry.

        Returns:
            str: The Redis key where the embedding was stored.

        .. code-block:: python

            key = await cache.aset(
                text="What is machine learning?",
                model_name="text-embedding-ada-002",
                embedding=[0.1, 0.2, 0.3, ...],
                metadata={"source": "user_query"}
            )
        """
        # Prepare data
        key, cache_entry = self._prepare_entry_data(
            text, model_name, embedding, metadata
        )

        # Store in Redis
        client = await self._get_async_redis_client()
        await client.hset(name=key, mapping=cache_entry)

        # Set TTL if specified
        await self.aexpire(key, ttl)

        return key

    def drop(self, text: str, model_name: str) -> None:
        """Remove an embedding from the cache.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.

        .. code-block:: python

            cache.drop(
                text="What is machine learning?",
                model_name="text-embedding-ada-002"
            )
        """
        client = self._get_redis_client()
        key = self._make_cache_key(text, model_name)
        client.delete(key)

    async def adrop(self, text: str, model_name: str) -> None:
        """Async remove an embedding from the cache.

        Asynchronously removes an embedding from the cache.

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.

        .. code-block:: python

            await cache.adrop(
                text="What is machine learning?",
                model_name="text-embedding-ada-002"
            )
        """
        client = await self._get_async_redis_client()
        key = self._make_cache_key(text, model_name)
        await client.delete(key)
