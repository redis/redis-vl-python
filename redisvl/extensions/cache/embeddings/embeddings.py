"""Embeddings cache implementation for RedisVL."""

from typing import Any, Dict, List, Optional, Tuple

from redis import Redis

from redisvl.extensions.cache.base import BaseCache
from redisvl.extensions.cache.embeddings.schema import CacheEntry
from redisvl.redis.utils import convert_bytes, hashify


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
    ) -> Tuple[str, Dict[str, Any]]:
        """Prepare data for storage in Redis

        Args:
            text (str): The text input that was embedded.
            model_name (str): The name of the embedding model.
            embedding (List[float]): The embedding vector.
            metadata (Optional[Dict[str, Any]]): Optional metadata.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple of (key, entry_data)
        """
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

    def _process_cache_data(
        self, data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Process Redis hash data into a cache entry response.

        Args:
            data (Optional[Dict[str, Any]]): Raw Redis hash data.

        Returns:
            Optional[Dict[str, Any]]: Processed cache entry or None if no data.
        """
        if not data:
            return None

        cache_hit = CacheEntry(**convert_bytes(data))
        return cache_hit.model_dump(exclude_none=True)

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
        key = self._make_cache_key(text, model_name)
        return self.get_by_key(key)

    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Get embedding by its full Redis key.

        Retrieves a cached embedding for the given Redis key.
        If found, refreshes the TTL of the entry.

        Args:
            key (str): The full Redis key for the embedding.

        Returns:
            Optional[Dict[str, Any]]: Embedding cache entry or None if not found.

        .. code-block:: python

            embedding_data = cache.get_by_key("embedcache:1234567890abcdef")
        """
        client = self._get_redis_client()

        # Get all fields
        data = client.hgetall(key)

        # Refresh TTL if data exists
        if data:
            self.expire(key)

        return self._process_cache_data(data)

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
        key = self._make_cache_key(text, model_name)
        return await self.aget_by_key(key)

    async def aget_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Async get embedding by its full Redis key.

        Asynchronously retrieves a cached embedding for the given Redis key.
        If found, refreshes the TTL of the entry.

        Args:
            key (str): The full Redis key for the embedding.

        Returns:
            Optional[Dict[str, Any]]: Embedding cache entry or None if not found.

        .. code-block:: python

            embedding_data = await cache.aget_by_key("embedcache:1234567890abcdef")
        """
        client = await self._get_async_redis_client()

        # Get all fields
        data = await client.hgetall(key)

        # Refresh TTL if data exists
        if data:
            await self.aexpire(key)

        return self._process_cache_data(data)

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

    def exists_by_key(self, key: str) -> bool:
        """Check if an embedding exists for the given Redis key.

        Args:
            key (str): The full Redis key for the embedding.

        Returns:
            bool: True if the embedding exists in the cache, False otherwise.

        .. code-block:: python

            if cache.exists_by_key("embedcache:1234567890abcdef"):
                print("Embedding is in cache")
        """
        client = self._get_redis_client()
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
        key = self._make_cache_key(text, model_name)
        return await self.aexists_by_key(key)

    async def aexists_by_key(self, key: str) -> bool:
        """Async check if an embedding exists for the given Redis key.

        Asynchronously checks if an embedding exists for the given Redis key.

        Args:
            key (str): The full Redis key for the embedding.

        Returns:
            bool: True if the embedding exists in the cache, False otherwise.

        .. code-block:: python

            if await cache.aexists_by_key("embedcache:1234567890abcdef"):
                print("Embedding is in cache")
        """
        client = await self._get_async_redis_client()
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
        client.hset(name=key, mapping=cache_entry)  # type: ignore

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
        await client.hset(name=key, mapping=cache_entry)  # type: ignore

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
        key = self._make_cache_key(text, model_name)
        self.drop_by_key(key)

    def drop_by_key(self, key: str) -> None:
        """Remove an embedding from the cache by its Redis key.

        Args:
            key (str): The full Redis key for the embedding.

        .. code-block:: python

            cache.drop_by_key("embedcache:1234567890abcdef")
        """
        client = self._get_redis_client()
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
        key = self._make_cache_key(text, model_name)
        await self.adrop_by_key(key)

    async def adrop_by_key(self, key: str) -> None:
        """Async remove an embedding from the cache by its Redis key.

        Asynchronously removes an embedding from the cache by its Redis key.

        Args:
            key (str): The full Redis key for the embedding.

        .. code-block:: python

            await cache.adrop_by_key("embedcache:1234567890abcdef")
        """
        client = await self._get_async_redis_client()
        await client.delete(key)
