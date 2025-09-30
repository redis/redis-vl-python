"""Embeddings cache implementation for RedisVL."""

from typing import Any, Awaitable, Dict, List, Optional, Tuple, cast

from redisvl.extensions.cache.base import BaseCache
from redisvl.extensions.cache.embeddings.schema import CacheEntry
from redisvl.redis.utils import convert_bytes, hashify
from redisvl.types import AsyncRedisClient, SyncRedisClient
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


class EmbeddingsCache(BaseCache):
    """Embeddings Cache for storing embedding vectors with exact key matching."""

    _warning_shown: bool = False  # Class-level flag to prevent warning spam

    def __init__(
        self,
        name: str = "embedcache",
        ttl: Optional[int] = None,
        redis_client: Optional[SyncRedisClient] = None,
        async_redis_client: Optional[AsyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: Dict[str, Any] = {},
    ):
        """Initialize an embeddings cache.

        Args:
            name (str): The name of the cache. Defaults to "embedcache".
            ttl (Optional[int]): The time-to-live for cached embeddings. Defaults to None.
            redis_client (Optional[SyncRedisClient]): Redis client instance. Defaults to None.
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
            async_redis_client=async_redis_client,
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

    def _should_warn_for_async_only(self) -> bool:
        """Check if only async client is available (no sync client).

        Returns:
            bool: True if only async client is available (no sync client).
        """
        return self._owns_redis_client is False and self._redis_client is None

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
        if self._should_warn_for_async_only():
            if not EmbeddingsCache._warning_shown:
                logger.warning(
                    "EmbeddingsCache initialized with async_redis_client only. "
                    "Use async methods (aget_by_key) instead of sync methods (get_by_key)."
                )
                EmbeddingsCache._warning_shown = True

        client = self._get_redis_client()

        # Get all fields
        data = client.hgetall(key)

        # Refresh TTL if data exists
        if data:
            self.expire(key)

        return self._process_cache_data(data)  # type: ignore

    def mget_by_keys(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Get multiple embeddings by their Redis keys.

        Efficiently retrieves multiple cached embeddings in a single network roundtrip.
        If found, refreshes the TTL of each entry.

        Args:
            keys (List[str]): List of Redis keys to retrieve.

        Returns:
            List[Optional[Dict[str, Any]]]: List of embedding cache entries or None for keys not found.
            The order matches the input keys order.

        .. code-block:: python

            # Get multiple embeddings
            embedding_data = cache.mget_by_keys([
                "embedcache:key1",
                "embedcache:key2"
            ])
        """
        if not keys:
            return []

        if self._should_warn_for_async_only():
            if not EmbeddingsCache._warning_shown:
                logger.warning(
                    "EmbeddingsCache initialized with async_redis_client only. "
                    "Use async methods (amget_by_keys) instead of sync methods (mget_by_keys)."
                )
                EmbeddingsCache._warning_shown = True

        client = self._get_redis_client()

        with client.pipeline(transaction=False) as pipeline:
            # Queue all hgetall operations
            for key in keys:
                pipeline.hgetall(key)
            results = pipeline.execute()

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if result:  # If cache hit, refresh TTL separately
                self.expire(keys[i])
            processed_results.append(self._process_cache_data(result))

        return processed_results

    def mget(self, texts: List[str], model_name: str) -> List[Optional[Dict[str, Any]]]:
        """Get multiple embeddings by their texts and model name.

        Efficiently retrieves multiple cached embeddings in a single operation.
        If found, refreshes the TTL of each entry.

        Args:
            texts (List[str]): List of text inputs that were embedded.
            model_name (str): The name of the embedding model.

        Returns:
            List[Optional[Dict[str, Any]]]: List of embedding cache entries or None for texts not found.

        .. code-block:: python

            # Get multiple embeddings
            embedding_data = cache.mget(
                texts=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        if not texts:
            return []

        # Generate keys for each text
        keys = [self._make_cache_key(text, model_name) for text in texts]

        # Use the key-based batch operation
        return self.mget_by_keys(keys)

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

        if self._should_warn_for_async_only():
            if not EmbeddingsCache._warning_shown:
                logger.warning(
                    "EmbeddingsCache initialized with async_redis_client only. "
                    "Use async methods (aset) instead of sync methods (set)."
                )
                EmbeddingsCache._warning_shown = True

        # Store in Redis
        client = self._get_redis_client()
        client.hset(name=key, mapping=cache_entry)  # type: ignore

        # Set TTL if specified
        self.expire(key, ttl)

        return key

    def mset(
        self,
        items: List[Dict[str, Any]],
        ttl: Optional[int] = None,
    ) -> List[str]:
        """Store multiple embeddings in a batch operation.

        Each item in the input list should be a dictionary with the following fields:
        - 'text': The text input that was embedded
        - 'model_name': The name of the embedding model
        - 'embedding': The embedding vector
        - 'metadata': Optional metadata to store with the embedding

        Args:
            items: List of dictionaries, each containing text, model_name, embedding, and optional metadata.
            ttl: Optional TTL override for these entries.

        Returns:
            List[str]: List of Redis keys where the embeddings were stored.

        .. code-block:: python

            # Store multiple embeddings
            keys = cache.mset([
                {
                    "text": "What is ML?",
                    "model_name": "text-embedding-ada-002",
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {"source": "user"}
                },
                {
                    "text": "What is AI?",
                    "model_name": "text-embedding-ada-002",
                    "embedding": [0.4, 0.5, 0.6],
                    "metadata": {"source": "docs"}
                }
            ])
        """
        if not items:
            return []

        if self._should_warn_for_async_only():
            if not EmbeddingsCache._warning_shown:
                logger.warning(
                    "EmbeddingsCache initialized with async_redis_client only. "
                    "Use async methods (amset) instead of sync methods (mset)."
                )
                EmbeddingsCache._warning_shown = True

        client = self._get_redis_client()
        keys = []

        with client.pipeline(transaction=False) as pipeline:
            # Process all entries
            for item in items:
                # Prepare and store
                key, cache_entry = self._prepare_entry_data(**item)
                keys.append(key)
                pipeline.hset(name=key, mapping=cache_entry)  # type: ignore

            pipeline.execute()

        # Set TTLs
        for key in keys:
            self.expire(key, ttl)

        return keys

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

    def mexists_by_keys(self, keys: List[str]) -> List[bool]:
        """Check if multiple embeddings exist by their Redis keys.

        Efficiently checks existence of multiple keys in a single operation.

        Args:
            keys (List[str]): List of Redis keys to check.

        Returns:
            List[bool]: List of boolean values indicating whether each key exists.
            The order matches the input keys order.

        .. code-block:: python

            # Check if multiple keys exist
            exists_results = cache.mexists_by_keys(["embedcache:key1", "embedcache:key2"])
        """
        if not keys:
            return []

        client = self._get_redis_client()

        with client.pipeline(transaction=False) as pipeline:
            # Queue all exists operations
            for key in keys:
                pipeline.exists(key)
            results = pipeline.execute()

        # Convert to boolean values
        return [bool(result) for result in results]

    def mexists(self, texts: List[str], model_name: str) -> List[bool]:
        """Check if multiple embeddings exist by their texts and model name.

        Efficiently checks existence of multiple embeddings in a single operation.

        Args:
            texts (List[str]): List of text inputs that were embedded.
            model_name (str): The name of the embedding model.

        Returns:
            List[bool]: List of boolean values indicating whether each embedding exists.

        .. code-block:: python

            # Check if multiple embeddings exist
            exists_results = cache.mexists(
                texts=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        if not texts:
            return []

        # Generate keys for each text
        keys = [self._make_cache_key(text, model_name) for text in texts]

        # Use the key-based batch operation
        return self.mexists_by_keys(keys)

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

    def mdrop_by_keys(self, keys: List[str]) -> None:
        """Remove multiple embeddings from the cache by their Redis keys.

        Efficiently removes multiple embeddings in a single operation.

        Args:
            keys (List[str]): List of Redis keys to remove.

        .. code-block:: python

            # Remove multiple embeddings
            cache.mdrop_by_keys(["embedcache:key1", "embedcache:key2"])
        """
        if not keys:
            return

        client = self._get_redis_client()

        with client.pipeline(transaction=False) as pipeline:
            for key in keys:
                pipeline.delete(key)
            pipeline.execute()

    def mdrop(self, texts: List[str], model_name: str) -> None:
        """Remove multiple embeddings from the cache by their texts and model name.

        Efficiently removes multiple embeddings in a single operation.

        Args:
            texts (List[str]): List of text inputs that were embedded.
            model_name (str): The name of the embedding model.

        .. code-block:: python

            # Remove multiple embeddings
            cache.mdrop(
                texts=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        if not texts:
            return

        # Generate keys for each text
        keys = [self._make_cache_key(text, model_name) for text in texts]

        # Use the key-based batch operation
        self.mdrop_by_keys(keys)

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
        data = await client.hgetall(key)  # type: ignore

        # Refresh TTL if data exists
        if data:
            await self.aexpire(key)

        return self._process_cache_data(data)

    async def amget_by_keys(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Async get multiple embeddings by their Redis keys.

        Asynchronously retrieves multiple cached embeddings in a single network roundtrip.
        If found, refreshes the TTL of each entry.

        Args:
            keys (List[str]): List of Redis keys to retrieve.

        Returns:
            List[Optional[Dict[str, Any]]]: List of embedding cache entries or None for keys not found.
            The order matches the input keys order.

        .. code-block:: python

            # Get multiple embeddings asynchronously
            embedding_data = await cache.amget_by_keys([
                "embedcache:key1",
                "embedcache:key2"
            ])
        """
        if not keys:
            return []

        client = await self._get_async_redis_client()

        # Use pipeline only for retrieval
        async with client.pipeline(transaction=False) as pipeline:
            # Queue all hgetall operations
            for key in keys:
                pipeline.hgetall(key)
            results = await pipeline.execute()

        # Process results and refresh TTLs separately
        processed_results = []
        for i, result in enumerate(results):
            if result:  # If cache hit, refresh TTL
                await self.aexpire(keys[i])
            processed_results.append(self._process_cache_data(result))

        return processed_results

    async def amget(
        self, texts: List[str], model_name: str
    ) -> List[Optional[Dict[str, Any]]]:
        """Async get multiple embeddings by their texts and model name.

        Asynchronously retrieves multiple cached embeddings in a single operation.
        If found, refreshes the TTL of each entry.

        Args:
            texts (List[str]): List of text inputs that were embedded.
            model_name (str): The name of the embedding model.

        Returns:
            List[Optional[Dict[str, Any]]]: List of embedding cache entries or None for texts not found.

        .. code-block:: python

            # Get multiple embeddings asynchronously
            embedding_data = await cache.amget(
                texts=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        if not texts:
            return []

        # Generate keys for each text
        keys = [self._make_cache_key(text, model_name) for text in texts]

        # Use the key-based batch operation
        return await self.amget_by_keys(keys)

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

    async def amset(
        self,
        items: List[Dict[str, Any]],
        ttl: Optional[int] = None,
    ) -> List[str]:
        """Async store multiple embeddings in a batch operation.

        Each item in the input list should be a dictionary with the following fields:
        - 'text': The text input that was embedded
        - 'model_name': The name of the embedding model
        - 'embedding': The embedding vector
        - 'metadata': Optional metadata to store with the embedding

        Args:
            items: List of dictionaries, each containing text, model_name, embedding, and optional metadata.
            ttl: Optional TTL override for these entries.

        Returns:
            List[str]: List of Redis keys where the embeddings were stored.

        .. code-block:: python

            # Store multiple embeddings asynchronously
            keys = await cache.amset([
                {
                    "text": "What is ML?",
                    "model_name": "text-embedding-ada-002",
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {"source": "user"}
                },
                {
                    "text": "What is AI?",
                    "model_name": "text-embedding-ada-002",
                    "embedding": [0.4, 0.5, 0.6],
                    "metadata": {"source": "docs"}
                }
            ])
        """
        if not items:
            return []

        client = await self._get_async_redis_client()
        keys = []

        async with client.pipeline(transaction=False) as pipeline:
            # Process all entries
            for item in items:
                # Prepare and store
                key, cache_entry = self._prepare_entry_data(**item)
                keys.append(key)
                await pipeline.hset(name=key, mapping=cache_entry)  # type: ignore

            await pipeline.execute()

        # Set TTLs
        for key in keys:
            await self.aexpire(key, ttl)

        return keys

    async def amexists_by_keys(self, keys: List[str]) -> List[bool]:
        """Async check if multiple embeddings exist by their Redis keys.

        Asynchronously checks existence of multiple keys in a single operation.

        Args:
            keys (List[str]): List of Redis keys to check.

        Returns:
            List[bool]: List of boolean values indicating whether each key exists.
            The order matches the input keys order.

        .. code-block:: python

            # Check if multiple keys exist asynchronously
            exists_results = await cache.amexists_by_keys(["embedcache:key1", "embedcache:key2"])
        """
        if not keys:
            return []

        client = await self._get_async_redis_client()

        async with client.pipeline(transaction=False) as pipeline:
            # Queue all exists operations
            for key in keys:
                await pipeline.exists(key)
            results = await pipeline.execute()

        # Convert to boolean values
        return [bool(result) for result in results]

    async def amexists(self, texts: List[str], model_name: str) -> List[bool]:
        """Async check if multiple embeddings exist by their texts and model name.

        Asynchronously checks existence of multiple embeddings in a single operation.

        Args:
            texts (List[str]): List of text inputs that were embedded.
            model_name (str): The name of the embedding model.

        Returns:
            List[bool]: List of boolean values indicating whether each embedding exists.

        .. code-block:: python

            # Check if multiple embeddings exist asynchronously
            exists_results = await cache.amexists(
                texts=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        if not texts:
            return []

        # Generate keys for each text
        keys = [self._make_cache_key(text, model_name) for text in texts]

        # Use the key-based batch operation
        return await self.amexists_by_keys(keys)

    async def amdrop_by_keys(self, keys: List[str]) -> None:
        """Async remove multiple embeddings from the cache by their Redis keys.

        Asynchronously removes multiple embeddings in a single operation.

        Args:
            keys (List[str]): List of Redis keys to remove.

        .. code-block:: python

            # Remove multiple embeddings asynchronously
            await cache.amdrop_by_keys(["embedcache:key1", "embedcache:key2"])
        """
        if not keys:
            return

        client = await self._get_async_redis_client()
        await client.delete(*keys)

    async def amdrop(self, texts: List[str], model_name: str) -> None:
        """Async remove multiple embeddings from the cache by their texts and model name.

        Asynchronously removes multiple embeddings in a single operation.

        Args:
            texts (List[str]): List of text inputs that were embedded.
            model_name (str): The name of the embedding model.

        .. code-block:: python

            # Remove multiple embeddings asynchronously
            await cache.amdrop(
                texts=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        if not texts:
            return

        # Generate keys for each text
        keys = [self._make_cache_key(text, model_name) for text in texts]

        # Use the key-based batch operation
        await self.amdrop_by_keys(keys)

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
