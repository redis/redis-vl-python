"""Embeddings cache implementation for RedisVL."""

import json
from collections.abc import Mapping
from typing import Any, Iterable

from redisvl.extensions.cache.base import BaseCache
from redisvl.extensions.cache.embeddings.schema import CacheEntry
from redisvl.redis.utils import convert_bytes, hashify
from redisvl.types import AsyncRedisClient, SyncRedisClient
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


class EmbeddingsCache(BaseCache):
    """Embeddings Cache for storing embedding vectors with exact key matching.

    Note:
        Entries written by RedisVL 0.32.0 and earlier use an older key format.
        Content-based reads (``get``, ``mget``, ``exists``, ``mexists`` and
        their async versions) still find them, and move each one to its new
        key as they do. The ``*_by_key`` methods do not, so a key returned by
        ``set`` or ``mset`` before the upgrade stops resolving once its entry
        has moved; look such entries up, or drop them, by content instead.
    """

    _warning_shown: bool = False  # Class-level flag to prevent warning spam

    def __init__(
        self,
        name: str = "embedcache",
        ttl: int | None = None,
        redis_client: SyncRedisClient | None = None,
        async_redis_client: AsyncRedisClient | None = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: dict[str, Any] | None = None,
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
        connection_kwargs = connection_kwargs or {}
        super().__init__(
            name=name,
            ttl=ttl,
            redis_client=redis_client,
            async_redis_client=async_redis_client,
            redis_url=redis_url,
            connection_kwargs=connection_kwargs,
        )
        self._legacy_warning_shown = False

    def _make_entry_id(self, content: bytes | str, model_name: str) -> str:
        """Generate a deterministic entry ID for the given content and model name.

        The content's type, the content and the model name are hashed as a JSON
        array, so no two distinct inputs share an ID: ``("a:b", "c")`` differs
        from ``("a", "b:c")``, and bytes content differs from a string that
        happens to equal its hex form.

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            str: A deterministic entry ID based on the content and model name.
        """
        if isinstance(content, bytes):
            parts = ["bytes", content.hex(), model_name]
        else:
            parts = ["str", content, model_name]
        return hashify(json.dumps(parts, separators=(",", ":")))

    def _make_legacy_entry_id(self, content: bytes | str, model_name: str) -> str:
        """Generate the entry ID that RedisVL 0.32.0 and earlier used.

        That format joined content and model name with ``:``, so distinct
        inputs could share an ID. It is kept only to find entries written
        before the upgrade; remove it together with the legacy fallback.
        """
        if isinstance(content, bytes):
            content = content.hex()
        return hashify(f"{content}:{model_name}")

    def _make_legacy_cache_key(self, content: bytes | str, model_name: str) -> str:
        """Generate the full Redis key that RedisVL 0.32.0 and earlier used."""
        return self._make_key(self._make_legacy_entry_id(content, model_name))

    @staticmethod
    def _legacy_entry_matches(
        data: Mapping[bytes | str, Any], content: bytes | str, model_name: str
    ) -> bool:
        """Whether a legacy entry was stored for exactly this content and model.

        Legacy IDs can collide, so a legacy entry is only trusted when the
        content and model name stored in it equal the ones requested.
        """
        stored = convert_bytes(dict(data))
        return stored.get("model_name") == model_name and stored.get(
            "content"
        ) == convert_bytes(content)

    def _plan_legacy_migrations(
        self,
        contents: list[bytes | str],
        model_name: str,
        misses: list[int],
        replies: list[Any],
    ) -> list[tuple[int, str, str, dict[Any, Any], int]]:
        """Pick the legacy entries to move to their new keys.

        Args:
            contents: The requested contents.
            model_name: The requested model name.
            misses: Indexes into ``contents`` whose new key was missing.
            replies: Alternating HGETALL and PTTL replies for each miss's
                legacy key, in ``misses`` order.

        Returns:
            One ``(index, legacy_key, new_key, entry_data, pttl)`` tuple per
            legacy entry that matches its request.
        """
        migrations = []
        for i, data, pttl in zip(misses, replies[::2], replies[1::2]):
            content = contents[i]
            # PTTL -2: the key expired after HGETALL read it, so treat it as gone.
            if not data or pttl == -2:
                continue
            if not self._legacy_entry_matches(data, content, model_name):
                continue
            entry_id = self._make_entry_id(content, model_name)
            entry_data = convert_bytes(dict(data))
            entry_data["entry_id"] = entry_id
            migrations.append(
                (
                    i,
                    self._make_legacy_cache_key(content, model_name),
                    self._make_key(entry_id),
                    entry_data,
                    pttl,
                )
            )
        return migrations

    def _queue_legacy_migration(
        self, pipeline: Any, migration: tuple[int, str, str, dict[Any, Any], int]
    ) -> None:
        """Queue the commands that move one legacy entry to its new key.

        The new key gets this cache's TTL, or keeps the legacy key's remaining
        TTL when the cache has none.

        Nothing ties these writes to the earlier read: a ``set`` of the same
        content and model in between is overwritten with the legacy fields,
        and a ``drop`` in between is undone. Both inputs embed the same
        content with the same model, so this is accepted for a fallback that
        will be removed.
        """
        _, legacy_key, new_key, entry_data, pttl = migration
        pipeline.hset(new_key, mapping=entry_data)
        _ttl = self._resolve_ttl()
        if _ttl:
            pipeline.expire(new_key, _ttl)
        elif pttl and pttl > 0:
            pipeline.pexpire(new_key, pttl)
        pipeline.delete(legacy_key)

    def _warn_legacy_migration(self, replies: list[Any]) -> None:
        """Log, once per cache instance, that legacy entries are being migrated.

        Args:
            replies: The replies of the migration write pipeline, which runs
                with ``raise_on_error=False`` so that a client without write
                access still gets its hit.
        """
        if self._legacy_warning_shown:
            return
        error = next((r for r in replies if isinstance(r, Exception)), None)
        logger.warning(
            "EmbeddingsCache '%s' is reading entries stored under the key "
            "format used by RedisVL 0.32.0 and earlier, and moving each to the "
            "new format as it is read. A future release will stop reading the "
            "old format, and entries still in it will then miss.%s",
            self.name,
            (
                f" Moving some entries failed, so they stay under the old key: {error}"
                if error
                else ""
            ),
        )
        self._legacy_warning_shown = True

    def _migrate_legacy_entries(
        self, contents: list[bytes | str], model_name: str, misses: list[int]
    ) -> dict[int, dict[str, Any] | None]:
        """Look up missed contents under their legacy keys and migrate matches.

        Args:
            contents: The requested contents.
            model_name: The requested model name.
            misses: Indexes into ``contents`` whose new key was missing.

        Returns:
            The migrated cache entries, keyed by index into ``contents``.
        """
        if not misses:
            return {}
        client = self._get_redis_client()
        with client.pipeline(transaction=False) as pipeline:
            for i in misses:
                legacy_key = self._make_legacy_cache_key(contents[i], model_name)
                pipeline.hgetall(legacy_key)
                pipeline.pttl(legacy_key)
            replies = pipeline.execute()

        migrations = self._plan_legacy_migrations(contents, model_name, misses, replies)
        if not migrations:
            return {}
        with client.pipeline(transaction=False) as pipeline:
            for migration in migrations:
                self._queue_legacy_migration(pipeline, migration)
            replies = pipeline.execute(raise_on_error=False)
        self._warn_legacy_migration(replies)
        return {m[0]: self._process_cache_data(m[3]) for m in migrations}

    async def _amigrate_legacy_entries(
        self, contents: list[bytes | str], model_name: str, misses: list[int]
    ) -> dict[int, dict[str, Any] | None]:
        """Async version of :meth:`_migrate_legacy_entries`."""
        if not misses:
            return {}
        client = await self._get_async_redis_client()
        async with client.pipeline(transaction=False) as pipeline:
            for i in misses:
                legacy_key = self._make_legacy_cache_key(contents[i], model_name)
                pipeline.hgetall(legacy_key)
                pipeline.pttl(legacy_key)
            replies = await pipeline.execute()

        migrations = self._plan_legacy_migrations(contents, model_name, misses, replies)
        if not migrations:
            return {}
        async with client.pipeline(transaction=False) as pipeline:
            for migration in migrations:
                self._queue_legacy_migration(pipeline, migration)
            replies = await pipeline.execute(raise_on_error=False)
        self._warn_legacy_migration(replies)
        return {m[0]: self._process_cache_data(m[3]) for m in migrations}

    def _make_cache_key(self, content: bytes | str, model_name: str) -> str:
        """Generate a full Redis key for the given content and model name.

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            str: The full Redis key.
        """
        entry_id = self._make_entry_id(content, model_name)
        return self._make_key(entry_id)

    def _prepare_entry_data(
        self,
        content: bytes | str,
        model_name: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Prepare data for storage in Redis

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.
            embedding (List[float]): The embedding vector.
            metadata (Optional[Dict[str, Any]]): Optional metadata.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple of (key, entry_data)
        """
        # Create cache entry with entry_id
        entry_id = self._make_entry_id(content, model_name)
        key = self._make_key(entry_id)
        entry = CacheEntry(
            entry_id=entry_id,
            content=content,
            model_name=model_name,
            embedding=embedding,
            metadata=metadata,
        )
        return key, entry.to_dict()

    def _process_cache_data(
        self, data: Mapping[bytes | str, Any] | None
    ) -> dict[str, Any] | None:
        """Process Redis hash data into a cache entry response.

        Args:
            data (Optional[Dict[str, Any]]): Raw Redis hash data.

        Returns:
            Optional[Dict[str, Any]]: Processed cache entry or None if no data.
        """
        if not data:
            return None

        cache_hit = CacheEntry(**convert_bytes(dict(data)))
        return cache_hit.model_dump(exclude_none=True)

    def _should_warn_for_async_only(self) -> bool:
        """Check if only async client is available (no sync client).

        Returns:
            bool: True if only async client is available (no sync client).
        """
        return self._owns_redis_client is False and self._redis_client is None

    def get(
        self,
        content: bytes | str,
        model_name: str,
    ) -> dict[str, Any] | None:
        """Get embedding by content and model name.

        Retrieves a cached embedding for the given content and model name.
        If found, refreshes the TTL of the entry.

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            Optional[Dict[str, Any]]: Embedding cache entry or None if not found.

        .. code-block:: python

            embedding_data = cache.get(
                content="What is machine learning?",
                model_name="text-embedding-ada-002"
            )
        """
        return self.mget([content], model_name)[0]

    def get_by_key(self, key: str) -> dict[str, Any] | None:
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

        return self.mget_by_keys([key])[0]

    def mget_by_keys(self, keys: list[str]) -> list[dict[str, Any] | None]:
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
        _ttl = self._resolve_ttl()

        # Not transactional, so on a Redis Cluster the keys may span slots.
        with client.pipeline(transaction=False) as pipeline:
            # Refresh each TTL in the same pipeline; EXPIRE on a missing key is a no-op.
            for key in keys:
                pipeline.hgetall(key)
                if _ttl:
                    pipeline.expire(key, _ttl)
            results = pipeline.execute()

        # Replies alternate HGETALL, EXPIRE; keep the lookups.
        if _ttl:
            results = results[::2]
        return [self._process_cache_data(result) for result in results]

    def mget(
        self, contents: Iterable[bytes | str], model_name: str
    ) -> list[dict[str, Any] | None]:
        """Get multiple embeddings by their content and model name.

        Efficiently retrieves multiple cached embeddings in a single operation.
        If found, refreshes the TTL of each entry.

        Args:
            contents (Iterable[bytes | str]): Iterable of content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            List[Optional[Dict[str, Any]]]: List of embedding cache entries or None for contents not found.

        .. code-block:: python

            # Get multiple embeddings
            embedding_data = cache.mget(
                contents=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        contents = list(contents)
        keys = [self._make_cache_key(content, model_name) for content in contents]

        if not keys:
            return []

        results = self.mget_by_keys(keys)
        # Fall back to the legacy key format; remove in a future release.
        misses = [i for i, result in enumerate(results) if result is None]
        migrated = self._migrate_legacy_entries(contents, model_name, misses)
        for i, entry in migrated.items():
            results[i] = entry
        return results

    def set(
        self,
        content: bytes | str,
        model_name: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """Store an embedding with its content and model name.

        Args:
            content (Union[bytes, str]): The content to be embedded.
            model_name (str): The name of the embedding model.
            embedding (List[float]): The embedding vector to store.
            metadata (Optional[Dict[str, Any]]): Optional metadata to store with the embedding.
            ttl (Optional[int]): Optional TTL override for this specific entry.

        Returns:
            str: The Redis key where the embedding was stored.

        .. code-block:: python

            key = cache.set(
                content="What is machine learning?",
                model_name="text-embedding-ada-002",
                embedding=[0.1, 0.2, 0.3, ...],
                metadata={"source": "user_query"}
            )
        """
        if self._should_warn_for_async_only():
            if not EmbeddingsCache._warning_shown:
                logger.warning(
                    "EmbeddingsCache initialized with async_redis_client only. "
                    "Use async methods (aset) instead of sync methods (set)."
                )
                EmbeddingsCache._warning_shown = True

        item = {
            "content": content,
            "model_name": model_name,
            "embedding": embedding,
            "metadata": metadata,
        }
        return self.mset([item], ttl=ttl)[0]

    def mset(
        self,
        items: list[dict[str, Any]],
        ttl: int | None = None,
    ) -> list[str]:
        """Store multiple embeddings in a batch operation.

        Each item in the input list should be a dictionary with the following fields:
        - 'content': The input that was embedded
        - 'model_name': The name of the embedding model
        - 'embedding': The embedding vector
        - 'metadata': Optional metadata to store with the embedding

        Args:
            items: List of dictionaries, each containing content, model_name, embedding, and optional metadata.
            ttl: Optional TTL override for these entries.

        Returns:
            List[str]: List of Redis keys where the embeddings were stored.

        Note:
            The batch is pipelined, not transactional, so on a Redis Cluster it
            fans out across shards. If it fails partway, some entries will have
            been written; the operation is idempotent, so simply retry it.

        .. code-block:: python

            # Store multiple embeddings
            keys = cache.mset([
                {
                    "content": "What is ML?",
                    "model_name": "text-embedding-ada-002",
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {"source": "user"}
                },
                {
                    "content": "What is AI?",
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
        _ttl = self._resolve_ttl(ttl)

        with client.pipeline(transaction=False) as pipeline:
            # Process all entries
            for item in items:
                # Prepare and store
                key, cache_entry = self._prepare_entry_data(**item)
                keys.append(key)
                pipeline.hset(name=key, mapping=cache_entry)  # type: ignore[arg-type]
                # Queue the expiry with its write so no entry is ever left
                # unexpiring. HSET on its own leaves a key's TTL untouched.
                if _ttl:
                    pipeline.expire(key, _ttl)

            pipeline.execute()

        return keys

    def exists(self, content: bytes | str, model_name: str) -> bool:
        """Check if an embedding exists for the given content and model.

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            bool: True if the embedding exists in the cache, False otherwise.

        .. code-block:: python

            if cache.exists("What is machine learning?", "text-embedding-ada-002"):
                print("Embedding is in cache")
        """
        return self.mexists([content], model_name)[0]

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

    def mexists_by_keys(self, keys: list[str]) -> list[bool]:
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

    def mexists(self, contents: Iterable[bytes | str], model_name: str) -> list[bool]:
        """Check if multiple embeddings exist by their contents and model name.

        Efficiently checks existence of multiple embeddings in a single operation.

        Args:
            contents (Iterable[bytes | str]): Iterable of content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            List[bool]: List of boolean values indicating whether each embedding exists.

        .. code-block:: python

            # Check if multiple embeddings exist
            exists_results = cache.mexists(
                contents=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        contents = list(contents)
        keys = [self._make_cache_key(content, model_name) for content in contents]

        if not keys:
            return []

        results = self.mexists_by_keys(keys)
        # Fall back to the legacy key format; remove in a future release.
        misses = [i for i, found in enumerate(results) if not found]
        migrated = self._migrate_legacy_entries(contents, model_name, misses)
        for i in migrated:
            results[i] = True
        return results

    def drop(self, content: bytes | str, model_name: str) -> None:
        """Remove an embedding from the cache.

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.

        .. code-block:: python

            cache.drop(
                content="What is machine learning?",
                model_name="text-embedding-ada-002"
            )
        """
        self.mdrop([content], model_name)

    def drop_by_key(self, key: str) -> None:
        """Remove an embedding from the cache by its Redis key.

        Args:
            key (str): The full Redis key for the embedding.

        .. code-block:: python

            cache.drop_by_key("embedcache:1234567890abcdef")
        """
        client = self._get_redis_client()
        client.delete(key)

    def mdrop_by_keys(self, keys: list[str]) -> None:
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

    def mdrop(self, contents: Iterable[bytes | str], model_name: str) -> None:
        """Remove multiple embeddings from the cache by their contents and model name.

        Efficiently removes multiple embeddings in a single operation.

        Args:
            contents (Iterable[bytes | str]): Iterable of content that was embedded.
            model_name (str): The name of the embedding model.

        .. code-block:: python

            # Remove multiple embeddings
            cache.mdrop(
                contents=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        contents = list(contents)
        if not contents:
            return

        client = self._get_redis_client()
        with client.pipeline(transaction=False) as pipeline:
            for content in contents:
                pipeline.delete(self._make_cache_key(content, model_name))
                # Legacy key format fallback; remove in a future release.
                pipeline.hgetall(self._make_legacy_cache_key(content, model_name))
            replies = pipeline.execute()

        # Legacy IDs can collide, so only drop legacy entries stored for this input.
        legacy_keys = [
            self._make_legacy_cache_key(content, model_name)
            for content, data in zip(contents, replies[1::2])
            if data and self._legacy_entry_matches(data, content, model_name)
        ]
        self.mdrop_by_keys(legacy_keys)

    async def aget(
        self,
        content: bytes | str,
        model_name: str,
    ) -> dict[str, Any] | None:
        """Async get embedding by content and model name.

        Asynchronously retrieves a cached embedding for the given content and model name.
        If found, refreshes the TTL of the entry.

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            Optional[Dict[str, Any]]: Embedding cache entry or None if not found.

        .. code-block:: python

            embedding_data = await cache.aget(
                content="What is machine learning?",
                model_name="text-embedding-ada-002"
            )
        """
        return (await self.amget([content], model_name))[0]

    async def aget_by_key(self, key: str) -> dict[str, Any] | None:
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
        return (await self.amget_by_keys([key]))[0]

    async def amget_by_keys(self, keys: list[str]) -> list[dict[str, Any] | None]:
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
        _ttl = self._resolve_ttl()

        # Not transactional, so on a Redis Cluster the keys may span slots.
        async with client.pipeline(transaction=False) as pipeline:
            # Refresh each TTL in the same pipeline; EXPIRE on a missing key is a no-op.
            for key in keys:
                pipeline.hgetall(key)
                if _ttl:
                    pipeline.expire(key, _ttl)
            results = await pipeline.execute()

        # Replies alternate HGETALL, EXPIRE; keep the lookups.
        if _ttl:
            results = results[::2]
        return [self._process_cache_data(result) for result in results]

    async def amget(
        self, contents: Iterable[bytes | str], model_name: str
    ) -> list[dict[str, Any] | None]:
        """Async get multiple embeddings by their contents and model name.

        Asynchronously retrieves multiple cached embeddings in a single operation.
        If found, refreshes the TTL of each entry.

        Args:
            contents (Iterable[bytes | str]): Iterable of content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            List[Optional[Dict[str, Any]]]: List of embedding cache entries or None for contents not found.

        .. code-block:: python

            # Get multiple embeddings asynchronously
            embedding_data = await cache.amget(
                contents=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        contents = list(contents)
        keys = [self._make_cache_key(content, model_name) for content in contents]

        if not keys:
            return []

        results = await self.amget_by_keys(keys)
        # Fall back to the legacy key format; remove in a future release.
        misses = [i for i, result in enumerate(results) if result is None]
        migrated = await self._amigrate_legacy_entries(contents, model_name, misses)
        for i, entry in migrated.items():
            results[i] = entry
        return results

    async def aset(
        self,
        content: bytes | str,
        model_name: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """Async store an embedding with its content and model name.

        Asynchronously stores an embedding with its content and model name.

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.
            embedding (List[float]): The embedding vector to store.
            metadata (Optional[Dict[str, Any]]): Optional metadata to store with the embedding.
            ttl (Optional[int]): Optional TTL override for this specific entry.

        Returns:
            str: The Redis key where the embedding was stored.

        .. code-block:: python

            key = await cache.aset(
                content="What is machine learning?",
                model_name="text-embedding-ada-002",
                embedding=[0.1, 0.2, 0.3, ...],
                metadata={"source": "user_query"}
            )
        """
        item = {
            "content": content,
            "model_name": model_name,
            "embedding": embedding,
            "metadata": metadata,
        }
        return (await self.amset([item], ttl=ttl))[0]

    async def amset(
        self,
        items: list[dict[str, Any]],
        ttl: int | None = None,
    ) -> list[str]:
        """Async store multiple embeddings in a batch operation.

        Each item in the input list should be a dictionary with the following fields:
        - 'content': The content that was embedded
        - 'model_name': The name of the embedding model
        - 'embedding': The embedding vector
        - 'metadata': Optional metadata to store with the embedding

        Args:
            items: List of dictionaries, each containing content, model_name, embedding, and optional metadata.
            ttl: Optional TTL override for these entries.

        Returns:
            List[str]: List of Redis keys where the embeddings were stored.

        Note:
            The batch is pipelined, not transactional, so on a Redis Cluster it
            fans out across shards. If it fails partway, some entries will have
            been written; the operation is idempotent, so simply retry it.

        .. code-block:: python

            # Store multiple embeddings asynchronously
            keys = await cache.amset([
                {
                    "content": "What is ML?",
                    "model_name": "text-embedding-ada-002",
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {"source": "user"}
                },
                {
                    "content": "What is AI?",
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
        _ttl = self._resolve_ttl(ttl)

        async with client.pipeline(transaction=False) as pipeline:
            # Process all entries
            for item in items:
                # Prepare and store
                key, cache_entry = self._prepare_entry_data(**item)
                keys.append(key)
                # Never await a queued command: queueing is synchronous, and
                # awaiting a cluster pipeline calls initialize(), which clears
                # the queue. Only execute() is awaited.
                pipeline.hset(name=key, mapping=cache_entry)  # type: ignore[arg-type]
                # Queue the expiry with its write so no entry is ever left
                # unexpiring. HSET on its own leaves a key's TTL untouched.
                if _ttl:
                    pipeline.expire(key, _ttl)

            await pipeline.execute()

        return keys

    async def amexists_by_keys(self, keys: list[str]) -> list[bool]:
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
                pipeline.exists(key)
            results = await pipeline.execute()

        # Convert to boolean values
        return [bool(result) for result in results]

    async def amexists(
        self, contents: Iterable[bytes | str], model_name: str
    ) -> list[bool]:
        """Async check if multiple embeddings exist by their contents and model name.

        Asynchronously checks existence of multiple embeddings in a single operation.

        Args:
            contents (Iterable[bytes | str]): Iterable of content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            List[bool]: List of boolean values indicating whether each embedding exists.

        .. code-block:: python

            # Check if multiple embeddings exist asynchronously
            exists_results = await cache.amexists(
                contents=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        contents = list(contents)
        keys = [self._make_cache_key(content, model_name) for content in contents]

        if not keys:
            return []

        results = await self.amexists_by_keys(keys)
        # Fall back to the legacy key format; remove in a future release.
        misses = [i for i, found in enumerate(results) if not found]
        migrated = await self._amigrate_legacy_entries(contents, model_name, misses)
        for i in migrated:
            results[i] = True
        return results

    async def amdrop_by_keys(self, keys: list[str]) -> None:
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

    async def amdrop(self, contents: Iterable[bytes | str], model_name: str) -> None:
        """Async remove multiple embeddings from the cache by their contents and model name.

        Asynchronously removes multiple embeddings in a single operation.

        Args:
            contents (Iterable[bytes | str]): Iterable of content that was embedded.
            model_name (str): The name of the embedding model.

        .. code-block:: python

            # Remove multiple embeddings asynchronously
            await cache.amdrop(
                contents=["What is machine learning?", "What is deep learning?"],
                model_name="text-embedding-ada-002"
            )
        """
        contents = list(contents)
        if not contents:
            return

        client = await self._get_async_redis_client()
        async with client.pipeline(transaction=False) as pipeline:
            for content in contents:
                pipeline.delete(self._make_cache_key(content, model_name))
                # Legacy key format fallback; remove in a future release.
                pipeline.hgetall(self._make_legacy_cache_key(content, model_name))
            replies = await pipeline.execute()

        # Legacy IDs can collide, so only drop legacy entries stored for this input.
        legacy_keys = [
            self._make_legacy_cache_key(content, model_name)
            for content, data in zip(contents, replies[1::2])
            if data and self._legacy_entry_matches(data, content, model_name)
        ]
        await self.amdrop_by_keys(legacy_keys)

    async def aexists(self, content: bytes | str, model_name: str) -> bool:
        """Async check if an embedding exists.

        Asynchronously checks if an embedding exists for the given content and model.

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.

        Returns:
            bool: True if the embedding exists in the cache, False otherwise.

        .. code-block:: python

            if await cache.aexists("What is machine learning?", "text-embedding-ada-002"):
                print("Embedding is in cache")
        """
        return (await self.amexists([content], model_name))[0]

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

    async def adrop(self, content: bytes | str, model_name: str) -> None:
        """Async remove an embedding from the cache.

        Asynchronously removes an embedding from the cache.

        Args:
            content (bytes | str): The content that was embedded.
            model_name (str): The name of the embedding model.

        .. code-block:: python

            await cache.adrop(
                content="What is machine learning?",
                model_name="text-embedding-ada-002"
            )
        """
        await self.amdrop([content], model_name)

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
