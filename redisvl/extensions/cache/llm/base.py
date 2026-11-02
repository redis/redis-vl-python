"""Base LLM cache interface for RedisVL.

This module defines the abstract base interface for LLM caches, which store
prompt-response pairs with semantic retrieval capabilities.
"""

import time
from typing import Any

from redisvl.extensions.cache.base import BaseCache
from redisvl.extensions.cache.llm.stats import CacheStats, _CacheCounters
from redisvl.query.filter import FilterExpression


class BaseLLMCache(BaseCache):
    """Base abstract LLM cache interface.

    This class defines the core functionality for caching LLM responses
    with semantic similarity search capabilities.
    """

    def __init__(self, name: str, ttl: int | None = None, **kwargs):
        """Initialize an LLM cache.

        Args:
            name (str): The name of the cache.
            ttl (Optional[int]): The time-to-live for cached responses. Defaults to None.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(name=name, ttl=ttl, **kwargs)
        self._stats = _CacheCounters()

    @property
    def stats(self) -> CacheStats:
        """A snapshot of this cache's own check and store counters.

        The counters are updated by the bundled caches' ``check``/``acheck`` and
        ``store``/``astore`` calls, need no setup, and cost one
        ``perf_counter()`` pair per check. A subclass that overrides one of them
        is counted only if it calls ``super()``, and must not add the counting
        decorators itself or the call is counted twice. The counters are per
        process: workers that share a cache name each count their own traffic.
        Nothing is written to Redis, and no prompt, response or metadata content
        is recorded.

        Returns:
            CacheStats: hits, misses, errors, stores and lookup time, plus the
            derived ``hit_rate`` and ``avg_lookup_seconds``.

        .. code-block:: python

            cache.check(prompt="What is the capital of France?")
            cache.store(prompt="What is the capital of Italy?", response="Rome")

            stats = cache.stats
            print(stats.hits, stats.misses, stats.hit_rate)
        """
        return self._stats.snapshot()

    def _record_check(self, start: float, results: list[dict[str, Any]]) -> None:
        """Count a check that returned; any returned entry makes it a hit."""
        self._stats.checks += 1
        self._stats.lookup_seconds += time.perf_counter() - start
        if results:
            self._stats.hits += 1
        else:
            self._stats.misses += 1

    def _record_error(self, start: float) -> None:
        """Count a check that raised, so that checks = hits + misses + errors."""
        self._stats.checks += 1
        self._stats.errors += 1
        self._stats.lookup_seconds += time.perf_counter() - start

    def delete(self) -> None:
        """Delete the cache and its index entirely."""
        raise NotImplementedError

    async def adelete(self) -> None:
        """Async delete the cache and its index entirely."""
        raise NotImplementedError

    def check(
        self,
        prompt: str | None = None,
        vector: list[float] | None = None,
        num_results: int = 1,
        return_fields: list[str] | None = None,
        filter_expression: FilterExpression | None = None,
        distance_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Check the cache for semantically similar prompts.

        Args:
            prompt (Optional[str]): The text prompt to search for in the cache.
            vector (Optional[List[float]]): Vector representation to search for.
            num_results (int): Number of results to return. Defaults to 1.
            return_fields (Optional[List[str]]): Fields to return in results.
            filter_expression (Optional[FilterExpression]): Optional filter to apply.
            distance_threshold (Optional[float]): Override for semantic distance threshold.

        Returns:
            List[Dict[str, Any]]: List of matching cache entries.
        """
        raise NotImplementedError

    async def acheck(
        self,
        prompt: str | None = None,
        vector: list[float] | None = None,
        num_results: int = 1,
        return_fields: list[str] | None = None,
        filter_expression: FilterExpression | None = None,
        distance_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Async check the cache for semantically similar prompts."""
        raise NotImplementedError

    def store(
        self,
        prompt: str,
        response: str,
        vector: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """Store a prompt-response pair in the cache.

        Args:
            prompt (str): The user prompt to cache.
            response (str): The LLM response to cache.
            vector (Optional[List[float]]): Optional embedding vector.
            metadata (Optional[Dict[str, Any]]): Optional metadata.
            filters (Optional[Dict[str, Any]]): Optional filters for retrieval.
            ttl (Optional[int]): Optional TTL override.

        Returns:
            str: The Redis key for the cached entry.
        """
        raise NotImplementedError

    async def astore(
        self,
        prompt: str,
        response: str,
        vector: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """Async store a prompt-response pair in the cache."""
        raise NotImplementedError

    def update(self, key: str, **kwargs) -> None:
        """Update specific fields within an existing cache entry.

        Args:
            key (str): The key of the document to update.
            **kwargs: Field-value pairs to update.
        """
        raise NotImplementedError

    async def aupdate(self, key: str, **kwargs) -> None:
        """Async update specific fields within an existing cache entry."""
        raise NotImplementedError
