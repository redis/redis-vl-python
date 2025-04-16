"""Base LLM cache interface for RedisVL.

This module defines the abstract base interface for LLM caches, which store
prompt-response pairs with semantic retrieval capabilities.
"""

from typing import Any, Dict, List, Optional

from redisvl.extensions.cache.base import BaseCache
from redisvl.query.filter import FilterExpression


class BaseLLMCache(BaseCache):
    """Base abstract LLM cache interface.

    This class defines the core functionality for caching LLM responses
    with semantic similarity search capabilities.
    """

    def __init__(self, name: str, ttl: Optional[int] = None, **kwargs):
        """Initialize an LLM cache.

        Args:
            name (str): The name of the cache.
            ttl (Optional[int]): The time-to-live for cached responses. Defaults to None.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(name=name, ttl=ttl, **kwargs)

    def delete(self) -> None:
        """Delete the cache and its index entirely."""
        raise NotImplementedError

    async def adelete(self) -> None:
        """Async delete the cache and its index entirely."""
        raise NotImplementedError

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
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
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Async check the cache for semantically similar prompts."""
        raise NotImplementedError

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
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
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
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
