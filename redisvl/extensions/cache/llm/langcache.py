"""LangCache API-based LLM cache implementation for RedisVL.

This module provides an LLM cache implementation that uses the LangCache
managed service via the langcache Python SDK.
"""

from typing import Any, Dict, List, Optional

from redisvl.extensions.cache.llm.base import BaseLLMCache
from redisvl.extensions.cache.llm.schema import CacheHit
from redisvl.query.filter import FilterExpression
from redisvl.utils.log import get_logger

logger = get_logger(__name__)

try:
    from langcache import LangCache as LangCacheClient
    from langcache.models import SearchStrategy

    LANGCACHE_AVAILABLE = True
except ImportError:
    LANGCACHE_AVAILABLE = False
    LangCacheClient = None  # type: ignore
    SearchStrategy = None  # type: ignore


class LangCacheWrapper(BaseLLMCache):
    """LLM Cache implementation using the LangCache managed service.

    This cache uses the LangCache API service for semantic caching of LLM
    responses. It requires a LangCache account and API key.

    Example:
        .. code-block:: python

            from redisvl.extensions.cache.llm import LangCacheWrapper

            cache = LangCacheWrapper(
                name="my_cache",
                server_url="https://api.langcache.com",
                cache_id="your-cache-id",
                api_key="your-api-key",
                ttl=3600
            )

            # Store a response
            cache.store(
                prompt="What is the capital of France?",
                response="Paris"
            )

            # Check for cached responses
            results = cache.check(prompt="What is the capital of France?")
    """

    def __init__(
        self,
        name: str = "langcache",
        server_url: str = "https://api.langcache.com",
        cache_id: str = "",
        api_key: str = "",
        ttl: Optional[int] = None,
        use_exact_search: bool = True,
        use_semantic_search: bool = True,
        **kwargs,
    ):
        """Initialize a LangCache wrapper.

        Args:
            name (str): The name of the cache. Defaults to "langcache".
            server_url (str): The LangCache server URL.
            cache_id (str): The LangCache cache ID.
            api_key (str): The LangCache API key.
            ttl (Optional[int]): Time-to-live for cache entries in seconds.
            use_exact_search (bool): Whether to use exact matching. Defaults to True.
            use_semantic_search (bool): Whether to use semantic search. Defaults to True.
            **kwargs: Additional arguments (ignored for compatibility).

        Raises:
            ImportError: If the langcache package is not installed.
            ValueError: If cache_id or api_key is not provided.
        """
        if not LANGCACHE_AVAILABLE:
            raise ImportError(
                "The langcache package is required to use LangCacheWrapper. "
                "Install it with: pip install langcache"
            )

        if not cache_id:
            raise ValueError("cache_id is required for LangCacheWrapper")
        if not api_key:
            raise ValueError("api_key is required for LangCacheWrapper")

        super().__init__(name=name, ttl=ttl, **kwargs)

        self._server_url = server_url
        self._cache_id = cache_id
        self._api_key = api_key

        # Determine search strategies
        self._search_strategies = []
        if use_exact_search:
            self._search_strategies.append(SearchStrategy.EXACT)
        if use_semantic_search:
            self._search_strategies.append(SearchStrategy.SEMANTIC)

        if not self._search_strategies:
            raise ValueError(
                "At least one of use_exact_search or use_semantic_search must be True"
            )

        # Initialize the LangCache client
        self._client = LangCacheClient(
            server_url=self._server_url,
            cache_id=self._cache_id,
            api_key=self._api_key,
        )

    def _convert_to_cache_hit(self, result: Dict[str, Any]) -> CacheHit:
        """Convert a LangCache result to a CacheHit object.

        Args:
            result (Dict[str, Any]): The result from LangCache.

        Returns:
            CacheHit: The converted cache hit.
        """
        # Extract attributes (metadata) from the result
        attributes = result.get("attributes", {})

        # LangCache returns similarity (0-1, higher is better)
        # Convert to distance (lower is better) for consistency
        similarity = result.get("similarity", 0.0)
        distance = 1.0 - similarity if similarity else 0.0

        return CacheHit(
            entry_id=result.get("id", ""),
            prompt=result.get("prompt", ""),
            response=result.get("response", ""),
            vector_distance=distance,
            inserted_at=result.get("created_at", 0.0),
            updated_at=result.get("updated_at", 0.0),
            metadata=attributes if attributes else None,
        )

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Check the cache for semantically similar prompts.

        Args:
            prompt (Optional[str]): The text prompt to search for.
            vector (Optional[List[float]]): Not supported by LangCache API.
            num_results (int): Number of results to return. Defaults to 1.
            return_fields (Optional[List[str]]): Not used (for compatibility).
            filter_expression (Optional[FilterExpression]): Not supported.
            distance_threshold (Optional[float]): Maximum distance threshold.
                Converted to similarity_threshold (1.0 - distance_threshold).
            attributes (Optional[Dict[str, str]]): LangCache attributes to filter by.
                Note: Attributes must be pre-configured in your LangCache instance.

        Returns:
            List[Dict[str, Any]]: List of matching cache entries.

        Raises:
            ValueError: If prompt is not provided.
        """
        if not prompt:
            raise ValueError("prompt is required for LangCache search")

        if vector is not None:
            logger.warning("LangCache does not support vector search directly")

        if filter_expression is not None:
            logger.warning("LangCache does not support filter expressions")

        # Convert distance_threshold to similarity_threshold
        # Distance: lower is better (0.0 = exact match)
        # Similarity: higher is better (1.0 = exact match)
        similarity_threshold = None
        if distance_threshold is not None:
            similarity_threshold = 1.0 - distance_threshold

        # Search using the LangCache client
        # The client itself is the context manager
        search_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "search_strategies": self._search_strategies,
            "similarity_threshold": similarity_threshold,
        }

        # Add attributes if provided
        if attributes:
            search_kwargs["attributes"] = attributes

        response = self._client.search(**search_kwargs)

        # Convert results to cache hits
        # Response is a SearchResponse Pydantic model with a 'data' attribute
        results = response.data if hasattr(response, "data") else []
        cache_hits = []
        for result in results[:num_results]:
            # Convert CacheEntry to dict
            result_dict: Dict[str, Any]
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            else:
                result_dict = dict(result)  # type: ignore[arg-type]
            hit = self._convert_to_cache_hit(result_dict)
            cache_hits.append(hit.to_dict())

        return cache_hits

    async def acheck(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Async check the cache for semantically similar prompts.

        Args:
            prompt (Optional[str]): The text prompt to search for.
            vector (Optional[List[float]]): Not supported by LangCache API.
            num_results (int): Number of results to return. Defaults to 1.
            return_fields (Optional[List[str]]): Not used (for compatibility).
            filter_expression (Optional[FilterExpression]): Not supported.
            distance_threshold (Optional[float]): Maximum distance threshold.
                Converted to similarity_threshold (1.0 - distance_threshold).
            attributes (Optional[Dict[str, str]]): LangCache attributes to filter by.
                Note: Attributes must be pre-configured in your LangCache instance.

        Returns:
            List[Dict[str, Any]]: List of matching cache entries.

        Raises:
            ValueError: If prompt is not provided.
        """
        if not prompt:
            raise ValueError("prompt is required for LangCache search")

        if vector is not None:
            logger.warning("LangCache does not support vector search directly")

        if filter_expression is not None:
            logger.warning("LangCache does not support filter expressions")

        # Convert distance_threshold to similarity_threshold
        # Distance: lower is better (0.0 = exact match)
        # Similarity: higher is better (1.0 = exact match)
        similarity_threshold = None
        if distance_threshold is not None:
            similarity_threshold = 1.0 - distance_threshold

        # Search using the LangCache client (async)
        # The client itself is the context manager
        search_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "search_strategies": self._search_strategies,
            "similarity_threshold": similarity_threshold,
        }

        # Add attributes if provided
        if attributes:
            search_kwargs["attributes"] = attributes

        response = await self._client.search_async(**search_kwargs)

        # Convert results to cache hits
        # Response is a SearchResponse Pydantic model with a 'data' attribute
        results = response.data if hasattr(response, "data") else []
        cache_hits = []
        for result in results[:num_results]:
            # Convert CacheEntry to dict
            result_dict: Dict[str, Any]
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            else:
                result_dict = dict(result)  # type: ignore[arg-type]
            hit = self._convert_to_cache_hit(result_dict)
            cache_hits.append(hit.to_dict())

        return cache_hits

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
            vector (Optional[List[float]]): Not supported by LangCache API.
            metadata (Optional[Dict[str, Any]]): Optional metadata (stored as attributes).
            filters (Optional[Dict[str, Any]]): Not supported.
            ttl (Optional[int]): Optional TTL override (not supported by LangCache).

        Returns:
            str: The entry ID for the cached entry.

        Raises:
            ValueError: If prompt or response is empty.
        """
        if not prompt:
            raise ValueError("prompt is required")
        if not response:
            raise ValueError("response is required")

        if vector is not None:
            logger.warning("LangCache does not support custom vectors")

        if filters is not None:
            logger.warning("LangCache does not support filters")

        if ttl is not None:
            logger.warning("LangCache does not support per-entry TTL")

        # Store using the LangCache client
        # The client itself is the context manager
        # Only pass attributes if metadata is provided
        # Some caches may not have attributes configured
        if metadata:
            result = self._client.set(
                prompt=prompt, response=response, attributes=metadata
            )
        else:
            result = self._client.set(prompt=prompt, response=response)

        # Return the entry ID
        # Result is a SetResponse Pydantic model with entry_id attribute
        return result.entry_id if hasattr(result, "entry_id") else ""

    async def astore(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Async store a prompt-response pair in the cache.

        Args:
            prompt (str): The user prompt to cache.
            response (str): The LLM response to cache.
            vector (Optional[List[float]]): Not supported by LangCache API.
            metadata (Optional[Dict[str, Any]]): Optional metadata (stored as attributes).
            filters (Optional[Dict[str, Any]]): Not supported.
            ttl (Optional[int]): Optional TTL override (not supported by LangCache).

        Returns:
            str: The entry ID for the cached entry.

        Raises:
            ValueError: If prompt or response is empty.
        """
        if not prompt:
            raise ValueError("prompt is required")
        if not response:
            raise ValueError("response is required")

        if vector is not None:
            logger.warning("LangCache does not support custom vectors")

        if filters is not None:
            logger.warning("LangCache does not support filters")

        if ttl is not None:
            logger.warning("LangCache does not support per-entry TTL")

        # Store using the LangCache client (async)
        # The client itself is the context manager
        # Only pass attributes if metadata is provided
        # Some caches may not have attributes configured
        if metadata:
            result = await self._client.set_async(
                prompt=prompt, response=response, attributes=metadata
            )
        else:
            result = await self._client.set_async(prompt=prompt, response=response)

        # Return the entry ID
        # Result is a SetResponse Pydantic model with entry_id attribute
        return result.entry_id if hasattr(result, "entry_id") else ""

    def update(self, key: str, **kwargs) -> None:
        """Update specific fields within an existing cache entry.

        Note: LangCache API does not support updating individual entries.
        This method will raise NotImplementedError.

        Args:
            key (str): The key of the document to update.
            **kwargs: Field-value pairs to update.

        Raises:
            NotImplementedError: LangCache does not support entry updates.
        """
        raise NotImplementedError(
            "LangCache API does not support updating individual entries. "
            "Delete and re-create the entry instead."
        )

    async def aupdate(self, key: str, **kwargs) -> None:
        """Async update specific fields within an existing cache entry.

        Note: LangCache API does not support updating individual entries.
        This method will raise NotImplementedError.

        Args:
            key (str): The key of the document to update.
            **kwargs: Field-value pairs to update.

        Raises:
            NotImplementedError: LangCache does not support entry updates.
        """
        raise NotImplementedError(
            "LangCache API does not support updating individual entries. "
            "Delete and re-create the entry instead."
        )

    def delete(self) -> None:
        """Delete the entire cache.

        This deletes all entries in the cache by calling delete_query
        with no attributes.
        """
        self._client.delete_query(attributes={})

    async def adelete(self) -> None:
        """Async delete the entire cache.

        This deletes all entries in the cache by calling delete_query
        with no attributes.
        """
        await self._client.delete_query_async(attributes={})

    def clear(self) -> None:
        """Clear the cache of all entries.

        This is an alias for delete() to match the BaseCache interface.
        """
        self.delete()

    async def aclear(self) -> None:
        """Async clear the cache of all entries.

        This is an alias for adelete() to match the BaseCache interface.
        """
        await self.adelete()

    def delete_by_id(self, entry_id: str) -> None:
        """Delete a single cache entry by ID.

        Args:
            entry_id (str): The ID of the entry to delete.
        """
        self._client.delete_by_id(entry_id=entry_id)

    async def adelete_by_id(self, entry_id: str) -> None:
        """Async delete a single cache entry by ID.

        Args:
            entry_id (str): The ID of the entry to delete.
        """
        await self._client.delete_by_id_async(entry_id=entry_id)

    def delete_by_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Delete cache entries matching the given attributes.

        Args:
            attributes (Dict[str, Any]): Attributes to match for deletion.

        Returns:
            Dict[str, Any]: Result of the deletion operation.
        """
        result = self._client.delete_query(attributes=attributes)
        # Convert DeleteQueryResponse to dict
        return result.model_dump() if hasattr(result, "model_dump") else {}

    async def adelete_by_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Async delete cache entries matching the given attributes.

        Args:
            attributes (Dict[str, Any]): Attributes to match for deletion.

        Returns:
            Dict[str, Any]: Result of the deletion operation.
        """
        result = await self._client.delete_query_async(attributes=attributes)
        # Convert DeleteQueryResponse to dict
        return result.model_dump() if hasattr(result, "model_dump") else {}
