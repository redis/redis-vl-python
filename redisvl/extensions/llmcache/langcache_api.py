import json
from typing import Any, Dict, List, Optional

from langcache import LangCache as LangCacheSDK

from redisvl.extensions.llmcache.base import BaseLLMCache
from redisvl.query.filter import FilterExpression
from redisvl.utils.utils import current_timestamp, hashify


class LangCache(BaseLLMCache):
    """Redis LangCache Service: API for managing a Redis LangCache"""

    def __init__(
        self,
        redis_client=None,
        name: str = "llmcache",
        distance_threshold: float = 0.1,
        ttl: Optional[int] = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: Dict[str, Any] = {},
        overwrite: bool = False,
        **kwargs,
    ):
        """Initialize a LangCache client.

        Args:
            redis_client: A Redis client instance.
            name: Name of the cache.
            distance_threshold: Threshold for semantic similarity (0.0 to 1.0).
            ttl: Time-to-live for cache entries in seconds.
            redis_url: URL for Redis connection if no client is provided.
            connection_kwargs: Additional Redis connection parameters.
            overwrite: Whether to overwrite an existing cache with the same name.
        """
        # Initialize the base class
        super().__init__(ttl)

        # Store configuration
        self._name = name
        self._redis_client = redis_client
        self._redis_url = redis_url
        self._distance_threshold = distance_threshold
        self._ttl = ttl
        self._cache_id = name

        # Initialize LangCache SDK client
        self._api = LangCacheSDK(server_url=redis_url, client=redis_client)

        # Create cache if it doesn't exist or if overwrite is True
        try:
            existing_cache = self._api.cache.get(cache_id=self._cache_id)
            if not existing_cache and overwrite:
                self._api.cache.create(
                    index_name=self._name,
                    redis_urls=[self._redis_url],
                )
        except Exception:
            # If the cache doesn't exist, create it
            if overwrite:
                self._api.cache.create(
                    index_name=self._name,
                    redis_urls=[self._redis_url],
                )

    @property
    def distance_threshold(self) -> float:
        """Get the current distance threshold for semantic similarity."""
        return self._distance_threshold

    def set_threshold(self, distance_threshold: float) -> None:
        """Sets the semantic distance threshold for the cache.

        Args:
            distance_threshold: The semantic distance threshold.

        Raises:
            ValueError: If the threshold is not between 0 and 2.
        """
        if not 0 <= float(distance_threshold) <= 2:
            raise ValueError("Distance threshold must be between 0 and 2")
        self._distance_threshold = float(distance_threshold)

    @property
    def ttl(self) -> Optional[int]:
        """Get the current TTL setting for cache entries."""
        return self._ttl

    def set_ttl(self, ttl: Optional[int] = None) -> None:
        """Set the TTL for cache entries.

        Args:
            ttl: Time-to-live in seconds, or None to disable expiration.

        Raises:
            ValueError: If ttl is negative.
        """
        if ttl is not None and ttl < 0:
            raise ValueError("TTL must be a positive integer or None")
        self._ttl = ttl

    def clear(self) -> None:
        """Clear all entries from the cache while preserving the cache configuration."""
        self._api.entries.delete_all(cache_id=self._cache_id, attributes={}, scope={})

    async def aclear(self) -> None:
        """Asynchronously clear all entries from the cache."""
        # Currently using synchronous implementation since langcache doesn't have async API
        self.clear()

    def delete(self) -> None:
        """Delete the cache and all its entries."""
        self.clear()
        self._api.cache.delete(cache_id=self._cache_id)

    async def adelete(self) -> None:
        """Asynchronously delete the cache and all its entries."""
        # Currently using synchronous implementation since langcache doesn't have async API
        self.delete()

    def drop(
        self, ids: Optional[List[str]] = None, keys: Optional[List[str]] = None
    ) -> None:
        """Remove specific entries from the cache.

        Args:
            ids: List of entry IDs to remove.
            keys: List of Redis keys to remove.
        """
        if ids:
            for entry_id in ids:
                self._api.entries.delete(entry_id=entry_id, cache_id=self._cache_id)

    async def adrop(
        self, ids: Optional[List[str]] = None, keys: Optional[List[str]] = None
    ) -> None:
        """Asynchronously remove specific entries from the cache.

        Args:
            ids: List of entry IDs to remove.
            keys: List of Redis keys to remove.
        """
        # Currently using synchronous implementation since langcache doesn't have async API
        self.drop(ids, keys)

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Check the cache for semantically similar entries.

        Args:
            prompt: The text prompt to search for.
            vector: The vector representation of the prompt.
            num_results: Maximum number of results to return.
            return_fields: Fields to include in the response.
            filter_expression: Optional filter for the search.
            distance_threshold: Override the default distance threshold.

        Returns:
            List of matching cache entries.

        Raises:
            ValueError: If neither prompt nor vector is provided.
            TypeError: If return_fields is not a list when provided.
        """
        if not any([prompt, vector]):
            raise ValueError("Either prompt or vector must be provided")

        if return_fields and not isinstance(return_fields, list):
            raise TypeError("return_fields must be a list")

        # Use provided threshold or default
        threshold = distance_threshold or self._distance_threshold

        # Search the cache - note we don't use scope since FilterExpression conversion would be complex
        # and require proper implementation for CacheEntryScope format
        results = self._api.entries.search(
            cache_id=self._cache_id,
            prompt=prompt or "",  # Ensure prompt is never None
            similarity_threshold=threshold,
        )

        # If we need to limit results and have more than requested, slice the list
        if num_results < len(results):
            results = results[:num_results]

        # Process and format results
        cache_hits = []
        for result in results:
            # Create a basic hit dict with required fields
            hit = {
                "key": result.id,
                "entry_id": result.id,
                "prompt": result.prompt,
                "response": result.response,
                "vector_distance": result.similarity,
                "inserted_at": current_timestamp(),  # Not available in the model
                "updated_at": current_timestamp(),  # Not available in the model
            }

            # Add metadata if it exists
            if hasattr(result, "metadata") and result.metadata:
                try:
                    metadata_dict = {}
                    # Convert metadata object to dict if possible
                    if hasattr(result.metadata, "__dict__"):
                        metadata_dict = {
                            k: v
                            for k, v in result.metadata.__dict__.items()
                            if not k.startswith("_")
                        }
                    hit["metadata"] = metadata_dict
                except Exception:
                    hit["metadata"] = {}

            # Filter return fields if specified
            if return_fields:
                hit = {k: v for k, v in hit.items() if k in return_fields or k == "key"}

            cache_hits.append(hit)

        return cache_hits

    async def acheck(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously check the cache for semantically similar entries."""
        # Currently using synchronous implementation since langcache doesn't have async API
        return self.check(
            prompt,
            vector,
            num_results,
            return_fields,
            filter_expression,
            distance_threshold,
        )

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store a new entry in the cache.

        Args:
            prompt: The prompt text.
            response: The response text.
            vector: Optional vector representation of the prompt.
            metadata: Optional metadata to store with the entry.
            filters: Optional filters to associate with the entry.
            ttl: Optional custom TTL for this entry.

        Returns:
            The ID of the created entry.
        """
        # Validate metadata
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        # Create entry with optional TTL
        entry_ttl = ttl if ttl is not None else self._ttl

        # Convert ttl to ttl_millis (milliseconds) if provided
        ttl_millis = entry_ttl * 1000 if entry_ttl is not None else None

        # Process additional attributes from filters
        attributes = {}
        if filters:
            attributes.update(filters)

        # Add metadata to attributes if provided
        if metadata:
            attributes["metadata"] = (
                json.dumps(metadata) if isinstance(metadata, dict) else metadata
            )

        # Store the entry and get the response
        create_response = self._api.entries.create(
            cache_id=self._cache_id,
            prompt=prompt,
            response=response,
            attributes=attributes,
            ttl_millis=ttl_millis,
        )

        # Return the entry ID from the response
        return create_response.entry_id

    async def astore(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Asynchronously store a new entry in the cache."""
        # Currently using synchronous implementation since langcache doesn't have async API
        return self.store(prompt, response, vector, metadata, filters, ttl)

    def update(self, key: str, **kwargs) -> None:
        """Update an existing cache entry.

        Args:
            key: The entry ID to update.
            **kwargs: Fields to update (prompt, response, metadata, etc.)
        """
        # Find the entry to update
        existing_entries = self._api.entries.search(
            cache_id=self._cache_id,
            prompt="",  # Required parameter but we're searching by ID
            attributes={"id": key},  # Search by ID as an attribute
            similarity_threshold=1.0,  # We're not doing semantic search
        )

        if not existing_entries:
            return

        existing_entry = existing_entries[0]

        # Prepare updated values
        # CacheEntry objects are Pydantic models, access their attributes directly
        prompt = kwargs.get(
            "prompt", existing_entry.prompt if hasattr(existing_entry, "prompt") else ""
        )
        response = kwargs.get(
            "response",
            existing_entry.response if hasattr(existing_entry, "response") else "",
        )

        # Prepare attributes for update
        attributes = {}
        if "metadata" in kwargs:
            attributes["metadata"] = (
                json.dumps(kwargs["metadata"])
                if isinstance(kwargs["metadata"], dict)
                else kwargs["metadata"]
            )

        # Convert TTL to milliseconds if provided
        ttl = kwargs.get("ttl", None)
        ttl_millis = ttl * 1000 if ttl is not None else None

        # Re-create the entry with updated values
        self._api.entries.create(
            cache_id=self._cache_id,
            prompt=prompt,
            response=response,
            attributes=attributes,
            ttl_millis=ttl_millis,
        )

    async def aupdate(self, key: str, **kwargs) -> None:
        """Asynchronously update an existing cache entry."""
        # Currently using synchronous implementation since langcache doesn't have async API
        self.update(key, **kwargs)

    def disconnect(self) -> None:
        """Close the Redis connection."""
        # Redis clients typically don't need explicit disconnection,
        # as they use connection pooling
        pass

    async def adisconnect(self) -> None:
        """Asynchronously close the Redis connection."""
        self.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.adisconnect()
