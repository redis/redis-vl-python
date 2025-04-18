import json
from typing import Any, Dict, List, Optional, Union

from langcache import LangCache as LangCacheSDK
from langcache.models import CacheEntryScope, CacheEntryScopeTypedDict

from redisvl.extensions.cache.llm.base import BaseLLMCache
from redisvl.query.filter import FilterExpression
from redisvl.utils.utils import current_timestamp

Scope = Optional[Union[CacheEntryScope, CacheEntryScopeTypedDict]]


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
        entry_scope: Scope = None,
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
            entry_scope: Optional scope for cache entries.
        """
        # Initialize the base class
        super().__init__(
            name=name,
            ttl=ttl,
            redis_client=redis_client,
            redis_url=redis_url,
            connection_kwargs=connection_kwargs,
        )

        # Store configuration
        self._name = name
        self._redis_client = redis_client
        self._redis_url = redis_url
        self._distance_threshold = distance_threshold
        self._ttl = ttl
        self._cache_id = name
        self._entry_scope = entry_scope
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
        self._api.entries.delete_all(
            cache_id=self._cache_id,
            attributes={},
            scope=(
                self._entry_scope
                if self._entry_scope is not None
                else CacheEntryScope()
            ),
        )

    async def aclear(self) -> None:
        """Async clear all entries from the cache while preserving the cache configuration."""
        await self._api.entries.delete_all_async(
            cache_id=self._cache_id,
            attributes={},
            scope=(
                self._entry_scope
                if self._entry_scope is not None
                else CacheEntryScope()
            ),
        )

    def delete(self) -> None:
        """Delete the cache and its index entirely."""
        # First delete all entries
        self.clear()
        # Then delete the cache configuration
        self._api.cache.delete(cache_id=self._cache_id)

    async def adelete(self) -> None:
        """Async delete the cache and its index entirely."""
        # First delete all entries
        await self.aclear()
        # Then delete the cache configuration
        await self._api.cache.delete_async(cache_id=self._cache_id)

    def drop(self, ids: List[str]) -> None:
        """Delete specific entries from the cache by their IDs.

        Args:
            ids: List of entry IDs to delete.
        """
        for entry_id in ids:
            self._api.entries.delete(
                cache_id=self._cache_id,
                entry_id=entry_id,
            )

    async def adrop(self, ids: List[str]) -> None:
        """Async delete specific entries from the cache by their IDs.

        Args:
            ids: List of entry IDs to delete.
        """
        for entry_id in ids:
            await self._api.entries.delete_async(
                cache_id=self._cache_id,
                entry_id=entry_id,
            )

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
        entry_scope: Scope = None,
    ) -> List[Dict[str, Any]]:
        """Check the cache for semantically similar prompts.

        Args:
            prompt: The text prompt to search for in the cache.
            vector: Vector representation to search for (not supported in LangCache).
            num_results: Number of results to return. Defaults to 1.
            return_fields: Fields to return in results.
            filter_expression: Optional filter to apply (not supported in LangCache).
            distance_threshold: Override for semantic distance threshold.
            entry_scope: Optional scope for cache entries.

        Returns:
            List of matching cache entries.

        Raises:
            ValueError: If neither prompt nor vector is provided.
            TypeError: If return_fields is not a list.
        """
        if prompt is None:
            raise ValueError("Must provide a prompt to check the cache")

        if vector is not None:
            # LangCache doesn't support direct vector search
            raise ValueError("Vector search is not supported in LangCache")

        if return_fields is not None and not isinstance(return_fields, list):
            raise TypeError("return_fields must be a list of field names")

        # Use the provided threshold or fall back to the instance default
        threshold = (
            distance_threshold
            if distance_threshold is not None
            else self._distance_threshold
        )

        # Use the provided scope or fall back to the instance default
        scope = entry_scope if entry_scope is not None else self._entry_scope

        # Search for similar entries
        entries = self._api.entries.search(
            cache_id=self._cache_id,
            prompt=prompt,
            similarity_threshold=threshold,
            scope=scope,
        )

        # Format the results
        results = []
        for entry in entries[:num_results]:
            # Create a base result with required fields
            result = {
                "key": entry.id,
                "entry_id": entry.id,
                "prompt": entry.prompt,
                "response": entry.response,
                "vector_distance": entry.similarity,
            }

            # Add metadata if available
            if hasattr(entry, "metadata") and entry.metadata is not None:
                # Convert metadata object to dict
                if not isinstance(entry.metadata, dict):
                    metadata_dict = {}
                    for key in dir(entry.metadata):
                        if not key.startswith("_"):
                            metadata_dict[key] = getattr(entry.metadata, key)
                    result["metadata"] = metadata_dict
                else:
                    result["metadata"] = entry.metadata

            # Add timestamps if available
            if hasattr(entry, "inserted_at") and entry.inserted_at is not None:
                result["inserted_at"] = entry.inserted_at
            if hasattr(entry, "updated_at") and entry.updated_at is not None:
                result["updated_at"] = entry.updated_at

            # Filter fields if requested
            if return_fields:
                result = {
                    k: v for k, v in result.items() if k in return_fields or k == "key"
                }

            results.append(result)

        return results

    async def acheck(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
        entry_scope: Scope = None,
    ) -> List[Dict[str, Any]]:
        """Async check the cache for semantically similar prompts.

        Args:
            prompt: The text prompt to search for in the cache.
            vector: Vector representation to search for (not supported in LangCache).
            num_results: Number of results to return. Defaults to 1.
            return_fields: Fields to return in results.
            filter_expression: Optional filter to apply (not supported in LangCache).
            distance_threshold: Override for semantic distance threshold.
            entry_scope: Optional scope for cache entries.

        Returns:
            List of matching cache entries.

        Raises:
            ValueError: If neither prompt nor vector is provided.
            TypeError: If return_fields is not a list.
        """
        if prompt is None:
            raise ValueError("Must provide a prompt to check the cache")

        if vector is not None:
            # LangCache doesn't support direct vector search
            raise ValueError("Vector search is not supported in LangCache")

        if return_fields is not None and not isinstance(return_fields, list):
            raise TypeError("return_fields must be a list of field names")

        # Use the provided threshold or fall back to the instance default
        threshold = (
            distance_threshold
            if distance_threshold is not None
            else self._distance_threshold
        )

        # Use the provided scope or fall back to the instance default
        scope = entry_scope if entry_scope is not None else self._entry_scope

        # Search for similar entries
        entries = await self._api.entries.search_async(
            cache_id=self._cache_id,
            prompt=prompt,
            similarity_threshold=threshold,
            scope=scope,
        )

        # Format the results
        results = []
        for entry in entries[:num_results]:
            # Create a base result with required fields
            result = {
                "key": entry.id,
                "entry_id": entry.id,
                "prompt": entry.prompt,
                "response": entry.response,
                "vector_distance": entry.similarity,
            }

            # Add metadata if available
            if hasattr(entry, "metadata") and entry.metadata is not None:
                # Convert metadata object to dict
                if not isinstance(entry.metadata, dict):
                    metadata_dict = {}
                    for key in dir(entry.metadata):
                        if not key.startswith("_"):
                            metadata_dict[key] = getattr(entry.metadata, key)
                    result["metadata"] = metadata_dict
                else:
                    result["metadata"] = entry.metadata

            # Add timestamps if available
            if hasattr(entry, "inserted_at") and entry.inserted_at is not None:
                result["inserted_at"] = entry.inserted_at
            if hasattr(entry, "updated_at") and entry.updated_at is not None:
                result["updated_at"] = entry.updated_at

            # Filter fields if requested
            if return_fields:
                result = {
                    k: v for k, v in result.items() if k in return_fields or k == "key"
                }

            results.append(result)

        return results

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        entry_scope: Scope = None,
    ) -> str:
        """Store a prompt-response pair in the cache.

        Args:
            prompt: The user prompt to cache.
            response: The LLM response to cache.
            vector: Optional embedding vector (not used in LangCache).
            metadata: Optional metadata to store with the entry.
            filters: Optional filters for retrieval (used as attributes in LangCache).
            ttl: Optional TTL override in seconds.
            entry_scope: Optional scope for the cache entry.

        Returns:
            The ID of the cached entry.

        Raises:
            ValueError: If metadata is not a dictionary.
        """
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        # Prepare attributes (filters in LangCache terminology)
        attributes = {}
        if filters:
            attributes.update(filters)

        # Add metadata as a serialized JSON string
        if metadata:
            attributes["metadata"] = json.dumps(metadata)

        # Use the provided scope or fall back to the instance default
        scope = entry_scope if entry_scope is not None else self._entry_scope

        # Convert TTL from seconds to milliseconds for LangCache SDK
        ttl_millis = ttl * 1000 if ttl is not None else None

        # Store the entry
        result = self._api.entries.create(
            cache_id=self._cache_id,
            prompt=prompt,
            response=response,
            attributes=attributes,
            ttl_millis=ttl_millis,
            scope=scope,
        )

        return result.entry_id

    async def astore(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        entry_scope: Scope = None,
    ) -> str:
        """Async store a prompt-response pair in the cache.

        Args:
            prompt: The user prompt to cache.
            response: The LLM response to cache.
            vector: Optional embedding vector (not used in LangCache).
            metadata: Optional metadata to store with the entry.
            filters: Optional filters for retrieval (used as attributes in LangCache).
            ttl: Optional TTL override in seconds.
            entry_scope: Optional scope for the cache entry.

        Returns:
            The ID of the cached entry.

        Raises:
            ValueError: If metadata is not a dictionary.
        """
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        # Prepare attributes (filters in LangCache terminology)
        attributes = {}
        if filters:
            attributes.update(filters)

        # Add metadata as a serialized JSON string
        if metadata:
            attributes["metadata"] = json.dumps(metadata)

        # Use the provided scope or fall back to the instance default
        scope = entry_scope if entry_scope is not None else self._entry_scope

        # Convert TTL from seconds to milliseconds for LangCache SDK
        ttl_millis = ttl * 1000 if ttl is not None else None

        # Store the entry
        result = await self._api.entries.create_async(
            cache_id=self._cache_id,
            prompt=prompt,
            response=response,
            attributes=attributes,
            ttl_millis=ttl_millis,
            scope=scope,
        )

        return result.entry_id

    def update(self, key: str, **kwargs) -> None:
        """Update specific fields within an existing cache entry.

        Args:
            key: The ID of the entry to update.
            **kwargs: Field-value pairs to update.
        """
        # LangCache doesn't support partial updates, so we need to get the entry first
        entry = self._api.entries.get(  # type: ignore[attr-defined]
            cache_id=self._cache_id,
            entry_id=key,
        )

        # Prepare the update data
        update_data = {
            "prompt": entry.prompt,
            "response": entry.response,
            "attributes": {},
        }

        # Update with new values
        if "prompt" in kwargs:
            update_data["prompt"] = kwargs["prompt"]
        if "response" in kwargs:
            update_data["response"] = kwargs["response"]

        # Handle metadata and filters
        if hasattr(entry, "attributes") and entry.attributes:
            update_data["attributes"] = entry.attributes

        if "metadata" in kwargs:
            if kwargs["metadata"] is not None:
                update_data["attributes"]["metadata"] = json.dumps(kwargs["metadata"])
            elif "metadata" in update_data["attributes"]:
                del update_data["attributes"]["metadata"]

        if "filters" in kwargs and kwargs["filters"]:
            update_data["attributes"].update(kwargs["filters"])

        # Update the entry
        self._api.entries.update(  # type: ignore[attr-defined]
            cache_id=self._cache_id,
            entry_id=key,
            prompt=update_data["prompt"],
            response=update_data["response"],
            attributes=update_data["attributes"],
        )

    async def aupdate(self, key: str, **kwargs) -> None:
        """Async update specific fields within an existing cache entry.

        Args:
            key: The ID of the entry to update.
            **kwargs: Field-value pairs to update.
        """
        # LangCache doesn't support partial updates, so we need to get the entry first
        entry = await self._api.entries.get_async(  # type: ignore[attr-defined]
            cache_id=self._cache_id,
            entry_id=key,
        )

        # Prepare the update data
        update_data = {
            "prompt": entry.prompt,
            "response": entry.response,
            "attributes": {},
        }

        # Update with new values
        if "prompt" in kwargs:
            update_data["prompt"] = kwargs["prompt"]
        if "response" in kwargs:
            update_data["response"] = kwargs["response"]

        # Handle metadata and filters
        if hasattr(entry, "attributes") and entry.attributes:
            update_data["attributes"] = entry.attributes

        if "metadata" in kwargs:
            if kwargs["metadata"] is not None:
                update_data["attributes"]["metadata"] = json.dumps(kwargs["metadata"])
            elif "metadata" in update_data["attributes"]:
                del update_data["attributes"]["metadata"]

        if "filters" in kwargs and kwargs["filters"]:
            update_data["attributes"].update(kwargs["filters"])

        # Update the entry
        await self._api.entries.update_async(  # type: ignore[attr-defined]
            cache_id=self._cache_id,
            entry_id=key,
            prompt=update_data["prompt"],
            response=update_data["response"],
            attributes=update_data["attributes"],
        )

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if (
            hasattr(self._api.sdk_configuration, "client")
            and self._api.sdk_configuration.client
        ):
            self._api.sdk_configuration.client.close()

    async def adisconnect(self) -> None:
        """Async disconnect from Redis."""
        if (
            hasattr(self._api.sdk_configuration, "client")
            and self._api.sdk_configuration.client
        ):
            self._api.sdk_configuration.client.close()
        if (
            hasattr(self._api.sdk_configuration, "async_client")
            and self._api.sdk_configuration.async_client
        ):
            await self._api.sdk_configuration.async_client.aclose()

    def __enter__(self):
        """Context manager entry."""
        return self._api.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return self._api.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._api.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._api.__aexit__(exc_type, exc_val, exc_tb)
