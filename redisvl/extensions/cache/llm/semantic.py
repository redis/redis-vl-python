import asyncio
from typing import Any, Dict, List, Optional, Tuple

from redis import Redis

from redisvl.extensions.cache.llm.base import BaseLLMCache
from redisvl.extensions.cache.llm.schema import (
    CacheEntry,
    CacheHit,
    SemanticCacheIndexSchema,
)
from redisvl.extensions.constants import (
    CACHE_VECTOR_FIELD_NAME,
    ENTRY_ID_FIELD_NAME,
    INSERTED_AT_FIELD_NAME,
    METADATA_FIELD_NAME,
    PROMPT_FIELD_NAME,
    REDIS_KEY_FIELD_NAME,
    RESPONSE_FIELD_NAME,
    UPDATED_AT_FIELD_NAME,
)
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import VectorRangeQuery
from redisvl.query.filter import FilterExpression
from redisvl.redis.utils import hashify
from redisvl.utils.log import get_logger
from redisvl.utils.utils import (
    current_timestamp,
    deprecated_argument,
    serialize,
    validate_vector_dims,
)
from redisvl.utils.vectorize.base import BaseVectorizer
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

logger = get_logger("[RedisVL]")


class SemanticCache(BaseLLMCache):
    """Semantic Cache for Large Language Models."""

    _index: SearchIndex
    _aindex: Optional[AsyncSearchIndex] = None

    @deprecated_argument("dtype", "vectorizer")
    def __init__(
        self,
        name: str = "llmcache",
        distance_threshold: float = 0.1,
        ttl: Optional[int] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        filterable_fields: Optional[List[Dict[str, Any]]] = None,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: Dict[str, Any] = {},
        overwrite: bool = False,
        **kwargs,
    ):
        """Semantic Cache for Large Language Models.

        Args:
            name (str, optional): The name of the semantic cache search index.
                Defaults to "llmcache".
            distance_threshold (float, optional): Semantic threshold for the
                cache. Defaults to 0.1.
            ttl (Optional[int], optional): The time-to-live for records cached
                in Redis. Defaults to None.
            vectorizer (Optional[BaseVectorizer], optional): The vectorizer for the cache.
                Defaults to HFTextVectorizer.
            filterable_fields (Optional[List[Dict[str, Any]]]): An optional list of RedisVL fields
                that can be used to customize cache retrieval with filters.
            redis_client(Optional[Redis], optional): A redis client connection instance.
                Defaults to None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.
            overwrite (bool): Whether or not to force overwrite the schema for
                the semantic cache index. Defaults to false.

        Raises:
            TypeError: If an invalid vectorizer is provided.
            TypeError: If the TTL value is not an int.
            ValueError: If the threshold is not between 0 and 1.
            ValueError: If existing schema does not match new schema and overwrite is False.
        """
        # Call parent class with all shared parameters
        super().__init__(
            name=name,
            ttl=ttl,
            redis_client=redis_client,
            redis_url=redis_url,
            connection_kwargs=connection_kwargs,
        )

        # Handle the deprecated dtype parameter
        dtype = kwargs.pop("dtype", None)

        # Set up vectorizer - either use the provided one or create a default
        if vectorizer:
            if not isinstance(vectorizer, BaseVectorizer):
                raise TypeError("Must provide a valid redisvl.vectorizer class.")
            if dtype and vectorizer.dtype != dtype:
                raise ValueError(
                    f"Provided dtype {dtype} does not match vectorizer dtype {vectorizer.dtype}"
                )
            self._vectorizer = vectorizer
        else:
            # Create the default vectorizer
            vectorizer_kwargs = kwargs

            if dtype:
                vectorizer_kwargs.update(dtype=dtype)

            self._vectorizer = HFTextVectorizer(
                model="redis/langcache-embed-v1",
                **vectorizer_kwargs,
            )

        # Set threshold for semantic matching
        self.set_threshold(distance_threshold)

        # Define the fields to return in search results
        self.return_fields = [
            ENTRY_ID_FIELD_NAME,
            PROMPT_FIELD_NAME,
            RESPONSE_FIELD_NAME,
            INSERTED_AT_FIELD_NAME,
            UPDATED_AT_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        # Create semantic cache schema and index
        schema = SemanticCacheIndexSchema.from_params(
            name, name, self._vectorizer.dims, self._vectorizer.dtype  # type: ignore
        )
        schema = self._modify_schema(schema, filterable_fields)

        # Initialize the search index
        self._index = SearchIndex(
            schema=schema,
            redis_client=self._redis_client,
            redis_url=self.redis_kwargs["redis_url"],
            **self.redis_kwargs["connection_kwargs"],
        )
        self._aindex = None

        # Check for existing cache index and handle schema mismatch
        self.overwrite = overwrite
        if not self.overwrite and self._index.exists():

            if not vectorizer:
                # user hasn't specified a vectorizer and an index already exists they're not overwriting
                # raise a warning to inform users we changed the default embedding model
                # remove this warning in future releases
                logger.warning(
                    "The default vectorizer has changed from `sentence-transformers/all-mpnet-base-v2` "
                    "to `redis/langcache-embed-v1` in version 0.6.0 of RedisVL. "
                    "For more information about this model, please refer to https://arxiv.org/abs/2504.02268 "
                    "or visit https://huggingface.co/redis/langcache-embed-v1. "
                    "To continue using the old vectorizer, please specify it explicitly in the constructor as: "
                    "vectorizer=HFTextVectorizer(model='sentence-transformers/all-mpnet-base-v2')"
                )

            existing_index = SearchIndex.from_existing(
                name, redis_client=self._index.client
            )
            if existing_index.schema.to_dict() != self._index.schema.to_dict():
                raise ValueError(
                    f"Existing index {name} schema does not match the user provided schema for the semantic cache. "
                    "If you wish to overwrite the index schema, set overwrite=True during initialization."
                )

        # Create the search index in Redis
        self._index.create(overwrite=self.overwrite, drop=False)

    def _modify_schema(
        self,
        schema: SemanticCacheIndexSchema,
        filterable_fields: Optional[List[Dict[str, Any]]] = None,
    ) -> SemanticCacheIndexSchema:
        """Modify the base cache schema using the provided filterable fields"""

        if filterable_fields is not None:
            protected_field_names = set(self.return_fields + [REDIS_KEY_FIELD_NAME])
            for filter_field in filterable_fields:
                field_name = filter_field["name"]
                if field_name in protected_field_names:
                    raise ValueError(
                        f"{field_name} is a reserved field name for the semantic cache schema"
                    )
                # Add to schema
                schema.add_field(filter_field)
                # Add to return fields too
                self.return_fields.append(field_name)

        return schema

    async def _get_async_index(self) -> AsyncSearchIndex:
        """Lazily construct the async search index class."""
        # Construct async index if necessary
        if self._aindex is None:
            async_client = await self._get_async_redis_client()
            self._aindex = AsyncSearchIndex(
                schema=self._index.schema,
                redis_client=async_client,
                redis_url=self.redis_kwargs["redis_url"],
                **self.redis_kwargs["connection_kwargs"],
            )
        return self._aindex

    @property
    def index(self) -> SearchIndex:
        """The underlying SearchIndex for the cache.

        Returns:
            SearchIndex: The search index.
        """
        return self._index

    @property
    def aindex(self) -> Optional[AsyncSearchIndex]:
        """The underlying AsyncSearchIndex for the cache.

        Returns:
            AsyncSearchIndex: The async search index.
        """
        return self._aindex

    @property
    def distance_threshold(self) -> float:
        """The semantic distance threshold for the cache.

        Returns:
            float: The semantic distance threshold.
        """
        return self._distance_threshold

    def set_threshold(self, distance_threshold: float) -> None:
        """Sets the semantic distance threshold for the cache.

        Args:
            distance_threshold (float): The semantic distance threshold for
                the cache.

        Raises:
            ValueError: If the threshold is not between 0 and 1.
        """
        if not 0 <= float(distance_threshold) <= 2:
            raise ValueError(
                f"Distance must be between 0 and 2, got {distance_threshold}"
            )
        self._distance_threshold = float(distance_threshold)

    def delete(self) -> None:
        """Delete the cache and its index entirely."""
        self._index.delete(drop=True)

    async def adelete(self) -> None:
        """Async delete the cache and its index entirely."""
        aindex = await self._get_async_index()
        await aindex.delete(drop=True)

    def drop(
        self, ids: Optional[List[str]] = None, keys: Optional[List[str]] = None
    ) -> None:
        """Drop specific entries from the cache by ID or Redis key.

        Args:
            ids (Optional[List[str]]): List of entry IDs to remove from the cache.
                Entry IDs are the unique identifiers without the cache prefix.
            keys (Optional[List[str]]): List of full Redis keys to remove from the cache.
                Keys are the complete Redis keys including the cache prefix.

        Note:
            At least one of ids or keys must be provided.

        Raises:
            ValueError: If neither ids nor keys is provided.
        """
        if ids is None and keys is None:
            raise ValueError("At least one of ids or keys must be provided.")

        # Convert entry IDs to full Redis keys if provided
        if ids is not None:
            self._index.drop_keys([self._index.key(id) for id in ids])
        if keys is not None:
            self._index.drop_keys(keys)

    async def adrop(
        self, ids: Optional[List[str]] = None, keys: Optional[List[str]] = None
    ) -> None:
        """Async drop specific entries from the cache by ID or Redis key.

        Args:
            ids (Optional[List[str]]): List of entry IDs to remove from the cache.
                Entry IDs are the unique identifiers without the cache prefix.
            keys (Optional[List[str]]): List of full Redis keys to remove from the cache.
                Keys are the complete Redis keys including the cache prefix.

        Note:
            At least one of ids or keys must be provided.

        Raises:
            ValueError: If neither ids nor keys is provided.
        """
        aindex = await self._get_async_index()

        if ids is None and keys is None:
            raise ValueError("At least one of ids or keys must be provided.")

        # Convert entry IDs to full Redis keys if provided
        if ids is not None:
            await aindex.drop_keys([self._index.key(id) for id in ids])
        if keys is not None:
            await aindex.drop_keys(keys)

    def _vectorize_prompt(self, prompt: Optional[str]) -> List[float]:
        """Converts a text prompt to its vector representation using the
        configured vectorizer."""
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        result = self._vectorizer.embed(prompt)
        return result  # type: ignore

    async def _avectorize_prompt(self, prompt: Optional[str]) -> List[float]:
        """Converts a text prompt to its vector representation using the
        configured vectorizer."""
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        result = await self._vectorizer.aembed(prompt)
        return result  # type: ignore

    def _check_vector_dims(self, vector: List[float]):
        """Checks the size of the provided vector and raises an error if it
        doesn't match the search index vector dimensions."""
        schema_vector_dims = self._index.schema.fields[
            CACHE_VECTOR_FIELD_NAME
        ].attrs.dims  # type: ignore
        validate_vector_dims(len(vector), schema_vector_dims)

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Checks the semantic cache for results similar to the specified prompt
        or vector.

        This method searches the cache using vector similarity with
        either a raw text prompt (converted to a vector) or a provided vector as
        input. It checks for semantically similar prompts and fetches the cached
        LLM responses.

        Args:
            prompt (Optional[str], optional): The text prompt to search for in
                the cache.
            vector (Optional[List[float]], optional): The vector representation
                of the prompt to search for in the cache.
            num_results (int, optional): The number of cached results to return.
                Defaults to 1.
            return_fields (Optional[List[str]], optional): The fields to include
                in each returned result. If None, defaults to all available
                fields in the cached entry.
            filter_expression (Optional[FilterExpression]) : Optional filter expression
                that can be used to filter cache results. Defaults to None and
                the full cache will be searched.
            distance_threshold (Optional[float]): The threshold for semantic
                vector distance.

        Returns:
            List[Dict[str, Any]]: A list of dicts containing the requested
                return fields for each similar cached response.

        Raises:
            ValueError: If neither a `prompt` nor a `vector` is specified.
            ValueError: if 'vector' has incorrect dimensions.
            TypeError: If `return_fields` is not a list when provided.

        .. code-block:: python

            response = cache.check(
                prompt="What is the capital city of France?"
            )
        """
        if not any([prompt, vector]):
            raise ValueError("Either prompt or vector must be specified.")
        if return_fields and not isinstance(return_fields, list):
            raise TypeError("Return fields must be a list of values.")

        # Use overrides or defaults
        distance_threshold = distance_threshold or self._distance_threshold

        # Vectorize prompt if not provided
        if vector is None and prompt is not None:
            vector = self._vectorize_prompt(prompt)

        # Validate the vector dimensions
        if vector is not None:
            self._check_vector_dims(vector)
        else:
            raise ValueError("Failed to generate a valid vector for the query.")

        # Create the vector search query
        query = VectorRangeQuery(
            vector=vector,
            vector_field_name=CACHE_VECTOR_FIELD_NAME,
            return_fields=self.return_fields,
            distance_threshold=distance_threshold,
            num_results=num_results,
            return_score=True,
            filter_expression=filter_expression,
            dtype=self._vectorizer.dtype,
        )

        # Search the cache!
        cache_search_results = self._index.query(query)
        redis_keys, cache_hits = self._process_cache_results(
            cache_search_results,
            return_fields,  # type: ignore
        )

        # Refresh TTL on all found keys
        for key in redis_keys:
            self.expire(key)

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
        """Async check the semantic cache for results similar to the specified prompt
        or vector.

        This method searches the cache using vector similarity with
        either a raw text prompt (converted to a vector) or a provided vector as
        input. It checks for semantically similar prompts and fetches the cached
        LLM responses.

        Args:
            prompt (Optional[str], optional): The text prompt to search for in
                the cache.
            vector (Optional[List[float]], optional): The vector representation
                of the prompt to search for in the cache.
            num_results (int, optional): The number of cached results to return.
                Defaults to 1.
            return_fields (Optional[List[str]], optional): The fields to include
                in each returned result. If None, defaults to all available
                fields in the cached entry.
            filter_expression (Optional[FilterExpression]) : Optional filter expression
                that can be used to filter cache results. Defaults to None and
                the full cache will be searched.
            distance_threshold (Optional[float]): The threshold for semantic
                vector distance.

        Returns:
            List[Dict[str, Any]]: A list of dicts containing the requested
                return fields for each similar cached response.

        Raises:
            ValueError: If neither a `prompt` nor a `vector` is specified.
            ValueError: if 'vector' has incorrect dimensions.
            TypeError: If `return_fields` is not a list when provided.

        .. code-block:: python

            response = await cache.acheck(
                prompt="What is the capital city of France?"
            )
        """
        aindex = await self._get_async_index()

        if not any([prompt, vector]):
            raise ValueError("Either prompt or vector must be specified.")
        if return_fields and not isinstance(return_fields, list):
            raise TypeError("Return fields must be a list of values.")

        # Use overrides or defaults
        distance_threshold = distance_threshold or self._distance_threshold

        # Vectorize prompt if not provided
        if vector is None and prompt is not None:
            vector = await self._avectorize_prompt(prompt)

        # Validate the vector dimensions
        if vector is not None:
            self._check_vector_dims(vector)
        else:
            raise ValueError("Failed to generate a valid vector for the query.")

        # Create the vector search query
        query = VectorRangeQuery(
            vector=vector,
            vector_field_name=CACHE_VECTOR_FIELD_NAME,
            return_fields=self.return_fields,
            distance_threshold=distance_threshold,
            num_results=num_results,
            return_score=True,
            filter_expression=filter_expression,
            normalize_vector_distance=True,
        )

        # Search the cache!
        cache_search_results = await aindex.query(query)
        redis_keys, cache_hits = self._process_cache_results(
            cache_search_results,
            return_fields,  # type: ignore
        )

        # Refresh TTL on all found keys async
        await asyncio.gather(*[self.aexpire(key) for key in redis_keys])

        return cache_hits

    def _process_cache_results(
        self,
        cache_search_results: List[Dict[str, Any]],
        return_fields: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process raw search results into cache hits."""
        redis_keys: List[str] = []
        cache_hits: List[Dict[Any, str]] = []

        for cache_search_result in cache_search_results:
            # Pop the redis key from the result
            redis_key = cache_search_result.pop("id")
            redis_keys.append(redis_key)

            # Create and process cache hit
            cache_hit = CacheHit(**cache_search_result)
            cache_hit_dict = cache_hit.to_dict()

            # Filter down to only selected return fields if needed
            if isinstance(return_fields, list) and return_fields:
                cache_hit_dict = {
                    k: v for k, v in cache_hit_dict.items() if k in return_fields
                }

            # Add the Redis key to the result
            cache_hit_dict[REDIS_KEY_FIELD_NAME] = redis_key
            cache_hits.append(cache_hit_dict)

        return redis_keys, cache_hits

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Stores the specified key-value pair in the cache along with metadata.

        Args:
            prompt (str): The user prompt to cache.
            response (str): The LLM response to cache.
            vector (Optional[List[float]], optional): The prompt vector to
                cache. Defaults to None, and the prompt vector is generated on
                demand.
            metadata (Optional[Dict[str, Any]], optional): The optional metadata to cache
                alongside the prompt and response. Defaults to None.
            filters (Optional[Dict[str, Any]]): The optional tag to assign to the cache entry.
                Defaults to None.
            ttl (Optional[int]): The optional TTL override to use on this individual cache
                entry. Defaults to the global TTL setting.

        Returns:
            str: The Redis key for the entries added to the semantic cache.

        Raises:
            ValueError: If neither prompt nor vector is specified.
            ValueError: if vector has incorrect dimensions.
            TypeError: If provided metadata is not a dictionary.

        .. code-block:: python

            key = cache.store(
                prompt="What is the capital city of France?",
                response="Paris",
                metadata={"city": "Paris", "country": "France"}
            )
        """
        # Vectorize prompt if necessary
        vector = vector or self._vectorize_prompt(prompt)
        self._check_vector_dims(vector)

        # Generate the entry ID
        entry_id = self._make_entry_id(prompt, filters)

        # Build cache entry for the cache
        cache_entry = CacheEntry(
            entry_id=entry_id,
            prompt=prompt,
            response=response,
            prompt_vector=vector,
            metadata=metadata,
            filters=filters,
        )

        # Load cache entry with TTL
        ttl = ttl or self._ttl
        keys = self._index.load(
            data=[cache_entry.to_dict(self._vectorizer.dtype)],
            ttl=ttl,
            id_field=ENTRY_ID_FIELD_NAME,
        )

        # Return the key where the entry was stored
        return keys[0]

    async def astore(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Async stores the specified key-value pair in the cache along with metadata.

        Args:
            prompt (str): The user prompt to cache.
            response (str): The LLM response to cache.
            vector (Optional[List[float]], optional): The prompt vector to
                cache. Defaults to None, and the prompt vector is generated on
                demand.
            metadata (Optional[Dict[str, Any]], optional): The optional metadata to cache
                alongside the prompt and response. Defaults to None.
            filters (Optional[Dict[str, Any]]): The optional tag to assign to the cache entry.
                Defaults to None.
            ttl (Optional[int]): The optional TTL override to use on this individual cache
                entry. Defaults to the global TTL setting.

        Returns:
            str: The Redis key for the entries added to the semantic cache.

        Raises:
            ValueError: If neither prompt nor vector is specified.
            ValueError: if vector has incorrect dimensions.
            TypeError: If provided metadata is not a dictionary.

        .. code-block:: python

            key = await cache.astore(
                prompt="What is the capital city of France?",
                response="Paris",
                metadata={"city": "Paris", "country": "France"}
            )
        """
        aindex = await self._get_async_index()

        # Vectorize prompt if necessary
        vector = vector or await self._avectorize_prompt(prompt)
        self._check_vector_dims(vector)

        # Generate the entry ID
        entry_id = self._make_entry_id(prompt, filters)

        # Build cache entry for the cache
        cache_entry = CacheEntry(
            entry_id=entry_id,
            prompt=prompt,
            response=response,
            prompt_vector=vector,
            metadata=metadata,
            filters=filters,
        )

        # Load cache entry with TTL
        ttl = ttl or self._ttl
        keys = await aindex.load(
            data=[cache_entry.to_dict(self._vectorizer.dtype)],
            ttl=ttl,
            id_field=ENTRY_ID_FIELD_NAME,
        )

        # Return the key where the entry was stored
        return keys[0]

    def update(self, key: str, **kwargs) -> None:
        """Update specific fields within an existing cache entry. If no fields
        are passed, then only the document TTL is refreshed.

        Args:
            key (str): the key of the document to update using kwargs.

        Raises:
            ValueError if an incorrect mapping is provided as a kwarg.
            TypeError if metadata is provided and not of type dict.

        .. code-block:: python

            key = cache.store('this is a prompt', 'this is a response')
            cache.update(key, metadata={"hit_count": 1, "model_name": "Llama-2-7b"})
        """
        if kwargs:
            for k, v in kwargs.items():
                # Make sure the item is in the index schema
                if k not in set(self._index.schema.field_names + [METADATA_FIELD_NAME]):
                    raise ValueError(f"{k} is not a valid field within the cache entry")

                # Check for metadata and serialize
                if k == METADATA_FIELD_NAME:
                    if isinstance(v, dict):
                        kwargs[k] = serialize(v)
                    else:
                        raise TypeError(
                            "If specified, cached metadata must be a dictionary."
                        )

            # Add updated timestamp
            kwargs.update({UPDATED_AT_FIELD_NAME: current_timestamp()})

            # Update the hash in Redis - ensure client exists and handle type properly
            client = self._get_redis_client()
            client.hset(key, mapping=kwargs)  # type: ignore

        # Refresh TTL regardless of whether fields were updated
        self.expire(key)

    async def aupdate(self, key: str, **kwargs) -> None:
        """Async update specific fields within an existing cache entry. If no fields
        are passed, then only the document TTL is refreshed.

        Args:
            key (str): the key of the document to update using kwargs.

        Raises:
            ValueError if an incorrect mapping is provided as a kwarg.
            TypeError if metadata is provided and not of type dict.

        .. code-block:: python

            key = await cache.astore('this is a prompt', 'this is a response')
            await cache.aupdate(
                key,
                metadata={"hit_count": 1, "model_name": "Llama-2-7b"}
            )
        """
        if kwargs:
            for k, v in kwargs.items():
                # Make sure the item is in the index schema
                if k not in set(self._index.schema.field_names + [METADATA_FIELD_NAME]):
                    raise ValueError(f"{k} is not a valid field within the cache entry")

                # Check for metadata and serialize
                if k == METADATA_FIELD_NAME:
                    if isinstance(v, dict):
                        kwargs[k] = serialize(v)
                    else:
                        raise TypeError(
                            "If specified, cached metadata must be a dictionary."
                        )

            # Add updated timestamp
            kwargs.update({UPDATED_AT_FIELD_NAME: current_timestamp()})

            # Update the hash in Redis - ensure client exists and handle type properly
            client = await self._get_async_redis_client()
            # Convert dict values to proper types for Redis
            await client.hset(key, mapping=kwargs)  # type: ignore

        # Refresh TTL regardless of whether fields were updated
        await self.aexpire(key)

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.disconnect()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.adisconnect()

    def disconnect(self):
        """Disconnect from Redis and search index.

        Closes all Redis connections and index connections.
        """
        # Close the search index connections
        if hasattr(self, "_index") and self._index:
            self._index.disconnect()

        # Close the async search index connections
        if hasattr(self, "_aindex") and self._aindex:
            self._aindex.disconnect_sync()

        # Close the base Redis connections
        super().disconnect()

    async def adisconnect(self):
        """Asynchronously disconnect from Redis and search index.

        Closes all Redis connections and index connections.
        """
        # Close the async search index connections
        if hasattr(self, "_aindex") and self._aindex:
            await self._aindex.disconnect()
            self._aindex = None

        # Close the base Redis connections
        await super().adisconnect()

    def _make_entry_id(
        self, prompt: str, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a deterministic entry ID for the given prompt and optional filters.

        Args:
            prompt (str): The prompt text.
            filters (Optional[Dict[str, Any]]): Optional filter dictionary.

        Returns:
            str: A deterministic entry ID based on the prompt and filters.
        """
        return hashify(prompt, filters)
