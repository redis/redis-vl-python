from typing import Any, Dict, List, Optional

from redis import Redis

from redisvl.extensions.llmcache.base import BaseLLMCache
from redisvl.extensions.llmcache.schema import (
    CacheEntry,
    CacheHit,
    SemanticCacheIndexSchema,
)
from redisvl.index import SearchIndex
from redisvl.query import RangeQuery
from redisvl.query.filter import FilterExpression
from redisvl.utils.utils import current_timestamp, serialize, validate_vector_dims
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer


class SemanticCache(BaseLLMCache):
    """Semantic Cache for Large Language Models."""

    redis_key_field_name: str = "key"
    entry_id_field_name: str = "entry_id"
    prompt_field_name: str = "prompt"
    response_field_name: str = "response"
    vector_field_name: str = "prompt_vector"
    inserted_at_field_name: str = "inserted_at"
    updated_at_field_name: str = "updated_at"
    metadata_field_name: str = "metadata"

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

        Raises:
            TypeError: If an invalid vectorizer is provided.
            TypeError: If the TTL value is not an int.
            ValueError: If the threshold is not between 0 and 1.
        """
        super().__init__(ttl)

        # Use the index name as the key prefix by default
        if "prefix" in kwargs:
            prefix = kwargs["prefix"]
        else:
            prefix = name

        # Set vectorizer default
        if vectorizer is None:
            vectorizer = HFTextVectorizer(
                model="sentence-transformers/all-mpnet-base-v2"
            )

        # Process fields
        self.return_fields = [
            self.entry_id_field_name,
            self.prompt_field_name,
            self.response_field_name,
            self.inserted_at_field_name,
            self.updated_at_field_name,
            self.metadata_field_name,
        ]

        # Create semantic cache schema and index
        schema = SemanticCacheIndexSchema.from_params(name, prefix, vectorizer.dims)
        schema = self._modify_schema(schema, filterable_fields)

        self._index = SearchIndex(schema=schema)

        # Handle redis connection
        if redis_client:
            self._index.set_client(redis_client)
        elif redis_url:
            self._index.connect(redis_url=redis_url, **connection_kwargs)

        # Initialize other components
        self._set_vectorizer(vectorizer)
        self.set_threshold(distance_threshold)
        self._index.create(overwrite=False)

    def _modify_schema(
        self,
        schema: SemanticCacheIndexSchema,
        filterable_fields: Optional[List[Dict[str, Any]]] = None,
    ) -> SemanticCacheIndexSchema:
        """Modify the base cache schema using the provided filterable fields"""

        if filterable_fields is not None:
            protected_field_names = set(
                self.return_fields + [self.redis_key_field_name]
            )
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

    @property
    def index(self) -> SearchIndex:
        """The underlying SearchIndex for the cache.

        Returns:
            SearchIndex: The search index.
        """
        return self._index

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
        if not 0 <= float(distance_threshold) <= 1:
            raise ValueError(
                f"Distance must be between 0 and 1, got {distance_threshold}"
            )
        self._distance_threshold = float(distance_threshold)

    def _set_vectorizer(self, vectorizer: BaseVectorizer) -> None:
        """Sets the vectorizer for the LLM cache.

        Must be a valid subclass of BaseVectorizer and have equivalent
        dimensions to the vector field defined in the schema.

        Args:
            vectorizer (BaseVectorizer): The RedisVL vectorizer to use for
                vectorizing cache entries.

        Raises:
            TypeError: If the vectorizer is not a valid type.
            ValueError: If the vector dimensions are mismatched.
        """
        if not isinstance(vectorizer, BaseVectorizer):
            raise TypeError("Must provide a valid redisvl.vectorizer class.")

        schema_vector_dims = self._index.schema.fields[self.vector_field_name].attrs.dims  # type: ignore
        validate_vector_dims(vectorizer.dims, schema_vector_dims)
        self._vectorizer = vectorizer

    def clear(self) -> None:
        """Clear the cache of all keys while preserving the index."""
        self._index.clear()

    def delete(self) -> None:
        """Clear the semantic cache of all keys and remove the underlying search
        index."""
        self._index.delete(drop=True)

    def drop(
        self, ids: Optional[List[str]] = None, keys: Optional[List[str]] = None
    ) -> None:
        """Manually expire specific entries from the cache by id or specific
        Redis key.

        Args:
            ids (Optional[str]): The document ID or IDs to remove from the cache.
            keys (Optional[str]): The Redis keys to remove from the cache.
        """
        if ids is not None:
            self._index.drop_keys([self._index.key(id) for id in ids])
        if keys is not None:
            self._index.drop_keys(keys)

    def _refresh_ttl(self, key: str) -> None:
        """Refresh the time-to-live for the specified key."""
        if self._ttl:
            self._index.client.expire(key, self._ttl)  # type: ignore

    def _vectorize_prompt(self, prompt: Optional[str]) -> List[float]:
        """Converts a text prompt to its vector representation using the
        configured vectorizer."""
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        return self._vectorizer.embed(prompt)

    def _check_vector_dims(self, vector: List[float]):
        """Checks the size of the provided vector and raises an error if it
        doesn't match the search index vector dimensions."""
        schema_vector_dims = self._index.schema.fields[self.vector_field_name].attrs.dims  # type: ignore
        validate_vector_dims(len(vector), schema_vector_dims)

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
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

        Returns:
            List[Dict[str, Any]]: A list of dicts containing the requested
                return fields for each similar cached response.

        Raises:
            ValueError: If neither a `prompt` nor a `vector` is specified.
            ValueError: if 'vector' has incorrect dimensions.
            TypeError: If `return_fields` is not a list when provided.

        .. code-block:: python

            response = cache.check(
                prompt="What is the captial city of France?"
            )
        """
        if not (prompt or vector):
            raise ValueError("Either prompt or vector must be specified.")

        vector = vector or self._vectorize_prompt(prompt)
        self._check_vector_dims(vector)
        return_fields = return_fields or self.return_fields

        if not isinstance(return_fields, list):
            raise TypeError("return_fields must be a list of field names")

        query = RangeQuery(
            vector=vector,
            vector_field_name=self.vector_field_name,
            return_fields=self.return_fields,
            distance_threshold=self._distance_threshold,
            num_results=num_results,
            return_score=True,
            filter_expression=filter_expression,
        )

        cache_hits: List[Dict[Any, str]] = []

        # Search the cache!
        cache_search_results = self._index.query(query)

        for cache_search_result in cache_search_results:
            key = cache_search_result["id"]
            self._refresh_ttl(key)

            print(cache_search_result, flush=True)

            # Create cache hit
            cache_hit = CacheHit(**cache_search_result)
            cache_hit_dict = {
                k: v for k, v in cache_hit.to_dict().items() if k in return_fields
            }
            cache_hit_dict["key"] = key
            cache_hits.append(cache_hit_dict)

        return cache_hits

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
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

        Returns:
            str: The Redis key for the entries added to the semantic cache.

        Raises:
            ValueError: If neither prompt nor vector is specified.
            ValueError: if vector has incorrect dimensions.
            TypeError: If provided metadata is not a dictionary.

        .. code-block:: python

            key = cache.store(
                prompt="What is the captial city of France?",
                response="Paris",
                metadata={"city": "Paris", "country": "France"}
            )
        """
        # Vectorize prompt if necessary and create cache payload
        vector = vector or self._vectorize_prompt(prompt)

        self._check_vector_dims(vector)

        # Build cache entry for the cache
        cache_entry = CacheEntry(
            prompt=prompt,
            response=response,
            prompt_vector=vector,
            metadata=metadata,
            filters=filters,
        )

        # Load cache entry with TTL
        keys = self._index.load(
            data=[cache_entry.to_dict()],
            ttl=self._ttl,
            id_field=self.entry_id_field_name,
        )
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
            )
        """
        if kwargs:
            for k, v in kwargs.items():

                # Make sure the item is in the index schema
                if k not in set(
                    self._index.schema.field_names + [self.metadata_field_name]
                ):
                    raise ValueError(f"{k} is not a valid field within the cache entry")

                # Check for metadata and deserialize
                if k == self.metadata_field_name:
                    if isinstance(v, dict):
                        kwargs[k] = serialize(v)
                    else:
                        raise TypeError(
                            "If specified, cached metadata must be a dictionary."
                        )

            kwargs.update({self.updated_at_field_name: current_timestamp()})

            self._index.client.hset(key, mapping=kwargs)  # type: ignore

        self._refresh_ttl(key)
