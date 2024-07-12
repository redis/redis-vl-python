from time import time
from typing import Any, Dict, List, Optional, Union

from redis import Redis

from redisvl.extensions.llmcache.base import BaseLLMCache
from redisvl.index import SearchIndex
from redisvl.query import RangeQuery
from redisvl.query.filter import Tag, FilterExpression
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer


class SemanticCacheIndexSchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, vector_dims: int):

        return cls(
                index={"name": name, "prefix": name},  # type: ignore
                fields=[  # type: ignore
                {"name": "cache_name", "type": "tag"},
                {"name": "prompt", "type": "text"},
                {"name": "response", "type": "text"},
                {"name": "inserted_at", "type": "numeric"},
                {"name": "updated_at", "type": "numeric"},
                {"name": "scope_tag", "type": "tag"},
                {
                    "name": "prompt_vector",
                    "type": "vector",
                    "attrs": {
                        "dims": vector_dims,
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ],
        )


class SemanticCache(BaseLLMCache):
    """Semantic Cache for Large Language Models."""

    entry_id_field_name: str = "id"
    prompt_field_name: str = "prompt"
    vector_field_name: str = "prompt_vector"
    inserted_at_field_name: str = "inserted_at"
    updated_at_field_name: str = "updated_at"
    tag_field_name: str = "scope_tag"
    response_field_name: str = "response"
    metadata_field_name: str = "metadata"

    def __init__(
        self,
        name: str = "llmcache",
        prefix: Optional[str] = None,
        distance_threshold: float = 0.1,
        ttl: Optional[int] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        connection_args: Dict[str, Any] = {},
        **kwargs,
    ):
        """Semantic Cache for Large Language Models.

        Args:
            name (str, optional): The name of the semantic cache search index.
                Defaults to "llmcache".
            prefix (Optional[str], optional): The prefix for Redis keys
                associated with the semantic cache search index. Defaults to
                None, and the index name will be used as the key prefix.
            distance_threshold (float, optional): Semantic threshold for the
                cache. Defaults to 0.1.
            ttl (Optional[int], optional): The time-to-live for records cached
                in Redis. Defaults to None.
            vectorizer (BaseVectorizer, optional): The vectorizer for the cache.
                Defaults to HFTextVectorizer.
            redis_client(Redis, optional): A redis client connection instance.
                Defaults to None.
            redis_url (str, optional): The redis url. Defaults to
                "redis://localhost:6379".
            connection_args (Dict[str, Any], optional): The connection arguments
                for the redis client. Defaults to None.

        Raises:
            TypeError: If an invalid vectorizer is provided.
            TypeError: If the TTL value is not an int.
            ValueError: If the threshold is not between 0 and 1.
            ValueError: If the index name is not provided
        """
        super().__init__(ttl)

        # Use the index name as the key prefix by default
        if prefix is None:
            prefix = name

        # Set vectorizer default
        if vectorizer is None:
            vectorizer = HFTextVectorizer(
                model="sentence-transformers/all-mpnet-base-v2"
            )

        schema = SemanticCacheIndexSchema.from_params(name, vectorizer.dims)
        self._index = SearchIndex(schema=schema)

        # handle redis connection
        if redis_client:
            self._index.set_client(redis_client)
        else:
            self._index.connect(redis_url=redis_url, **connection_args)

        # initialize other components
        self.default_return_fields = [
            self.entry_id_field_name,
            self.prompt_field_name,
            self.response_field_name,
            self.tag_field_name,
            self.vector_field_name,
            self.metadata_field_name,
        ]
        self.set_vectorizer(vectorizer)
        self.set_threshold(distance_threshold)
        self._index.create(overwrite=False)

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

    def set_vectorizer(self, vectorizer: BaseVectorizer) -> None:
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

        if schema_vector_dims != vectorizer.dims:
            raise ValueError(
                "Invalid vector dimensions! "
                f"Vectorizer has dims defined as {vectorizer.dims}",
                f"Vector field has dims defined as {schema_vector_dims}",
            )

        self._vectorizer = vectorizer

    def clear(self) -> None:
        """Clear the cache of all keys while preserving the index."""
        self._index.clear()

    def delete(self) -> None:
        """Clear the semantic cache of all keys and remove the underlying search
        index."""
        self._index.delete(drop=True)

    def drop(self, document_ids: Union[str, List[str]]) -> None:
        """Remove a specific entry or entries from the cache by it's ID.

        Args:
            document_ids (Union[str, List[str]]): The document ID or IDs to remove from the cache.
        """
        if isinstance(document_ids, List):
            with self._index.client.pipeline(transaction=False) as pipe:  # type: ignore
                for key in document_ids:  # type: ignore
                    pipe.delete(key)
                pipe.execute()
        else:
            self._index.client.delete(document_ids)  # type: ignore

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

    def _search_cache(
        self,
        vector: List[float],
        num_results: int,
        return_fields: Optional[List[str]],
        ##tags: Optional[Union[List[str], str]],
        filters: Optional[FilterExpression],
    ) -> List[Dict[str, Any]]:
        """Searches the semantic cache for similar prompt vectors and returns
        the specified return fields for each cache hit."""
        # Setup and type checks
        if not isinstance(vector, list):
            raise TypeError("Vector must be a list of floats")

        return_fields = return_fields or self.default_return_fields

        if not isinstance(return_fields, list):
            raise TypeError("return_fields must be a list of field names")

        # Construct vector RangeQuery for the cache check
        query = RangeQuery(
            vector=vector,
            vector_field_name=self.vector_field_name,
            return_fields=return_fields,
            distance_threshold=self._distance_threshold,
            num_results=num_results,
            return_score=True,
        )
        ##if tags:
        ##    query.set_filter(self.get_filter(tags))  # type: ignore
        if filters:
            query.set_filter(filters)  # type: ignore

        # Gather and return the cache hits
        cache_hits: List[Dict[str, Any]] = self._index.query(query)
        # Process cache hits
        for hit in cache_hits:
            self._refresh_ttl(hit[self.entry_id_field_name])
            # Check for metadata and deserialize
            if self.metadata_field_name in hit:
                hit[self.metadata_field_name] = self.deserialize(
                    hit[self.metadata_field_name]
                )
        return cache_hits

    def _check_vector_dims(self, vector: List[float]):
        """Checks the size of the provided vector and raises an error if it
        doesn't match the search index vector dimensions."""
        schema_vector_dims = self._index.schema.fields[self.vector_field_name].attrs.dims  # type: ignore
        if schema_vector_dims != len(vector):
            raise ValueError(
                "Invalid vector dimensions! "
                f"Vector has dims defined as {len(vector)}",
                f"Vector field has dims defined as {schema_vector_dims}",
            )

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        ##tags: Optional[Union[List[str], str]] = None,
        filters: Optional[FilterExpression] = None,
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
            tags (Optional[Union[List[str], str]) : the tag or tags to filter
            results by. Default is None and full cache is searched.

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

        # Use provided vector or create from prompt
        vector = vector or self._vectorize_prompt(prompt)
        self._check_vector_dims(vector)

        # Check for cache hits by searching the cache
        ##cache_hits = self._search_cache(vector, num_results, return_fields, tags)
        cache_hits = self._search_cache(vector, num_results, return_fields, filters)
        return cache_hits

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[dict] = None,
        tag: Optional[str] = None,
    ) -> str:
        """Stores the specified key-value pair in the cache along with metadata.

        Args:
            prompt (str): The user prompt to cache.
            response (str): The LLM response to cache.
            vector (Optional[List[float]], optional): The prompt vector to
                cache. Defaults to None, and the prompt vector is generated on
                demand.
            metadata (Optional[dict], optional): The optional metadata to cache
                alongside the prompt and response. Defaults to None.
            tag (Optional[str]): The optional tag to assign to the cache entry.
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

        # Construct semantic cache payload
        now = time()
        id_field = self.entry_id_field_name
        payload = {
            id_field: self.hash_input(prompt),
            self.prompt_field_name: prompt,
            self.response_field_name: response,
            self.vector_field_name: array_to_buffer(vector),
            self.inserted_at_field_name: now,
            self.updated_at_field_name: now,
        }
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("If specified, cached metadata must be a dictionary.")
            # Serialize the metadata dict and add to cache payload
            payload[self.metadata_field_name] = self.serialize(metadata)
        if tag is not None:
            payload[self.tag_field_name] = tag

        # Load LLMCache entry with TTL
        keys = self._index.load(data=[payload], ttl=self._ttl, id_field=id_field)
        return keys[0]

    def update(self, key: str, **kwargs) -> None:
        """Update specific fields within an existing cache entry. If no fields
        are passed, then only the document TTL is refreshed.

        Args:
            key (str): the key of the document to update.
            kwargs:

        Raises:
            ValueError if an incorrect mapping is provided as a kwarg.
            TypeError if metadata is provided and not of type dict.

        .. code-block:: python
            key = cache.store('this is a prompt', 'this is a response')
            cache.update(key, metadata={"hit_count": 1, "model_name": "Llama-2-7b"})
            )
        """
        if not kwargs:
            self._refresh_ttl(key)
            return

        for _key, val in kwargs.items():
            if _key not in {
                self.prompt_field_name,
                self.vector_field_name,
                self.response_field_name,
                self.tag_field_name,
                self.metadata_field_name,
            }:
                raise ValueError(f" {key} is not a valid field within document")

            # Check for metadata and deserialize
            if _key == self.metadata_field_name:
                if isinstance(val, dict):
                    kwargs[_key] = self.serialize(val)
                else:
                    raise TypeError(
                        "If specified, cached metadata must be a dictionary."
                    )
        kwargs.update({self.updated_at_field_name: time()})
        self._index.client.hset(key, mapping=kwargs)  # type: ignore

    def get_filter(
        self,
        tags: Optional[Union[List[str], str]] = None,
    ) -> Tag:
        """Set the tags filter to apply to querries based on the desired scope.

        Args:
            tags (Optional[Union[List[str], str]]): name of the specific tag or
                tags to filter to. Default is None, which means all cache data
                will be in scope.
        """
        default_filter = Tag(self.tag_field_name) == []
        if not (tags):
            return default_filter

        tag_filter = Tag(self.tag_field_name) == tags
        return tag_filter
