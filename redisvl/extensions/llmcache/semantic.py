from typing import Any, Dict, List, Optional

from redis import Redis

from redisvl.extensions.llmcache.base import BaseLLMCache
from redisvl.index import SearchIndex
from redisvl.query import RangeQuery
from redisvl.redis.utils import array_to_buffer
from redisvl.schema.schema import IndexSchema
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer


class SemanticCache(BaseLLMCache):
    """Semantic Cache for Large Language Models."""

    entry_id_field_name: str = "id"
    prompt_field_name: str = "prompt"
    vector_field_name: str = "prompt_vector"
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

        # build cache index schema
        schema = IndexSchema.from_dict({"index": {"name": name, "prefix": prefix}})
        # add fields
        schema.add_fields(
            [
                {"name": self.prompt_field_name, "type": "text"},
                {"name": self.response_field_name, "type": "text"},
                {
                    "name": self.vector_field_name,
                    "type": "vector",
                    "attrs": {
                        "dims": vectorizer.dims,
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ]
        )

        # build search index
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
        with self._index.client.pipeline(transaction=False) as pipe:  # type: ignore
            for key in self._index.client.scan_iter(match=f"{self._index.prefix}:*"):  # type: ignore
                pipe.delete(key)
            pipe.execute()

    def delete(self) -> None:
        """Clear the semantic cache of all keys and remove the underlying search
        index."""
        self._index.delete(drop=True)

    def _refresh_ttl(self, key: str) -> None:
        """Refresh the time-to-live for the specified key."""
        if self.ttl:
            self._index.client.expire(key, self.ttl)  # type: ignore

    def _vectorize_prompt(self, prompt: Optional[str]) -> List[float]:
        """Converts a text prompt to its vector representation using the
        configured vectorizer."""
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")
        return self._vectorizer.embed(prompt)

    def _search_cache(
        self, vector: List[float], num_results: int, return_fields: Optional[List[str]]
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

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
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

        Returns:
            List[Dict[str, Any]]: A list of dicts containing the requested
                return fields for each similar cached response.

        Raises:
            ValueError: If neither a `prompt` nor a `vector` is specified.
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

        # Check for cache hits by searching the cache
        cache_hits = self._search_cache(vector, num_results, return_fields)
        return cache_hits

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[dict] = None,
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

        Returns:
            str: The Redis key for the entries added to the semantic cache.

        Raises:
            ValueError: If neither prompt nor vector is specified.
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
        # Construct semantic cache payload
        id_field = self.entry_id_field_name
        payload = {
            id_field: self.hash_input(prompt),
            self.prompt_field_name: prompt,
            self.response_field_name: response,
            self.vector_field_name: array_to_buffer(vector),
        }
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("If specified, cached metadata must be a dictionary.")
            # Serialize the metadata dict and add to cache payload
            payload[self.metadata_field_name] = self.serialize(metadata)

        # Load LLMCache entry with TTL
        keys = self._index.load(data=[payload], ttl=self._ttl, id_field=id_field)
        return keys[0]
