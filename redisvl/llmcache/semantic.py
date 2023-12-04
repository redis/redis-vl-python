import warnings
from typing import Any, Dict, List, Optional

from redis.commands.search.field import Field, VectorField

from redisvl.index import SearchIndex
from redisvl.llmcache.base import BaseLLMCache
from redisvl.query import VectorQuery
from redisvl.utils.utils import array_to_buffer
from redisvl.schema import Schema
from redisvl.vectorize.base import BaseVectorizer
from redisvl.vectorize.text import HFTextVectorizer


class SemanticCache(BaseLLMCache):
    """Semantic Cache for Large Language Models."""

    _default_vector_field_name: str = "prompt_vector"
    _default_vector_field: Dict[str, str] = {
        "name": _default_vector_field_name,
        "dims": 768,
        "datatype": "float32",
        "distance_metric": "cosine",
        "algorithm": "flat"
    }

    def __init__(
        self,
        name: str = "cache",
        prefix: str = "llmcache",
        distance_threshold: float = 0.1,
        ttl: Optional[int] = None,
        vectorizer: BaseVectorizer = HFTextVectorizer(
            "sentence-transformers/all-mpnet-base-v2"
        ),
        custom_vector_field: Dict[str, str] = {},
        redis_url: str = "redis://localhost:6379",
        **kwargs,
    ):
        """Semantic Cache for Large Language Models.

        Args:
            name (str, optional): The name of the index. Defaults to "cache".
            prefix (str, optional): The prefix for Redis keys associated with
                the semantic cache search index. Defaults to "llmcache".
            distance_threshold (float, optional): Semantic threshold for the
                cache. Defaults to 0.1.
            ttl (Optional[int], optional): The time-to-live for records cached
                in Redis. Defaults to None.
            vectorizer (BaseVectorizer, optional): The vectorizer for the cache.
                Defaults to HFTextVectorizer.
            redis_url (str, optional): The redis url. Defaults to
                "redis://localhost:6379".
            kwargs (Optional[dict], optional): The connection arguments for the
                redis client. Defaults to None.

        Raises:
            TypeError: If an invalid vectorizer is provided.
            TypeError: If the TTL value is not an int.
            ValueError: If the threshold is not between 0 and 1.
            ValueError: If the index name or prefix is not provided
        """
        # Check for index_name in kwargs
        if "index_name" in kwargs:
            name = kwargs.pop("index_name")
            warnings.warn(
                message="index_name kwarg is deprecated in favor of name.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        # Check for threshold in kwargs
        if "threshold" in kwargs:
            distance_threshold = 1 - kwargs.pop("threshold")
            warnings.warn(
                message="threshold kwarg is deprecated in favor of distance_threshold. "
                + "Setting distance_threshold to 1 - threshold.",
                category=DeprecationWarning,
                stacklevel=2,
            )


        if name is None or prefix is None:
            raise ValueError("Index name and prefix must be provided.")

        # set cache attributes
        self._vector_field = custom_vector_field or self._default_vector_field
        self.set_vectorizer(vectorizer)
        self.set_ttl(ttl)
        self.set_threshold(distance_threshold)

        # create the underlying index
        llm_cache_index_schema = Schema(**{
            "index": {
                "name": name,
                "prefix": prefix,
            },
            "fields": {
                "vector": [self._vector_field]
            }
        })
        self._index = SearchIndex(
            schema=llm_cache_index_schema,
            redis_url=redis_url,
            **kwargs
        )
        self._index.create(overwrite=False)


    @classmethod
    def from_index(cls, index: SearchIndex, **kwargs):
        """Create a SemanticCache from a pre-existing SearchIndex.

        Args:
            index (SearchIndex): The SearchIndex object to use as the backbone of the cache.

        Returns:
            SemanticCache: A SemanticCache object.
        """
        raise DeprecationWarning(
            "This method is deprecated since 0.0.4. Use the constructor instead."
        )
        # TODO should we do these for the other methods in index.py that we removed?

    @property
    def index(self) -> SearchIndex:
        """Returns the index for the cache.

        Returns:
            SearchIndex: The index for the cache.
        """
        return self._index

    @property
    def distance_threshold(self) -> float:
        """Returns the semantic distance threshold for the cache."""
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
        """Sets the vectorizer for the LLM cache. Must be a valid subclass of
           BaseVectorizer and have equivalent dimensions to the vector field
           defined int he schema.

        Args:
            vectorizer (BaseVectorizer): The RedisVL vectorizer to use for
                vectorizing cache entries.

        Raises:
            TypeError: If the vectorizer is not a valid type.
            ValueError: If the vector dimensions are mismatched.
        """
        if not isinstance(vectorizer, BaseVectorizer):
            raise TypeError("Must provide a valid redisvl.vectorizer class.")

        if self._vector_field.get("dims") != vectorizer.dims:
            raise ValueError(
                "Invalid vector dimensions!"
                f"Vectorizer has dims defined as {vectorizer.dims}",
                f"Vector field has dims defined as {self._vector_field.get('dims')}"
            )

        self._vectorizer = vectorizer

    def clear(self) -> None:
        """Clear the cache of all keys while preserving the index"""
        with self._index.client.pipeline(transaction=False) as pipe:
            for key in self._index.client.scan_iter(match=f"{self._index.prefix}:*"):
                pipe.delete(key)
            pipe.execute()

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Checks the cache for results similar to the specified prompt or vector.

        This method searches the semantic cache using either a raw text prompt
        or a precomputed vector, and retrieves cached responses based on
        semantic similarity.

        Args:
            prompt (Optional[str], optional): The text prompt to search for in
                the cache.
            vector (Optional[List[float]], optional): The vector representation
                of the prompt to search for in the cache.
            num_results (int, optional): The number of similar results to
                return.
            return_fields (Optional[List[str]], optional): The fields to include
                in each returned result. If None, defaults to ['response'].

        Raises:
            ValueError: If neither a prompt nor a vector is specified.
            TypeError: If 'return_fields' is not a list when provided.

        Returns:
            List[Dict[str, Any]]: A list of dicts containing the requested
                return fields for each similar cached response.
        """
        # Handle deprecated keyword argument 'fields'
        if "fields" in kwargs:
            return_fields = kwargs.pop("fields")
            warnings.warn(
                message="The 'fields' keyword argument is now deprecated; use 'return_fields' instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        if return_fields is None:
            return_fields = ["response"]

        if not isinstance(return_fields, list):
            raise TypeError("return_fields must be a list of field names")

        if not (prompt or vector):
            raise ValueError("Either prompt or vector must be specified.")

        # Use provided vector or create from prompt
        vector = vector or self._vectorize_prompt(prompt)

        # Check for cache hits by searching the cache
        cache_hits = self._search_cache(vector, return_fields, num_results)

        if cache_hits == []:
            # TODO: I think an exception here is too chatty from a user perspective. Let's just return empty?
            pass

        return cache_hits

    def _vectorize_prompt(self, prompt: Optional[str]) -> List[float]:
        """Converts a text prompt to its vector representation using the
        configured vectorizer."""
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")
        return self._vectorizer.embed(prompt)

    def _search_cache(
        self, vector: List[float], return_fields: List[str], num_results: int
    ) -> List[Dict[str, Any]]:
        """Searches the cache for similar vectors and returns the specified
        fields for each hit."""
        if not isinstance(vector, list):
            raise TypeError("Vector must be a list of floats")

        # Construct vector query for the cache
        query = VectorQuery(
            vector=vector,
            # TODO how to handle the field name if they changed the schema?
            vector_field_name=self._default_vector_field_name,
            return_fields=return_fields,
            num_results=num_results,
            return_score=True,
        )

        # Gather and return the cache hits
        cache_hits: List[Dict[str, Any]] = []
        results = self._index.query(query)
        for result in results:
            # Check against semantic distance threshold
            if float(result["vector_distance"]) < self._distance_threshold:
                self._refresh_ttl(result["id"])
                cache_hits.append({key: result[key] for key in return_fields})
        return cache_hits

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[dict] = {},
    ) -> None:
        """Stores the specified key-value pair in the cache along with metadata.

        Args:
            prompt (str): The prompt to store.
            response (str): The response to store.
            vector (Optional[List[float]], optional): The vector to store. Defaults to None.
            metadata (Optional[dict], optional): The metadata to store. Defaults to {}.

        Raises:
            ValueError: If neither prompt nor vector is specified.
        """
        # Vectorize prompt if necessary and create cache payload
        vector = vector or self._vectorize_prompt(prompt)
        # TODO should we work to make this flexible?
        payload = {
            "id": self.hash_input(prompt),
            "prompt": prompt,
            "response": response,
            self._vector_field_name: array_to_buffer(vector),
        }
        if metadata:
            payload.update(metadata)

        # Load LLMCache entry with TTL
        self._index.load(data=[payload], ttl=self._ttl, key_field="id")

    def _refresh_ttl(self, key: str) -> None:
        """Refreshes the time-to-live for the specified key."""
        if self.ttl:
            self._index.client.expire(key, self.ttl)
