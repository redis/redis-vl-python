from typing import List, Optional, Union

from redis.commands.search.field import Field, VectorField
from redis.exceptions import ResponseError

from redisvl.index import SearchIndex, check_connected
from redisvl.llmcache.base import BaseLLMCache
from redisvl.query import VectorQuery
from redisvl.utils.utils import array_to_buffer
from redisvl.vectorize.base import BaseVectorizer
from redisvl.vectorize.text import HFTextVectorizer


def similarity(vector_distance: float) -> float:
    return 1 - float(vector_distance)


class SemanticCache(BaseLLMCache):
    """Cache for Large Language Models."""

    # TODO allow for user to change default fields
    _vector_field_name: str = "prompt_vector"
    _default_fields: List[Field] = [
        VectorField(
            _vector_field_name,
            "FLAT",
            {"DIM": 768, "TYPE": "FLOAT32", "DISTANCE_METRIC": "COSINE"},
        ),
    ]

    def __init__(
        self,
        name: str = "cache",
        prefix: str = "llmcache",
        threshold: float = 0.9,
        ttl: Optional[int] = None,
        vectorizer: BaseVectorizer = HFTextVectorizer(
            "sentence-transformers/all-mpnet-base-v2"
        ),
        redis_url: str = "redis://localhost:6379",
        kwargs: Optional[dict] = None,
    ):
        """Semantic Cache for Large Language Models.

        Args:
            name (str, optional): The name of the index. Defaults to "cache".
            prefix (str, optional): The prefix for the index. Defaults to
                "llmcache".
            threshold (float, optional): Semantic threshold for the cache.
                Defaults to 0.9.
            ttl (Optional[int], optional): The TTL for the cache. Defaults to
                None.
            vectorizer (BaseVectorizer, optional): The vectorizer for the cache.
                Defaults to
                HFTextVectorizer("sentence-transformers/all-mpnet-base-v2").
            redis_url (str, optional): The redis url. Defaults to
                "redis://localhost:6379".
            kwargs (Optional[dict], optional): The connection arguments for the
                redis client. Defaults to None.

        Raises:
            TypeError: If an invalid vectorizer is passed in.
            TypeError: If the non-null TTL value is not an int.
            ValueError: If the threshold is not between 0 and 1.
            ValueError: If the index name or prefix is not supplied when
                constructing index manually.
        """
        if "index_name" in kwargs:
            name = kwargs.pop("index_name")
            print("WARNING: index_name is deprecated in favor of name.")

        if not isinstance(vectorizer, BaseVectorizer):
            raise TypeError("Must provide a RedisVL vectorizer class.")

        if ttl is not None and not isinstance(ttl, int):
            raise TypeError("Must provide TTL as an integer.")

        if name is None or prefix is None:
            raise ValueError("Index name and prefix must be provided.")

        # Create the underlying index
        self._index = SearchIndex(
            name=name, prefix=prefix, fields=self._default_fields
        )
        self._index.connect(redis_url=redis_url, **kwargs)
        self._index.create(overwrite=False)

        # Set other attributes
        self._vectorizer = vectorizer
        self.set_ttl(ttl)
        self.set_threshold(threshold)

    # @classmethod
    # def from_index(cls, index: SearchIndex, **kwargs):
    #     """Create a SemanticCache from a pre-existing SearchIndex.

    #     Args:
    #         index (SearchIndex): The SearchIndex object to use as the backbone of the cache.

    #     Returns:
    #         SemanticCache: A SemanticCache object.
    #     """
    #     # TODO: discuss this use case
    #     return cls(index=index, **kwargs)

    @property
    def ttl(self) -> Optional[int]:
        """Returns the TTL for the cache.

        Returns:
            Optional[int]: The TTL for the cache.
        """
        return self._ttl

    def set_ttl(self, ttl: int):
        """Sets the TTL for the cache.

        Args:
            ttl (int): The TTL for the cache.

        Raises:
            ValueError: If the TTL is not an integer.
        """
        self._ttl = int(ttl)

    @property
    def index(self) -> SearchIndex:
        """Returns the index for the cache.

        Returns:
            SearchIndex: The index for the cache.
        """
        return self._index

    @property
    def threshold(self) -> float:
        """Returns the threshold for the cache."""
        return self._threshold

    def set_threshold(self, threshold: float):
        """Sets the threshold for the cache.

        Args:
            threshold (float): The threshold for the cache.

        Raises:
            ValueError: If the threshold is not between 0 and 1.
        """
        if not 0 <= float(threshold) <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        self._threshold = float(threshold)

    @check_connected("_index.client")
    def clear(self):
        """Clear the LLMCache of all keys in the index."""
        client = self._index.client
        prefix = self._index.prefix
        if client:
            with client.pipeline(transaction=False) as pipe:
                for key in client.scan_iter(match=f"{prefix}:*"):
                    pipe.delete(key)
                pipe.execute()
        else:
            raise RuntimeError("LLMCache is not connected to a Redis instance.")

    @check_connected("_index.client")
    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: List[str] = ["response"],
        **kwargs
    ) -> List[str]:
        """Checks whether the cache contains the specified prompt or vector.

        Args:
            prompt (Optional[str], optional): The prompt to check. Defaults to None.
            vector (Optional[List[float]], optional): The vector to check. Defaults to None.
            num_results (int, optional): The number of results to return. Defaults to 1.
            return_fields (List[str], optional): The fields to return. Defaults to ["response"].

        Raises:
            ValueError: If neither prompt nor vector is specified.

        Returns:
            List[str]: The response(s) if the cache contains the prompt or vector.
        """
        # handle backwards compatability
        if "fields" in kwargs:
            return_fields = kwargs.pop("fields")

        if not prompt and not vector:
            raise ValueError("Either prompt or vector must be specified.")

        if not vector:
            vector = self._vectorizer.embed(prompt)  # type: ignore

        # define vector query for semantic cache lookup
        v = VectorQuery(
            vector=vector,
            vector_field_name=self._vector_field_name,
            return_fields=return_fields,
            num_results=num_results,
            return_score=True,
        )

        cache_hits: List[str] = []

        results = self._index.query(v)
        for result in results:
            if similarity(result["vector_distance"]) > self.threshold:
                self._refresh_ttl(result["id"])
                # TODO: discuss
                # Allow for selecting return fields and yielding those objs
                cache_hits.append({
                    key: result[key] for key in return_fields
                })

        if cache_hits == []:
            # TODO: do we need to catch this? An exception here feels noisy from a user perspective.
            pass

        return cache_hits

    @check_connected("_index.client")
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
        # TODO - foot gun for schema mismatch if user has a different index
        vector = vector or self._vectorizer.embed(prompt)

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

    @check_connected("_index.client")
    def _refresh_ttl(self, key: str):
        """Refreshes the TTL for the specified key."""
        client = self._index.client
        if client:
            if self.ttl:
                client.expire(key, self.ttl)
        else:
            raise RuntimeError("LLMCache is not connected to a Redis instance.")
