from typing import List, Optional

from redis.commands.search.field import VectorField

from redisvl.index import SearchIndex
from redisvl.llmcache.base import BaseLLMCache
from redisvl.providers import HuggingfaceProvider
from redisvl.providers.base import BaseProvider
from redisvl.query import create_vector_query
from redisvl.utils.log import get_logger
from redisvl.utils.utils import array_to_buffer

_logger = get_logger(__name__)


class SemanticCache(BaseLLMCache):
    """Cache for Large Language Models."""

    _default_fields = [
        VectorField(
            "prompt_vector",
            "FLAT",
            {"DIM": 768, "TYPE": "FLOAT32", "DISTANCE_METRIC": "COSINE"},
        ),
    ]
    _default_provider = HuggingfaceProvider("sentence-transformers/all-mpnet-base-v2")

    def __init__(
        self,
        index_name: str = "cache",
        prefix: str = "llmcache",
        threshold: float = 0.9,
        provider: Optional[BaseProvider] = None,
        redis_url: Optional[str] = "redis://localhost:6379",
        connection_args: Optional[dict] = None,
        ttl: Optional[int] = None,
    ):

        self.ttl = ttl
        self._provider = provider or self._default_provider
        self._threshold = threshold

        # TODO - configure logging based on verbosity
        self._index = SearchIndex(
            index_name, prefix=prefix, fields=self._default_fields
        )
        connection_args = connection_args or {}
        self._index.connect(redis_url=redis_url, **connection_args)
        self._index.create()

    @property
    def index(self) -> SearchIndex:
        """Returns the index for the cache."""
        return self._index

    @property
    def threshold(self) -> float:
        """Returns the threshold for the cache."""
        return self._threshold

    def set_threshold(self, threshold: float):
        """Sets the threshold for the cache."""
        self._threshold = threshold

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        fields: List[str] = ["response"],
    ) -> Optional[List[str]]:
        """Checks whether the cache contains the specified key."""
        if not prompt and not vector:
            raise ValueError("Either prompt or vector must be specified.")

        query = create_vector_query(
            return_fields=fields,
            vector_field_name="prompt_vector",
            number_of_results=num_results,
        )
        if vector:
            prompt_vector = array_to_buffer(vector)
        else:
            prompt_vector = array_to_buffer(self._provider.embed(prompt)) # type: ignore
        results = self._index.search(query, query_params={"vector": prompt_vector})

        cache_hits = []
        for doc in results.docs:
            sim = similarity(doc.vector_score)
            _logger.info("Similarity: %s", sim)
            if similarity(doc.vector_score) > self.threshold:
                cache_hits.append(doc.response)
        return cache_hits

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[dict] = {},
        key: Optional[str] = None,
    ) -> None:
        """Stores the specified key-value pair in the cache along with metadata."""
        if not key:
            key = self.hash_input(prompt)
        if vector:
            prompt_vector = array_to_buffer(vector)
        else:
            prompt_vector = array_to_buffer(self._provider.embed(prompt))

        payload = {"id": key, "prompt_vector": prompt_vector, "response": response}
        if metadata:
            payload.update(metadata)
        self._index.load([payload])

    def _refresh_ttl(self, key: str):
        """Refreshes the TTL for the specified key."""
        client = self._index.get_client()
        if client:
            client.expire(key, self.ttl)
        else:
            raise RuntimeError("LLMCache is not connected to a Redis instance.")


def similarity(distance):
    return 1 - float(distance)
