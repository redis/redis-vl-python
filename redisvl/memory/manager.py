from typing import List, Optional

from redis.commands.search.field import TagField, VectorField

from redisvl.index import SearchIndex
from redisvl.memory.interaction import Interaction
from redisvl.query import TagFilter, VectorQuery
from redisvl.utils.utils import array_to_buffer, convert_bytes, similarity
from redisvl.vectorize.base import BaseVectorizer
from redisvl.vectorize.text import HFTextVectorizer


class MemoryManager:
    """Memory manager for Large Language Models applications."""

    _return_fields = list(Interaction.model_fields.keys())
    _tag_field_name = "session_id"
    _vector_field_name = "memory_vector"
    _default_fields = [
        TagField(_tag_field_name),
        VectorField(
            _vector_field_name,
            "FLAT",
            {"DIM": 768, "TYPE": "FLOAT32", "DISTANCE_METRIC": "COSINE"},
        ),
    ]

    def __init__(
        self,
        index_name: Optional[str] = "llm_memory",
        prefix: Optional[str] = "llm_interaction",
        max_session_len: Optional[int] = 30,
        semantic_threshold: Optional[float] = 0.9,
        vectorizer: Optional[BaseVectorizer] = HFTextVectorizer(
            "sentence-transformers/all-mpnet-base-v2"
        ),
        redis_url: Optional[str] = "redis://localhost:6379",
        connection_args: Optional[dict] = None,
    ):
        """Multi-faceted memory manager for Large Language Model (LLM) applications.

        Args:
            index_name (Optional[str], optional): The name of the index. Defaults to "memory".
            prefix (Optional[str], optional): The prefix for the index. Defaults to "memory".
            max_session_len (Optional[int], optional): Strict max number of interactions to
                persist per session. Defaults to 30.
            semantic_threshold (Optional[float], optional): Semantic threshold for the cache. Defaults to 0.9.
            ttl (Optional[int], optional): The TTL for the cache. Defaults to None.
            vectorizer (Optional[Basevectorizer], optional): The vectorizer for the cache.
                Defaults to HFTextVectorizer("sentence-transformers/all-mpnet-base-v2").
            redis_url (Optional[str], optional): The redis url. Defaults to "redis://localhost:6379".
            connection_args (Optional[dict], optional): The connection arguments for the redis client. Defaults to None.

        Raises:
            ValueError: If the threshold is not between 0 and 1.

        """
        self._max_session_len = max_session_len
        self._vectorizer = vectorizer
        self.set_threshold(semantic_threshold)

        index = SearchIndex(name=index_name, prefix=prefix, fields=self._default_fields)
        connection_args = connection_args or {}
        index.connect(url=redis_url, **connection_args)

        # create index or connect to existing index
        if not index.exists():
            index.create()

        self._index = index

    @property
    def index(self) -> SearchIndex:
        """Returns the index for the long term memory.

        Returns:
            SearchIndex: The index for the long term memory.
        """
        return self._index

    @property
    def semantic_threshold(self) -> float:
        """Returns the semantic threshold for the long term memory."""
        return self._semantic_threshold

    def set_threshold(self, semantic_threshold: float):
        """Sets the semantic threshold for the long term memory.

        Args:
            threshold (float): The threshold for the long term memory.

        Raises:
            ValueError: If the threshold is not between 0 and 1.
        """
        if not 0 <= float(semantic_threshold) <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        self._semantic_threshold = float(semantic_threshold)

    def _session_key(self, session_id: str) -> str:
        return f"{self._index._name}:session:{session_id}"

    def add(self, interaction: Interaction) -> None:
        """Stores the LLM exchange in the memory store.

        Args:
            interaction (Interaction): Interaction object to persist
        """
        # construct payload
        payload = interaction.model_dump()
        payload[self._vector_field_name] = array_to_buffer(
            self._vectorizer.embed(interaction.content)
        )
        # write memory
        keys = self._index.load([payload], return_keys=True)
        self._session_append(interaction.session_id, interaction_key=keys[0])
        return True

    def _session_append(self, session_id: str, interaction_key: str) -> None:
        # create session key
        session_key = self._session_key(session_id)
        # add interaction to session and get session len
        pipe = self._index.client.pipeline()
        pipe.lpush(session_key, interaction_key).llen(session_key)
        _, session_len = pipe.execute()
        # check session len and clean up old interactions
        session_overhang = session_len - self._max_session_len
        if session_overhang > 0:
            removed_interaction_keys = self._index.client.rpop(
                session_key, session_overhang
            )
            self._index.client.delete(*removed_interaction_keys)

    def seek(self, session_id: str, n: int) -> List[Interaction]:
        """_summary_

        Args:
            session_id (str): _description_
            n (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            List[Interaction]: _description_
        """
        if n < 1:
            raise ValueError("Must seek atleast 1 recent interaction")
        # use seek range
        return self.seek_range(session_id, start=0, end=n - 1)

    def seek_range(self, session_id: str, start: int, end: int) -> List[Interaction]:
        """_summary_

        Args:
            session_id (str): _description_
            start (int): _description_
            end (int): _description_

        Returns:
            List[Interaction]: _description_
        """
        # create session key
        session_key = self._session_key(session_id)
        # fetch recent interactions
        recent_interactions = self._index.client.lrange(session_key, start, end)
        # grab interaction data and fetch with pipeline
        pipe = self._index.client.pipeline()
        for i in recent_interactions:
            pipe.hmget(i, *self._return_fields)
        # unpack and return interactions
        interactions: List[Interaction] = [
            Interaction(**convert_bytes(dict(zip(self._return_fields, res))))
            for res in pipe.execute()
        ]
        return interactions

    def seek_relevant(
        self, session_id: str, context: str, n: Optional[int] = 3
    ) -> List[Interaction]:
        """_summary_

        Args:
            session_id (str): _description_
            context (str): _description_
            n (Optional, optional): _description_. Defaults to 3.

        Returns:
            List[Interaction]: _description_
        """
        # create vector from context
        vector = self._vectorizer.embed(context)
        # create redis vector query
        v = VectorQuery(
            vector=vector,
            vector_field_name=self._vector_field_name,
            return_fields=self._return_fields,
            # TODO similar to below - how to manage using topN vs "range"
            num_results=n,
            return_score=True,
        )
        v.set_filter(TagFilter(self._tag_field_name, session_id))
        results = self._index.query(v)
        # unpack results
        memory_hits = []
        for doc in results.docs:
            print("DOC", doc, flush=True)
            if similarity(doc.vector_distance) > self.semantic_threshold:
                memory_hits.append(Interaction(**doc.__dict__))
        return memory_hits

    def len(self, session_id: str) -> int:
        """_summary_

        Args:
            session_id (str): _description_

        Returns:
            int: _description_
        """
        return self._index.client.llen(self._session_key(session_id))

    def clear(self, session_id: str):
        """_summary_

        Args:
            session_id (str): _description_
        """
        session_key = self._session_key(session_id)
        pipe = self._index.client.pipeline()
        for key in self._index.client.lrange(session_key, 0, -1):
            pipe.delete(key)
        pipe.delete(session_key)
        pipe.execute()
