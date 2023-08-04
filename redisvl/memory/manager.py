from typing import List, Optional

from redis.commands.search.field import VectorField, TagField

from redisvl.index import SearchIndex
from redisvl.vectorize.text import HFTextVectorizer
from redisvl.vectorize.base import BaseVectorizer
from redisvl.query import VectorQuery
from redisvl.utils.utils import array_to_buffer, similarity
from redisvl.memory.interaction import Interaction
import hashlib


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
        prefix: Optional[str] = "memory",
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

    def hash_input(self, prompt: str):
        """Hashes the input using SHA256."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _interaction_id(self, session_id: str, content: str) -> str:
        return f"{session_id}:{self.hash_input(content)}"

    def add(self, interaction: Interaction) -> None:
        """Stores the LLM exchange in the memory store.

        Args:
            interaction (Interaction): Interaction object to persist
        """
        id = self._interaction_id(interaction.session_id, interaction.content)
        vector = array_to_buffer(
            self._vectorizer.embed(interaction.content)
        )
        # construct payload
        payload = interaction.as_dict()
        payload = {
            "id": id,
            **payload,
            self._vector_field_name: vector,
        }
        # write memory
        self._index.load([payload], key_field="id")
        self._append_session(interaction.session_id, id)
        return True

    def _append_session(self, session_id: str, interaction_id: str) -> None:
        # create list_key
        list_key = f"{self._index._name}:session:{session_id}"
        # add interaction to session and get session len
        pipe = self._index.client.pipeline()
        pipe.lpush(list_key, interaction_id).llen(list_key)
        res = pipe.execute()
        # check session len and clean up old interactions
        session_len = res[1]
        session_overhang = session_len - self._max_session_len
        if session_overhang > 0:
            old_interactions = self._index.client.rpop(list_key, session_overhang)
            self._index.client.delete(*old_interactions)

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
        return self.seek_range(session_id, start=0, end=n-1)

    def seek_range(self, session_id: str, start: int, end: int) -> List[Interaction]:
        """_summary_

        Args:
            session_id (str): _description_
            start (int): _description_
            end (int): _description_

        Returns:
            List[Interaction]: _description_
        """
        # fetch recent interactions
        list_key = f"{self._index._name}:session:{session_id}"
        recent_interactions = self._index.client.lrange(list_key, start, end)
        # grab interaction data and return
        pipe = self._index.client.pipeline()
        for i in recent_interactions:
            pipe.hmget(i, *self._return_fields)
        # unpack and return interactions
        res = pipe.execute()
        print(res)
        return [
            Interaction.from_dict(interaction) for interaction in pipe.execute()
        ]

    def seek_relevant(
        self,
        session_id: str,
        context: str,
        n: Optional[int] = 3
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
        # TODO - implement filter on session_id
        v = VectorQuery(
            vector=vector,
            vector_field_name=self._vector_field_name,
            return_fields=self._return_fields,
            # TODO similar to below - how to manage using topN vs "range"
            num_results=n,
            return_score=True,
        )
        results = self._index.query(v)
        # unpack results
        memory_hits = []
        for doc in results.docs:
            # TODO should we enforce this threshold? Trying to think through use cases
            if similarity(doc.vector_distance) > self.semantic_threshold:
                memory_hits.append(Interaction.from_dict(doc.__dict__))
        return memory_hits


    def len(self, session_id: str) -> int:
        list_key = f"{self._index._name}:session:{session_id}"
        return self._index.client.llen(list_key)

    def clear(self, session_id: str):
        list_key = f"{self._index._name}:session:{session_id}"
        self._index.client.delete(list_key)
