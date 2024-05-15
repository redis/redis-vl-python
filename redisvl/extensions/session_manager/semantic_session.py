import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from redis import Redis

from redisvl.extensions.session_manager import BaseSessionManager
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, RangeQuery
from redisvl.query.filter import Num, Tag
from redisvl.redis.utils import array_to_buffer
from redisvl.schema.schema import IndexSchema
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer


class SemanticSessionManager(BaseSessionManager):
    def __init__(
        self,
        name: str,
        session_id: str,
        user_id: str,
        prefix: Optional[str] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        distance_threshold: float = 0.3,
        redis_client: Optional[Redis] = None,
        preamble: str = "",
    ):
        """Initialize session memory with index

        Session Manager stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Session history is stored in prompt:response pairs referred to
        as exchanges.

        Args:
            name (str): The name of the session manager index.
            session_id (str): Tag to be added to entries to link to a specific
                session.
            user_id (str): Tag to be added to entries to link to a specific user.
            prefix (Optional[str]): Prefix for the keys for this session data.
                Defaults to None and will be replaced with the index name.
            vectorizer (Vectorizer): The vectorizer to create embeddings with.
            distance_threshold (float): The maximum semantic distance to be
                included in the context. Defaults to 0.3.
            redis_client (Optional[Redis]): A Redis client instance. Defaults to
                None.
            preamble (str): System level prompt to be included in all context.


        The proposed schema will support a single combined vector embedding
        constructed from the prompt & response in a single string.

        """
        super().__init__(name, session_id, user_id, prefix, preamble)

        prefix = prefix or name

        if vectorizer is None:
            self._vectorizer = HFTextVectorizer(
                model="sentence-transformers/msmarco-distilbert-cos-v5"
            )

        self.set_distance_threshold(distance_threshold)

        schema = IndexSchema.from_dict({"index": {"name": name, "prefix": prefix}})

        schema.add_fields(
            [
                {"name": "prompt", "type": "text"},
                {"name": "response", "type": "text"},
                {"name": "timestamp", "type": "numeric"},
                {"name": "session_id", "type": "tag"},
                {"name": "user_id", "type": "tag"},
                {"name": "count", "type": "numeric"},
                {"name": "token_count", "type": "numeric"},
                {
                    "name": "combined_vector_field",
                    "type": "vector",
                    "attrs": {
                        "dims": self._vectorizer.dims,
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ]
        )

        self._index = SearchIndex(schema=schema)

        if redis_client:
            self._index.set_client(redis_client)
            self._redis_client = redis_client
        else:
            self._index.connect(redis_url="redis://localhost:6379")
            self._redis_client = Redis(decode_responses=True)

        self._index.create(overwrite=False)

        self.set_scope(session_id, user_id)

    def set_scope(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Set the tag filter to apply to querries based on the desired scope.

        This new scope persists until another call to set_scope is made, or if
        scope specified in calls to fetch_recent or fetch_relevant.

        Args:
            session_id (str): Id of the specific session to filter to. Default is
                None, which means all sessions will be in scope.
            user_id (str): Id of the specific user to filter to. Default is None,
                which means all users will be in scope.
        """
        if not (session_id or user_id):
            return

        tag_filter = Tag("user_id") == []
        if user_id:
            tag_filter = tag_filter & (Tag("user_id") == self._user_id)
        if session_id:
            tag_filter = tag_filter & (Tag("session_id") == self._session_id)

        self._tag_filter = tag_filter

    def clear(self) -> None:
        """Clears the chat session history."""
        with self._index.client.pipeline(transaction=False) as pipe:  # type: ignore
            for key in self._index.client.scan_iter(match=f"{self._index.prefix}:*"):  # type: ignore
                pipe.delete(key)
            pipe.execute()

    def delete(self) -> None:
        """Clear all conversation keys and remove the search index."""
        self._index.delete(drop=True)

    def drop(self, timestamp: Optional[int]) -> None:
        """Remove a specific exchange from the conversation history."""
        raise NotImplementedError

    def fetch_relevant(
        self,
        prompt: str,
        as_text: bool = False,
        top_k: int = 3,
        fall_back: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        raw: bool = False,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Searches the chat history for information semantically related to
        the specified prompt.

        This method uses vector similarity search with a text prompt as input.
        It checks for semantically similar prompt:response pairs and fetches
        the top k most relevant previous prompt:response pairs to include as
        context to the next LLM call.

        Args:
            prompt (str): The text prompt to search for in session memory
            as_text (bool): Whether to return the prompt:response pairs as text
            or as JSON
            top_k (int): The number of previous exchanges to return. Default is 3.
            fallback (bool): Whether to drop back to recent conversation history
                if no relevant context is found.
            session_id (str): Tag to be added to entries to link to a specific
                session.
            user_id (str): Tag to be added to entries to link to a specific user.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response.

        Returns:
            Union[List[str], List[Dict[str,str]]: Either a list of strings, or a
            list of prompts and responses in JSON containing the most relevant.
        """
        self.set_scope(session_id, user_id)
        return_fields = [
            "session_id",
            "user_id",
            "count",
            "prompt",
            "response",
            "timestamp",
            "combined_vector_field",
        ]

        query = RangeQuery(
            vector=self._vectorizer.embed(prompt),
            vector_field_name="combined_vector_field",
            return_fields=return_fields,
            distance_threshold=self._distance_threshold,
            num_results=top_k,
            return_score=True,
            filter_expression=self._tag_filter,
        )
        hits = self._index.query(query)

        # if we don't find semantic matches fallback to returning recent context
        if not hits and fall_back:
            return self.fetch_recent(as_text=as_text, top_k=top_k, raw=raw)
        if raw:
            return hits
        return self._format_context(hits, as_text)

    def fetch_recent(
        self,
        top_k: int = 3,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        as_text: bool = False,
        raw: bool = False,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Retreive the recent conversation history in sequential order.

        Args:
            as_text (bool): Whether to return the conversation as a single string,
                          or list of alternating prompts and responses.
            top_k (int): The number of previous exchanges to return. Default is 3
            session_id (str): Tag to be added to entries to link to a specific
                session.
            user_id (str): Tag to be added to entries to link to a specific user.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response
        Returns:
            Union[str, List[str]]: A single string transcription of the session
                                   or list of strings if as_text is false.
        """
        self.set_scope(session_id, user_id)
        return_fields = [
            "session_id",
            "user_id",
            "count",
            "prompt",
            "response",
            "timestamp",
        ]

        count_key = ":".join([self._user_id, self._session_id, "count"])
        count = self._redis_client.get(count_key) or 0
        last_k_filter = Num("count") > int(count) - top_k
        combined = self._tag_filter & last_k_filter

        query = FilterQuery(
            return_fields=return_fields,
            filter_expression=combined,
            params={"sortby": "timestamp"},
        )
        hits = self._index.query(query)
        if raw:
            return hits
        return self._format_context(hits, as_text)

    @property
    def distance_threshold(self):
        return self._distance_threshold

    def set_distance_threshold(self, threshold):
        self._distance_threshold = threshold

    def store(self, prompt: str, response: str) -> None:
        """Insert a prompt:response pair into the session memory. A timestamp
        is associated with each exchange so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            prompt (str): The user prompt to the LLM.
            response (str): The corresponding LLM response.
        """
        count_key = ":".join([self._user_id, self._session_id, "count"])
        count = self._redis_client.incr(count_key)
        vector = self._vectorizer.embed(prompt + response)
        timestamp = int(datetime.now().timestamp())
        payload = {
            "id": self.hash_input(prompt + str(timestamp)),
            "prompt": prompt,
            "response": response,
            "timestamp": timestamp,
            "session_id": self._session_id,
            "user_id": self._user_id,
            "count": count,
            "token_count": 1,  # TODO get actual token count
            "combined_vector_field": array_to_buffer(vector),
        }
        self._index.load(data=[payload])

    def set_preamble(self, prompt: str) -> None:
        """Add a preamble statement to the the begining of each session to be
        included in each subsequent LLM call.
        """
        self._preamble = {"role": "_preamble", "_content": prompt}
        # TODO store this in Redis with asigned scope?

    def hash_input(self, prompt: str):
        """Hashes the input using SHA256."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
