from time import time
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
    session_field_name: str = "session_tag"
    user_field_name: str = "user_tag"
    vector_field_name: str = "vector_field"

    def __init__(
        self,
        name: str,
        session_tag: str,
        user_tag: str,
        prefix: Optional[str] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        distance_threshold: float = 0.3,
        redis_client: Optional[Redis] = None,
    ):
        """Initialize session memory with index

        Session Manager stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Session history is stored in prompt:response pairs referred to
        as exchanges.

        Args:
            name (str): The name of the session manager index.
            session_tag (str): Tag to be added to entries to link to a specific
                session.
            user_tag (str): Tag to be added to entries to link to a specific user.
            prefix (Optional[str]): Prefix for the keys for this session data.
                Defaults to None and will be replaced with the index name.
            vectorizer (Vectorizer): The vectorizer to create embeddings with.
            distance_threshold (float): The maximum semantic distance to be
                included in the context. Defaults to 0.3.
            redis_client (Optional[Redis]): A Redis client instance. Defaults to
                None.

        The proposed schema will support a single vector embedding constructed
        from either the prompt or response in a single string.

        """
        super().__init__(name, session_tag, user_tag)

        prefix = prefix or name

        self._vectorizer = vectorizer or HFTextVectorizer(
            model="sentence-transformers/msmarco-distilbert-cos-v5"
        )

        self.set_distance_threshold(distance_threshold)

        schema = IndexSchema.from_dict({"index": {"name": name, "prefix": prefix}})

        schema.add_fields(
            [
                {"name": "role", "type": "text"},
                {"name": "content", "type": "text"},
                {"name": "tool_call_id", "type": "text"},
                {"name": "timestamp", "type": "numeric"},
                {"name": "session_tag", "type": "tag"},
                {"name": "user_tag", "type": "tag"},
                {
                    "name": "vector_field",
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
            self._client = redis_client
        else:
            self._index.connect(redis_url="redis://localhost:6379")
            self._client = Redis(decode_responses=True)

        self._index.create(overwrite=False)

        self.set_scope(session_tag, user_tag)

    def set_scope(
        self,
        session_tag: Optional[str] = None,
        user_tag: Optional[str] = None,
    ) -> None:
        """Set the tag filter to apply to querries based on the desired scope.

        This new scope persists until another call to set_scope is made, or if
        scope specified in calls to get_recent or get_relevant.

        Args:
            session_tag (str): Id of the specific session to filter to. Default is
                None, which means all sessions will be in scope.
            user_tag (str): Id of the specific user to filter to. Default is None,
                which means all users will be in scope.
        """
        if not (session_tag or user_tag):
            return
        self._session_tag = session_tag or self._session_tag
        self._user_tag = user_tag or self._user_tag
        tag_filter = Tag(self.user_field_name) == []
        if user_tag:
            tag_filter = tag_filter & (Tag(self.user_field_name) == self._user_tag)
        if session_tag:
            tag_filter = tag_filter & (
                Tag(self.session_field_name) == self._session_tag
            )

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

    def drop(self, id_field: Optional[str] = None) -> None:
        """Remove a specific exchange from the conversation history.

        Args:
            id_field Optional[str]: The id_field of the entry to delete.
                If None then the last entry is deleted.
        """
        if id_field:
            sep = self._index.key_separator
            key = sep.join([self._index.schema.index.name, id_field])
        else:
            key = self.get_recent(top_k=1, raw=True)[0]["id"]  # type: ignore
        self._client.delete(key)

    @property
    def messages(self) -> Union[List[str], List[Dict[str, str]]]:
        """Returns the full chat history."""
        # TODO raw or as_text?
        return_fields = [
            self.id_field_name,
            self.session_field_name,
            self.user_field_name,
            self.role_field_name,
            self.content_field_name,
            self.tool_field_name,
            self.timestamp_field_name,
        ]

        query = FilterQuery(
            filter_expression=self._tag_filter,
            return_fields=return_fields,
        )

        sorted_query = query.query
        sorted_query.sort_by(self.timestamp_field_name, asc=True)
        hits = self._index.search(sorted_query, query.params).docs

        return self._format_context(hits, as_text=False)

    def get_relevant(
        self,
        prompt: str,
        as_text: bool = False,
        top_k: int = 5,
        fall_back: bool = False,
        session_tag: Optional[str] = None,
        user_tag: Optional[str] = None,
        raw: bool = False,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Searches the chat history for information semantically related to
        the specified prompt.

        This method uses vector similarity search with a text prompt as input.
        It checks for semantically similar prompts and responses and gets
        the top k most relevant previous prompts or responses to include as
        context to the next LLM call.

        Args:
            prompt (str): The text prompt to search for in session memory
            as_text (bool): Whether to return the prompts and responses as text
            or as JSON
            top_k (int): The number of previous exchanges to return. Default is 5.
                Note that one exchange contains both a prompt and a response.
            fallback (bool): Whether to drop back to recent conversation history
                if no relevant context is found.
            session_tag (str): Tag to be added to entries to link to a specific
                session.
            user_tag (str): Tag to be added to entries to link to a specific user.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response.

        Returns:
            Union[List[str], List[Dict[str,str]]: Either a list of strings, or a
            list of prompts and responses in JSON containing the most relevant.

        Raises ValueError: if top_k is not an integer greater or equal to 1.
        """
        if type(top_k) != int or top_k < 1:
            raise ValueError("top_k must be an integer greater than or equal to 1")

        self.set_scope(session_tag, user_tag)
        return_fields = [
            self.session_field_name,
            self.user_field_name,
            self.role_field_name,
            self.content_field_name,
            self.timestamp_field_name,
            self.tool_field_name,
            self.vector_field_name,
        ]

        query = RangeQuery(
            vector=self._vectorizer.embed(prompt),
            vector_field_name=self.vector_field_name,
            return_fields=return_fields,
            distance_threshold=self._distance_threshold,
            num_results=top_k,
            return_score=True,
            filter_expression=self._tag_filter,
        )
        hits = self._index.query(query)

        # if we don't find semantic matches fallback to returning recent context
        if not hits and fall_back:
            return self.get_recent(as_text=as_text, top_k=top_k, raw=raw)
        if raw:
            return hits
        return self._format_context(hits, as_text)

    def get_recent(
        self,
        top_k: int = 5,
        session_tag: Optional[str] = None,
        user_tag: Optional[str] = None,
        as_text: bool = False,
        raw: bool = False,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Retreive the recent conversation history in sequential order.

        Args:
            as_text (bool): Whether to return the conversation as a single string,
                          or list of alternating prompts and responses.
            top_k (int): The number of previous exchanges to return. Default is 5.
                Note that one exchange contains both a prompt and a respoonse.
            session_tag (str): Tag to be added to entries to link to a specific
                session.
            user_tag (str): Tag to be added to entries to link to a specific user.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response

        Returns:
            Union[str, List[str]]: A single string transcription of the session
                                   or list of strings if as_text is false.

        Raises:
            ValueError: if top_k is not an integer greater than or equal to 1.
        """
        if type(top_k) != int or top_k < 1:
            raise ValueError("top_k must be an integer greater than or equal to 1")

        self.set_scope(session_tag, user_tag)
        return_fields = [
            self.id_field_name,
            self.session_field_name,
            self.user_field_name,
            self.role_field_name,
            self.content_field_name,
            self.tool_field_name,
            self.timestamp_field_name,
        ]

        query = FilterQuery(
            filter_expression=self._tag_filter,
            return_fields=return_fields,
            num_results=top_k,
        )

        sorted_query = query.query
        sorted_query.sort_by(self.timestamp_field_name, asc=False)
        hits = self._index.search(sorted_query, query.params).docs

        if raw:
            return hits[::-1]
        return self._format_context(hits[::-1], as_text)

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
        self.add_messages(
            [
                {self.role_field_name: "user", self.content_field_name: prompt},
                {self.role_field_name: "llm", self.content_field_name: response},
            ]
        )

    def add_messages(self, messages: List[Dict[str, str]]) -> None:
        """Insert a list of prompts and responses into the session memory.
        A timestamp is associated with each so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            messages (List[Dict[str, str]]): The list of user prompts and LLM responses.
        """
        sep = self._index.key_separator
        payloads = []
        for message in messages:
            vector = self._vectorizer.embed(message[self.content_field_name])
            timestamp = time()
            id_field = sep.join([self._user_tag, self._session_tag, str(timestamp)])
            payload = {
                self.id_field_name: id_field,
                self.role_field_name: message[self.role_field_name],
                self.content_field_name: message[self.content_field_name],
                self.timestamp_field_name: timestamp,
                self.session_field_name: self._session_tag,
                self.user_field_name: self._user_tag,
                self.vector_field_name: array_to_buffer(vector),
            }
            if self.tool_field_name in message:
                payload.update({self.tool_field_name: message[self.tool_field_name]})
            payloads.append(payload)
        self._index.load(data=payloads, id_field=self.id_field_name)

    def add_message(self, message: Dict[str, str]) -> None:
        """Insert a single prompt or response into the session memory.
        A timestamp is associated with it so that it can be later sorted
        in sequential ordering after retrieval.

        Args:
            message (Dict[str,str]): The user prompt or LLM response.
        """
        self.add_messages([message])
