from typing import Any, Dict, List, Optional, Union

from redis import Redis

from redisvl.extensions.constants import (
    CONTENT_FIELD_NAME,
    ID_FIELD_NAME,
    ROLE_FIELD_NAME,
    SESSION_FIELD_NAME,
    SESSION_VECTOR_FIELD_NAME,
    TIMESTAMP_FIELD_NAME,
    TOOL_FIELD_NAME,
)
from redisvl.extensions.session_manager import BaseSessionManager
from redisvl.extensions.session_manager.schema import (
    ChatMessage,
    SemanticSessionIndexSchema,
)
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, RangeQuery
from redisvl.query.filter import Tag
from redisvl.utils.utils import deprecated_argument, validate_vector_dims
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer


class SemanticSessionManager(BaseSessionManager):

    @deprecated_argument("dtype", "vectorizer")
    def __init__(
        self,
        name: str,
        session_tag: Optional[str] = None,
        prefix: Optional[str] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        distance_threshold: float = 0.3,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: Dict[str, Any] = {},
        overwrite: bool = False,
        **kwargs,
    ):
        """Initialize session memory with index

        Session Manager stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Session history is stored in individual user or LLM prompts and
        responses.


        Args:
            name (str): The name of the session manager index.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance ULID.
            prefix (Optional[str]): Prefix for the keys for this session data.
                Defaults to None and will be replaced with the index name.
            vectorizer (Optional[BaseVectorizer]): The vectorizer used to create embeddings.
            distance_threshold (float): The maximum semantic distance to be
                included in the context. Defaults to 0.3.
            redis_client (Optional[Redis]): A Redis client instance. Defaults to
                None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.
            overwrite (bool): Whether or not to force overwrite the schema for
                the semantic session index. Defaults to false.

        The proposed schema will support a single vector embedding constructed
        from either the prompt or response in a single string.

        """
        super().__init__(name, session_tag)

        prefix = prefix or name
        dtype = kwargs.pop("dtype", None)

        # Validate a provided vectorizer or set the default
        if vectorizer:
            if not isinstance(vectorizer, BaseVectorizer):
                raise TypeError("Must provide a valid redisvl.vectorizer class.")
            if dtype and vectorizer.dtype != dtype:
                raise ValueError(
                    f"Provided dtype {dtype} does not match vectorizer dtype {vectorizer.dtype}"
                )
        else:
            vectorizer_kwargs = kwargs

            if dtype:
                vectorizer_kwargs.update(**{"dtype": dtype})

            vectorizer = HFTextVectorizer(
                model="sentence-transformers/all-mpnet-base-v2",
                **vectorizer_kwargs,
            )

        self._vectorizer = vectorizer

        self.set_distance_threshold(distance_threshold)

        schema = SemanticSessionIndexSchema.from_params(
            name, prefix, vectorizer.dims, vectorizer.dtype  # type: ignore
        )

        self._index = SearchIndex(
            schema=schema,
            redis_client=redis_client,
            redis_url=redis_url,
            **connection_kwargs,
        )

        # Check for existing session index
        if not overwrite and self._index.exists():
            existing_index = SearchIndex.from_existing(
                name, redis_client=self._index.client
            )
            if existing_index.schema.to_dict() != self._index.schema.to_dict():
                raise ValueError(
                    f"Existing index {name} schema does not match the user provided schema for the semantic session. "
                    "If you wish to overwrite the index schema, set overwrite=True during initialization."
                )
        self._index.create(overwrite=overwrite, drop=False)

        self._default_session_filter = Tag(SESSION_FIELD_NAME) == self._session_tag

    def clear(self) -> None:
        """Clears the chat session history."""
        self._index.clear()

    def delete(self) -> None:
        """Clear all conversation keys and remove the search index."""
        self._index.delete(drop=True)

    def drop(self, id: Optional[str] = None) -> None:
        """Remove a specific exchange from the conversation history.

        Args:
            id (Optional[str]): The id of the session entry to delete.
                If None then the last entry is deleted.
        """
        if id is None:
            id = self.get_recent(top_k=1, raw=True)[0][ID_FIELD_NAME]  # type: ignore

        self._index.client.delete(self._index.key(id))  # type: ignore

    @property
    def messages(self) -> Union[List[str], List[Dict[str, str]]]:
        """Returns the full chat history."""
        # TODO raw or as_text?
        # TODO refactor method to use get_recent and support other session tags
        return_fields = [
            ID_FIELD_NAME,
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TOOL_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
        ]

        query = FilterQuery(
            filter_expression=self._default_session_filter,
            return_fields=return_fields,
        )
        query.sort_by(TIMESTAMP_FIELD_NAME, asc=True)
        messages = self._index.query(query)

        return self._format_context(messages, as_text=False)

    def get_relevant(
        self,
        prompt: str,
        as_text: bool = False,
        top_k: int = 5,
        fall_back: bool = False,
        session_tag: Optional[str] = None,
        raw: bool = False,
        distance_threshold: Optional[float] = None,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Searches the chat history for information semantically related to
        the specified prompt.

        This method uses vector similarity search with a text prompt as input.
        It checks for semantically similar prompts and responses and gets
        the top k most relevant previous prompts or responses to include as
        context to the next LLM call.

        Args:
            prompt (str): The message text to search for in session memory
            as_text (bool): Whether to return the prompts and responses as text
            or as JSON
            top_k (int): The number of previous messages to return. Default is 5.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance ULID.
            distance_threshold (Optional[float]): The threshold for semantic
                vector distance.
            fall_back (bool): Whether to drop back to recent conversation history
                if no relevant context is found.
            raw (bool): Whether to return the full Redis hash entry or just the
                message.

        Returns:
            Union[List[str], List[Dict[str,str]]: Either a list of strings, or a
            list of prompts and responses in JSON containing the most relevant.

        Raises ValueError: if top_k is not an integer greater or equal to 0.
        """
        if type(top_k) != int or top_k < 0:
            raise ValueError("top_k must be an integer greater than or equal to -1")
        if top_k == 0:
            return []

        # override distance threshold
        distance_threshold = distance_threshold or self._distance_threshold

        return_fields = [
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
            TOOL_FIELD_NAME,
        ]

        session_filter = (
            Tag(SESSION_FIELD_NAME) == session_tag
            if session_tag
            else self._default_session_filter
        )

        query = RangeQuery(
            vector=self._vectorizer.embed(prompt),
            vector_field_name=SESSION_VECTOR_FIELD_NAME,
            return_fields=return_fields,
            distance_threshold=distance_threshold,
            num_results=top_k,
            return_score=True,
            filter_expression=session_filter,
            dtype=self._vectorizer.dtype,
        )
        messages = self._index.query(query)

        # if we don't find semantic matches fallback to returning recent context
        if not messages and fall_back:
            return self.get_recent(as_text=as_text, top_k=top_k, raw=raw)
        if raw:
            return messages
        return self._format_context(messages, as_text)

    def get_recent(
        self,
        top_k: int = 5,
        as_text: bool = False,
        raw: bool = False,
        session_tag: Optional[str] = None,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Retreive the recent conversation history in sequential order.

        Args:
            top_k (int): The number of previous exchanges to return. Default is 5.
            as_text (bool): Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance ULID.

        Returns:
            Union[str, List[str]]: A single string transcription of the session
                or list of strings if as_text is false.

        Raises:
            ValueError: if top_k is not an integer greater than or equal to 0.
        """
        if type(top_k) != int or top_k < 0:
            raise ValueError("top_k must be an integer greater than or equal to 0")

        return_fields = [
            ID_FIELD_NAME,
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TOOL_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
        ]

        session_filter = (
            Tag(SESSION_FIELD_NAME) == session_tag
            if session_tag
            else self._default_session_filter
        )

        query = FilterQuery(
            filter_expression=session_filter,
            return_fields=return_fields,
            num_results=top_k,
        )
        query.sort_by(TIMESTAMP_FIELD_NAME, asc=False)
        messages = self._index.query(query)

        if raw:
            return messages[::-1]
        return self._format_context(messages[::-1], as_text)

    @property
    def distance_threshold(self):
        return self._distance_threshold

    def set_distance_threshold(self, threshold):
        self._distance_threshold = threshold

    def store(
        self, prompt: str, response: str, session_tag: Optional[str] = None
    ) -> None:
        """Insert a prompt:response pair into the session memory. A timestamp
        is associated with each message so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            prompt (str): The user prompt to the LLM.
            response (str): The corresponding LLM response.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance ULID.
        """
        self.add_messages(
            [
                {ROLE_FIELD_NAME: "user", CONTENT_FIELD_NAME: prompt},
                {ROLE_FIELD_NAME: "llm", CONTENT_FIELD_NAME: response},
            ],
            session_tag,
        )

    def add_messages(
        self, messages: List[Dict[str, str]], session_tag: Optional[str] = None
    ) -> None:
        """Insert a list of prompts and responses into the session memory.
        A timestamp is associated with each so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            messages (List[Dict[str, str]]): The list of user prompts and LLM responses.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance ULID.
        """
        session_tag = session_tag or self._session_tag
        chat_messages: List[Dict[str, Any]] = []

        for message in messages:
            content_vector = self._vectorizer.embed(message[CONTENT_FIELD_NAME])
            validate_vector_dims(
                len(content_vector),
                self._index.schema.fields[SESSION_VECTOR_FIELD_NAME].attrs.dims,  # type: ignore
            )

            chat_message = ChatMessage(
                role=message[ROLE_FIELD_NAME],
                content=message[CONTENT_FIELD_NAME],
                session_tag=session_tag,
                vector_field=content_vector,  # type: ignore
            )

            if TOOL_FIELD_NAME in message:
                chat_message.tool_call_id = message[TOOL_FIELD_NAME]

            chat_messages.append(chat_message.to_dict(dtype=self._vectorizer.dtype))

        self._index.load(data=chat_messages, id_field=ID_FIELD_NAME)

    def add_message(
        self, message: Dict[str, str], session_tag: Optional[str] = None
    ) -> None:
        """Insert a single prompt or response into the session memory.
        A timestamp is associated with it so that it can be later sorted
        in sequential ordering after retrieval.

        Args:
            message (Dict[str,str]): The user prompt or LLM response.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance ULID.
        """
        self.add_messages([message], session_tag)
