from typing import Any, Dict, List, Optional, Union

from redis import Redis

from redisvl.extensions.constants import (
    CONTENT_FIELD_NAME,
    ID_FIELD_NAME,
    METADATA_FIELD_NAME,
    ROLE_FIELD_NAME,
    SESSION_FIELD_NAME,
    TIMESTAMP_FIELD_NAME,
    TOOL_FIELD_NAME,
)
from redisvl.extensions.message_history import BaseMessageHistory
from redisvl.extensions.message_history.schema import ChatMessage, MessageHistorySchema
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag
from redisvl.utils.utils import serialize


class MessageHistory(BaseMessageHistory):

    def __init__(
        self,
        name: str,
        session_tag: Optional[str] = None,
        prefix: Optional[str] = None,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """Initialize message history

        Message History stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Message history is stored in individual user or LLM prompts and
        responses.

        Args:
            name (str): The name of the message history index.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
            prefix (Optional[str]): Prefix for the keys for this conversation data.
                Defaults to None and will be replaced with the index name.
            redis_client (Optional[Redis]): A Redis client instance. Defaults to
                None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.

        """
        super().__init__(name, session_tag)

        prefix = prefix or name

        schema = MessageHistorySchema.from_params(name, prefix)

        self._index = SearchIndex(
            schema=schema,
            redis_client=redis_client,
            redis_url=redis_url,
            **connection_kwargs,
        )

        self._index.create(overwrite=False)

        self._default_session_filter = Tag(SESSION_FIELD_NAME) == self._session_tag

    def clear(self) -> None:
        """Clears the conversation message history."""
        self._index.clear()

    def delete(self) -> None:
        """Clear all conversation keys and remove the search index."""
        self._index.delete(drop=True)

    def drop(self, id: Optional[str] = None) -> None:
        """Remove a specific exchange from the conversation history.

        Args:
            id (Optional[str]): The id of the message entry to delete.
                If None then the last entry is deleted.
        """
        if id is None:
            id = self.get_recent(top_k=1, raw=True)[0][ID_FIELD_NAME]  # type: ignore

        self._index.client.delete(self._index.key(id))  # type: ignore

    @property
    def messages(self) -> Union[List[str], List[Dict[str, str]]]:
        """Returns the full message history."""
        # TODO raw or as_text?
        # TODO refactor this method to use get_recent and support other session tags?
        return_fields = [
            ID_FIELD_NAME,
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TOOL_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        query = FilterQuery(
            filter_expression=self._default_session_filter,
            return_fields=return_fields,
            num_results=1000,
        )
        query.sort_by(TIMESTAMP_FIELD_NAME, asc=True)
        messages = self._index.query(query)

        return self._format_context(messages, as_text=False)

    def get_recent(
        self,
        top_k: int = 5,
        as_text: bool = False,
        raw: bool = False,
        session_tag: Optional[str] = None,
        role: Optional[Union[str, List[str]]] = None,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Retrieve the recent message history in sequential order.

        Args:
            top_k (int): The number of previous messages to return. Default is 5.
            as_text (bool): Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response.
            session_tag (Optional[str]): Tag of the entries linked to a specific
                conversation session. Defaults to instance ULID.
            role (Optional[Union[str, List[str]]]): Filter messages by role(s).
                Can be a single role string ("system", "user", "llm", "tool") or
                a list of roles. If None, all roles are returned.

        Returns:
            Union[str, List[str]]: A single string transcription of the messages
                or list of strings if as_text is false.

        Raises:
            ValueError: if top_k is not an integer greater than or equal to 0,
                or if role contains invalid values.
        """
        if type(top_k) != int or top_k < 0:
            raise ValueError("top_k must be an integer greater than or equal to 0")

        # Validate and normalize role parameter
        roles_to_filter = self._validate_roles(role)

        return_fields = [
            ID_FIELD_NAME,
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TOOL_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        session_filter = (
            Tag(SESSION_FIELD_NAME) == session_tag
            if session_tag
            else self._default_session_filter
        )

        # Combine session filter with role filter if provided
        filter_expression = session_filter
        if roles_to_filter is not None:
            if len(roles_to_filter) == 1:
                role_filter = Tag(ROLE_FIELD_NAME) == roles_to_filter[0]
            else:
                # Multiple roles - use OR logic
                role_filters = [Tag(ROLE_FIELD_NAME) == r for r in roles_to_filter]
                role_filter = role_filters[0]
                for rf in role_filters[1:]:
                    role_filter = role_filter | rf

            filter_expression = session_filter & role_filter

        query = FilterQuery(
            filter_expression=filter_expression,
            return_fields=return_fields,
            num_results=top_k,
        )
        query.sort_by(TIMESTAMP_FIELD_NAME, asc=False)
        messages = self._index.query(query)

        if raw:
            return messages[::-1]
        return self._format_context(messages[::-1], as_text)

    def store(
        self, prompt: str, response: str, session_tag: Optional[str] = None
    ) -> None:
        """Insert a prompt:response pair into the message history. A timestamp
        is associated with each exchange so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            prompt (str): The user prompt to the LLM.
            response (str): The corresponding LLM response.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
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
        """Insert a list of prompts and responses into the message history.
        A timestamp is associated with each so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            messages (List[Dict[str, str]]): The list of user prompts and LLM responses.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
        """
        session_tag = session_tag or self._session_tag
        chat_messages: List[Dict[str, Any]] = []

        for message in messages:

            chat_message = ChatMessage(
                role=message[ROLE_FIELD_NAME],
                content=message[CONTENT_FIELD_NAME],
                session_tag=session_tag,
            )

            if TOOL_FIELD_NAME in message:
                chat_message.tool_call_id = message[TOOL_FIELD_NAME]
            if METADATA_FIELD_NAME in message:
                chat_message.metadata = serialize(message[METADATA_FIELD_NAME])
            chat_messages.append(chat_message.to_dict())

        self._index.load(data=chat_messages, id_field=ID_FIELD_NAME)

    def add_message(
        self, message: Dict[str, str], session_tag: Optional[str] = None
    ) -> None:
        """Insert a single prompt or response into the message history.
        A timestamp is associated with it so that it can be later sorted
        in sequential ordering after retrieval.

        Args:
            message (Dict[str,str]): The user prompt or LLM response.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
        """
        self.add_messages([message], session_tag)
