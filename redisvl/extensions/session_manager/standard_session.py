from typing import Any, Dict, List, Optional, Union

from redis import Redis

from redisvl.extensions.session_manager import BaseSessionManager
from redisvl.extensions.session_manager.schema import (
    ChatMessage,
    StandardSessionIndexSchema,
)
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag


class StandardSessionManager(BaseSessionManager):
    session_field_name: str = "session_tag"

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
        """Initialize session memory

        Session Manager stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context.Session history is stored in individual user or LLM prompts and
        responses.

        Args:
            name (str): The name of the session manager index.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance uuid.
            prefix (Optional[str]): Prefix for the keys for this session data.
                Defaults to None and will be replaced with the index name.
            redis_client (Optional[Redis]): A Redis client instance. Defaults to
                None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.

        The proposed schema will support a single combined vector embedding
        constructed from the prompt & response in a single string.

        """
        super().__init__(name, session_tag)

        prefix = prefix or name

        schema = StandardSessionIndexSchema.from_params(name, prefix)

        self._index = SearchIndex(schema=schema)

        if redis_client:
            self._index.set_client(redis_client)
        else:
            self._index.connect(redis_url=redis_url, **connection_kwargs)

        self._index.create(overwrite=False)

        self._default_session_filter = Tag(self.session_field_name) == self._session_tag

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
            id = self.get_recent(top_k=1, raw=True)[0][self.id_field_name]  # type: ignore

        self._index.client.delete(self._index.key(id))  # type: ignore

    @property
    def messages(self) -> Union[List[str], List[Dict[str, str]]]:
        """Returns the full chat history."""
        # TODO raw or as_text?
        # TODO refactor this method to use get_recent and support other session tags?
        return_fields = [
            self.id_field_name,
            self.session_field_name,
            self.role_field_name,
            self.content_field_name,
            self.tool_field_name,
            self.timestamp_field_name,
        ]

        query = FilterQuery(
            filter_expression=self._default_session_filter,
            return_fields=return_fields,
        )

        sorted_query = query.query
        sorted_query.sort_by(self.timestamp_field_name, asc=True)
        messages = [
            doc.__dict__ for doc in self._index.search(sorted_query, query.params).docs
        ]

        return self._format_context(messages, as_text=False)

    def get_recent(
        self,
        top_k: int = 5,
        as_text: bool = False,
        raw: bool = False,
        session_tag: Optional[str] = None,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Retreive the recent conversation history in sequential order.

        Args:
            top_k (int): The number of previous messages to return. Default is 5.
            as_text (bool): Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance uuid.

        Returns:
            Union[str, List[str]]: A single string transcription of the session
                or list of strings if as_text is false.

        Raises:
            ValueError: if top_k is not an integer greater than or equal to 0.
        """
        if type(top_k) != int or top_k < 0:
            raise ValueError("top_k must be an integer greater than or equal to 0")

        return_fields = [
            self.id_field_name,
            self.session_field_name,
            self.role_field_name,
            self.content_field_name,
            self.tool_field_name,
            self.timestamp_field_name,
        ]

        session_filter = (
            Tag(self.session_field_name) == session_tag
            if session_tag
            else self._default_session_filter
        )

        query = FilterQuery(
            filter_expression=session_filter,
            return_fields=return_fields,
            num_results=top_k,
        )

        sorted_query = query.query
        sorted_query.sort_by(self.timestamp_field_name, asc=False)
        messages = [
            doc.__dict__ for doc in self._index.search(sorted_query, query.params).docs
        ]

        if raw:
            return messages[::-1]
        return self._format_context(messages[::-1], as_text)

    def store(
        self, prompt: str, response: str, session_tag: Optional[str] = None
    ) -> None:
        """Insert a prompt:response pair into the session memory. A timestamp
        is associated with each exchange so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            prompt (str): The user prompt to the LLM.
            response (str): The corresponding LLM response.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance uuid.
        """
        self.add_messages(
            [
                {self.role_field_name: "user", self.content_field_name: prompt},
                {self.role_field_name: "llm", self.content_field_name: response},
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
                session. Defaults to instance uuid.
        """
        session_tag = session_tag or self._session_tag
        chat_messages: List[Dict[str, Any]] = []

        for message in messages:

            chat_message = ChatMessage(
                role=message[self.role_field_name],
                content=message[self.content_field_name],
                session_tag=session_tag,
            )

            if self.tool_field_name in message:
                chat_message.tool_call_id = message[self.tool_field_name]

            chat_messages.append(chat_message.to_dict())

        self._index.load(data=chat_messages, id_field=self.id_field_name)

    def add_message(
        self, message: Dict[str, str], session_tag: Optional[str] = None
    ) -> None:
        """Insert a single prompt or response into the session memory.
        A timestamp is associated with it so that it can be later sorted
        in sequential ordering after retrieval.

        Args:
            message (Dict[str,str]): The user prompt or LLM response.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                session. Defaults to instance uuid.
        """
        self.add_messages([message], session_tag)
