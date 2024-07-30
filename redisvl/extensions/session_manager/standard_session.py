import json
from time import time
from typing import Any, Dict, List, Optional, Union

from redis import Redis

from redisvl.extensions.session_manager import BaseSessionManager
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag
from redisvl.schema.schema import IndexSchema


class StandardSessionIndexSchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, prefix: str):

        return cls(
            index={"name": name, "prefix": prefix},  # type: ignore
            fields=[  # type: ignore
                {"name": "role", "type": "text"},
                {"name": "content", "type": "text"},
                {"name": "tool_call_id", "type": "text"},
                {"name": "timestamp", "type": "numeric"},
                {"name": "session_tag", "type": "tag"},
                {"name": "user_tag", "type": "tag"},
            ],
        )


class StandardSessionManager(BaseSessionManager):
    session_field_name: str = "session_tag"
    user_field_name: str = "user_tag"

    def __init__(
        self,
        name: str,
        session_tag: str,
        user_tag: str,
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
                session.
            user_tag (Optional[str]): Tag to be added to entries to link to a specific user.
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
        super().__init__(name, session_tag, user_tag)

        prefix = prefix or name

        schema = StandardSessionIndexSchema.from_params(name, prefix)
        self._index = SearchIndex(schema=schema)

        # handle redis connection
        if redis_client:
            self._index.set_client(redis_client)
        elif redis_url:
            self._index.connect(redis_url=redis_url, **connection_kwargs)

        self._index.create(overwrite=False)

        self.set_scope(session_tag, user_tag)

    def set_scope(
        self,
        session_tag: Optional[str] = None,
        user_tag: Optional[str] = None,
    ) -> None:
        """Set the filter to apply to queries based on the desired scope.

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
        self._index.clear()

    def delete(self) -> None:
        """Clear all conversation keys and remove the search index."""
        self._index.delete(drop=True)

    def drop(self, id_field: Optional[str] = None) -> None:
        """Remove a specific exchange from the conversation history.

        Args:
            id_field (Optional[str]): The id_field of the entry to delete.
                If None then the last entry is deleted.
        """
        if id_field:
            sep = self._index.key_separator
            key = sep.join([self._index.schema.index.name, id_field])
        else:
            key = self.get_recent(top_k=1, raw=True)[0]["id"]  # type: ignore
        self._index.client.delete(key)  # type: ignore

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
            top_k (int): The number of previous messages to return. Default is 5.
            session_tag (str): Tag to be added to entries to link to a specific
                session.
            user_tag (str): Tag to be added to entries to link to a specific user.
            as_text (bool): Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response

        Returns:
            Union[str, List[str]]: A single string transcription of the session
                or list of strings if as_text is false.

        Raises:
            ValueError: if top_k is not an integer greater than or equal to 0.
        """
        if type(top_k) != int or top_k < 0:
            raise ValueError("top_k must be an integer greater than or equal to 0")

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
            timestamp = time()
            id_field = sep.join([self._user_tag, self._session_tag, str(timestamp)])
            payload = {
                self.id_field_name: id_field,
                self.role_field_name: message[self.role_field_name],
                self.content_field_name: message[self.content_field_name],
                self.timestamp_field_name: timestamp,
                self.session_field_name: self._session_tag,
                self.user_field_name: self._user_tag,
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
