import json
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

from redis import Redis

from redisvl.extensions.session_manager import BaseSessionManager
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag
from redisvl.schema.schema import IndexSchema


class StandardSessionManager(BaseSessionManager):

    def __init__(
        self,
        name: str,
        session_tag: str,
        user_tag: str,
        redis_client: Optional[Redis] = None,
    ):
        """Initialize session memory

        Session Manager stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Session history is stored in prompt:response pairs referred to
        as exchanges.

        Args:
            name str: The name of the session manager index.
            session_tag str: Tag to be added to entries to link to a specific
                session.
            user_tag str: Tag to be added to entries to link to a specific user.
            redis_client Optional[Redis]: A Redis client instance. Defaults to
                None.

        The proposed schema will support a single combined vector embedding
        constructed from the prompt & response in a single string.

        """
        super().__init__(name, session_tag, user_tag)

        if redis_client:
            self._client = redis_client
        else:
            # TODO make this configurable
            self._client = Redis(host="localhost", port=6379, decode_responses=True)

        self.set_scope(session_tag, user_tag)

    def set_scope(
        self,
        session_tag: Optional[str] = None,
        user_tag: Optional[str] = None,
    ) -> None:
        """Set the filter to apply to querries based on the desired scope.

        This new scope persists until another call to set_scope is made, or if
        scope is specified in calls to get_recent.

        Args:
            session_tag str: Id of the specific session to filter to. Default is
                None, which means session_tag will be unchanged.
            user_tag str: Id of the specific user to filter to. Default is None,
                which means user_tag will be unchanged.
        """
        if not (session_tag or user_tag):
            return

        self._session_tag = session_tag or self._session_tag
        self._user_tag = user_tag or self._user_tag

    def clear(self) -> None:
        """Clears the chat session history."""
        self._client.delete(self.key)

    def delete(self) -> None:
        """Clears the chat session history."""
        self._client.delete(self.key)

    def drop(self, id_field: Optional[str] = None) -> None:
        """Remove a specific exchange from the conversation history.

        Args:
            id_field Optional[str]: The id_field of the entry to delete.
                If None then the last entry is deleted.
        """
        if id_field:
            messages = self._client.lrange(self.key, 0, -1)
            messages = [json.loads(msg) for msg in messages]
            messages = [msg for msg in messages if msg["id_field"] != id_field]
            messages = [json.dumps(msg) for msg in messages]
            self.clear()
            self._client.rpush(self.key, *messages)
        else:
            self._client.rpop(self.key)

    @property
    def messages(self) -> Union[List[str], List[Dict[str, str]]]:
        """Returns the full chat history."""
        # TODO raw or as_text?
        return self.get_recent(top_k=0)

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
            top_k int: The number of previous exchanges to return. Default is 5
            session_tag str: Tag to be added to entries to link to a specific
                session.
            user_tag str: Tag to be added to entries to link to a specific user.
            as_text bool: Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
            raw bool: Whether to return the full Redis hash entry or just the
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
        messages = self._client.lrange(self.key, -top_k, -1)
        messages = [json.loads(msg) for msg in messages]
        if raw:
            return messages
        return self._format_context(messages, as_text)

    @property
    def key(self):
        return ":".join([self._name, self._user_tag, self._session_tag])

    def store(self, prompt: str, response: str) -> None:
        """Insert a prompt:response pair into the session memory. A timestamp
        is associated with each exchange so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            prompt str: The user prompt to the LLM.
            response str: The corresponding LLM response.
        """
        self.add_messages(
            [{"role": "user", "content": prompt}, {"role": "llm", "content": response}]
        )

    def add_messages(self, messages: List[Dict[str, str]]) -> None:
        """Insert a list of prompts and responses into the session memory.
        A timestamp is associated with each so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            messages (List[Dict[str, str]]): The list of user prompts and LLM responses.
        """
        payloads = []
        for message in messages:
            timestamp = time()
            payload = {
                self.id_field_name: ":".join(
                    [self._user_tag, self._session_tag, str(timestamp)]
                ),
                self.role_field_name: message[self.role_field_name],
                self.content_field_name: message[self.content_field_name],
                self.timestamp_field_name: timestamp,
            }
            if self.tool_field_name in message:
                payload.update({self.tool_field_name: message[self.tool_field_name]})

            payloads.append(json.dumps(payload))

        self._client.rpush(self.key, *payloads)

    def add_message(self, message: Dict[str, str]) -> None:
        """Insert a single prompt or response into the session memory.
        A timestamp is associated with it so that it can be later sorted
        in sequential ordering after retrieval.

        Args:
            message (Dict[str,str]): The user prompt or LLM response.
        """
        self.add_messages([message])
