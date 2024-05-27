import hashlib
import json
from datetime import datetime
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
        session_id: str,
        user_id: str,
        redis_client: Optional[Redis] = None,
        preamble: str = "",
    ):
        """Initialize session memory

        Session Manager stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Session history is stored in prompt:response pairs referred to
        as exchanges.

        Args:
            name str: The name of the session manager index.
            session_id str: Tag to be added to entries to link to a specific
                session.
            user_id str: Tag to be added to entries to link to a specific user.
            redis_client Optional[Redis]: A Redis client instance. Defaults to
                None.
            preamble str: System level prompt to be included in all context.


        The proposed schema will support a single combined vector embedding
        constructed from the prompt & response in a single string.

        """
        super().__init__(name, session_id, user_id, preamble)

        if redis_client:
            self._client = redis_client
        else:
            # TODO make this configurable
            self._client = Redis(host="localhost", port=6379, decode_responses=True)

        self.set_scope(session_id, user_id)

    def set_scope(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Set the filter to apply to querries based on the desired scope.

        This new scope persists until another call to set_scope is made, or if
        scope is specified in calls to fetch_recent.

        Args:
            session_id str: Id of the specific session to filter to. Default is
                None, which means session_id will be unchanged.
            user_id str: Id of the specific user to filter to. Default is None,
                which means user_id will be unchanged.
        """
        if not (session_id or user_id):
            return

        self._session_id = session_id or self._session_id
        self._user_id = user_id or self._user_id

    def clear(self) -> None:
        """Clears the chat session history."""
        self._client.delete(self.key)

    def delete(self) -> None:
        """Clears the chat session history."""
        self._client.delete(self.key)

    def drop(self, id_field: Optional[str]=None) -> None:
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
            top_k int: The number of previous exchanges to return. Default is 3
            session_id str: Tag to be added to entries to link to a specific
                session.
            user_id str: Tag to be added to entries to link to a specific user.
            as_text bool: Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
            raw bool: Whether to return the full Redis hash entry or just the
                prompt and response
        Returns:
            Union[str, List[str]]: A single string transcription of the session
                or list of strings if as_text is false.
        """
        if top_k == 0:
            return self._format_context([], as_text)
        if top_k == -1:
            top_k = 0
        self.set_scope(session_id, user_id)
        messages = self._client.lrange(self.key, -top_k, -1)
        messages = [json.loads(msg) for msg in messages]
        if raw:
            return messages
        return self._format_context(messages, as_text)

    @property
    def key(self):
        return ":".join([self._name, self._user_id, self._session_id])

    def store(self, prompt: str, response: str) -> None:
        """Insert a prompt:response pair into the session memory. A timestamp
        is associated with each exchange so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            prompt str: The user prompt to the LLM.
            response str: The corresponding LLM response.
        """
        timestamp = datetime.now().timestamp()
        payload = {
            "id_field": ":".join([self._user_id, self._session_id, str(timestamp)]),
            "prompt": prompt,
            "response": response,
            "timestamp": timestamp,
            "token_count": 1,  # TODO get actual token count
        }
        self._client.rpush(self.key, json.dumps(payload))

    def set_preamble(self, prompt: str) -> None:
        """Add a preamble statement to the the begining of each session to be
        included in each subsequent LLM call.
        """
        self._preamble = {"role": "_preamble", "_content": prompt}
        # TODO store this in Redis with asigned scope?

    def hash_input(self, prompt: str):
        """Hashes the input using SHA256."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
