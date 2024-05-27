import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from redis import Redis

from redisvl.redis.utils import array_to_buffer


class BaseSessionManager:
    def __init__(
        self,
        name: str,
        session_id: str,
        user_id: str,
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
            preamble (str): System level prompt to be included in all context.
        """
        self._name = name
        self._user_id = user_id
        self._session_id = session_id
        self.set_preamble(preamble)

    def set_scope(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Set the filter to apply to querries based on the desired scope.

        This new scope persists until another call to set_scope is made, or if
        scope specified in calls to fetch_recent.

        Args:
            session_id (str): Id of the specific session to filter to. Default is
                None.
            user_id (str): Id of the specific user to filter to. Default is None.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Clears the chat session history."""
        raise NotImplementedError

    def delete(self) -> None:
        """Clear all conversation history and remove any search indices."""
        raise NotImplementedError

    def drop(self, id_field: Optional[str]=None) -> None:
        """Remove a specific exchange from the conversation history.

        Args:
            id_field Optional[str]: The id_field of the entry to delete.
                If None then the last entry is deleted.
        """
        raise NotImplementedError

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
            top_k (int): The number of previous exchanges to return. Default is 3.
                Note that one exchange contains both a prompt and response.
            session_id (str): Tag to be added to entries to link to a specific
                session.
            user_id (str): Tag to be added to entries to link to a specific user.
            as_text (bool): Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response
        Returns:
            Union[str, List[str]]: A single string transcription of the session
                                   or list of strings if as_text is false.
        """
        raise NotImplementedError

    def _format_context(
        self, hits: List[Dict[str, Any]], as_text: bool
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Extracts the prompt and response fields from the Redis hashes and
           formats them as either flat dictionaries oor strings.

        Args:
            hits (List): The hashes containing prompt & response pairs from
                recent conversation history.
            as_text (bool): Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
        Returns:
            Union[str, List[str]]: A single string transcription of the session
                or list of strings if as_text is false.
        """
        if as_text:
            text_statements = [self._preamble["_content"]]
            for hit in hits:
                text_statements.append(hit["prompt"])
                text_statements.append(hit["response"])
            return text_statements
        else:
            statements = [self._preamble]
            for hit in hits:
                statements.append({"role": "_user", "_content": hit["prompt"]})
                statements.append({"role": "_llm", "_content": hit["response"]})
            return statements

    def store(self, prompt: str, response: str) -> None:
        """Insert a prompt:response pair into the session memory. A timestamp
        is associated with each exchange so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            prompt (str): The user prompt to the LLM.
            response (str): The corresponding LLM response.
        """
        raise NotImplementedError

    def set_preamble(self, prompt: str) -> None:
        """Add a preamble statement to the the begining of each session to be
        included in each subsequent LLM call.
        """
        self._preamble = {"role": "_preamble", "_content": prompt}
        # TODO store this in Redis with asigned scope?

    def hash_input(self, prompt: str):
        """Hashes the input using SHA256."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
