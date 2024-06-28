from typing import Any, Dict, List, Optional, Union

from redis import Redis


class BaseSessionManager:
    id_field_name: str = "id_field"
    role_field_name: str = "role"
    content_field_name: str = "content"
    tool_field_name: str = "tool_call_id"
    timestamp_field_name: str = "timestamp"

    def __init__(
        self,
        name: str,
        session_tag: str,
        user_tag: str,
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
        """
        self._name = name
        self._user_tag = user_tag
        self._session_tag = session_tag

    def set_scope(
        self,
        session_tag: Optional[str] = None,
        user_tag: Optional[str] = None,
    ) -> None:
        """Set the filter to apply to querries based on the desired scope.

        This new scope persists until another call to set_scope is made, or if
        scope specified in calls to get_recent.

        Args:
            session_tag (str): Id of the specific session to filter to. Default is
                None.
            user_tag (str): Id of the specific user to filter to. Default is None.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Clears the chat session history."""
        raise NotImplementedError

    def delete(self) -> None:
        """Clear all conversation history and remove any search indices."""
        raise NotImplementedError

    def drop(self, id_field: Optional[str] = None) -> None:
        """Remove a specific exchange from the conversation history.

        Args:
            id_field Optional[str]: The id_field of the entry to delete.
                If None then the last entry is deleted.
        """
        raise NotImplementedError

    @property
    def messages(self) -> Union[List[str], List[Dict[str, str]]]:
        """Returns the full chat history."""
        raise NotImplementedError

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
            top_k (int): The number of previous exchanges to return. Default is 5.
                Note that one exchange contains both a prompt and response.
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
            ValueError: If top_k is not an integer greater than or equal to 0.
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
            text_statements = []
            for hit in hits:
                text_statements.append(hit[self.content_field_name])
            return text_statements
        else:
            statements = []
            for hit in hits:
                statements.append(
                    {
                        self.role_field_name: hit[self.role_field_name],
                        self.content_field_name: hit[self.content_field_name],
                    }
                )
                if (
                    hasattr(hit, self.tool_field_name)
                    or isinstance(hit, dict)
                    and self.tool_field_name in hit
                ):
                    statements[-1].update(
                        {self.tool_field_name: hit[self.tool_field_name]}
                    )
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

    def add_messages(self, messages: List[Dict[str, str]]) -> None:
        """Insert a list of prompts and responses into the session memory.
        A timestamp is associated with each so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            messages (List[Dict[str, str]]): The list of user prompts and LLM responses.
        """
        raise NotImplementedError

    def add_message(self, message: Dict[str, str]) -> None:
        """Insert a single prompt or response into the session memory.
        A timestamp is associated with it so that it can be later sorted
        in sequential ordering after retrieval.

        Args:
            message (Dict[str,str]): The user prompt or LLM response.
        """
        raise NotImplementedError
