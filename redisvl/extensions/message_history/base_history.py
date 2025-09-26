from typing import Any, Dict, List, Optional, Union

from redisvl.extensions.constants import (
    CONTENT_FIELD_NAME,
    METADATA_FIELD_NAME,
    ROLE_FIELD_NAME,
    TOOL_FIELD_NAME,
)
from redisvl.extensions.message_history.schema import ChatMessage
from redisvl.utils.utils import create_ulid, deserialize


class BaseMessageHistory:

    def __init__(
        self,
        name: str,
        session_tag: Optional[str] = None,
    ):
        """Initialize message history with index

        Message History stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Message history is stored in individual user or LLM prompts and
        responses.

        Args:
            name (str): The name of the message history index.
            session_tag (str): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
        """
        self._name = name
        self._session_tag = session_tag or create_ulid()

    def clear(self) -> None:
        """Clears the chat message history."""
        raise NotImplementedError

    def delete(self) -> None:
        """Clear all conversation history and remove any search indices."""
        raise NotImplementedError

    def drop(self, id_field: Optional[str] = None) -> None:
        """Remove a specific exchange from the conversation history.

        Args:
            id_field (Optional[str]): The id_field of the entry to delete.
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
        as_text: bool = False,
        raw: bool = False,
        session_tag: Optional[str] = None,
        role: Optional[Union[str, List[str]]] = None,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Retrieve the recent conversation history in sequential order.

        Args:
            top_k (int): The number of previous exchanges to return. Default is 5.
                Note that one exchange contains both a prompt and response.
            as_text (bool): Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response
            session_tag (str): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
            role (Optional[Union[str, List[str]]]): Filter messages by role(s).
                Can be a single role string ("system", "user", "llm", "tool") or
                a list of roles. If None, all roles are returned.

        Returns:
            Union[str, List[str]]: A single string transcription of the messages
                                   or list of strings if as_text is false.

        Raises:
            ValueError: If top_k is not an integer greater than or equal to 0,
                or if role contains invalid values.
        """
        raise NotImplementedError

    def _validate_roles(
        self, role: Optional[Union[str, List[str]]]
    ) -> Optional[List[str]]:
        """Validate and normalize role parameter.

        Args:
            role: A single role string, list of roles, or None.

        Returns:
            List of valid role strings if role is provided, None otherwise.

        Raises:
            ValueError: If role contains invalid values.
        """
        if role is None:
            return None

        valid_roles = {"system", "user", "llm", "tool"}

        # Handle single role string
        if isinstance(role, str):
            if role not in valid_roles:
                raise ValueError(
                    f"Invalid role '{role}'. Valid roles are: {valid_roles}"
                )
            return [role]

        # Handle list of roles
        if isinstance(role, list):
            if not role:  # Empty list
                raise ValueError("roles cannot be empty")
            for r in role:
                if r not in valid_roles:
                    raise ValueError(
                        f"Invalid role '{r}'. Valid roles are: {valid_roles}"
                    )
            return role

        raise ValueError("role must be a string or list of strings")

    def _format_context(
        self, messages: List[Dict[str, Any]], as_text: bool
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Extracts the prompt and response fields from the Redis hashes and
           formats them as either flat dictionaries or strings.

        Args:
            messages (List[Dict[str, Any]]): The messages from the message history index.
            as_text (bool): Whether to return the conversation as a single string,
                or list of alternating prompts and responses.

        Returns:
            Union[str, List[str]]: A single string transcription of the messages
                or list of strings if as_text is false.
        """
        context = []

        for message in messages:

            chat_message = ChatMessage(**message)

            if as_text:
                context.append(chat_message.content)
            else:
                chat_message_dict = {
                    ROLE_FIELD_NAME: chat_message.role,
                    CONTENT_FIELD_NAME: chat_message.content,
                }
                if chat_message.tool_call_id is not None:
                    chat_message_dict[TOOL_FIELD_NAME] = chat_message.tool_call_id
                if chat_message.metadata is not None:
                    chat_message_dict[METADATA_FIELD_NAME] = deserialize(
                        chat_message.metadata
                    )

                context.append(chat_message_dict)  # type: ignore

        return context

    def store(
        self, prompt: str, response: str, session_tag: Optional[str] = None
    ) -> None:
        """Insert a prompt:response pair into the message history. A timestamp
        is associated with each exchange so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            prompt (str): The user prompt to the LLM.
            response (str): The corresponding LLM response.
            session_tag (Optional[str]): The tag to mark the message with. Defaults to None.
        """
        raise NotImplementedError

    def add_messages(
        self, messages: List[Dict[str, str]], session_tag: Optional[str] = None
    ) -> None:
        """Insert a list of prompts and responses into the message history.
        A timestamp is associated with each so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            messages (List[Dict[str, str]]): The list of user prompts and LLM responses.
            session_tag (Optional[str]): The tag to mark the messages with. Defaults to None.
        """
        raise NotImplementedError

    def add_message(
        self, message: Dict[str, str], session_tag: Optional[str] = None
    ) -> None:
        """Insert a single prompt or response into the message history.
        A timestamp is associated with it so that it can be later sorted
        in sequential ordering after retrieval.

        Args:
            message (Dict[str,str]): The user prompt or LLM response.
            session_tag (Optional[str]): The tag to mark the message with. Defaults to None.
        """
        raise NotImplementedError
