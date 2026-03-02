import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from redisvl.extensions.constants import (
    CONTENT_FIELD_NAME,
    ID_FIELD_NAME,
    MESSAGE_VECTOR_FIELD_NAME,
    METADATA_FIELD_NAME,
    ROLE_FIELD_NAME,
    SESSION_FIELD_NAME,
    TIMESTAMP_FIELD_NAME,
    TOOL_FIELD_NAME,
)
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema
from redisvl.utils.utils import current_timestamp


class ChatRole(str, Enum):
    """Enumeration of valid roles for a chat message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A single chat message exchanged between a user and an LLM."""

    entry_id: Optional[str] = Field(default=None)
    """A unique identifier for the message."""
    role: Union[ChatRole, str]  # str allows deprecated values with warning
    """The role of the message sender (e.g. 'user' or 'assistant')."""
    content: str
    """The content of the message."""
    session_tag: str
    """Tag associated with the current conversation session."""
    timestamp: Optional[float] = Field(default=None)
    """The time the message was sent, in UTC, rounded to milliseconds."""
    tool_call_id: Optional[str] = Field(default=None)
    """An optional identifier for a tool call associated with the message."""
    vector_field: Optional[List[float]] = Field(default=None)
    """The vector representation of the message content."""
    metadata: Optional[str] = Field(default=None)
    """Optional additional data to store alongside the message"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def generate_id(cls, values):
        if TIMESTAMP_FIELD_NAME not in values:
            values[TIMESTAMP_FIELD_NAME] = current_timestamp()
        if ID_FIELD_NAME not in values:
            # Add UUID suffix to prevent timestamp collisions when creating
            # multiple messages rapidly (e.g., in add_messages or store)
            unique_suffix = uuid4().hex[:8]
            values[ID_FIELD_NAME] = (
                f"{values[SESSION_FIELD_NAME]}:{values[TIMESTAMP_FIELD_NAME]}:{unique_suffix}"
            )
        return values

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: Any) -> Union[ChatRole, str]:
        if isinstance(v, str):
            try:
                return ChatRole(v)
            except ValueError:
                warnings.warn(
                    f"Role '{v}' is a deprecated value. Update to valid roles: {[r.value for r in ChatRole]}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        return v

    def to_dict(self, dtype: Optional[str] = None) -> Dict:
        data = self.model_dump(exclude_none=True)

        # handle optional fields
        if MESSAGE_VECTOR_FIELD_NAME in data:
            data[MESSAGE_VECTOR_FIELD_NAME] = array_to_buffer(
                data[MESSAGE_VECTOR_FIELD_NAME], dtype  # type: ignore[arg-type]
            )

        return data


class MessageHistorySchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, prefix: str):

        return cls(
            index={"name": name, "prefix": prefix},  # type: ignore
            fields=[  # type: ignore
                {"name": ROLE_FIELD_NAME, "type": "tag"},
                {"name": CONTENT_FIELD_NAME, "type": "text"},
                {"name": TOOL_FIELD_NAME, "type": "tag"},
                {"name": TIMESTAMP_FIELD_NAME, "type": "numeric"},
                {"name": SESSION_FIELD_NAME, "type": "tag"},
                {"name": METADATA_FIELD_NAME, "type": "text"},
            ],
        )


class SemanticMessageHistorySchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, prefix: str, vectorizer_dims: int, dtype: str):

        return cls(
            index={"name": name, "prefix": prefix},  # type: ignore
            fields=[  # type: ignore
                {"name": ROLE_FIELD_NAME, "type": "tag"},
                {"name": CONTENT_FIELD_NAME, "type": "text"},
                {"name": TOOL_FIELD_NAME, "type": "tag"},
                {"name": TIMESTAMP_FIELD_NAME, "type": "numeric"},
                {"name": SESSION_FIELD_NAME, "type": "tag"},
                {"name": METADATA_FIELD_NAME, "type": "text"},
                {
                    "name": MESSAGE_VECTOR_FIELD_NAME,
                    "type": "vector",
                    "attrs": {
                        "dims": vectorizer_dims,
                        "datatype": dtype,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ],
        )
