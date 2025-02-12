from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redisvl.extensions.constants import (
    CONTENT_FIELD_NAME,
    ID_FIELD_NAME,
    ROLE_FIELD_NAME,
    SESSION_FIELD_NAME,
    SESSION_VECTOR_FIELD_NAME,
    TIMESTAMP_FIELD_NAME,
    TOOL_FIELD_NAME,
)
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema
from redisvl.utils.utils import current_timestamp


class ChatMessage(BaseModel):
    """A single chat message exchanged between a user and an LLM."""

    entry_id: Optional[str] = Field(default=None)
    """A unique identifier for the message."""
    role: str  # TODO -- do we enumify this?
    """The role of the message sender (e.g., 'user' or 'llm')."""
    content: str
    """The content of the message."""
    session_tag: str
    """Tag associated with the current session."""
    timestamp: Optional[float] = Field(default=None)
    """The time the message was sent, in UTC, rounded to milliseconds."""
    tool_call_id: Optional[str] = Field(default=None)
    """An optional identifier for a tool call associated with the message."""
    vector_field: Optional[List[float]] = Field(default=None)
    """The vector representation of the message content."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def generate_id(cls, values):
        if TIMESTAMP_FIELD_NAME not in values:
            values[TIMESTAMP_FIELD_NAME] = current_timestamp()
        if ID_FIELD_NAME not in values:
            values[ID_FIELD_NAME] = (
                f"{values[SESSION_FIELD_NAME]}:{values[TIMESTAMP_FIELD_NAME]}"
            )
        return values

    def to_dict(self, dtype: Optional[str] = None) -> Dict:
        data = self.model_dump(exclude_none=True)

        # handle optional fields
        if SESSION_VECTOR_FIELD_NAME in data:
            data[SESSION_VECTOR_FIELD_NAME] = array_to_buffer(
                data[SESSION_VECTOR_FIELD_NAME], dtype  # type: ignore[arg-type]
            )
        return data


class StandardSessionIndexSchema(IndexSchema):

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
            ],
        )


class SemanticSessionIndexSchema(IndexSchema):

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
                {
                    "name": SESSION_VECTOR_FIELD_NAME,
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
