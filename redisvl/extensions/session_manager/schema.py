from typing import Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator

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

    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    @classmethod
    def generate_id(cls, values):
        if "timestamp" not in values:
            values["timestamp"] = current_timestamp()
        if "entry_id" not in values:
            values["entry_id"] = f'{values["session_tag"]}:{values["timestamp"]}'
        return values

    def to_dict(self) -> Dict:
        data = self.dict(exclude_none=True)

        # handle optional fields
        if "vector_field" in data:
            data["vector_field"] = array_to_buffer(data["vector_field"])

        return data


class StandardSessionIndexSchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, prefix: str):

        return cls(
            index={"name": name, "prefix": prefix},  # type: ignore
            fields=[  # type: ignore
                {"name": "role", "type": "tag"},
                {"name": "content", "type": "text"},
                {"name": "tool_call_id", "type": "tag"},
                {"name": "timestamp", "type": "numeric"},
                {"name": "session_tag", "type": "tag"},
            ],
        )


class SemanticSessionIndexSchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, prefix: str, vectorizer_dims: int):

        return cls(
            index={"name": name, "prefix": prefix},  # type: ignore
            fields=[  # type: ignore
                {"name": "role", "type": "tag"},
                {"name": "content", "type": "text"},
                {"name": "tool_call_id", "type": "tag"},
                {"name": "timestamp", "type": "numeric"},
                {"name": "session_tag", "type": "tag"},
                {
                    "name": "vector_field",
                    "type": "vector",
                    "attrs": {
                        "dims": vectorizer_dims,
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ],
        )
