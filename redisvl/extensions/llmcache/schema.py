from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from redisvl.extensions.constants import (
    CACHE_VECTOR_FIELD_NAME,
    INSERTED_AT_FIELD_NAME,
    PROMPT_FIELD_NAME,
    RESPONSE_FIELD_NAME,
    UPDATED_AT_FIELD_NAME,
)
from redisvl.redis.utils import array_to_buffer, hashify
from redisvl.schema import IndexSchema
from redisvl.utils.utils import current_timestamp, deserialize, serialize


class CacheEntry(BaseModel):
    """A single cache entry in Redis"""

    entry_id: Optional[str] = Field(default=None)
    """Cache entry identifier"""
    prompt: str
    """Input prompt or question cached in Redis"""
    response: str
    """Response or answer to the question, cached in Redis"""
    prompt_vector: List[float]
    """Text embedding representation of the prompt"""
    inserted_at: float = Field(default_factory=current_timestamp)
    """Timestamp of when the entry was added to the cache"""
    updated_at: float = Field(default_factory=current_timestamp)
    """Timestamp of when the entry was updated in the cache"""
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    """Optional metadata stored on the cache entry"""
    filters: Optional[Dict[str, Any]] = Field(default=None)
    """Optional filter data stored on the cache entry for customizing retrieval"""

    @model_validator(mode="before")
    @classmethod
    def generate_id(cls, values):
        # Ensure entry_id is set
        if not values.get("entry_id"):
            values["entry_id"] = hashify(values["prompt"], values.get("filters"))
        return values

    @field_validator("metadata")
    @classmethod
    def non_empty_metadata(cls, v):
        if v is not None and not isinstance(v, dict):
            raise TypeError("Metadata must be a dictionary.")
        return v

    def to_dict(self, dtype: str) -> Dict:
        data = self.model_dump(exclude_none=True)
        data["prompt_vector"] = array_to_buffer(self.prompt_vector, dtype)
        if self.metadata is not None:
            data["metadata"] = serialize(self.metadata)
        if self.filters is not None:
            data.update(self.filters)
            del data["filters"]
        return data


class CacheHit(BaseModel):
    """A cache hit based on some input query"""

    entry_id: str
    """Cache entry identifier"""
    prompt: str
    """Input prompt or question cached in Redis"""
    response: str
    """Response or answer to the question, cached in Redis"""
    vector_distance: float
    """The semantic distance between the query vector and the stored prompt vector"""
    inserted_at: float
    """Timestamp of when the entry was added to the cache"""
    updated_at: float
    """Timestamp of when the entry was updated in the cache"""
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    """Optional metadata stored on the cache entry"""
    filters: Optional[Dict[str, Any]] = Field(default=None)
    """Optional filter data stored on the cache entry for customizing retrieval"""

    # Allow extra fields to simplify handling filters
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def validate_cache_hit(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Deserialize metadata if necessary
        if "metadata" in values and isinstance(values["metadata"], str):
            values["metadata"] = deserialize(values["metadata"])

        # Collect any extra fields and store them as filters
        extra_data = values.pop("__pydantic_extra__", {}) or {}
        if extra_data:
            current_filters = values.get("filters") or {}
            if not isinstance(current_filters, dict):
                current_filters = {}
            current_filters.update(extra_data)
            values["filters"] = current_filters

        return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert this model to a dictionary, merging filters into the result."""
        data = self.model_dump(exclude_none=True)
        if data.get("filters"):
            data.update(data["filters"])
            del data["filters"]
        return data


class SemanticCacheIndexSchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, prefix: str, vector_dims: int, dtype: str):

        return cls(
            index={"name": name, "prefix": prefix},  # type: ignore
            fields=[  # type: ignore
                {"name": PROMPT_FIELD_NAME, "type": "text"},
                {"name": RESPONSE_FIELD_NAME, "type": "text"},
                {"name": INSERTED_AT_FIELD_NAME, "type": "numeric"},
                {"name": UPDATED_AT_FIELD_NAME, "type": "numeric"},
                {
                    "name": CACHE_VECTOR_FIELD_NAME,
                    "type": "vector",
                    "attrs": {
                        "dims": vector_dims,
                        "datatype": dtype,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ],
        )
