from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from redisvl.extensions.constants import (
    CACHE_VECTOR_FIELD_NAME,
    ENTRY_ID_FIELD_NAME,
    INSERTED_AT_FIELD_NAME,
    METADATA_FIELD_NAME,
    PROMPT_FIELD_NAME,
    RESPONSE_FIELD_NAME,
    UPDATED_AT_FIELD_NAME,
)
from redisvl.redis.utils import array_to_buffer, hashify
from redisvl.schema import IndexSchema
from redisvl.utils.utils import current_timestamp, deserialize, serialize


class CacheEntry(BaseModel):
    """A single LLM cache entry in Redis."""

    entry_id: str
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
    def validate_and_set_defaults(cls, values):
        """Ensure the entry_id is set and validate data types."""
        # Set entry_id if not provided
        if not values.get(ENTRY_ID_FIELD_NAME):
            if PROMPT_FIELD_NAME not in values:
                raise ValueError("Prompt is required for cache entry")

            filters = values.get("filters")
            values[ENTRY_ID_FIELD_NAME] = hashify(values[PROMPT_FIELD_NAME], filters)

        # Set timestamps if not provided
        if INSERTED_AT_FIELD_NAME not in values:
            values[INSERTED_AT_FIELD_NAME] = current_timestamp()
        if UPDATED_AT_FIELD_NAME not in values:
            values[UPDATED_AT_FIELD_NAME] = current_timestamp()

        return values

    @field_validator("metadata", "filters", mode="before")
    @classmethod
    def deserialize_if_string(cls, v):
        """Deserialize metadata or filters if they are serialized strings."""
        if isinstance(v, str):
            try:
                return deserialize(v)
            except:
                pass
        return v

    def to_dict(self, dtype: str = "float32") -> Dict[str, Any]:
        """Convert to a dictionary for storage.

        Args:
            dtype (str): The data type for vector conversion.

        Returns:
            Dict[str, Any]: The dictionary representation ready for storage.
        """
        data = self.model_dump(exclude_none=True)

        # Convert vector to binary format
        vector_field_name = CACHE_VECTOR_FIELD_NAME
        if "prompt_vector" in data:
            data[vector_field_name] = array_to_buffer(data.pop("prompt_vector"), dtype)

        # Serialize dictionary fields
        if "metadata" in data and data["metadata"] is not None:
            data[METADATA_FIELD_NAME] = serialize(data["metadata"])
        if "filters" in data and data["filters"] is not None:
            data["filters"] = serialize(data["filters"])

        # Set entry_id if needed
        if ENTRY_ID_FIELD_NAME not in data:
            data[ENTRY_ID_FIELD_NAME] = self.entry_id

        return data


class CacheHit(BaseModel):
    """A cache hit result from searching the semantic cache."""

    entry_id: str = Field(alias="entry_id")
    """Cache entry identifier"""
    prompt: str
    """Input prompt or question cached in Redis"""
    response: str
    """Response or answer to the question, cached in Redis"""
    score: Optional[float] = None
    """Optional similarity score returned by vector search"""
    inserted_at: Optional[float] = None
    """Timestamp of when the entry was added to the cache"""
    updated_at: Optional[float] = None
    """Timestamp of when the entry was updated in the cache"""
    metadata: Optional[Dict[str, Any]] = None
    """Optional metadata stored on the cache entry"""

    @model_validator(mode="before")
    @classmethod
    def deserialize_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize fields as needed."""
        # Deserialize metadata if it's a string
        if METADATA_FIELD_NAME in values and isinstance(
            values[METADATA_FIELD_NAME], str
        ):
            try:
                values[METADATA_FIELD_NAME] = deserialize(values[METADATA_FIELD_NAME])
            except:
                # If deserialization fails, keep as string
                pass

        # Convert string timestamps to floats
        for timestamp_field in [INSERTED_AT_FIELD_NAME, UPDATED_AT_FIELD_NAME]:
            if timestamp_field in values and isinstance(values[timestamp_field], str):
                try:
                    values[timestamp_field] = float(values[timestamp_field])
                except:
                    # If conversion fails, keep as string
                    pass

        return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert the cache hit to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the cache hit result.
        """
        return self.model_dump(exclude_none=True)


class SemanticCacheIndexSchema(IndexSchema):
    """Schema for the semantic cache index.

    This defines the Redis search index structure used by SemanticCache.
    """

    @classmethod
    def from_params(
        cls,
        name: str,
        prefix: str,
        dims: int,
        dtype: str,
    ) -> "SemanticCacheIndexSchema":
        """Create a semantic cache index schema from parameters.

        Args:
            name (str): The name of the index.
            prefix (str): The prefix for Redis keys.
            dims (int): The dimensions of the vector field.
            dtype (str): The data type for the vector field.

        Returns:
            SemanticCacheIndexSchema: The semantic cache index schema.

        Raises:
            ValueError: If the dimensions are invalid.
        """
        if dims <= 0:
            raise ValueError(f"Vector dimensions must be positive, got {dims}")

        schema_dict = {
            "index": {
                "name": name,
                "prefix": f"{prefix}:",
                "storage_type": "hash",
            },
            "fields": [
                {"name": ENTRY_ID_FIELD_NAME, "type": "tag"},
                {"name": PROMPT_FIELD_NAME, "type": "text"},
                {"name": RESPONSE_FIELD_NAME, "type": "text"},
                {"name": INSERTED_AT_FIELD_NAME, "type": "numeric", "sortable": True},
                {"name": UPDATED_AT_FIELD_NAME, "type": "numeric", "sortable": True},
                {"name": METADATA_FIELD_NAME, "type": "text"},
                {
                    "name": CACHE_VECTOR_FIELD_NAME,
                    "type": "vector",
                    "attrs": {
                        "dims": dims,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": dtype,
                    },
                },
            ],
        }

        return cls.from_dict(schema_dict)

    def add_field(self, field_dict: Dict[str, Any]) -> None:
        """Add a field to the schema.

        Args:
            field_dict (Dict[str, Any]): The field definition to add.

        Raises:
            ValueError: If a field with the same name already exists.
        """
        field_name = field_dict.get("name")
        if not field_name:
            raise ValueError("Field must have a 'name' attribute")

        # Check if field already exists
        if field_name in self.field_names:
            raise ValueError(f"Field '{field_name}' already exists in schema")

        # Add field to schema
        self.fields.append(field_dict)  # type: ignore
