"""Schema definitions for embeddings cache in RedisVL.

This module defines the Pydantic models used for embedding cache entries and
related data structures.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redisvl.extensions.constants import EMBEDDING_FIELD_NAME, METADATA_FIELD_NAME
from redisvl.utils.utils import current_timestamp, deserialize, serialize


class CacheEntry(BaseModel):
    """Embedding cache entry data model"""

    model_config = ConfigDict(protected_namespaces=())

    entry_id: str
    """Cache entry identifier"""
    text: str
    """The text input that was embedded"""
    model_name: str
    """The name of the embedding model used"""
    embedding: List[float]
    """The embedding vector representation"""
    inserted_at: float = Field(default_factory=current_timestamp)
    """Timestamp of when the entry was added to the cache"""
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    """Optional metadata stored on the cache entry"""

    @model_validator(mode="before")
    @classmethod
    def deserialize_cache_entry(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Deserialize metadata if necessary
        if METADATA_FIELD_NAME in values and isinstance(
            values[METADATA_FIELD_NAME], str
        ):
            values[METADATA_FIELD_NAME] = deserialize(values[METADATA_FIELD_NAME])
        # Deserialize embeddings if necessary
        if EMBEDDING_FIELD_NAME in values and isinstance(
            values[EMBEDDING_FIELD_NAME], str
        ):
            values[EMBEDDING_FIELD_NAME] = deserialize(values[EMBEDDING_FIELD_NAME])

        return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert the cache entry to a dictionary for storage"""
        data = self.model_dump(exclude_none=True)
        data[EMBEDDING_FIELD_NAME] = serialize(self.embedding)
        if self.metadata is not None:
            data[METADATA_FIELD_NAME] = serialize(self.metadata)
        return data
