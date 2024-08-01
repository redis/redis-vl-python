from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from redisvl.redis.utils import array_to_buffer, hashify
from redisvl.schema import IndexSchema
from redisvl.utils.utils import current_timestamp, deserialize, serialize


class CacheEntry(BaseModel):
    entry_id: Optional[str] = Field(default=None)
    prompt: str
    response: str
    prompt_vector: List[float]
    inserted_at: float = Field(default_factory=current_timestamp)
    updated_at: float = Field(default_factory=current_timestamp)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    filters: Optional[Dict[str, Any]] = Field(default=None)

    @root_validator(pre=True)
    @classmethod
    def generate_id(cls, values):
        # Ensure entry_id is set
        if not values.get("entry_id"):
            values["entry_id"] = hashify(values["prompt"])
        return values

    @validator("metadata")
    def non_empty_metadata(cls, v):
        if v is not None and not isinstance(v, dict):
            raise TypeError("Metadata must be a dictionary.")
        return v

    def to_dict(self) -> Dict:
        data = self.dict(exclude_none=True)
        data["prompt_vector"] = array_to_buffer(self.prompt_vector)
        if self.metadata:
            data["metadata"] = serialize(self.metadata)
        if self.filters:
            data.update(self.filters)
            del data["filters"]
        return data


class CacheHit(BaseModel):
    entry_id: str
    prompt: str
    response: str
    vector_distance: float
    inserted_at: float
    updated_at: float
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    filters: Optional[Dict[str, Any]] = Field(default=None)

    @root_validator(pre=True)
    @classmethod
    def validate_cache_hit(cls, values):
        # Deserialize metadata if necessary
        if "metadata" in values and isinstance(values["metadata"], str):
            values["metadata"] = deserialize(values["metadata"])

        # Separate filters from other fields
        known_fields = set(cls.__fields__.keys())
        filters = {k: v for k, v in values.items() if k not in known_fields}

        # Add filters to values
        if filters:
            values["filters"] = filters

        # Remove filter fields from the main values
        for k in filters:
            values.pop(k)

        return values

    def to_dict(self) -> Dict:
        data = self.dict(exclude_none=True)
        if self.filters:
            data.update(self.filters)
            del data["filters"]

        return data


class SemanticCacheIndexSchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, prefix: str, vector_dims: int):

        return cls(
            index={"name": name, "prefix": prefix},  # type: ignore
            fields=[  # type: ignore
                {"name": "prompt", "type": "text"},
                {"name": "response", "type": "text"},
                {"name": "inserted_at", "type": "numeric"},
                {"name": "updated_at", "type": "numeric"},
                {
                    "name": "prompt_vector",
                    "type": "vector",
                    "attrs": {
                        "dims": vector_dims,
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ],
        )
