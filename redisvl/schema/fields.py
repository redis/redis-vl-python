from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, validator
from redis.commands.search.field import Field as RedisField
from redis.commands.search.field import GeoField as RedisGeoField
from redis.commands.search.field import NumericField as RedisNumericField
from redis.commands.search.field import TagField as RedisTagField
from redis.commands.search.field import TextField as RedisTextField
from redis.commands.search.field import VectorField as RedisVectorField
from typing_extensions import Literal


class BaseField(BaseModel):
    name: str = Field(...)
    sortable: Optional[bool] = False
    as_name: Optional[str] = None


class TextField(BaseField):
    weight: Optional[float] = 1
    no_stem: Optional[bool] = False
    phonetic_matcher: Optional[str] = None
    withsuffixtrie: Optional[bool] = False

    def as_field(self) -> RedisField:
        return RedisTextField(
            self.name,
            weight=self.weight,  # type: ignore
            no_stem=self.no_stem,  # type: ignore
            phonetic_matcher=self.phonetic_matcher, # type: ignore
            sortable=self.sortable,
            as_name=self.as_name,
        )


class TagField(BaseField):
    separator: Optional[str] = ","
    case_sensitive: Optional[bool] = False

    def as_field(self) -> RedisField:
        return RedisTagField(
            self.name,
            separator=self.separator,  # type: ignore
            case_sensitive=self.case_sensitive,  # type: ignore
            sortable=self.sortable,
            as_name=self.as_name,
        )


class NumericField(BaseField):
    def as_field(self) -> RedisField:
        return RedisNumericField(
            self.name, sortable=self.sortable, as_name=self.as_name
        )


class GeoField(BaseField):
    def as_field(self) -> RedisField:
        return RedisGeoField(self.name, sortable=self.sortable, as_name=self.as_name)


class BaseVectorField(BaseModel):
    name: str = Field(...)
    dims: int = Field(...)
    algorithm: object = Field(...)
    datatype: str = Field(default="FLOAT32")
    distance_metric: str = Field(default="COSINE")
    initial_cap: Optional[int] = None
    as_name: Optional[str] = None

    @validator("algorithm", "datatype", "distance_metric", pre=True)
    def uppercase_strings(cls, v):
        return v.upper()

    @property
    def field_data(self) -> Dict[str, Any]:
        field_data = {
            "TYPE": self.datatype,
            "DIM": self.dims,
            "DISTANCE_METRIC": self.distance_metric,
        }
        if self.initial_cap is not None:  # Only include it if it's set
            field_data["INITIAL_CAP"] = self.initial_cap
        return field_data


class FlatVectorField(BaseVectorField):
    algorithm: Literal["FLAT"] = "FLAT"
    block_size: Optional[int] = None

    def as_field(self) -> RedisField:
        # grab base field params and augment with flat-specific fields
        field_data = super().field_data
        if self.block_size is not None:
            field_data["BLOCK_SIZE"] = self.block_size
        return RedisVectorField(
            self.name, self.algorithm, field_data, as_name=self.as_name
        )


class HNSWVectorField(BaseVectorField):
    algorithm: Literal["HNSW"] = "HNSW"
    m: int = Field(default=16)
    ef_construction: int = Field(default=200)
    ef_runtime: int = Field(default=10)
    epsilon: float = Field(default=0.01)

    def as_field(self) -> RedisField:
        # grab base field params and augment with hnsw-specific fields
        field_data = super().field_data
        field_data.update(
            {
                "M": self.m,
                "EF_CONSTRUCTION": self.ef_construction,
                "EF_RUNTIME": self.ef_runtime,
                "EPSILON": self.epsilon,
            }
        )
        return RedisVectorField(
            self.name, self.algorithm, field_data, as_name=self.as_name
        )


class FieldFactory:
    """
    Factory class to create fields from client data and kwargs.
    """

    FIELD_TYPE_MAP = {
        "tag": TagField,
        "text": TextField,
        "numeric": NumericField,
        "geo": GeoField,
    }

    VECTOR_FIELD_TYPE_MAP = {
        "flat": FlatVectorField,
        "hnsw": HNSWVectorField,
    }

    @classmethod
    def _get_vector_type(cls, **field_data) -> BaseVectorField:
        """Get the vector field type from the field data."""
        if "algorithm" not in field_data:
            raise ValueError("Must provide algorithm param for the vector field.")

        if "dims" not in field_data:
            raise ValueError("Must provide dims param for the vector field.")

        algorithm = field_data["algorithm"].lower()
        if algorithm not in cls.VECTOR_FIELD_TYPE_MAP:
            raise ValueError(f"Unknown vector field algorithm: {algorithm}")

        # default to FLAT
        return cls.VECTOR_FIELD_TYPE_MAP.get(algorithm, FlatVectorField)(**field_data)

    @classmethod
    def create_field(
        cls, field_type: str, name: str, **kwargs
    ) -> Union[BaseField, BaseVectorField]:
        """Create a field of a given type with provided attributes."""

        if field_type == "vector":
            return cls._get_vector_type(name=name, **kwargs)

        if field_type not in cls.FIELD_TYPE_MAP:
            raise ValueError(f"Unknown field type: {field_type}")

        field_class = cls.FIELD_TYPE_MAP[field_type]
        return field_class(name=name, **kwargs)
