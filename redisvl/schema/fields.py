from typing import Any, Dict, Optional, Tuple, Type, Union

from pydantic.v1 import BaseModel, Field, validator
from redis.commands.search.field import Field as RedisField
from redis.commands.search.field import GeoField as RedisGeoField
from redis.commands.search.field import NumericField as RedisNumericField
from redis.commands.search.field import TagField as RedisTagField
from redis.commands.search.field import TextField as RedisTextField
from redis.commands.search.field import VectorField as RedisVectorField
from typing_extensions import Literal


class BaseFieldAttributes(BaseModel):
    sortable: Optional[bool] = False


class TextFieldAttributes(BaseFieldAttributes):
    weight: Optional[float] = 1
    no_stem: Optional[bool] = False
    phonetic_matcher: Optional[str] = None
    withsuffixtrie: Optional[bool] = False


class TagFieldAttributes(BaseFieldAttributes):
    separator: Optional[str] = ","
    case_sensitive: Optional[bool] = False


class NumericFieldAttributes(BaseFieldAttributes):
    pass


class GeoFieldAttributes(BaseFieldAttributes):
    pass


class BaseVectorFieldAttributes(BaseModel):
    dims: int = Field(...)
    algorithm: object = Field(...)
    datatype: str = Field(default="FLOAT32")
    distance_metric: str = Field(default="COSINE")
    initial_cap: Optional[int] = None

    @validator("algorithm", "datatype", "distance_metric", pre=True)
    @classmethod
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


class HNSWVectorFieldAttributes(BaseVectorFieldAttributes):
    algorithm: Literal["HNSW"] = "HNSW"
    m: int = Field(default=16)
    ef_construction: int = Field(default=200)
    ef_runtime: int = Field(default=10)
    epsilon: float = Field(default=0.01)


class FlatVectorFieldAttributes(BaseVectorFieldAttributes):
    algorithm: Literal["FLAT"] = "FLAT"
    block_size: Optional[int] = None


### Field Classes ###


class BaseField(BaseModel):
    """Base field"""

    name: str
    """Field name"""
    type: str
    """Field type"""
    path: Optional[str] = None
    """Field path (within JSON object)"""
    attrs: Optional[Union[BaseFieldAttributes, BaseVectorFieldAttributes]] = None
    """Specified field attributes"""

    def _handle_names(self) -> Tuple[str, Optional[str]]:
        if self.path:
            return self.path, self.name
        return self.name, None

    def as_redis_field(self) -> RedisField:
        raise NotImplementedError


class TextField(BaseField):
    """Text field supporting a full text search index"""

    type: str = Field(default="text", const=True)
    attrs: TextFieldAttributes = Field(default_factory=TextFieldAttributes)

    def as_redis_field(self) -> RedisField:
        name, as_name = self._handle_names()
        return RedisTextField(
            name,
            as_name=as_name,
            weight=self.attrs.weight,  # type: ignore
            no_stem=self.attrs.no_stem,  # type: ignore
            phonetic_matcher=self.attrs.phonetic_matcher,  # type: ignore
            sortable=self.attrs.sortable,
        )


class TagField(BaseField):
    """Tag field for simple boolean filtering"""

    type: str = Field(default="tag", const=True)
    attrs: TagFieldAttributes = Field(default_factory=TagFieldAttributes)

    def as_redis_field(self) -> RedisField:
        name, as_name = self._handle_names()
        return RedisTagField(
            name,
            as_name=as_name,
            separator=self.attrs.separator,  # type: ignore
            case_sensitive=self.attrs.case_sensitive,  # type: ignore
            sortable=self.attrs.sortable,
        )


class NumericField(BaseField):
    """Numeric field for numeric range filtering"""

    type: str = Field(default="numeric", const=True)
    attrs: NumericFieldAttributes = Field(default_factory=NumericFieldAttributes)

    def as_redis_field(self) -> RedisField:
        name, as_name = self._handle_names()
        return RedisNumericField(
            name,
            as_name=as_name,
            sortable=self.attrs.sortable,
        )


class GeoField(BaseField):
    """Geo field with a geo-spatial index for location based search"""

    type: str = Field(default="geo", const=True)
    attrs: GeoFieldAttributes = Field(default_factory=GeoFieldAttributes)

    def as_redis_field(self) -> RedisField:
        name, as_name = self._handle_names()
        return RedisGeoField(
            name,
            as_name=as_name,
            sortable=self.attrs.sortable,
        )


class FlatVectorField(BaseField):
    "Vector field with a FLAT index (brute force nearest neighbors search)"
    type: str = Field(default="vector", const=True)
    attrs: FlatVectorFieldAttributes

    def as_redis_field(self) -> RedisField:
        # grab base field params and augment with flat-specific fields
        name, as_name = self._handle_names()
        field_data = self.attrs.field_data
        if self.attrs.block_size is not None:
            field_data["BLOCK_SIZE"] = self.attrs.block_size
        return RedisVectorField(name, self.attrs.algorithm, field_data, as_name=as_name)


class HNSWVectorField(BaseField):
    """Vector field with an HNSW index (approximate nearest neighbors search)"""

    type: str = Field(default="vector", const=True)
    attrs: HNSWVectorFieldAttributes

    def as_redis_field(self) -> RedisField:
        # grab base field params and augment with hnsw-specific fields
        name, as_name = self._handle_names()
        field_data = self.attrs.field_data
        field_data.update(
            {
                "M": self.attrs.m,
                "EF_CONSTRUCTION": self.attrs.ef_construction,
                "EF_RUNTIME": self.attrs.ef_runtime,
                "EPSILON": self.attrs.epsilon,
            }
        )
        return RedisVectorField(name, self.attrs.algorithm, field_data, as_name=as_name)


class FieldFactory:
    """Factory class to create fields from client data and kwargs."""

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
    def pick_vector_field_type(cls, attrs: Dict[str, Any]) -> Type[BaseField]:
        """Get the vector field type from the field data."""
        if "algorithm" not in attrs:
            raise ValueError("Must provide algorithm param for the vector field.")

        if "dims" not in attrs:
            raise ValueError("Must provide dims param for the vector field.")

        algorithm = attrs["algorithm"].lower()
        if algorithm not in cls.VECTOR_FIELD_TYPE_MAP:
            raise ValueError(f"Unknown vector field algorithm: {algorithm}")

        return cls.VECTOR_FIELD_TYPE_MAP[algorithm]  # type: ignore

    @classmethod
    def create_field(
        cls,
        type: str,
        name: str,
        attrs: Dict[str, Any] = {},
        path: Optional[str] = None,
    ) -> BaseField:
        """Create a field of a given type with provided attributes."""

        if type == "vector":
            field_class = cls.pick_vector_field_type(attrs)
        else:
            if type not in cls.FIELD_TYPE_MAP:
                raise ValueError(f"Unknown field type: {type}")
            field_class = cls.FIELD_TYPE_MAP[type]  # type: ignore

        return field_class(name=name, path=path, attrs=attrs)  # type: ignore
