from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field, validator
from redis.commands.search.field import (
    GeoField,
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from typing_extensions import Literal


class BaseField(BaseModel):
    name: str = Field(...)
    sortable: Optional[bool] = False
    as_name: Optional[str] = None


class TextFieldSchema(BaseField):
    weight: Optional[float] = 1
    no_stem: Optional[bool] = False
    phonetic_matcher: Optional[str] = None
    withsuffixtrie: Optional[bool] = False

    def as_field(self):
        return TextField(
            self.name,
            weight=self.weight,
            no_stem=self.no_stem,
            phonetic_matcher=self.phonetic_matcher,
            sortable=self.sortable,
            as_name=self.as_name,
        )


class TagFieldSchema(BaseField):
    separator: Optional[str] = ","
    case_sensitive: Optional[bool] = False

    def as_field(self):
        return TagField(
            self.name,
            separator=self.separator,
            case_sensitive=self.case_sensitive,
            sortable=self.sortable,
            as_name=self.as_name,
        )


class NumericFieldSchema(BaseField):
    def as_field(self):
        return NumericField(self.name, sortable=self.sortable, as_name=self.as_name)


class GeoFieldSchema(BaseField):
    def as_field(self):
        return GeoField(self.name, sortable=self.sortable, as_name=self.as_name)


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

    def as_field(self) -> Dict[str, Any]:
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

    def as_field(self):
        # grab base field params and augment with flat-specific fields
        field_data = super().as_field()
        if self.block_size is not None:
            field_data["BLOCK_SIZE"] = self.block_size
        return VectorField(self.name, self.algorithm, field_data, as_name=self.as_name)


class HNSWVectorField(BaseVectorField):
    algorithm: Literal["HNSW"] = "HNSW"
    m: int = Field(default=16)
    ef_construction: int = Field(default=200)
    ef_runtime: int = Field(default=10)
    epsilon: float = Field(default=0.01)

    def as_field(self):
        # grab base field params and augment with hnsw-specific fields
        field_data = super().as_field()
        field_data.update(
            {
                "M": self.m,
                "EF_CONSTRUCTION": self.ef_construction,
                "EF_RUNTIME": self.ef_runtime,
                "EPSILON": self.epsilon,
            }
        )
        return VectorField(self.name, self.algorithm, field_data, as_name=self.as_name)


class StorageType(Enum):
    HASH = "hash"
    JSON = "json"


class IndexModel(BaseModel):
    """Represents the schema for an index, including its name, optional prefix,
    and the storage type used."""

    name: str
    prefix: str
    key_separator: str = ":"
    storage_type: StorageType = StorageType.HASH

    class Config:
        use_enum_values = True

    @validator("name")
    def name_must_not_be_empty(cls, value):
        if not value:
            raise ValueError("name must not be empty")
        return value

    @validator("prefix", pre=True, always=True)
    def set_default_prefix(cls, v):
        return v if v is not None else ""

    @validator("key_separator", pre=True, always=True)
    def set_default_key_separator(cls, v):
        return v if v is not None else ":"

    @validator("storage_type", pre=True, always=True)
    def validate_storage_type(cls, v):
        if isinstance(v, StorageType):
            return v
        try:
            return StorageType(v)
        except ValueError:
            raise ValueError(f"Invalid storage type: {v}")


class FieldsModel(BaseModel):
    tag: Optional[List[TagFieldSchema]] = None
    text: Optional[List[TextFieldSchema]] = None
    numeric: Optional[List[NumericFieldSchema]] = None
    geo: Optional[List[GeoFieldSchema]] = None
    vector: Optional[List[Union[FlatVectorField, HNSWVectorField]]] = None


class SchemaModel(BaseModel):
    index: IndexModel = Field(...)
    fields: FieldsModel = Field(...)

    @property
    def index_fields(self):
        redis_fields = []
        for field_name in self.fields.__fields__.keys():
            field_group = getattr(self.fields, field_name)
            if field_group is not None:
                for field in field_group:
                    redis_fields.append(field.as_field())
        return redis_fields


def read_schema(file_path: str):
    fp = Path(file_path).resolve()
    if not fp.exists():
        raise FileNotFoundError(f"Schema file {file_path} does not exist")

    with open(fp, "r") as f:
        schema = yaml.safe_load(f)

    return SchemaModel(**schema)


class MetadataSchemaGenerator:
    """A class to generate a schema for metadata, categorizing fields into text,
    numeric, and tag types."""

    def _test_numeric(self, value) -> bool:
        """Test if a value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _infer_type(self, value) -> Optional[str]:
        """Infer the type of a value."""
        if value in [None, ""]:
            return None
        if self._test_numeric(value):
            return "numeric"
        if isinstance(value, (list, set, tuple)) and all(
           isinstance(v, str) for v in value):
            return "tag"
        return "text" if isinstance(value, str) else "unknown"

    def generate(self, metadata: Dict[str, Any],
                 strict: Optional[bool] = False) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a schema from metadata."""
        result = {"text": [], "numeric": [], "tag": []}
        field_classes = {
            "text": TextFieldSchema,
            "tag": TagFieldSchema,
            "numeric": NumericFieldSchema,
        }

        for key, value in metadata.items():
            field_type = self._infer_type(value)

            if field_type in [None, "unknown"]:
                if strict:
                    raise ValueError(
                        f"Unable to determine field type for key '{key}' with"
                        f" value '{value}'")
                print(f"Warning: Unable to determine field type for key '{key}'"
                      f" with value '{value}'")
                continue

            field_class = field_classes.get(field_type)
            if field_class:
                result[field_type].append(field_class(name=key).dict(exclude_none=True))

        return result
