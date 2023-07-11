import yaml
from typing import Any, Dict, List, Optional, Pattern, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
from typing import Union
from redis.commands.search.field import (
    GeoField,
    NumericField,
    TagField,
    TextField,
    VectorField,
)



class BaseField(BaseModel):
    name: str = Field(...)
    sortable: Optional[bool] = False


class TextFieldSchema(BaseField):
    weight: Optional[float] = 1
    no_stem: Optional[bool] = False
    phonetic_matcher: Optional[str]
    withsuffixtrie: Optional[bool] = False

    def as_field(self):
        return TextField(self.name, weight=self.weight, no_stem=self.no_stem,
                         phonetic_matcher=self.phonetic_matcher, sortable=self.sortable)


class TagFieldSchema(BaseField):
    separator: Optional[str] = ','
    case_sensitive: Optional[bool] = False

    def as_field(self):
        return TagField(self.name, separator=self.separator, case_sensitive=self.case_sensitive,
                        sortable=self.sortable)


class NumericFieldSchema(BaseField):

    def as_field(self):
        return NumericField(self.name, sortable=self.sortable)


class GeoFieldSchema(BaseField):

    def as_field(self):
        return GeoField(self.name, sortable=self.sortable)


class BaseVectorField(BaseModel):
    name: str = Field(...)
    dims: int = Field(...)
    algorithm: str = Field(...)
    datatype: str = Field(default="FLOAT32")
    distance_metric: str = Field(default="COSINE")
    initial_cap: int = Field(default=20000)

    @validator("algorithm", "datatype", "distance_metric", pre=True)
    def uppercase_strings(cls, v):
        return v.upper()


class FlatVectorField(BaseVectorField):
    algorithm: str = Field("FLAT", const=True)
    block_size: int = Field(default=1000)

    def as_field(self):
        return VectorField(
            self.name,
            self.algorithm,
            {
                "TYPE": self.datatype,
                "DIM": self.dims,
                "DISTANCE_METRIC": self.distance_metric,
                "INITIAL_CAP": self.initial_cap,
                "BLOCK_SIZE": self.block_size,
            },
        )

class HNSWVectorField(BaseVectorField):
    algorithm: str = Field("HNSW", const=True)
    m: int = Field(default=16)
    ef_construction: int = Field(default=200)
    ef_runtime: int = Field(default=10)
    epsilon: float = Field(default=0.8)

    def as_field(self):
        return VectorField(
            self.name,
            self.algorithm,
            {
                "TYPE": self.datatype,
                "DIM": self.dims,
                "DISTANCE_METRIC": self.distance_metric,
                "INITIAL_CAP": self.initial_cap,
                "M": self.m,
                "EF_CONSTRUCTION": self.ef_construction,
                "EF_RUNTIME": self.ef_runtime,
                "EPSILON": self.epsilon,
            },
        )


class IndexModel(BaseModel):
    name: str = Field(...)
    prefix: str = Field(...)
    key_field: str = Field(...)
    storage_type: str = Field(default="hash")


class FieldsModel(BaseModel):
    tag: Optional[List[TagFieldSchema]]
    text: Optional[List[TextFieldSchema]]
    numeric: Optional[List[NumericFieldSchema]]
    geo: Optional[List[GeoFieldSchema]]
    vector: Optional[List[Union[FlatVectorField, HNSWVectorField]]]


class SchemaModel(BaseModel):
    index: IndexModel = Field(...)
    fields: FieldsModel = Field(...)

    @validator("index")
    def validate_index(cls, v):
        if v.storage_type not in ["hash", "json"]:
            raise ValueError(f"Storage type {v.storage_type} not supported")
        return v

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
