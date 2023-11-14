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
        )


class NumericFieldSchema(BaseField):
    def as_field(self):
        return NumericField(self.name, sortable=self.sortable)


class GeoFieldSchema(BaseField):
    def as_field(self):
        return GeoField(self.name, sortable=self.sortable)


class BaseVectorField(BaseModel):
    name: str = Field(...)
    dims: int = Field(...)
    algorithm: object = Field(...)
    datatype: str = Field(default="FLOAT32")
    distance_metric: str = Field(default="COSINE")
    initial_cap: Optional[int] = None

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
        return VectorField(self.name, self.algorithm, field_data)


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
        return VectorField(self.name, self.algorithm, field_data)


class IndexModel(BaseModel):
    name: str = Field(...)
    prefix: Optional[str] = Field(default="")
    storage_type: Optional[str] = Field(default="hash")


class FieldsModel(BaseModel):
    tag: Optional[List[TagFieldSchema]] = None
    text: Optional[List[TextFieldSchema]] = None
    numeric: Optional[List[NumericFieldSchema]] = None
    geo: Optional[List[GeoFieldSchema]] = None
    vector: Optional[List[Union[FlatVectorField, HNSWVectorField]]] = None


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


class MetadataSchemaGenerator:
    """
    A class to generate a schema for metadata, categorizing fields into text, numeric, and tag types.
    """

    def _test_numeric(self, value) -> bool:
        """
        Test if the given value can be represented as a numeric value.

        Args:
            value: The value to test.

        Returns:
            bool: True if the value can be converted to float, False otherwise.
        """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _infer_type(self, value) -> Optional[str]:
        """
        Infer the type of the given value.

        Args:
            value: The value to infer the type of.

        Returns:
            Optional[str]: The inferred type of the value, or None if the type is unrecognized or the value is empty.
        """
        if value is None or value == "":
            return None
        elif self._test_numeric(value):
            return "numeric"
        elif isinstance(value, (list, set, tuple)) and all(
            isinstance(v, str) for v in value
        ):
            return "tag"
        elif isinstance(value, str):
            return "text"
        else:
            return "unknown"

    def generate(
        self, metadata: Dict[str, Any], strict: Optional[bool] = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a schema from the provided metadata.

        This method categorizes each metadata field into text, numeric, or tag types based on the field values.
        It also allows forcing strict type determination by raising an exception if a type cannot be inferred.

        Args:
            metadata: The metadata dictionary to generate the schema from.
            strict: If True, the method will raise an exception for fields where the type cannot be determined.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary with keys 'text', 'numeric', and 'tag', each mapping to a list of field schemas.

        Raises:
            ValueError: If the force parameter is True and a field's type cannot be determined.
        """
        result: Dict[str, List[Dict[str, Any]]] = {"text": [], "numeric": [], "tag": []}

        for key, value in metadata.items():
            field_type = self._infer_type(value)

            if field_type in ["unknown", None]:
                if strict:
                    raise ValueError(
                        f"Unable to determine field type for key '{key}' with value '{value}'"
                    )
                print(
                    f"Warning: Unable to determine field type for key '{key}' with value '{value}'"
                )
                continue

            # Extract the field class with defaults
            field_class = {
                "text": TextFieldSchema,
                "tag": TagFieldSchema,
                "numeric": NumericFieldSchema,
            }.get(field_type)  # type: ignore

            if field_class:
                result[field_type].append(field_class(name=key).dict(exclude_none=True))  # type: ignore

        return result
