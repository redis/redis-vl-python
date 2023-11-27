import os
import yaml

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ValidationError

from redisvl.schema.fields import (
    TagFieldSchema,
    TextFieldSchema,
    NumericFieldSchema,
    FlatVectorFieldSchema,
    HNSWVectorFieldSchema,
    GeoFieldSchema
)


class StorageType(Enum):
    HASH = "hash"
    JSON = "json"


class IndexModel(BaseModel):
    """Represents the schema for an index, including its name, optional prefix,
    and the storage type used."""
    name: str
    prefix: str = "rvl"
    key_separator: str = ":"
    storage_type: StorageType = StorageType.HASH

class FieldsModel(BaseModel):
    tag: Optional[List[TagFieldSchema]] = None
    text: Optional[List[TextFieldSchema]] = None
    numeric: Optional[List[NumericFieldSchema]] = None
    geo: Optional[List[GeoFieldSchema]] = None
    vector: Optional[List[Union[FlatVectorFieldSchema, HNSWVectorFieldSchema]]] = None


class SchemaValidationError(Exception):
    """Custom exception for schema validation errors."""
    pass


class Schema:

    def __init__(
        self,
        index: Union[Dict[str, Any], IndexModel],
        fields: Union[Dict[str, List[Any]], FieldsModel]
    ):
        self._index = self._validate_index_model(index)
        self._fields = self._validate_fields_model(fields)

    def _validate_index_model(self, index: Union[Dict[str, Any], IndexModel]) -> IndexModel:
        """
        Validate the index model schema.
        """
        try:
            if isinstance(index, dict):
                return IndexModel(**index)
            elif isinstance(index, IndexModel):
                return index
            else:
                raise TypeError("Index must be an IndexModel instance or a dictionary.")
        except ValidationError as e:
            raise SchemaValidationError(f"Invalid index model: {e}.") from e
        except Exception as e:
            raise SchemaValidationError("Failed to create schema.") from e

    def _validate_fields_model(self, fields: Union[Dict[str, Any], FieldsModel]) -> FieldsModel:
        """
        Validate the fields model schema.
        """
        try:
            if isinstance(fields, dict):
                return FieldsModel(**fields)
            elif isinstance(fields, FieldsModel):
                return fields
            else:
                raise TypeError("Fields must be a FieldsModel instance or a dictionary.")
        except ValidationError as e:
            raise SchemaValidationError(f"Invalid fields model: {e}") from e
        except Exception as e:
            raise SchemaValidationError("Failed to create schema.") from e

    @classmethod
    def from_params(
        cls,
        name: str,
        prefix: str = "rvl",
        key_separator: str = ":",
        storage_type: str = "hash",
        fields: Dict[str, List[Any]] = {},
        **kwargs
    ):
        """
        Create a Schema instance from provided parameters.
        Args:
            name: The index name.
            prefix: The index prefix.
            key_separator: The key separator.
            storage_type: The storage type.
            fields: The field definitions.
        Returns:
            A Schema instance.
        """
        index = {
            "name": name,
            "prefix": prefix,
            "key_separator": key_separator,
            "storage_type": StorageType(storage_type)
        }
        return cls(index=index, fields=fields)

    # @classmethod
    # def from_sample(cls, sample: Dict[str, Any]) -> None:
    #     """Construct a Schema from a sample of data."""
    #     generator = SchemaGenerator()
    #     schema = generator.generate(metadata=sample, strict=True)
    #     return cls(index = schema.index, fields = schema.fields)
    # TODO can't implement this until we have extended the schema generator

    @property
    def index_name(self) -> str:
        return self._index.name

    @property
    def index_prefix(self) -> str:
        return self._index.prefix

    @property
    def key_separator(self) -> str:
        return self._index.key_separator

    @property
    def storage_type(self) -> str:
        return self._index.storage_type

    @property
    def index_fields(self):
        redis_fields = []
        for field_name in self._fields.__fields__.keys():
            field_group = getattr(self._fields, field_name)
            if field_group is not None:
                for field in field_group:
                    redis_fields.append(field.as_field())
        return redis_fields

    def dump(self) -> Dict[str, Any]:
        """
        Dump the RedisVL schema to a dictionary.

        Returns:
            The RedisVL schema as a dictionary.
        """
        return {
            "index": self._index.dict(),
            "fields": self._fields.dict()
        }

    def write(self, path: Union[str, os.PathLike]) -> None:
        """
        Write the schema to a yaml file.

        Args:
            path (Union[str, os.PathLike], optional): The yaml file path where
                the schema will be written.

        Raises:
            TypeError: If the provided file path is not a valid YAML file.
        """
        if not path.endswith(".yaml"):
            raise TypeError("Invalid file path. Must be a YAML file.")

        schema = self.dump()
        with open(path, "w") as f:
            yaml.dump(schema, f)

    def add_field(self, field) -> None:
        """
        Add a new field to the schema.
        """
        raise NotImplementedError

    def remove_field(self, field_name) -> None:
        """
        Remove a field from the schema.
        """
        raise NotImplementedError

    def update_field(self, field) -> None:
        """
        Update an existing field in the schema.
        """
        raise NotImplementedError


    def read_schema(file_path: str) -> Schema:
        """
        Create a Schema instance from a YAML file.
        Args:
            file_path: The path to the YAML file.
        Returns:
            A Schema instance.
        Raises:
            ValueError: If the file path is not a YAML file.
            FileNotFoundError: If the YAML file does not exist.
        """
        if not file_path.endswith(".yaml"):
            raise ValueError("Must provide a valid YAML file path")

        fp = Path(file_path).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"Schema file {file_path} does not exist")

        with open(fp, "r") as f:
            schema = yaml.safe_load(f)

        return Schema(**schema)


class SchemaGenerator:
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
            isinstance(v, str) for v in value
        ):
            return "tag"
        return "text" if isinstance(value, str) else "unknown"

    def generate(
        self,
        metadata: Dict[str, Any],
        strict: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a RedisVL schema from metadata.

        Args:
            metadata (Dict[str, Any]): Metadata object to validate and
                generate schema.
            strict (bool, optional): Whether to generate schema in strict
                mode. Defaults to False.

        Raises:
            ValueError: Unable to determine schema field type for a
                key-value pair.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Output metadata schema.
        """
        result: Dict[str, List[Dict[str, Any]]] = {"text": [], "numeric": [], "tag": []}
        field_classes = {
            "text": TextFieldSchema,
            "tag": TagFieldSchema,
            "numeric": NumericFieldSchema,
            # TODO expand support for other metadata types including vector and geo
        }

        for key, value in metadata.items():
            field_type = self._infer_type(value)

            if field_type is None or field_type == "unknown":
                if strict:
                    raise ValueError(
                        f"Unable to determine field type for key '{key}' with"
                        f" value '{value}'"
                    )
                print(
                    f"Warning: Unable to determine field type for key '{key}'"
                    f" with value '{value}'"
                )
                continue

            if isinstance(field_type, str):
                field_class = field_classes.get(field_type)
                if field_class:
                    result[field_type].append(
                        field_class(name=key).dict(exclude_none=True)
                    )

        return result
