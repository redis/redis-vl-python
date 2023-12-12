import re
import yaml
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel, ValidationError

from redisvl.schema.fields import (
    BaseField,
    BaseVectorField,
    TagField,
    TextField,
    NumericField,
    FlatVectorField,
    HNSWVectorField,
    GeoField
)


class StorageType(Enum):
    HASH = "hash"
    JSON = "json"


def get_vector_type(**field_data: Dict[str, Any]) -> Union[FlatVectorField, HNSWVectorField]:
    """Get the vector field type from the field data."""

    vector_field_classes = {
        'flat': FlatVectorField,
        'hnsw': HNSWVectorField
    }
    algorithm = field_data.get('algorithm', '').lower()
    if algorithm not in vector_field_classes.keys():
        raise ValueError(f"Unknown vector field algorithm: {algorithm}")

    # default to FLAT
    return vector_field_classes.get(algorithm, FlatVectorField)(**field_data)

class Schema(BaseModel):
    name: str
    prefix: str = "rvl"
    key_separator: str = ":"
    storage_type: StorageType = StorageType.HASH
    fields: Dict[str, List[BaseField]] = {}

    _FIELD_TYPE_MAP = {
        "tag": TagField,
        "text": TextField,
        "numeric": NumericField,
        "geo": GeoField,
        "vector": get_vector_type
    }

    @property
    def index_fields(self) -> list:
        # TODO @tyler: Should this not return BaseFields?
        redis_fields = []
        for field_name in self.fields:
            if field_type := getattr(self.fields, field_name):
                for field in field_type:
                    redis_fields.append(field.as_field())
        return redis_fields

    def add_fields(self, fields: Dict[str, List[Dict[str, Any]]]):
        for field_type, field_list in fields.items():
            for field_data in field_list:
                self.add_field(field_type, **field_data)

    def add_field(self, field_type: str, **kwargs):
        """Add a field to the schema.

        Args:
            field_type: The type of field to add.
            kwargs: The keyword arguments for the field.

        Raises:
            ValueError: If the field name already exists.
        """
        name = kwargs.get('name')
        if not name:
            raise ValueError("Field name must be provided.")
        try:
            new_field = self._FIELD_TYPE_MAP[field_type](**kwargs)
        except KeyError:
            raise ValueError(f"Unknown field type: {field_type}")
        except ValidationError as e:
            raise ValueError(f"Error adding field with type {field_type}") from e

        # Ensure a field of the same name isn't already added
        existing_fields = self.fields.get(field_type, [])
        if any(field.name == name for field in existing_fields):
            raise ValueError(
                f"Field with name '{name}' already exists in {field_type} fields."
            )

        self.fields.setdefault(field_type, []).append(new_field)

    def remove_field(self, field_type: str, field_name: str):
        if field_type in self.fields:
            self.fields[field_type] = [
                field for field in self.fields[field_type] if field.name != field_name]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        schema = cls(**data['index'])
        for field_type, field_list in data['fields'].items():
            for field_data in field_list:
                # make use of our add field method!
                schema.add_field(field_type, **field_data)
        return schema

    def to_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Dump the RedisVL schema to a dictionary.

        Returns:
            The RedisVL schema as a dictionary.
        """
        index_data = {
            'name': self.name,
            'prefix': self.prefix,
            'key_separator': self.key_separator,
            'storage_type': self.storage_type.value
        }
        formatted_fields = {}
        for field_type, fields in self.fields.items():
            formatted_fields[field_type] = [field.dict(exclude_unset=True) for field in fields]
        return {'index': index_data, 'fields': formatted_fields}


    @classmethod
    def from_yaml(cls, file_path: str) -> "Schema":
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
            yaml_data = yaml.safe_load(f)

        return cls.from_dict(yaml_data)

    def to_yaml(self, file_path: str) -> None:
        """
        Write the schema to a yaml file.

        Args:
            file_path (str): The yaml file path where the RedisVL schema is written.

        """
        # TODO
        # - Ovewrite
        # - error if file exists

        schema = self.to_dict()
        with open(file_path, "w+") as f:
            f.write(yaml.dump(schema, sort_keys=False))

    def generate_fields(
        self,
        data: Dict[str, Any],
        strict: bool = False,
        ignore_fields: List[str] = [],
        field_args: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, List[Dict[str, Any]]]:

        fields = {}
        for field_name, value in data.items():
            field_kwargs = {"name": field_name, **field_args.get(field_name, {})}

            # ignore this field
            if field_name in ignore_fields:
                continue

            # infer the field type and get field class
            try:
                field_type = TypeInferrer.infer(value)
                field_class = self._FIELD_TYPE_MAP[field_type]
            except ValueError as e:
                if strict:
                    raise
                else:
                    print(e.message)
                    continue

            # check for JSON usage
            #if self.storage_type == "JSON":
            #    field_kwargs["as_name"] = field_name
            #    field_kwargs["name"] = f"$.{field_name.replace(' ', '')}"

            # run pydantic validation
            field_instance = field_class(**field_kwargs)

            # add new field
            fields.setdefault(field_type, []).append(
                field_instance.dict(exclude_unset=True)
            )

        return fields


class TypeInferrer:

    GEO_PATTERN = r"^\s*[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)\s*$"

    TYPE_METHOD_MAP = {
        "numeric": "_is_numeric",
        "geo": "_is_geographic",
        "tag": "_is_tag",
        "text": "_is_text",
    }

    @classmethod
    def infer(cls, value) -> str:
        for type_name, method_name in cls.TYPE_METHOD_MAP.items():
            method = getattr(cls, method_name)
            if method(value):
                return type_name
        raise ValueError(f"Unable to infer type for value: {value}")

    @classmethod
    def _is_numeric(cls, value) -> bool:
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @classmethod
    def _is_tag(cls, value) -> bool:
        return isinstance(value, (list, set, tuple)) and all(isinstance(v, str) for v in value)

    @classmethod
    def _is_text(cls, value) -> bool:
        return isinstance(value, str)

    @classmethod
    def _is_geographic(cls, value) -> bool:
        if isinstance(value, str):
            return bool(re.match(cls.GEO_PATTERN, value))


