import re
import yaml
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple, Optional, Type

from pydantic import BaseModel, ValidationError

from redisvl.schema.fields import (
    BaseField,
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

class FieldFactory:
    FIELD_TYPE_MAP = {
        "tag": TagField,
        "text": TextField,
        "numeric": NumericField,
        "geo": GeoField,
        "vector": get_vector_type
    }

    @staticmethod
    def create_field(field_type: str, name: str, **kwargs) -> BaseField:
        field_class = FieldFactory.FIELD_TYPE_MAP.get(field_type)
        if not field_class:
            raise ValueError(f"Unknown field type: {field_type}")
        return field_class(name=name, **kwargs)




class IndexSchema(BaseModel):
    """
    RedisVL index schema for storing and indexing vectors and metadata
    fields in Redis.

    Attributes:
        name (str): The name of the index.
        prefix (str): The key prefix used in the Redis database keys.
        key_separator (str): The key separator used in the Redis database keys.
        storage_type (StorageType): The Redis storage type for underlying data.
        fields (Dict[str, List[BaseField]]): The defined index fields.
    """
    name: str
    prefix: str = "rvl"
    key_separator: str = ":"
    storage_type: StorageType = StorageType.HASH
    fields: Dict[str, List[BaseField]] = {}

    @property
    def redis_fields(self) -> list:
        """Returns a list of index fields in the Redis database."""
        redis_fields = []
        for field_list in self.fields.values():
            redis_fields.extend(field.as_field() for field in field_list)
        return redis_fields

    def add_fields(self, fields: Dict[str, List[Dict[str, Any]]]):
        """Add fields to the index schema.

        Args:
            fields (Dict[str, List[Dict[str, Any]]]): The fields to
                add to the index schema.

        Raises:
            ValueError: If a field with the same name already exists.
        """
        for field_type, field_list in fields.items():
            for field_data in field_list:
                self.add_field(field_type, **field_data)

    def add_field(self, field_type: str, **kwargs):
        """Add a field to the index schema.

        Args:
            field_type (str): The field type.
            name (str): The field name.
            **kwargs: Additional keyword arguments for the field.

        Raises:
            ValueError: If a field with the same name already exists.
        """
        name = kwargs.get('name', None)
        if name is None:
            raise ValueError("Field name is required.")
        new_field = FieldFactory.create_field(field_type, **kwargs)
        if any(field.name == name for field in self.fields.get(field_type, [])):
            raise ValueError(
                f"Field with name '{name}' already exists in {field_type} fields."
            )

        self.fields.setdefault(field_type, []).append(new_field)

    def remove_field(self, field_type: str, field_name: str):
        """Remove a field from the index schema.

        Args:
            field_type (str): The field type (e.g. 'text', 'tag', 'numeric')
            field_name (str): The field name

        Raises:
            ValueError: If the field type or field name does not exist.
        """
        fields = self.fields.get(field_type)

        if fields is None:
            raise ValueError(f"Field type '{field_type}' does not exist.")

        filtered_fields = [field for field in fields if field.name != field_name]

        if len(filtered_fields) == len(fields):
            # field not found, raise Error
            raise ValueError(
                f"Field '{field_name}' does not exist in {field_type} fields."
            )
        self.fields[field_type] = filtered_fields

    def generate_fields(
        self,
        data: Dict[str, Any],
        strict: bool = False,
        ignore_fields: List[str] = [],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate fields from a sample of data.

        This method is commonly used to generated schema fields
        from metadata. For some datasets, there are a number of fields
        which makes it tedious to manually define each field. This method
        can be used to automatically generate fields from a sample of data.

        Note: Vector fields are not generated by this method
        Note: This method is a hueristic and may not always generate the
            correct field type.

        Args:
            data (Dict[str, Any]): The sample data to generate fields from.
            strict (bool): Whether to raise an error if a field type cannot be inferred.
            ignore_fields (List[str]): A list of field names to ignore.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary of fields.
        """

        fields = {}
        for field_name, value in data.items():
            if field_name in ignore_fields:
                continue
            try:
                field_type = TypeInferrer.infer(value)
                new_field = FieldFactory.create_field(
                    field_type,
                    field_name,
                )
                fields.setdefault(field_type, []).append(new_field.dict(exclude_unset=True))
            except ValueError as e:
                if strict:
                    raise
                else:
                    print(f"Error inferring field type for {field_name}: {e}")
        return fields

    # Class methods for serialization/deserialization
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexSchema":
        """Create an IndexSchema from a dictionary.

        Args:
            data (Dict[str, Any]): The index schema data.

        Returns:
            IndexSchema: The index schema.
        """
        schema = cls(**data['index'])
        for field_type, field_list in data['fields'].items():
            for field_data in field_list:
                schema.add_field(field_type, **field_data)
        return schema

    def to_dict(self) -> Dict[str, Any]:
        """Convert the index schema to a dictionary.

        Returns:
            Dict[str, Any]: The index schema as a dictionary.
        """
        index_data = {
            'name': self.name,
            'prefix': self.prefix,
            'key_separator': self.key_separator,
            'storage_type': self.storage_type.value
        }
        formatted_fields = {}
        for field_type, fields in self.fields.items():
            formatted_fields[field_type] = [
                field.dict(exclude_unset=True) for field in fields
            ]
        return {'index': index_data, 'fields': formatted_fields}

    @classmethod
    def from_yaml(cls, file_path: str) -> "IndexSchema":
        """Create an IndexSchema from a YAML file.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            IndexSchema: The index schema.
        """
        try:
            fp = Path(file_path).resolve()
        except OSError as e:
            raise ValueError(f"Invalid file path: {file_path}") from e

        if not fp.exists():
            raise FileNotFoundError(f"Schema file {file_path} does not exist")

        with open(fp, "r") as f:
            yaml_data = yaml.safe_load(f)
            return cls.from_dict(yaml_data)

    def to_yaml(self, file_path: str, overwrite: bool = True) -> None:
        """Write the index schema to a YAML file.

        Args:
            file_path (str): The path to the YAML file.
            overwrite (bool): Whether to overwrite the file if it already exists.

        Raises:
            FileExistsError: If the file already exists and overwrite is False.
        """
        fp = Path(file_path).resolve()
        if fp.exists() and not overwrite:
            raise FileExistsError(f"Schema file {file_path} already exists.")

        with open(fp, "w") as f:
            yaml_data = self.to_dict()
            yaml.dump(yaml_data, f, sort_keys=False)



class TypeInferrer:
    """
    Infers the type of a field based on its value.
    """

    GEO_PATTERN = re.compile(
        r"^\s*[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)\s*$"
    )

    TYPE_METHOD_MAP = {
        "numeric": "_is_numeric",
        "geo": "_is_geographic",
        "tag": "_is_tag",
        "text": "_is_text",
    }

    @classmethod
    def infer(cls, value: Any) -> str:
        """
        Infers the field type for a given value.

        Args:
            value: The value to infer the type of.

        Returns:
            The inferred field type as a string.

        Raises:
            ValueError: If the type cannot be inferred.
        """
        for type_name, method_name in cls.TYPE_METHOD_MAP.items():
            if getattr(cls, method_name)(value):
                return type_name
        raise ValueError(f"Unable to infer type for value: {value}")

    @classmethod
    def _is_numeric(cls, value: Any) -> bool:
        """Check if the value is numeric."""
        if not isinstance(value, (int, float, str)):
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @classmethod
    def _is_tag(cls, value: Any) -> bool:
        """Check if the value is a tag."""
        return isinstance(value, (list, set, tuple)) and all(isinstance(v, str) for v in value)

    @classmethod
    def _is_text(cls, value: Any) -> bool:
        """Check if the value is text."""
        return isinstance(value, str)

    @classmethod
    def _is_geographic(cls, value: Any) -> bool:
        """Check if the value is a geographic coordinate."""
        return isinstance(value, str) and cls.GEO_PATTERN.match(value) is not None
