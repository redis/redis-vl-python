import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from pydantic import BaseModel, validator
from redis.commands.search.field import Field as RedisField

from redisvl.schema.fields import BaseField, BaseVectorField, FieldFactory


class StorageType(Enum):
    HASH = "hash"
    JSON = "json"


class IndexSchema(BaseModel):
    """Represents a schema definition for an index in Redis, used in RedisVL for
    organizing and querying vector and metadata fields.

    This schema defines the structure of data stored in Redis, including
    information about the storage type, field definitions, and key formatting
    conventions used in the Redis database. Use the convenience class
    constructor methods `from_dict` and `from_yaml` to load and create an index
    schema from your definitions.

    Note: All field names MUST be unique in the index schema.

    Attributes:
        name (str): Unique name of the index.
        prefix (str): Prefix used for Redis keys. Defaults to "rvl".
        key_separator (str): Separator character used in Redis keys. Defaults
            to ":".
        storage_type (StorageType): Enum representing the underlying Redis data
            structure (e.g. hash or json). Defaults to hash.
        fields (Dict[str, List[Union[BaseField, BaseVectorField]]]): A dict
            mapping field types to lists of redisvl field definitions.

    .. code-block:: python

        from redisvl.schema import IndexSchema
        # From YAML
        schema = IndexSchema.from_yaml("schema.yaml")
        # From Dict
        schema = IndexSchema.from_dict({
            "index": {
                "name": "my-index",
                "prefix": "docs",
                "storage_type": "hash",
            },
            "fields": {
                "tag": [{"name": "doc-id"}],
                "vector": [
                    {"name": "doc-embedding", "algorithm": "flat", "dims": 1536}
                ]
            }
        })

    """

    name: str
    prefix: str = "rvl"
    key_separator: str = ":"
    storage_type: StorageType = StorageType.HASH
    fields: Dict[str, List[Union[BaseField, BaseVectorField]]] = {}

    @validator("fields", pre=True)
    def check_unique_field_names(cls, fields):
        """Validate that field names are all unique."""
        all_names = cls._get_field_names(fields)
        print(all_names, flush=True)
        if len(set(all_names)) != len(all_names):
            raise ValueError(
                f"Field names {all_names} must be unique across all fields."
            )
        return fields

    @staticmethod
    def _get_field_names(
        fields: Dict[str, List[Union[BaseField, BaseVectorField]]]
    ) -> List[str]:
        """Returns a list of field names from a fields object.

        Returns:
            List[str]: A list of field names from the fields object.
        """
        all_names: List[str] = []
        for field_list in fields.values():
            for field in field_list:
                all_names.append(field.name)
        return all_names

    @property
    def field_names(self) -> List[str]:
        """Returns a list of field names associated with the index schema.

        Returns:
            List[str]: A list of field names from the schema.
        """
        return self._get_field_names(self.fields)

    @property
    def redis_fields(self) -> List[RedisField]:
        """Returns a list of core redis-py field definitions based on the
        current schema fields.

        Converts field definitions into a format suitable for use with
        redis-py, facilitating the creation and management of index structures in
        the Redis database.

        Returns:
            List[RedisField]: A list of redis-py field definitions.
        """
        redis_fields: List[RedisField] = []
        for field_list in self.fields.values():
            redis_fields.extend(field.as_field() for field in field_list)  # type: ignore
        return redis_fields

    def add_fields(self, fields: Dict[str, List[Dict[str, Any]]]):
        """Extends the schema with additional fields.

        This method allows dynamically adding new fields to the index schema. It
        processes a dictionary where each key represents a field type, and the
        corresponding value is a list of field definitions to add.

        Args:
            fields (Dict[str, List[Dict[str, Any]]]): A dictionary mapping field
                types to lists of field attributes.

        .. code-block:: python

            schema.add_fields({})
            # From Dict
            schema = IndexSchema.from_dict({


        Raises:
            ValueError: If a field with the same name already exists in the
                schema.
        """
        for field_type, field_list in fields.items():
            for field_data in field_list:
                self.add_field(field_type, **field_data)

    def add_field(self, field_type: str, **kwargs):
        """Adds a single field to the index schema based on the specified field
        type and attributes.

        This method allows for the addition of individual fields to the schema,
        providing flexibility in defining the structure of the index.

        Args:
            field_type (str): Type of the field to be added
                (e.g., 'text', 'numeric', 'tag', 'vector', 'geo').
            **kwargs: A dictionary of attributes for the field, including the
                required 'name'.

        Raises:
            ValueError: If the field name is either not provided or already
                exists within the schema.
        """
        name = kwargs.get("name", None)
        if name is None:
            raise ValueError("Field name is required.")

        new_field = FieldFactory.create_field(field_type, **kwargs)
        if name in self.field_names:
            raise ValueError(
                f"Duplicate field '{name}' already present in index schema."
            )

        self.fields.setdefault(field_type, []).append(new_field)

    def remove_field(self, field_type: str, field_name: str):
        """Removes a field from the schema based on the specified field type and
        name.

        This method is useful for dynamically altering the schema by removing
        existing fields.

        Args:
            field_type (str): The type of the field to be removed.
            field_name (str): The name of the field to be removed.

        Raises:
            ValueError: If the field type or the specified field name does not
                exist in the schema.
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
        """Generates a set of field definitions from a sample data dictionary.

        This method simplifies the process of creating a schema by inferring
        field types and attributes from sample data. It's particularly useful
        during the development process while dealing with datasets containing
        numerous fields, reducing the need for manual specification.

        Args:
            data (Dict[str, Any]): Sample data used to infer field definitions.
            strict (bool, optional): If True, raises an error on failing to
                infer a field type. Defaults to False.
            ignore_fields (List[str], optional): A list of field names to
                exclude from processing. Defaults to an empty list.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary with inferred field
                types and attributes.

        Notes:
            - Vector fields are not generated by this method.
            - This method employs heuristics and may not always correctly infer
                field types.
        """
        fields: Dict[str, List[Dict[str, Any]]] = {}
        for field_name, value in data.items():
            if field_name in ignore_fields:
                continue
            try:
                field_type = TypeInferrer.infer(value)
                new_field = FieldFactory.create_field(
                    field_type,
                    field_name,
                )
                fields.setdefault(field_type, []).append(
                    new_field.dict(exclude_unset=True)
                )
            except ValueError as e:
                if strict:
                    raise
                else:
                    print(f"Error inferring field type for {field_name}: {e}")
        return fields

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexSchema":
        """Create an IndexSchema from a dictionary.

        Args:
            data (Dict[str, Any]): The index schema data.

        Returns:
            IndexSchema: The index schema.


        .. code-block:: python

            from redisvl.schema import IndexSchema
            schema = IndexSchema.from_dict({
                "index": {
                    "name": "my-index",
                    "prefix": "docs",
                    "storage_type": "hash",
                },
                "fields": {
                    "tag": [{"name": "doc-id"}],
                    "vector": [
                        {"name": "doc-embedding", "algorithm": "flat", "dims": 1536}
                    ]
                }
            })

        """
        schema = cls(**data["index"])
        for field_type, field_list in data["fields"].items():
            for field_data in field_list:
                schema.add_field(field_type, **field_data)
        return schema

    def to_dict(self) -> Dict[str, Any]:
        """Convert the index schema to a dictionary.

        Returns:
            Dict[str, Any]: The index schema as a dictionary.
        """
        index_data = {
            "name": self.name,
            "prefix": self.prefix,
            "key_separator": self.key_separator,
            "storage_type": self.storage_type.value,
        }
        formatted_fields = {}
        for field_type, fields in self.fields.items():
            formatted_fields[field_type] = [
                field.dict(exclude_unset=True) for field in fields
            ]
        return {"index": index_data, "fields": formatted_fields}

    @classmethod
    def from_yaml(cls, file_path: str) -> "IndexSchema":
        """Create an IndexSchema from a YAML file.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            IndexSchema: The index schema.

        .. code-block:: python

            from redisvl.schema import IndexSchema
            schema = IndexSchema.from_yaml("schema.yaml")

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
    """Infers the type of a field based on its value."""

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
        """Infers the field type for a given value.

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
        return isinstance(value, (list, set, tuple)) and all(
            isinstance(v, str) for v in value
        )

    @classmethod
    def _is_text(cls, value: Any) -> bool:
        """Check if the value is text."""
        return isinstance(value, str)

    @classmethod
    def _is_geographic(cls, value: Any) -> bool:
        """Check if the value is a geographic coordinate."""
        return isinstance(value, str) and cls.GEO_PATTERN.match(value) is not None
