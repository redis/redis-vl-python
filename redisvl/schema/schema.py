import re

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic.v1 import BaseModel, root_validator, validator
from redis.commands.search.field import Field as RedisField

from redisvl.schema.fields import BaseField, FieldFactory
from redisvl.utils.log import get_logger


logger = get_logger(__name__)
SCHEMA_VERSION = "0.1.0"


class StorageType(Enum):
    """
    Enumeration for the storage types supported in Redis.

    Attributes:
        HASH (str): Represents the 'hash' storage type in Redis.
        JSON (str): Represents the 'json' storage type in Redis.
    """
    HASH = "hash"
    JSON = "json"


class IndexInfo(BaseModel):
    """
    Represents the basic configuration information for an index in Redis.

    This class includes the essential details required to define an index, such as
    its name, prefix, key separator, and storage type.
    """
    name: str
    """The unique name of the index."""
    prefix: str = "rvl"
    """The prefix used for Redis keys associated with this index."""
    key_separator: str = ":"
    """The separator character used in Redis keys."""
    storage_type: StorageType = StorageType.HASH
    """The storage type used in Redis (e.g., 'hash' or 'json')."""


class IndexSchema(BaseModel):
    """Represents a schema definition for a search index in Redis, primarily
    used in RedisVL for organizing and querying vector and metadata fields.

    This schema provides a structured format to define the layout and types of
    fields stored in Redis, including details such as storage type, field
    definitions, and key formatting conventions.

    The class offers methods to create an index schema from a YAML file or a
    Python dictionary, supporting flexible schema definitions and easy
    integration into various workflows.

    .. code-block:: python

        from redisvl.schema import IndexSchema

        # From YAML
        schema = IndexSchema.from_yaml("schema.yaml")

        # From Dict
        schema = IndexSchema.from_dict({
            "index": {
                "name": "docs-index",
                "prefix": "docs",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "doc-id",
                    "type": "tag"
                },
                {
                    "name": "doc-embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 1536
                    }
                }
            ]
        })

    Note:
        The `fields` attribute in the schema must contain unique field names to ensure
        correct and unambiguous field references.

    """
    index: IndexInfo
    """Details of the basic index configurations."""
    fields: Dict[str, BaseField] = {}
    """Fields associated with the search index and their properties"""
    version: str = SCHEMA_VERSION
    """Version of the underlying index schema."""

    @staticmethod
    def _make_field(storage_type, **field_inputs) -> BaseField:
        """
        Parse raw field inputs derived from YAML or dict.

        Validates and sets the 'path' attribute for fields when using JSON storage type.
        """
        # Parse raw field inputs
        field_name = field_inputs.get("name")
        field_type = field_inputs.get("type")
        field_attrs = field_inputs.get("attrs", {})
        field_path = field_inputs.get("path")

        if not field_name or not field_type:
            raise ValueError("Fields must include a 'type' and 'name'.")

        # Handle field path and storage type
        if storage_type == StorageType.JSON:
            field_path = field_path if field_path else f"$.{field_name}"
        else:
            if field_path is not None:
                logger.warning(
                    f"Path attribute for field '{field_name}' will be ignored for HASH storage type."
                )
            field_path = None

        # Update attrs and create field instance
        field_attrs.update({
            "name": field_name,
            "path": field_path
        })
        return FieldFactory.create_field(field_type=field_type, **field_attrs)

    @staticmethod
    def _convert_old_format(storage_type: StorageType, raw_fields: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        updated_fields: List[Dict[str, Any]] = []
        for field_type, fields_list in raw_fields.items():
            for field in fields_list:
                if storage_type == StorageType.HASH:
                    field.pop("path", None)
                    updated_fields.append({
                        "name": field.pop("name", None),
                        "path": None,
                        "type": field_type,
                        "attrs": field
                    })
                else:
                   updated_fields.append({
                        "name": field.pop("as_name", None),
                        "path": field.pop("path", field.pop("name", None)),
                        "type": field_type,
                        "attrs": field
                    })
        return updated_fields

    @root_validator(pre=True)
    @classmethod
    def validate_and_create_fields(cls, values):
        """
        Validate uniqueness of field names and create valid field instances.
        """
        index = IndexInfo(**values.get('index'))
        raw_fields = values.get('fields', [])
        prepared_fields: Dict[str, BaseField] = {}
        # Process raw fields
        if isinstance(raw_fields, dict):
            # Need to handle backwards compat for the moment
            # TODO -- will remove this when 0.1.0 lands
            logger.warning("New schema format introduced; please update schema specs prior to 0.1.0")
            raw_fields = cls._convert_old_format(index.storage_type, raw_fields)
        for field_input in raw_fields:
            field = cls._make_field(index.storage_type, **field_input)
            if field.name in prepared_fields:
                raise ValueError(
                    f"Duplicate field name: {field.name}. Field names must be unique across all fields."
                )
            prepared_fields[field.name] = field

        values['fields'] = prepared_fields
        values['index'] = index
        return values

    @validator("version", pre=True)
    @classmethod
    def validate_version(cls, version: str):
        """Validate IndexSchema version."""
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"RedisVL IndexSchema version must be {SCHEMA_VERSION} but got {version}"
            )
        return version

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
            return cls(**yaml_data)

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
                    "name": "docs-index",
                    "prefix": "docs",
                    "storage_type": "hash",
                },
                "fields": [
                    {
                        "name": "doc-id",
                        "type": "tag"
                    },
                    {
                        "name": "doc-embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 1536
                        }
                    }
                ]
            })
        """
        return cls(**data)

    @property
    def field_names(self) -> List[str]:
        """Returns a list of field names associated with the index schema.

        Returns:
            List[str]: A list of field names from the schema.
        """
        return list(self.fields.keys())

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
        redis_fields: List[RedisField] = [
            field.as_redis_field() for _, field in self.fields.items()
        ]
        return redis_fields

    def add_field(self, field_inputs: Dict[str, Any]):
        """Adds a single field to the index schema based on the specified field
        type and attributes.

        This method allows for the addition of individual fields to the schema,
        providing flexibility in defining the structure of the index.

        Args:
            field_inputs (Dict[str, Any]): A field to add.

        Raises:
            ValueError: If the field name or type are not provided or if the name
                already exists within the schema.

        .. code-block:: python

            # Add a tag field
            schema.add_field({"name": "user", "type": "tag})

            # Add a vector field
            schema.add_field({
                "name": "user-embedding",
                "type": "vector",
                "attrs": {
                    "dims": 1024,
                    "algorithm": "flat",
                    "datatype": "float32"
                }
            })
        """
        # Parse field inputs
        field = self._make_field(self.index.storage_type, **field_inputs)
        # Check for duplicates
        if field.name in self.fields:
            raise ValueError(
                f"Duplicate field name: {field.name}. Field names must be unique across all fields for this index."
            )
        # Add field
        self.fields[field.name] = field

    def add_fields(self, fields: List[Dict[str, Any]]):
        """Extends the schema with additional fields.

        This method allows dynamically adding new fields to the index schema. It
        processes a list of field definitions.

        Args:
            fields (List[Dict[str, Any]]): A list of fields to add.

        Raises:
            ValueError: If a field with the same name already exists in the
                schema.

        .. code-block:: python

            schema.add_fields([
                {"name": "user", "type": "tag"},
                {"name": "bio", "type": "text"},
                {
                    "name": "user-embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 1024,
                        "algorithm": "flat",
                        "datatype": "float32"
                    }
                }
            ])
        """
        for field in fields:
            self.add_field(**field)

    def remove_field(self, field_name: str):
        """Removes a field from the schema based on the specified name.

        This method is useful for dynamically altering the schema by removing
        existing fields.

        Args:
            field_name (str): The name of the field to be removed.
        """
        if field_name not in self.fields:
            logger.warning(f"Field '{field_name}' does not exist in the schema")
            return
        del self.fields[field_name]

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
        fields: List[Dict[str, Any]]
        for field_name, value in data.items():
            if field_name in ignore_fields:
                continue
            try:
                field_type = TypeInferrer.infer(value)
                fields.append({
                    "name": field_name,
                    "type": field_type,
                    "attrs": FieldFactory.create_field(
                        field_type,
                        field_name,
                    ).dict(exclude_unset=True)
                })
            except ValueError as e:
                if strict:
                    raise
                else:
                    logger.warning(f"Error inferring field type for {field_name}: {e}")
        return fields

    def to_dict(self) -> Dict[str, Any]:
        """Convert the index schema to a dictionary.

        Returns:
            Dict[str, Any]: The index schema as a dictionary.
        """
        return self.dict(exclude_unset=True)

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
