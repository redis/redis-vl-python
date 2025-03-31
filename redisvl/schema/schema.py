import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from pydantic import BaseModel, model_validator
from redis.commands.search.field import Field as RedisField

from redisvl.schema.fields import BaseField, FieldFactory
from redisvl.schema.type_utils import TypeInferrer
from redisvl.utils.log import get_logger
from redisvl.utils.utils import model_to_dict

logger = get_logger(__name__)


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
    """Index info includes the essential details regarding index settings,
    such as its name, prefix, key separator, and storage type in Redis.

    In yaml format, the index info section looks like:

    .. code-block:: yaml

        index:
            name: user-index
            prefix: user
            key_separtor: ':'
            storage_type: json

    In dict format, the index info section looks like:

    .. code-block:: python

        {"index": {
            "name": "user-index",
            "prefix": "user",
            "key_separator": ":",
            "storage_type": "json"
        }}

    """

    name: str
    """The unique name of the index."""
    prefix: str = "rvl"
    """The prefix used for Redis keys associated with this index."""
    key_separator: str = ":"
    """The separator character used in designing Redis keys."""
    storage_type: StorageType = StorageType.HASH
    """The storage type used in Redis (e.g., 'hash' or 'json')."""


class IndexSchema(BaseModel):
    """A schema definition for a search index in Redis, used in RedisVL for
    configuring index settings and organizing vector and metadata fields.

    The class offers methods to create an index schema from a YAML file or a
    Python dictionary, supporting flexible schema definitions and easy
    integration into various workflows.

    An example `schema.yaml` file might look like this:

    .. code-block:: yaml

        version: '0.1.0'

        index:
            name: user-index
            prefix: user
            key_separator: ":"
            storage_type: json

        fields:
            - name: user
              type: tag
            - name: credit_score
              type: tag
            - name: embedding
              type: vector
              attrs:
                algorithm: flat
                dims: 3
                distance_metric: cosine
                datatype: float32

    Loading the schema for RedisVL from yaml is as simple as:

    .. code-block:: python

        from redisvl.schema import IndexSchema

        schema = IndexSchema.from_yaml("schema.yaml")

    Loading the schema for RedisVL from dict is as simple as:

    .. code-block:: python

        from redisvl.schema import IndexSchema

        schema = IndexSchema.from_dict({
            "index": {
                "name": "user-index",
                "prefix": "user",
                "key_separator": ":",
                "storage_type": "json",
            },
            "fields": [
                {"name": "user", "type": "tag"},
                {"name": "credit_score", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32"
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
    version: Literal["0.1.0"] = "0.1.0"
    """Version of the underlying index schema."""

    @staticmethod
    def _make_field(storage_type, **field_inputs) -> BaseField:
        """
        Parse raw field inputs derived from YAML or dict.

        Validates and sets the 'path' attribute for fields when using JSON storage type.
        """
        # Create field from inputs
        field = FieldFactory.create_field(**field_inputs)
        # Handle field path and storage type
        if storage_type == StorageType.JSON:
            field.path = field.path if field.path else f"$.{field.name}"
        else:
            if field.path is not None:
                logger.warning(
                    f"Path attribute for field '{field.name}' will be ignored for HASH storage type."
                )
            field.path = None
        return field

    @model_validator(mode="before")
    @classmethod
    def validate_and_create_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate uniqueness of field names and create valid field instances.
        """
        # Ensure index is a dictionary for validation
        index = values.get("index", {})
        if not isinstance(index, IndexInfo):
            index = IndexInfo(**index)

        input_fields = values.get("fields", [])
        prepared_fields: Dict[str, BaseField] = {}
        # Handle old fields format temporarily
        if isinstance(input_fields, dict):
            raise ValueError("New schema format introduced; please update schema spec.")
        # Process and create fields
        for field_input in input_fields:
            field = cls._make_field(index.storage_type, **field_input)
            if field.name in prepared_fields:
                raise ValueError(
                    f"Duplicate field name: {field.name}. Field names must be unique across all fields."
                )
            prepared_fields[field.name] = field

        values["fields"] = prepared_fields
        values["index"] = index
        return values

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
            return cls.model_validate(yaml_data)

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
        schema_dict = data.copy()
        return cls.model_validate(schema_dict)

    @property
    def field_names(self) -> List[str]:
        """A list of field names associated with the index schema.

        Returns:
            List[str]: A list of field names from the schema.
        """
        return list(self.fields.keys())

    @property
    def redis_fields(self) -> List[RedisField]:
        """A list of core redis-py field definitions based on the
        current schema fields.

        Converts RedisVL field definitions into a format suitable for use with
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
            self.add_field(field)

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
    ) -> List[Dict[str, Any]]:
        """Generates a list of extracted field specs from a sample data point.

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
        fields: List[Dict[str, Any]] = []
        for field_name, value in data.items():
            if field_name in ignore_fields:
                continue
            try:
                field_type = TypeInferrer.infer(value)
                fields.append(
                    FieldFactory.create_field(
                        field_type,
                        field_name,
                    ).model_dump()
                )
            except ValueError as e:
                if strict:
                    raise
                else:
                    logger.warn(
                        message=f"Error inferring field type for {field_name}: {e}"
                    )
        return fields

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the index schema model to a dictionary, handling Enums
        and other special cases properly.

        Returns:
            Dict[str, Any]: The index schema as a dictionary.
        """
        dict_schema = model_to_dict(self)
        # cast fields back to a pure list
        dict_schema["fields"] = [
            field for field_name, field in dict_schema["fields"].items()
        ]
        return dict_schema

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
