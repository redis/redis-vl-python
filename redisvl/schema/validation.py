"""
RedisVL Schema Validation Module

This module provides utilities for validating data against RedisVL schemas
using dynamically generated Pydantic models.
"""

import json
import re
import warnings
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError, create_model, field_validator

from redisvl.schema import IndexSchema
from redisvl.schema.fields import BaseField, FieldTypes, VectorDataType
from redisvl.schema.schema import StorageType
from redisvl.schema.type_utils import TypeInferrer
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


class SchemaModelGenerator:
    """
    Generates and caches Pydantic models based on Redis schema definitions.

    This class handles the conversion of RedisVL IndexSchema objects into
    Pydantic models with appropriate field types and validators.
    """

    _model_cache: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def get_model_for_schema(cls, schema: IndexSchema) -> Type[BaseModel]:
        """
        Get or create a Pydantic model for a schema.

        Args:
            schema: The IndexSchema to convert to a Pydantic model

        Returns:
            A Pydantic model class that can validate data against the schema
        """
        # Use schema identifier as cache key
        cache_key = schema.index.name

        if cache_key not in cls._model_cache:
            cls._model_cache[cache_key] = cls._create_model(schema)

        return cls._model_cache[cache_key]

    @classmethod
    def _map_field_to_pydantic_type(
        cls, field: BaseField, storage_type: StorageType
    ) -> Type:
        """
        Map Redis field types to appropriate Pydantic types.

        Args:
            field: The Redis field definition
            storage_type: The storage type (HASH or JSON)

        Returns:
            The Pydantic field type

        Raises:
            ValueError: If the field type is not supported
        """
        if field.type == FieldTypes.TEXT:
            return str
        elif field.type == FieldTypes.TAG:
            return str
        elif field.type == FieldTypes.NUMERIC:
            return Union[int, float]
        elif field.type == FieldTypes.GEO:
            return str
        elif field.type == FieldTypes.VECTOR:
            # For JSON storage, vectors are always lists
            if storage_type == StorageType.JSON:
                return List[Union[int, float]]
            else:
                return bytes

        # If we get here, the field type is not supported
        raise ValueError(f"Unsupported field type: {field.type}")

    @classmethod
    def _create_model(cls, schema: IndexSchema) -> Type[BaseModel]:
        """
        Create a Pydantic model from schema definition.

        Args:
            schema: The IndexSchema to convert

        Returns:
            A Pydantic model class with appropriate fields and validators
        """
        field_definitions = {}
        validators = {}

        # Get storage type from schema
        storage_type = schema.index.storage_type

        # Create field definitions dictionary for create_model
        for field_name, field in schema.fields.items():
            field_type = cls._map_field_to_pydantic_type(field, storage_type)

            # Create field definition (all fields are optional in the model)
            # this handles the cases where objects have missing fields (supported behavior)
            field_definitions[field_name] = (
                Optional[field_type],  # Make fields optional
                Field(
                    default=None,
                    json_schema_extra={
                        "field_type": field.type,
                    },
                ),
            )

            # Add field-specific validator info to our validator registry
            if field.type == FieldTypes.GEO:
                validators[field_name] = {"type": "geo"}

            elif field.type == FieldTypes.VECTOR:
                validators[field_name] = {
                    "type": "vector",
                    "dims": field.attrs.dims,
                    "datatype": field.attrs.datatype,
                    "storage_type": storage_type,
                }

        # First create the model class with field definitions
        model_name = f"{schema.index.name}__PydanticModel"
        model_class = create_model(model_name, **field_definitions)

        # Then add validators to the model class
        for field_name, validator_info in validators.items():
            if validator_info["type"] == "geo":
                # Add geo validator
                validator = cls._create_geo_validator(field_name)
                setattr(model_class, f"validate_{field_name}", validator)

            elif validator_info["type"] == "vector":
                # Add vector validator
                validator = cls._create_vector_validator(
                    field_name,
                    validator_info["dims"],
                    validator_info["datatype"],
                    validator_info["storage_type"],
                )
                setattr(model_class, f"validate_{field_name}", validator)

        return model_class

    @staticmethod
    def _create_geo_validator(field_name: str):
        """
        Create a validator for geo fields.

        Args:
            field_name: Name of the field to validate

        Returns:
            A validator function that can be attached to a Pydantic model
        """

        # Create the validator function
        def validate_geo_field(cls, value):
            # Skip validation for None values
            if value is not None:
                # Validate against pattern
                if not re.match(TypeInferrer.GEO_PATTERN.pattern, value):
                    raise ValueError(
                        f"Geo field '{field_name}' value '{value}' is not a valid 'lat,lon' format"
                    )
            return value

        # Add the field_validator decorator
        return field_validator(field_name, mode="after")(validate_geo_field)

    @staticmethod
    def _create_vector_validator(
        field_name: str, dims: int, datatype: VectorDataType, storage_type: StorageType
    ):
        """
        Create a validator for vector fields.

        Args:
            field_name: Name of the field to validate
            dims: Expected dimensions of the vector
            datatype: Expected datatype of the vector elements
            storage_type: Type of storage (HASH or JSON)

        Returns:
            A validator function that can be attached to a Pydantic model
        """

        # Create the validator function
        def validate_vector_field(cls, value):
            # Skip validation for None values
            if value is not None:

                # Handle list representation
                if isinstance(value, list):

                    # Validate dimensions
                    if len(value) != dims:
                        raise ValueError(
                            f"Vector field '{field_name}' must have {dims} dimensions, got {len(value)}"
                        )

                    # Validate data types
                    datatype_str = str(datatype).upper()

                    # Integer-based datatypes
                    if datatype_str in ("INT8", "UINT8"):
                        # Check type
                        if not all(isinstance(v, int) for v in value):
                            raise ValueError(
                                f"Vector field '{field_name}' must contain only integer values for {datatype_str}"
                            )

            return value

        return validate_vector_field


def extract_from_json_path(obj: Dict[str, Any], path: str) -> Any:
    """
    Extract a value from a nested JSON object using a path.

    Args:
        obj: The object to extract values from
        path: JSONPath-style path (e.g., $.field.subfield)

    Returns:
        The extracted value or None if not found
    """
    # Handle JSONPath syntax (e.g., $.field.subfield)
    if path.startswith("$."):
        path_parts = path[2:].split(".")
    else:
        path_parts = path.split(".")

    current = obj
    for part in path_parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None

    return current


def validate_object(schema: IndexSchema, obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an object against a schema.

    Args:
        schema: The IndexSchema to validate against
        obj: The object to validate

    Returns:
        Validated object with any type coercions applied

    Raises:
        ValueError: If validation fails with enhanced error message
    """
    # Get Pydantic model for this schema
    model_class = SchemaModelGenerator.get_model_for_schema(schema)

    # Prepare object for validation
    # Handle nested JSON if needed
    if schema.index.storage_type == StorageType.JSON:
        # Extract values from nested paths
        flat_obj = {}
        for field_name, field in schema.fields.items():
            if field.path:
                value = extract_from_json_path(obj, field.path)
                if value is not None:
                    flat_obj[field_name] = value
            elif field_name in obj:
                flat_obj[field_name] = obj[field_name]
    else:
        flat_obj = obj

    # Validate against model
    validated = model_class.model_validate(flat_obj)
    return validated.model_dump(exclude_none=True)
