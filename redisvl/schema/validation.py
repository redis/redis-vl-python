"""
RedisVL Schema Validation Module

This module provides utilities for validating data against RedisVL schemas
using dynamically generated Pydantic models.
"""

import json
from typing import Any, Dict, List, Optional, Type, Union

from jsonpath_ng import parse as jsonpath_parse
from pydantic import BaseModel, Field, field_validator

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
        cache_key = str(hash(json.dumps(schema.to_dict(), sort_keys=True).encode()))

        if cache_key not in cls._model_cache:
            cls._model_cache[cache_key] = cls._create_model(schema)

        return cls._model_cache[cache_key]

    @classmethod
    def _map_field_to_pydantic_type(
        cls, field: BaseField, storage_type: StorageType
    ) -> Type[Any]:
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
            return Union[int, float]  # type: ignore
        elif field.type == FieldTypes.GEO:
            return str
        elif field.type == FieldTypes.VECTOR:
            # For JSON storage, vectors are always lists
            if storage_type == StorageType.JSON:
                # For int data types, vectors must be ints, otherwise floats
                if field.attrs.datatype in (  # type: ignore
                    VectorDataType.INT8,
                    VectorDataType.UINT8,
                ):
                    return List[int]
                return List[float]
            else:
                return bytes

        # If we get here, the field type is not supported
        raise ValueError(f"Unsupported field type: {field.type}")

    @classmethod
    def _create_model(cls, schema: IndexSchema) -> Type[BaseModel]:
        """
        Create a Pydantic model from schema definition using type() approach.

        Args:
            schema: The IndexSchema to convert

        Returns:
            A Pydantic model class with appropriate fields and validators
        """
        # Get storage type from schema
        storage_type = schema.index.storage_type

        # Create annotations dictionary for the dynamic model
        annotations: Dict[str, Any] = {}
        class_dict: Dict[str, Any] = {}

        # Build annotations and field metadata
        for field_name, field in schema.fields.items():
            field_type = cls._map_field_to_pydantic_type(field, storage_type)

            # Make all fields optional in the model
            annotations[field_name] = Optional[field_type]

            # Add default=None to make fields truly optional (can be missing from input)
            class_dict[field_name] = Field(default=None)

            # Register validators for GEO fields
            if field.type == FieldTypes.GEO:

                def make_geo_validator(fname: str):
                    @field_validator(fname, mode="after")
                    def _validate_geo(cls, value):
                        # Skip validation for None values
                        if value is not None:
                            # Validate against pattern
                            if not TypeInferrer._is_geographic(value):
                                raise ValueError(
                                    f"Geo field '{fname}' value '{value}' is not a valid 'lat,lon' format"
                                )
                        return value

                    return _validate_geo

                class_dict[f"validate_{field_name}"] = make_geo_validator(field_name)

            # Register validators for NUMERIC fields
            elif field.type == FieldTypes.NUMERIC:

                def make_numeric_validator(fname: str):
                    # mode='before' so it catches bools before parsing
                    @field_validator(fname, mode="before")
                    def _disallow_bool(cls, value):
                        if isinstance(value, bool):
                            raise ValueError(f"Field '{fname}' cannot be boolean.")
                        return value

                    return _disallow_bool

                class_dict[f"validate_{field_name}"] = make_numeric_validator(
                    field_name
                )

            # Register validators for VECTOR fields
            elif field.type == FieldTypes.VECTOR:
                dims = field.attrs.dims  # type: ignore
                datatype = field.attrs.datatype  # type: ignore

                def make_vector_validator(
                    fname: str, dims: int, datatype: VectorDataType
                ):
                    @field_validator(fname, mode="after")
                    def _validate_vector(cls, value):
                        # Skip validation for None values
                        if value is not None:
                            # Handle list representation
                            if isinstance(value, list):
                                # Validate dimensions
                                if len(value) != dims:
                                    raise ValueError(
                                        f"Vector field '{fname}' must have {dims} dimensions, got {len(value)}"
                                    )
                                # Validate data types
                                datatype_str = str(datatype).upper()
                                # Integer-based datatypes
                                if datatype_str in ("INT8", "UINT8"):
                                    # Check range for INT8
                                    if datatype_str == "INT8":
                                        if any(v < -128 or v > 127 for v in value):
                                            raise ValueError(
                                                f"Vector field '{fname}' contains values outside the INT8 range (-128 to 127)"
                                            )
                                    # Check range for UINT8
                                    elif datatype_str == "UINT8":
                                        if any(v < 0 or v > 255 for v in value):
                                            raise ValueError(
                                                f"Vector field '{fname}' contains values outside the UINT8 range (0 to 255)"
                                            )
                        return value

                    return _validate_vector

                class_dict[f"validate_{field_name}"] = make_vector_validator(
                    field_name, dims, datatype
                )

        # Create class dictionary with annotations and field metadata
        class_dict.update(
            **{
                "__annotations__": annotations,
                "model_config": {"arbitrary_types_allowed": True, "extra": "allow"},
            }
        )

        # Create the model class using type()
        model_name = f"{schema.index.name}__PydanticModel"
        return type(model_name, (BaseModel,), class_dict)


def extract_from_json_path(obj: Dict[str, Any], path: str) -> Any:
    """
    Extract a value from a nested JSON object using a JSONPath expression.

    Args:
        obj: The object to extract values from
        path: JSONPath expression (e.g., $.field.subfield, $.[*].name)

    Returns:
        The extracted value or None if not found

    Notes:
        This function uses the jsonpath-ng library for proper JSONPath parsing
        and supports the full JSONPath specification including filters, wildcards,
        and array indexing.
    """
    # If path doesn't start with $, add it as per JSONPath spec
    if not path.startswith("$"):
        path = f"$.{path}"

    # Parse and find the JSONPath expression
    jsonpath_expr = jsonpath_parse(path)
    matches = jsonpath_expr.find(obj)

    # Return the first match value, or None if no matches
    if matches:
        return matches[0].value
    return None


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
