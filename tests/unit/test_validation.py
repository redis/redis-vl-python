"""
Tests for the RedisVL schema validation module.

This module tests the core validation functionality:
1. Model generation from schemas
2. Field-specific validators
3. JSON path extraction
4. Validation of various field types
"""

import re
from typing import Any, List, Optional, Tuple, Union

import pytest

from redisvl.schema import IndexSchema
from redisvl.schema.fields import FieldTypes, VectorDataType
from redisvl.schema.schema import StorageType
from redisvl.schema.type_utils import TypeInferrer
from redisvl.schema.validation import (
    SchemaModelGenerator,
    extract_from_json_path,
    validate_object,
)

# -------------------- FIXTURES --------------------


@pytest.fixture
def sample_hash_schema():
    """Create a sample schema with HASH storage for testing."""
    schema_dict = {
        "index": {
            "name": "test-hash-index",
            "prefix": "test",
            "key_separator": ":",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "test_id", "type": "tag"},
            {"name": "title", "type": "text"},
            {"name": "rating", "type": "numeric"},
            {"name": "location", "type": "geo"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 4,
                    "datatype": "float32",
                    "distance_metric": "cosine",
                },
            },
            {
                "name": "int_vector",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 3,
                    "datatype": "int8",
                    "distance_metric": "l2",
                },
            },
        ],
    }
    return IndexSchema.from_dict(schema_dict)


@pytest.fixture
def sample_json_schema():
    """Create a sample schema with JSON storage for testing."""
    schema_dict = {
        "index": {
            "name": "test-json-index",
            "prefix": "test",
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {"name": "test_id", "type": "tag", "path": "$.test_id"},
            {"name": "user", "type": "tag", "path": "$.metadata.user"},
            {"name": "title", "type": "text", "path": "$.content.title"},
            {"name": "rating", "type": "numeric", "path": "$.metadata.rating"},
            {
                "name": "embedding",
                "type": "vector",
                "path": "$.content.embedding",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 4,
                    "datatype": "float32",
                    "distance_metric": "cosine",
                },
            },
            {
                "name": "int_vector",
                "type": "vector",
                "path": "$.content.int_vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 3,
                    "datatype": "int8",
                    "distance_metric": "l2",
                },
            },
        ],
    }
    return IndexSchema.from_dict(schema_dict)


@pytest.fixture
def valid_hash_data():
    """Sample valid data for testing HASH storage validation."""
    return {
        "test_id": "doc1",
        "title": "Test Document",
        "rating": 4.5,
        "location": "37.7749,-122.4194",
        "embedding": b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",  # Bytes for HASH
        "int_vector": b"\x01\x02\x03",  # Bytes for HASH
    }


@pytest.fixture
def valid_json_data():
    """Sample valid data for testing JSON storage validation."""
    return {
        "test_id": "doc1",
        "metadata": {"user": "user123", "rating": 4.5},
        "content": {
            "title": "Test Document",
            "embedding": [0.1, 0.2, 0.3, 0.4],  # List for JSON
            "int_vector": [1, 2, 3],  # List for JSON
        },
    }


# -------------------- TEST HELPERS --------------------


def validate_field(
    schema: IndexSchema,
    field_name: str,
    value: Any,
    should_pass: bool,
    error_text: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Helper function to validate a field value against a schema.

    Args:
        schema: The schema to validate against
        field_name: The name of the field to validate
        value: The value to validate
        should_pass: Whether validation should pass
        error_text: Expected error text if validation should fail

    Returns:
        Tuple of (validation_success, error_message)
    """
    # Get model for schema
    model_class = SchemaModelGenerator.get_model_for_schema(schema)

    # Create test data with minimal viable fields
    test_data = {field_name: value}

    # Try to validate
    try:
        validated = model_class.model_validate(test_data)

        # If we got here, validation passed
        success = True
        error_msg = None

    except Exception as e:
        # Validation failed
        success = False
        error_msg = str(e)

    # Check if result matches expectation
    if success != should_pass:
        print("ERROR", error_msg, flush=True)
        print(validated, flush=True)
    assert (
        success == should_pass
    ), f"Validation {'passed' if success else 'failed'} but expected {'pass' if should_pass else 'fail'}"

    # Check error text if specified and validation failed
    if not success and error_text and error_msg:
        assert (
            error_text in error_msg
        ), f"Error '{error_msg}' does not contain expected text '{error_text}'"

    return success, error_msg


# -------------------- CATEGORY 1: BASIC UNIT TESTS --------------------


class TestSchemaModelGenerator:
    """Tests for the SchemaModelGenerator class."""

    @pytest.mark.parametrize("schema_type", ["hash", "json"])
    def test_get_model_for_schema(
        self, schema_type, sample_hash_schema, sample_json_schema
    ):
        """Test generating a model from a schema."""
        # Select schema based on type
        schema = sample_hash_schema if schema_type == "hash" else sample_json_schema

        # Get model for schema
        model_class = SchemaModelGenerator.get_model_for_schema(schema)

        # Verify model name matches the index name
        assert model_class.__name__ == f"{schema.index.name}__PydanticModel"

        # Verify model has expected fields
        for field_name in schema.field_names:
            assert field_name in model_class.model_fields

    def test_model_caching(self, sample_hash_schema):
        """Test that models are cached and reused."""
        # Get model twice
        model1 = SchemaModelGenerator.get_model_for_schema(sample_hash_schema)
        model2 = SchemaModelGenerator.get_model_for_schema(sample_hash_schema)

        # Verify same instance
        assert model1 is model2

    @pytest.mark.parametrize(
        "field_type,storage_type,expected_type",
        [
            (FieldTypes.TEXT, StorageType.HASH, str),
            (FieldTypes.TAG, StorageType.HASH, str),
            (FieldTypes.NUMERIC, StorageType.HASH, Union[int, float]),
            (FieldTypes.GEO, StorageType.HASH, str),
            (FieldTypes.VECTOR, StorageType.HASH, bytes),
            (FieldTypes.TEXT, StorageType.JSON, str),
            (FieldTypes.TAG, StorageType.JSON, str),
            (FieldTypes.NUMERIC, StorageType.JSON, Union[int, float]),
            (FieldTypes.GEO, StorageType.JSON, str),
            (FieldTypes.VECTOR, StorageType.JSON, List[float]),
        ],
    )
    def test_type_mapping(self, field_type, storage_type, expected_type):
        """Test mapping Redis field types to Pydantic types."""

        # Create a basic field of the specified type
        class SimpleField:
            def __init__(self, ftype):
                self.type = ftype
                # Add attrs for vector fields
                if ftype == FieldTypes.VECTOR:

                    class Attrs:
                        dims = 4
                        datatype = VectorDataType.FLOAT32

                    self.attrs = Attrs()

        field = SimpleField(field_type)
        field_type_result = SchemaModelGenerator._map_field_to_pydantic_type(
            field, storage_type
        )

        assert field_type_result == expected_type

    def test_unsupported_field_type(self):
        """Test that an error is raised for unsupported field types."""

        # Create a dummy field with unsupported type
        class DummyField:
            type = "unsupported_type"

        # Mapping should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            SchemaModelGenerator._map_field_to_pydantic_type(
                DummyField(), StorageType.HASH
            )

        assert "Unsupported field type" in str(exc_info.value)


class TestJsonPathExtraction:
    """Tests for JSON path extraction functionality."""

    @pytest.mark.parametrize(
        "path,expected_value",
        [
            ("$.test_id", "doc1"),
            ("$.metadata.user", "user123"),
            ("$.metadata.rating", 4.5),
            ("$.content.title", "Test Document"),
            ("$.content.embedding", [0.1, 0.2, 0.3, 0.4]),
            ("metadata.user", "user123"),  # alternate format
            ("$.nonexistent", None),  # nonexistent path
            ("$.metadata.nonexistent", None),  # nonexistent nested path
        ],
    )
    def test_extract_from_json_path(self, valid_json_data, path, expected_value):
        """Test extracting values using JSON paths."""
        assert extract_from_json_path(valid_json_data, path) == expected_value


# # -------------------- CATEGORY 2: PARAMETRIZED VALIDATOR TESTS --------------------


class TestBasicFieldValidation:
    """Tests for validating non-vector field types."""

    @pytest.mark.parametrize(
        "field_type,field_name,valid_values,invalid_values",
        [
            # TEXT fields
            (
                "text",
                "title",
                [("Test Document", None), ("123", None), ("", None)],
                [(123, "string"), (True, "string"), ([], "string")],
            ),
            # TAG fields
            (
                "tag",
                "test_id",
                [("doc1", None), ("123", None), ("abc,def", None), ("", None)],
                [
                    (123, "string"),
                    (True, "string"),
                    ([], "string"),
                    ([1, 2, 3], "string"),
                ],
            ),
            # NUMERIC fields
            (
                "numeric",
                "rating",
                [(5, None), (4.5, None), (0, None), (-1.5, None), ("5.3", None)],
                [("high", "number"), (True, "boolean"), ([], "number")],
            ),
            # GEO fields
            (
                "geo",
                "location",
                [
                    ("0,0", None),
                    ("90,-180", None),
                    ("-90,180", None),
                    ("37.7749,-122.4194", None),
                ],
                [
                    ("invalid_geo", "lat,lon"),
                    ("37.7749", "lat,lon"),
                    ("37.7749,", "lat,lon"),
                    (",122.4194", "lat,lon"),
                    ("91,0", "lat,lon"),  # Latitude > 90
                    ("-91,0", "lat,lon"),  # Latitude < -90
                    ("0,181", "lat,lon"),  # Longitude > 180
                    ("0,-181", "lat,lon"),  # Longitude < -180
                    (123, "string"),
                    (True, "string"),
                ],
            ),
        ],
    )
    def test_basic_field_validation(
        self, sample_hash_schema, field_type, field_name, valid_values, invalid_values
    ):
        """
        Test validation of basic field types (text, tag, numeric, geo).

        This test consolidates previously separate tests for different field types.
        """
        # Test valid values
        for value, _ in valid_values:
            validate_field(sample_hash_schema, field_name, value, True)

            # For GEO fields, also verify pattern
            if field_type == "geo" and isinstance(value, str):
                assert re.match(TypeInferrer.GEO_PATTERN.pattern, value)

        # Test invalid values
        for value, error_text in invalid_values:
            validate_field(sample_hash_schema, field_name, value, False, error_text)

            # For GEO fields, also verify pattern failure
            if field_type == "geo" and isinstance(value, str):
                assert not re.match(TypeInferrer.GEO_PATTERN.pattern, value)

    @pytest.mark.parametrize(
        "test_case",
        [
            # Valid cases for HASH storage (bytes)
            {
                "storage": StorageType.HASH,
                "field_name": "embedding",
                "value": b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
                "valid": True,
                "error_text": None,
                "description": "Valid bytes for HASH storage",
            },
            {
                "storage": StorageType.HASH,
                "field_name": "int_vector",
                "value": b"\x01\x02\x03",
                "valid": True,
                "error_text": None,
                "description": "Valid bytes for HASH storage (int vector)",
            },
            # Invalid cases for HASH storage (trying to use lists)
            {
                "storage": StorageType.HASH,
                "field_name": "embedding",
                "value": [0.1, 0.2, 0.3, 0.4],
                "valid": False,
                "error_text": "bytes",
                "description": "List not valid for HASH storage",
            },
            # Valid cases for JSON storage (lists)
            {
                "storage": StorageType.JSON,
                "field_name": "embedding",
                "value": [0.1, 0.2, 0.3, 0.4],
                "valid": True,
                "error_text": None,
                "description": "Valid list for JSON storage",
            },
            {
                "storage": StorageType.JSON,
                "field_name": "int_vector",
                "value": [1, 2, 3],
                "valid": True,
                "error_text": None,
                "description": "Valid int list for JSON storage",
            },
            # Invalid cases for JSON storage (trying to use bytes)
            {
                "storage": StorageType.JSON,
                "field_name": "embedding",
                "value": b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
                "valid": False,
                "error_text": "list",
                "description": "Bytes not valid for JSON storage",
            },
            # Dimension validation
            {
                "storage": StorageType.JSON,
                "field_name": "embedding",
                "value": [0.1, 0.2, 0.3],  # Should be 4 dimensions
                "valid": False,
                "error_text": "dimensions",
                "description": "Wrong dimensions for vector",
            },
            # Type validation for int vectors
            {
                "storage": StorageType.JSON,
                "field_name": "int_vector",
                "value": [0.1, 0.2, 0.3],  # Should be integers
                "valid": False,
                "error_text": "integer",
                "description": "Float values in int vector",
            },
        ],
    )
    def test_vector_field_validation(
        self, sample_hash_schema, sample_json_schema, test_case
    ):
        """Test validation of vector fields with storage-specific requirements."""
        # Select the appropriate schema based on storage type
        schema = (
            sample_hash_schema
            if test_case["storage"] == StorageType.HASH
            else sample_json_schema
        )

        # Validate the field
        validate_field(
            schema,
            test_case["field_name"],
            test_case["value"],
            test_case["valid"],
            test_case["error_text"],
        )


class TestNestedJsonValidation:
    """Tests for JSON path-based validation with nested structures."""

    @pytest.mark.parametrize(
        "test_case",
        [
            # Complete valid data
            {
                "data": {
                    "test_id": "doc1",
                    "metadata": {"user": "user123", "rating": 4.5},
                    "content": {
                        "title": "Test Document",
                        "embedding": [0.1, 0.2, 0.3, 0.4],
                        "int_vector": [1, 2, 3],
                    },
                },
                "expected_fields": [
                    "test_id",
                    "user",
                    "title",
                    "rating",
                    "embedding",
                    "int_vector",
                ],
                "missing_fields": [],
            },
            # Partial data - missing some fields
            {
                "data": {
                    "test_id": "doc1",
                    "metadata": {"user": "user123"},
                    "content": {"title": "Test Document"},
                },
                "expected_fields": ["test_id", "user", "title"],
                "missing_fields": ["rating", "embedding", "int_vector"],
            },
            # Minimal data
            {
                "data": {"test_id": "doc1"},
                "expected_fields": ["test_id"],
                "missing_fields": [
                    "user",
                    "title",
                    "rating",
                    "embedding",
                    "int_vector",
                ],
            },
        ],
    )
    def test_nested_json_validation(self, sample_json_schema, test_case):
        """Test validating nested JSON with various data structures."""
        # Validate object
        validated = validate_object(sample_json_schema, test_case["data"])

        # Verify expected fields are present
        for field in test_case["expected_fields"]:
            assert field in validated

        # Verify missing fields are not present
        for field in test_case["missing_fields"]:
            assert field not in validated


class TestEndToEndValidation:
    """End-to-end tests for complete object validation against schema."""

    @pytest.mark.parametrize(
        "schema_type,data,expected_result",
        [
            # Valid HASH data
            (
                "hash",
                {
                    "test_id": "doc1",
                    "title": "Test Document",
                    "rating": 4.5,
                    "location": "37.7749,-122.4194",
                    "embedding": b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
                    "int_vector": b"\x01\x02\x03",
                },
                {
                    "success": True,
                    "fields": [
                        "test_id",
                        "title",
                        "rating",
                        "location",
                        "embedding",
                        "int_vector",
                    ],
                },
            ),
            # Partial HASH data
            (
                "hash",
                {"test_id": "doc1", "title": "Test Document"},
                {"success": True, "fields": ["test_id", "title"]},
            ),
            # Valid JSON data
            (
                "json",
                {
                    "test_id": "doc1",
                    "metadata": {"user": "user123", "rating": 4.5},
                    "content": {
                        "title": "Test Document",
                        "embedding": [0.1, 0.2, 0.3, 0.4],
                        "int_vector": [1, 2, 3],
                    },
                },
                {
                    "success": True,
                    "fields": [
                        "test_id",
                        "user",
                        "rating",
                        "title",
                        "embedding",
                        "int_vector",
                    ],
                },
            ),
            # Invalid HASH data - wrong vector type
            (
                "hash",
                {
                    "test_id": "doc1",
                    "embedding": [0.1, 0.2, 0.3, 0.4],  # Should be bytes for HASH
                },
                {"success": False, "error_field": "embedding"},
            ),
            # Invalid JSON data - wrong vector type
            (
                "json",
                {
                    "test_id": "doc1",
                    "content": {
                        "embedding": b"\x00\x00\x00\x00"  # Should be list for JSON
                    },
                },
                {"success": False, "error_field": "embedding"},
            ),
        ],
    )
    def test_end_to_end_validation(
        self, sample_hash_schema, sample_json_schema, schema_type, data, expected_result
    ):
        """Test validating complete objects with various data scenarios."""
        # Select schema based on type
        schema = sample_hash_schema if schema_type == "hash" else sample_json_schema

        if expected_result["success"]:
            # Validation should succeed
            validated = validate_object(schema, data)

            # Verify expected fields are present
            for field in expected_result["fields"]:
                assert field in validated
        else:
            # Validation should fail
            with pytest.raises(ValueError) as exc_info:
                validate_object(schema, data)

            # Error should mention the field
            assert expected_result["error_field"] in str(exc_info.value)


# -------------------- ADDITIONAL TESTS --------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_object_validation(self, sample_hash_schema, sample_json_schema):
        """Test validating an empty object."""
        # Empty object should validate for both storage types (all fields are optional)
        # TODO confirm if this is indeed true
        assert validate_object(sample_hash_schema, {}) == {}
        assert validate_object(sample_json_schema, {}) == {}

    def test_additional_fields(self, sample_hash_schema, valid_hash_data):
        """Test that additional fields not in schema are NOT ignored."""
        # Add extra field not in schema
        data_with_extra = valid_hash_data.copy()
        data_with_extra["extra_field"] = "some value"

        # Validation should succeed and ignore extra field
        validated = validate_object(sample_hash_schema, data_with_extra)
        assert "extra_field" in validated

    def test_explicit_none_fields_excluded(self, sample_hash_schema):
        """Test that fields explicitly set to None are excluded."""
        # Data with explicit None values
        data = {
            "test_id": "doc1",
            "title": "Test Document",
            "rating": None,
            "location": None,
        }

        # Validate and check fields
        validated = validate_object(sample_hash_schema, data)
        assert "test_id" in validated
        assert "title" in validated
        assert "rating" not in validated
        assert "location" not in validated
