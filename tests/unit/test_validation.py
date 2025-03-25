"""
Tests for the RedisVL schema validation module.

This module tests the core validation functionality:
1. Model generation from schemas
2. Field-specific validators
3. JSON path extraction
4. Validation of various field types
"""

import re
from typing import Any, Dict, List

import pytest

from redisvl.schema import IndexSchema
from redisvl.schema.fields import FieldTypes, VectorDataType
from redisvl.schema.type_utils import TypeInferrer
from redisvl.schema.validation import (
    SchemaModelGenerator,
    extract_from_json_path,
    validate_object,
)


@pytest.fixture
def sample_schema():
    """Create a sample schema with different field types for testing."""
    schema_dict = {
        "index": {
            "name": "test-index",
            "prefix": "test",
            "key_separator": ":",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "id", "type": "tag"},
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
        ],
    }
    return IndexSchema.from_dict(schema_dict)


@pytest.fixture
def sample_json_schema():
    """Create a sample schema with JSON storage and path fields."""
    schema_dict = {
        "index": {
            "name": "test-json-index",
            "prefix": "test",
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {"name": "id", "type": "tag", "path": "$.id"},
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
        ],
    }
    return IndexSchema.from_dict(schema_dict)


@pytest.fixture
def valid_data():
    """Sample valid data for testing validation."""
    return {
        "id": "doc1",
        "title": "Test Document",
        "rating": 4.5,
        "location": "37.7749,-122.4194",
        "embedding": [0.1, 0.2, 0.3, 0.4],
    }


@pytest.fixture
def valid_nested_data():
    """Sample valid nested data for testing JSON path validation."""
    return {
        "id": "doc1",
        "metadata": {"user": "user123", "rating": 4.5},
        "content": {"title": "Test Document", "embedding": [0.1, 0.2, 0.3, 0.4]},
    }


class TestSchemaModelGenerator:
    """Tests for the SchemaModelGenerator class."""

    def test_get_model_for_schema(self, sample_schema):
        """Test generating a model from a schema."""
        # Get model for schema
        model_class = SchemaModelGenerator.get_model_for_schema(sample_schema)

        # Verify model name matches the index name
        assert model_class.__name__ == "test-index__PydanticModel"

        # Verify model has expected fields
        for field_name in sample_schema.field_names:
            assert field_name in model_class.model_fields

    def test_model_caching(self, sample_schema):
        """Test that models are cached and reused."""
        # Get model twice
        model1 = SchemaModelGenerator.get_model_for_schema(sample_schema)
        model2 = SchemaModelGenerator.get_model_for_schema(sample_schema)

        # Verify same instance
        assert model1 is model2

    def test_type_mapping(self, sample_schema):
        """Test mapping Redis field types to Pydantic types."""
        for field_name, field in sample_schema.fields.items():
            field_type = SchemaModelGenerator._map_field_to_pydantic_type(field)

            # Verify each field type maps to expected Python type
            if field.type == FieldTypes.TEXT:
                assert field_type == str
            elif field.type == FieldTypes.TAG:
                assert field_type == str
            elif field.type == FieldTypes.NUMERIC:
                assert field_type.__origin__ == type(Union)  # Check it's a Union
            elif field.type == FieldTypes.VECTOR:
                assert field_type.__origin__ == type(Union)  # Check it's a Union

    def test_unsupported_field_type(self):
        """Test that an error is raised for unsupported field types."""

        # Create a dummy field with unsupported type
        class DummyField:
            type = "unsupported_type"

        # Mapping should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            SchemaModelGenerator._map_field_to_pydantic_type(DummyField())

        assert "Unsupported field type" in str(exc_info.value)


class TestFieldValidators:
    """Tests for field-specific validators."""

    def test_text_field_validation(self, sample_schema, valid_data):
        """Test validation of text fields."""
        model_class = SchemaModelGenerator.get_model_for_schema(sample_schema)

        # Valid text field
        valid = valid_data.copy()
        validated = model_class.model_validate(valid)
        assert validated.title == "Test Document"

        # Invalid text field (number)
        invalid = valid_data.copy()
        invalid["title"] = 123
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid)
        assert "title" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)

    def test_tag_field_validation(self, sample_schema, valid_data):
        """Test validation of tag fields."""
        model_class = SchemaModelGenerator.get_model_for_schema(sample_schema)

        # Valid tag field
        valid = valid_data.copy()
        validated = model_class.model_validate(valid)
        assert validated.id == "doc1"

        # Invalid tag field (number)
        invalid = valid_data.copy()
        invalid["id"] = 123
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid)
        assert "id" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)

    def test_numeric_field_validation(self, sample_schema, valid_data):
        """Test validation of numeric fields."""
        model_class = SchemaModelGenerator.get_model_for_schema(sample_schema)

        # Valid numeric field (integer)
        valid_int = valid_data.copy()
        valid_int["rating"] = 5
        validated = model_class.model_validate(valid_int)
        assert validated.rating == 5

        # Valid numeric field (float)
        valid_float = valid_data.copy()
        valid_float["rating"] = 4.5
        validated = model_class.model_validate(valid_float)
        assert validated.rating == 4.5

        # Invalid numeric field (string)
        invalid = valid_data.copy()
        invalid["rating"] = "high"
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid)
        assert "rating" in str(exc_info.value)
        assert "must be a number" in str(exc_info.value)

        # Invalid numeric field (string that looks like number)
        invalid_num_str = valid_data.copy()
        invalid_num_str["rating"] = "4.5"
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid_num_str)
        assert "rating" in str(exc_info.value)
        assert "must be a number" in str(exc_info.value)

    def test_geo_field_validation(self, sample_schema, valid_data):
        """Test validation of geo fields."""
        model_class = SchemaModelGenerator.get_model_for_schema(sample_schema)

        # Valid geo format
        valid_geo = valid_data.copy()
        valid_geo["location"] = "37.7749,-122.4194"
        validated = model_class.model_validate(valid_geo)
        assert validated.location == "37.7749,-122.4194"

        # Invalid geo format (not matching lat,lon pattern)
        invalid_geo = valid_data.copy()
        invalid_geo["location"] = "invalid_geo"
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid_geo)
        assert "location" in str(exc_info.value)
        assert "not a valid 'lat,lon' format" in str(exc_info.value)

        # Verify the geo pattern actually works with valid formats
        valid_formats = [
            "0,0",
            "90,-180",
            "-90,180",
            "37.7749,-122.4194",
            "37.7749,122.4194",
            "-37.7749,-122.4194",
        ]
        for format in valid_formats:
            assert re.match(TypeInferrer.GEO_PATTERN.pattern, format)

        # Verify invalid formats fail the pattern
        invalid_formats = [
            "invalid",
            "37.7749",
            "37.7749,",
            ",122.4194",
            "91,0",  # Latitude > 90
            "-91,0",  # Latitude < -90
            "0,181",  # Longitude > 180
            "0,-181",  # Longitude < -180
        ]
        for format in invalid_formats:
            assert not re.match(TypeInferrer.GEO_PATTERN.pattern, format)

    def test_vector_field_validation_float(self, sample_schema, valid_data):
        """Test validation of float vector fields."""
        model_class = SchemaModelGenerator.get_model_for_schema(sample_schema)

        # Valid vector
        valid_vector = valid_data.copy()
        valid_vector["embedding"] = [0.1, 0.2, 0.3, 0.4]
        validated = model_class.model_validate(valid_vector)
        assert validated.embedding == [0.1, 0.2, 0.3, 0.4]

        # Valid vector as bytes
        valid_bytes = valid_data.copy()
        valid_bytes["embedding"] = b"\x00\x01\x02\x03"
        validated = model_class.model_validate(valid_bytes)
        assert validated.embedding == b"\x00\x01\x02\x03"

        # Invalid vector type (string)
        invalid_type = valid_data.copy()
        invalid_type["embedding"] = "not a vector"
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid_type)
        assert "embedding" in str(exc_info.value)

        # Invalid dimensions
        invalid_dims = valid_data.copy()
        invalid_dims["embedding"] = [0.1, 0.2, 0.3]  # 3 dimensions instead of 4
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid_dims)
        assert "embedding" in str(exc_info.value)
        assert "dimensions" in str(exc_info.value)

        # Invalid vector values
        invalid_values = valid_data.copy()
        invalid_values["embedding"] = [0.1, "string", 0.3, 0.4]
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid_values)
        assert "embedding" in str(exc_info.value)

    def test_vector_field_validation_int(self, sample_schema, valid_data):
        """Test validation of integer vector fields."""
        model_class = SchemaModelGenerator.get_model_for_schema(sample_schema)

        # Valid integer vector
        valid_vector = valid_data.copy()
        valid_vector["int_vector"] = [1, 2, 3]
        validated = model_class.model_validate(valid_vector)
        assert validated.int_vector == [1, 2, 3]

        # Invalid: float values in int vector
        invalid_floats = valid_data.copy()
        invalid_floats["int_vector"] = [0.1, 0.2, 0.3]
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid_floats)
        assert "int_vector" in str(exc_info.value)
        assert "integer values" in str(exc_info.value)

        # Invalid: values outside INT8 range
        invalid_range = valid_data.copy()
        invalid_range["int_vector"] = [1000, 2000, 3000]  # Outside INT8 range
        with pytest.raises(ValueError) as exc_info:
            model_class.model_validate(invalid_range)
        assert "int_vector" in str(exc_info.value)
        assert "must be between" in str(exc_info.value)


class TestJsonPathValidation:
    """Tests for JSON path-based validation."""

    def test_extract_from_json_path(self, valid_nested_data):
        """Test extracting values using JSON paths."""
        # Test simple path
        assert extract_from_json_path(valid_nested_data, "$.id") == "doc1"

        # Test nested path
        assert extract_from_json_path(valid_nested_data, "$.metadata.user") == "user123"
        assert extract_from_json_path(valid_nested_data, "$.metadata.rating") == 4.5
        assert (
            extract_from_json_path(valid_nested_data, "$.content.title")
            == "Test Document"
        )
        assert extract_from_json_path(valid_nested_data, "$.content.embedding") == [
            0.1,
            0.2,
            0.3,
            0.4,
        ]

        # Test non-existent path
        assert extract_from_json_path(valid_nested_data, "$.nonexistent") is None
        assert (
            extract_from_json_path(valid_nested_data, "$.metadata.nonexistent") is None
        )

        # Test path with alternate formats
        assert extract_from_json_path(valid_nested_data, "metadata.user") == "user123"

    def test_validate_nested_json(self, sample_json_schema, valid_nested_data):
        """Test validating a nested JSON object."""
        # Validate nested object
        validated = validate_object(sample_json_schema, valid_nested_data)

        # Verify validation succeeds and flattens the structure
        assert validated is not None
        assert "id" in validated
        assert "user" in validated
        assert "title" in validated
        assert "rating" in validated
        assert "embedding" in validated

        # Verify values were extracted correctly
        assert validated["id"] == "doc1"
        assert validated["user"] == "user123"
        assert validated["title"] == "Test Document"
        assert validated["rating"] == 4.5
        assert validated["embedding"] == [0.1, 0.2, 0.3, 0.4]

    def test_validate_nested_json_missing_paths(self, sample_json_schema):
        """Test validating a nested JSON with missing paths."""
        # Nested object with missing paths
        partial_nested = {
            "id": "doc1",
            "metadata": {
                "user": "user123"
                # missing rating
            },
            "content": {
                "title": "Test Document"
                # missing embedding
            },
        }

        # Validate object
        validated = validate_object(sample_json_schema, partial_nested)

        # Verify validation succeeds with partial data
        assert validated is not None
        assert "id" in validated
        assert "user" in validated
        assert "title" in validated
        assert "rating" not in validated
        assert "embedding" not in validated


class TestObjectValidation:
    """Tests for complete object validation."""

    def test_validate_valid_object(self, sample_schema, valid_data):
        """Test validating a valid object."""
        # Validate object
        validated = validate_object(sample_schema, valid_data)

        # Verify no exceptions and data is returned
        assert validated is not None

        # Verify all fields are present
        for field_name in sample_schema.field_names:
            if field_name in valid_data:
                assert field_name in validated

    def test_validate_missing_optional_fields(self, sample_schema):
        """Test validating an object with missing optional fields."""
        # Object with only some fields
        partial_data = {"id": "doc1", "title": "Test Document"}

        # Validate object
        validated = validate_object(sample_schema, partial_data)

        # Verify validation passes with partial data
        assert validated is not None
        assert "id" in validated
        assert "title" in validated
        assert "rating" not in validated
        assert "location" not in validated
        assert "embedding" not in validated

    def test_explicit_none_fields_are_excluded(self, sample_schema):
        """Test that fields explicitly set to None are excluded from output."""
        # Object with some fields set to None
        data_with_none = {
            "id": "doc1",
            "title": "Test Document",
            "rating": None,
            "location": None,
        }

        # Validate object
        validated = validate_object(sample_schema, data_with_none)

        # Verify None fields are excluded
        assert validated is not None
        assert "id" in validated
        assert "title" in validated
        assert "rating" not in validated
        assert "location" not in validated

    def test_validate_with_multiple_invalid_fields(self, sample_schema, valid_data):
        """Test validation with multiple invalid fields."""
        # Create object with multiple invalid fields
        invalid_data = valid_data.copy()
        invalid_data["title"] = 123
        invalid_data["rating"] = "not a number"
        invalid_data["location"] = "invalid"

        # Validation should fail with the first error encountered
        with pytest.raises(ValueError) as exc_info:
            validate_object(sample_schema, invalid_data)

        # Error message should mention validation failure
        assert "Validation failed" in str(exc_info.value)

    @pytest.mark.parametrize(
        "case",
        [
            {"field": "title", "value": 123, "error_text": "must be a string"},
            {"field": "rating", "value": "high", "error_text": "must be a number"},
            {
                "field": "location",
                "value": "invalid_geo",
                "error_text": "not a valid 'lat,lon' format",
            },
            {
                "field": "embedding",
                "value": [0.1, 0.2, 0.3],
                "error_text": "dimensions",
            },
        ],
    )
    def test_validate_invalid_field_parametrized(self, sample_schema, valid_data, case):
        """Parametrized test for validating invalid fields."""
        # Create invalid data according to test case
        invalid_data = valid_data.copy()
        invalid_data[case["field"]] = case["value"]

        # Validate and check error
        with pytest.raises(ValueError) as exc_info:
            validate_object(sample_schema, invalid_data)

        # Error should mention the field and specific issue
        error_message = str(exc_info.value)
        assert case["field"] in error_message
        assert case["error_text"] in error_message
