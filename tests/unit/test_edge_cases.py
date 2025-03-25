"""
Tests for edge cases in the RedisVL validation module.

This module tests edge cases in the validation system that might not be
covered in the main test files, including:
1. Performance and caching behavior
2. Handling of unusual data types
3. Extreme values
4. Boundary conditions
"""

import time
from typing import Any, Dict, List

import pytest

from redisvl.index.storage import BaseStorage
from redisvl.schema.fields import Field, FieldTypes, VectorDataType
from redisvl.schema.index import Index, IndexSchema
from redisvl.schema.validation import SchemaModelGenerator, validate_object


class TestSchemaModelCaching:
    """Tests for model caching behavior."""

    def test_caching_improves_performance(self):
        """Test that caching improves model generation performance."""
        # Create a complex schema
        fields = {
            f"field_{i}": Field(name=f"field_{i}", type=FieldTypes.TEXT)
            for i in range(50)  # 50 fields should be enough to measure performance
        }

        schema = IndexSchema(
            index=Index(name="performance_test", prefix="doc"), fields=fields
        )

        # First generation (not cached)
        start_time = time.time()
        model1 = SchemaModelGenerator.get_model_for_schema(schema)
        first_time = time.time() - start_time

        # Second generation (should be cached)
        start_time = time.time()
        model2 = SchemaModelGenerator.get_model_for_schema(schema)
        second_time = time.time() - start_time

        # Verify second generation is faster
        assert second_time < first_time

        # Should be much faster (usually at least 10x)
        assert second_time < (first_time * 0.5)

        # Verify same model instance
        assert model1 is model2

    def test_different_schemas_get_different_models(self):
        """Test that different schemas get different model instances."""
        # Create two different schemas
        schema1 = IndexSchema(
            index=Index(name="test1", prefix="doc1"),
            fields={"field1": Field(name="field1", type=FieldTypes.TEXT)},
        )

        schema2 = IndexSchema(
            index=Index(name="test2", prefix="doc2"),
            fields={"field1": Field(name="field1", type=FieldTypes.TEXT)},
        )

        # Get models
        model1 = SchemaModelGenerator.get_model_for_schema(schema1)
        model2 = SchemaModelGenerator.get_model_for_schema(schema2)

        # Verify different model instances
        assert model1 is not model2
        assert model1.__name__ != model2.__name__


class TestUnusualDataTypes:
    """Tests for handling unusual data types during validation."""

    @pytest.fixture
    def basic_schema(self):
        """Create a basic schema for testing."""
        return IndexSchema(
            index=Index(name="test", prefix="doc"),
            fields={
                "text_field": Field(name="text_field", type=FieldTypes.TEXT),
                "tag_field": Field(name="tag_field", type=FieldTypes.TAG),
                "num_field": Field(name="num_field", type=FieldTypes.NUMERIC),
            },
        )

    def test_none_values(self, basic_schema):
        """Test handling of None values."""
        # Data with None values
        data = {"text_field": None, "tag_field": None, "num_field": None}

        # Validate
        result = validate_object(basic_schema, data)

        # None values should be excluded
        assert len(result) == 0

    def test_empty_string_values(self, basic_schema):
        """Test handling of empty strings."""
        # Data with empty strings
        data = {"text_field": "", "tag_field": "", "num_field": 0}

        # Validate
        result = validate_object(basic_schema, data)

        # Empty strings are valid for text and tag
        assert result["text_field"] == ""
        assert result["tag_field"] == ""
        assert result["num_field"] == 0

    def test_boolean_values(self, basic_schema):
        """Test handling of boolean values."""
        # Data with booleans
        data = {"text_field": True, "tag_field": False, "num_field": True}

        # Booleans aren't valid for text or tag
        with pytest.raises(ValueError) as exc_info:
            validate_object(basic_schema, data)

        assert "text_field" in str(exc_info.value)

        # Create new schema with only numeric
        num_schema = IndexSchema(
            index=Index(name="test", prefix="doc"),
            fields={"num_field": Field(name="num_field", type=FieldTypes.NUMERIC)},
        )

        # Validate with only the numeric field
        result = validate_object(num_schema, {"num_field": True})

        # Python converts True to 1, False to 0
        assert result["num_field"] == 1

    def test_list_for_text(self, basic_schema):
        """Test handling lists for text fields."""
        # Data with list for text
        data = {"text_field": ["item1", "item2"]}

        # Lists aren't valid for text
        with pytest.raises(ValueError) as exc_info:
            validate_object(basic_schema, data)

        assert "text_field" in str(exc_info.value)


class TestVectorEdgeCases:
    """Tests for edge cases with vector fields."""

    @pytest.fixture
    def vector_schema(self):
        """Create a schema with vector fields for testing."""
        return IndexSchema(
            index=Index(name="test_vectors", prefix="vec"),
            fields={
                "float_vec": Field(
                    name="float_vec",
                    type=FieldTypes.VECTOR,
                    attrs={"dims": 3, "datatype": VectorDataType.FLOAT32},
                ),
                "int_vec": Field(
                    name="int_vec",
                    type=FieldTypes.VECTOR,
                    attrs={"dims": 3, "datatype": VectorDataType.INT8},
                ),
            },
        )

    def test_large_vectors(self, vector_schema):
        """Test validation of very large vectors."""
        # Create a large vector (1000 dimensions)
        large_schema = IndexSchema(
            index=Index(name="large_vec", prefix="vec"),
            fields={
                "large_vec": Field(
                    name="large_vec",
                    type=FieldTypes.VECTOR,
                    attrs={"dims": 1000, "datatype": VectorDataType.FLOAT32},
                )
            },
        )

        # Valid large vector
        large_vector = {"large_vec": [0.1] * 1000}
        result = validate_object(large_schema, large_vector)
        assert len(result["large_vec"]) == 1000

        # Invalid dimensions
        invalid_dims = {"large_vec": [0.1] * 999}
        with pytest.raises(ValueError) as exc_info:
            validate_object(large_schema, invalid_dims)
        assert "dimensions" in str(exc_info.value)

    def test_mixed_vector_types(self, vector_schema):
        """Test validation of vectors with mixed element types."""
        # Float vector with mixed types
        mixed_float = {"float_vec": [1, 2.5, "3"]}
        with pytest.raises(ValueError) as exc_info:
            validate_object(vector_schema, mixed_float)
        assert "float_vec" in str(exc_info.value)

        # Int vector with mixed types
        mixed_int = {"int_vec": [1, 2.5, 3]}
        with pytest.raises(ValueError) as exc_info:
            validate_object(vector_schema, mixed_int)
        assert "int_vec" in str(exc_info.value)

    def test_empty_vector(self, vector_schema):
        """Test validation of empty vectors."""
        # Empty float vector
        empty_vec = {"float_vec": []}
        with pytest.raises(ValueError) as exc_info:
            validate_object(vector_schema, empty_vec)
        assert "float_vec" in str(exc_info.value)
        assert "dimensions" in str(exc_info.value)

    def test_vector_int_range(self, vector_schema):
        """Test validation of integer vectors with values outside allowed range."""
        # INT8 vector with values outside range
        out_of_range = {"int_vec": [100, 200, 300]}  # Valid int, but outside INT8 range
        with pytest.raises(ValueError) as exc_info:
            validate_object(vector_schema, out_of_range)
        assert "int_vec" in str(exc_info.value)
        assert "must be between" in str(exc_info.value)

        # INT8 vector with valid range
        valid_range = {"int_vec": [-128, 0, 127]}
        result = validate_object(vector_schema, valid_range)
        assert result["int_vec"] == [-128, 0, 127]


class TestGeoEdgeCases:
    """Tests for edge cases with geo fields."""

    @pytest.fixture
    def geo_schema(self):
        """Create a schema with geo fields for testing."""
        return IndexSchema(
            index=Index(name="test_geo", prefix="geo"),
            fields={"location": Field(name="location", type=FieldTypes.GEO)},
        )

    def test_geo_boundary_values(self, geo_schema):
        """Test validation of geo fields with boundary values."""
        # Valid boundary values
        valid_boundaries = [
            {"location": "90,180"},  # Max lat, max lon
            {"location": "-90,-180"},  # Min lat, min lon
            {"location": "0,0"},  # Zero point
            {"location": "90,0"},  # North pole
            {"location": "-90,0"},  # South pole
        ]

        for data in valid_boundaries:
            result = validate_object(geo_schema, data)
            assert result["location"] == data["location"]

    def test_geo_invalid_boundary_values(self, geo_schema):
        """Test validation of geo fields with invalid boundary values."""
        # Invalid boundary values
        invalid_boundaries = [
            {"location": "91,0"},  # Lat > 90
            {"location": "-91,0"},  # Lat < -90
            {"location": "0,181"},  # Lon > 180
            {"location": "0,-181"},  # Lon < -180
            {"location": "90.1,0"},  # Lat > 90 (decimal)
            {"location": "0,180.1"},  # Lon > 180 (decimal)
        ]

        for data in invalid_boundaries:
            with pytest.raises(ValueError) as exc_info:
                validate_object(geo_schema, data)
            assert "location" in str(exc_info.value)
            assert "not a valid" in str(exc_info.value)

    def test_geo_formats(self, geo_schema):
        """Test validation of geo fields with different formats."""
        # Various valid formats
        valid_formats = [
            {"location": "37.7749,-122.4194"},  # Decimal degrees
            {"location": "-37.7749,122.4194"},  # Negative latitude
            {"location": "37.7749,122.4194"},  # Positive longitude
            {"location": "0.0000,0.0000"},  # Zeros with decimal
            {"location": "37,-122"},  # Integer degrees
        ]

        for data in valid_formats:
            result = validate_object(geo_schema, data)
            assert result["location"] == data["location"]

        # Invalid formats
        invalid_formats = [
            {"location": "37.7749"},  # Missing longitude
            {"location": "37.7749,"},  # Missing longitude value
            {"location": ",122.4194"},  # Missing latitude value
            {"location": "37.7749:122.4194"},  # Wrong separator
            {"location": "37.7749, 122.4194"},  # Space after separator
            {"location": "North,South"},  # Non-numeric values
        ]

        for data in invalid_formats:
            with pytest.raises(ValueError) as exc_info:
                validate_object(geo_schema, data)
            assert "location" in str(exc_info.value)


class TestNestedJsonEdgeCases:
    """Tests for edge cases with nested JSON."""

    @pytest.fixture
    def nested_schema(self):
        """Create a schema with JSON paths for testing."""
        fields = {
            "id": Field(name="id", type=FieldTypes.TAG),
            "title": Field(name="title", type=FieldTypes.TEXT, path="$.content.title"),
            "rating": Field(
                name="rating", type=FieldTypes.NUMERIC, path="$.metadata.rating"
            ),
            "deeply_nested": Field(
                name="deeply_nested",
                type=FieldTypes.TEXT,
                path="$.level1.level2.level3.level4.value",
            ),
        }

        return IndexSchema(
            index=Index(name="test_nested", prefix="nested"), fields=fields
        )

    def test_very_deeply_nested_json(self, nested_schema):
        """Test validation with very deeply nested JSON."""
        # Create a deeply nested structure
        deeply_nested = {
            "id": "doc1",
            "level1": {
                "level2": {"level3": {"level4": {"value": "deeply nested value"}}}
            },
        }

        # Validate
        result = validate_object(nested_schema, deeply_nested)
        assert result["id"] == "doc1"
        assert result["deeply_nested"] == "deeply nested value"

    def test_partial_path_missing(self, nested_schema):
        """Test validation when part of a JSON path is missing."""
        # Create object with partial path missing
        partial_missing = {
            "id": "doc1",
            "level1": {
                "level2": {
                    # level3 missing
                }
            },
        }

        # Validate - should ignore missing path
        result = validate_object(nested_schema, partial_missing)
        assert result["id"] == "doc1"
        assert "deeply_nested" not in result

    def test_nested_arrays(self):
        """Test validation with nested arrays in JSON."""
        # Create schema with path to array element
        array_schema = IndexSchema(
            index=Index(name="test_arrays", prefix="arr"),
            fields={
                "id": Field(name="id", type=FieldTypes.TAG),
                "first_item": Field(
                    name="first_item", type=FieldTypes.TEXT, path="$.items[0]"
                ),
                "nested_item": Field(
                    name="nested_item",
                    type=FieldTypes.TEXT,
                    path="$.nested.items[1].name",
                ),
            },
        )

        # Note: JSONPath with array indexing is not supported currently
        # This test documents this limitation

        # Create data with arrays
        array_data = {
            "id": "arr1",
            "items": ["first", "second", "third"],
            "nested": {"items": [{"name": "item1"}, {"name": "item2"}]},
        }

        # Validate - array paths won't be found
        result = validate_object(array_schema, array_data)
        assert result["id"] == "arr1"
        assert "first_item" not in result
        assert "nested_item" not in result


class TestValidationIntegrationEdgeCases:
    """Tests for integration edge cases between storage and validation."""

    @pytest.fixture
    def storage_with_schema(self):
        """Create a storage instance with schema for testing."""
        schema = IndexSchema(
            index=Index(name="test_storage", prefix="doc"),
            fields={
                "id": Field(name="id", type=FieldTypes.TAG),
                "vec": Field(
                    name="vec",
                    type=FieldTypes.VECTOR,
                    attrs={"dims": 3, "datatype": VectorDataType.FLOAT32},
                ),
            },
        )

        return BaseStorage(schema=schema, client=None)

    def test_validation_with_bytes_no_client(self, storage_with_schema):
        """Test validation with bytes when no Redis client is available."""
        # No Redis client was provided, so hset won't be called
        # This just tests that validation works with bytes

        # Valid data with bytes
        data = {"id": "doc1", "vec": b"\x00\x01\x02"}  # 3 bytes

        # Validate - should work even without client
        validated = storage_with_schema.validate_object(data)
        assert validated["id"] == "doc1"
        assert validated["vec"] == b"\x00\x01\x02"

    def test_unexpected_field_is_ignored(self, storage_with_schema):
        """Test that unexpected fields are ignored during validation."""
        # Data with extra field
        data = {
            "id": "doc1",
            "vec": [0.1, 0.2, 0.3],
            "extra": "This field is not in the schema",
        }

        # Validate
        validated = storage_with_schema.validate_object(data)

        # Extra field should be ignored
        assert validated["id"] == "doc1"
        assert validated["vec"] == [0.1, 0.2, 0.3]
        assert "extra" not in validated
