"""
Tests for RedisVL storage classes with focus on validation integration.

This module tests how the storage classes integrate with the validation system:
1. How validation is used in storage operations
2. Preprocessing and validation flow
3. Error handling in write operations
"""

from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from redisvl.index.storage import HashStorage, JsonStorage
from redisvl.schema import IndexInfo, IndexSchema
from redisvl.schema.fields import (
    FlatVectorField,
    FlatVectorFieldAttributes,
    GeoField,
    HNSWVectorField,
    HNSWVectorFieldAttributes,
    NumericField,
    TagField,
    TextField,
    VectorDataType,
    VectorDistanceMetric,
)
from redisvl.schema.validation import validate_object


@pytest.fixture
def sample_schema():
    """Create a comprehensive schema for testing with all field types"""
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "test-index",
                "prefix": "test",
                "key_separator": ":",
                "storage_type": "hash",
            },
            "fields": [
                # Standard fields
                {"type": "text", "name": "text_field"},
                {"type": "numeric", "name": "num_field"},
                {"type": "tag", "name": "tag_field"},
                {"type": "geo", "name": "geo_field"},
                # Vector fields
                {
                    "type": "vector",
                    "name": "flat_vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "data_type": "float32",
                    },
                },
                {
                    "type": "vector",
                    "name": "hnsw_vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "data_type": "float32",
                        "m": 16,
                        "ef_construction": 200,
                        "ef_runtime": 10,
                        "epsilon": 0.01,
                    },
                },
            ],
        }
    )


@pytest.fixture(params=[JsonStorage, HashStorage])
def storage_instance(request, sample_schema):
    StorageClass = request.param
    instance = StorageClass(index_schema=sample_schema)
    return instance


def test_key_formatting(storage_instance):
    key = "1234"
    generated_key = storage_instance._key(key, "", "")
    assert generated_key == key
    generated_key = storage_instance._key(key, "", ":")
    assert generated_key == key
    generated_key = storage_instance._key(key, "test", ":")
    assert generated_key == f"test:{key}"


def test_create_key(storage_instance):
    id_field = "id"
    obj = {id_field: "1234"}
    expected_key = (
        f"{storage_instance.index_schema.index.prefix}"
        f"{storage_instance.index_schema.index.key_separator}"
        f"{obj[id_field]}"
    )
    generated_key = storage_instance._create_key(obj, id_field)
    assert (
        generated_key == expected_key
    ), "The generated key does not match the expected format."


def test_preprocess(storage_instance):
    data = {"key": "value"}
    preprocessed_data = storage_instance._preprocess(data, preprocess=None)
    assert preprocessed_data == data

    def fn(d):
        d["foo"] = "bar"
        return d

    preprocessed_data = storage_instance._preprocess(data, fn)
    assert "foo" in preprocessed_data
    assert preprocessed_data["foo"] == "bar"


def test_preprocess_and_validate_objects(storage_instance):
    """Test combined preprocessing and validation"""
    objects = [
        {"num_field": 123, "text_field": "valid text"},  # Valid
        {"num_field": "123", "text_field": "valid text"},  # Invalid numeric field
    ]

    def preprocess(obj):
        obj["processed"] = True
        return obj

    # When validate=True, should raise ValueError for invalid object
    with pytest.raises(ValueError) as exc_info:
        storage_instance._preprocess_and_validate_objects(
            objects, preprocess=preprocess, validate=True
        )

    # Error message should mention the issue
    assert "Validation failed" in str(exc_info.value)
    assert "must be a number" in str(exc_info.value)

    # When validate=False, should process both objects without errors
    prepared_objects = storage_instance._preprocess_and_validate_objects(
        objects, preprocess=preprocess, validate=False
    )

    assert len(prepared_objects) == 2
    # Preprocessing should have worked for both objects
    assert all(obj[1].get("processed") for obj in prepared_objects)


def test_validate_object(storage_instance):
    """Test validation of individual objects"""

    # Valid data should be returned unchanged (except for any type coercion)
    valid_data = {
        "text_field": "some text",
        "num_field": 123.45,
        "tag_field": "tag1,tag2,tag3",
        "geo_field": "37.7749,-122.4194",
        "flat_vector": [0.1, 0.2, 0.3],
        "hnsw_vector": [0.4, 0.5, 0.6],
    }

    validated = storage_instance.validate(valid_data)
    assert validated is not None
    assert validated["num_field"] == valid_data["num_field"]
    assert validated["text_field"] == valid_data["text_field"]

    # Invalid text field
    invalid_text = valid_data.copy()
    invalid_text["text_field"] = 123
    with pytest.raises(ValueError) as exc_info:
        storage_instance.validate(invalid_text)
    assert "text_field" in str(exc_info.value)

    # Invalid numeric field (string that looks like number)
    invalid_numeric = valid_data.copy()
    invalid_numeric["num_field"] = "123.45"
    with pytest.raises(ValueError) as exc_info:
        storage_instance.validate(invalid_numeric)
    assert "num_field" in str(exc_info.value)

    # Invalid geo field
    invalid_geo = valid_data.copy()
    invalid_geo["geo_field"] = "invalid-geo-format"
    with pytest.raises(ValueError) as exc_info:
        storage_instance.validate(invalid_geo)
    assert "geo_field" in str(exc_info.value)

    # Invalid vector field (wrong dimensions)
    invalid_vector_dims = valid_data.copy()
    invalid_vector_dims["flat_vector"] = [0.1, 0.2]
    with pytest.raises(ValueError) as exc_info:
        storage_instance.validate(invalid_vector_dims)
    assert "flat_vector" in str(exc_info.value)
    assert "dimensions" in str(exc_info.value)

    # Invalid vector field (non-numeric values)
    invalid_vector_values = valid_data.copy()
    invalid_vector_values["hnsw_vector"] = ["a", "b", "c"]
    with pytest.raises(ValueError) as exc_info:
        storage_instance.validate(invalid_vector_values)
    assert "hnsw_vector" in str(exc_info.value)
    assert "numeric values" in str(exc_info.value)


def test_partial_object_validation(storage_instance):
    """Test validation of partial objects (missing fields)"""

    # Object with only some fields
    partial_data = {
        "text_field": "valid text",
        # Missing num_field, tag_field, etc.
    }

    # Should validate successfully since fields are optional
    validated = storage_instance.validate(partial_data)
    assert validated is not None
    assert "text_field" in validated
    assert "num_field" not in validated

    # Explicitly setting a field to None should result in it being excluded
    null_field_data = {"text_field": "valid text", "num_field": None}

    validated = storage_instance.validate(null_field_data)
    assert "num_field" not in validated


def test_write_with_validation(storage_instance, mocker):
    """Test the write method with validation enabled"""
    # Mock the _set method to avoid actual Redis calls
    mocker.patch.object(storage_instance, "_set")

    # Mock pipeline execution
    mock_pipe = mocker.MagicMock()
    mock_pipe.execute = mocker.MagicMock()

    # Mock Redis client
    mock_client = mocker.MagicMock()
    mock_client.pipeline.return_value.__enter__.return_value = mock_pipe

    # Valid and invalid objects
    objects = [
        {"text_field": "valid", "num_field": 123},  # Valid
        {"text_field": 456, "num_field": 789},  # Invalid text field
    ]

    # With validation enabled, should raise error on first invalid object
    with pytest.raises(ValueError) as exc_info:
        storage_instance.write(mock_client, objects, validate=True)

    assert "Validation failed" in str(exc_info.value)
    assert "text_field" in str(exc_info.value)

    # With validation disabled, should process all objects
    keys = storage_instance.write(mock_client, objects, validate=False)

    assert len(keys) == 2
    assert storage_instance._set.call_count == 2


class TestBaseStorageValidation:
    """Tests for validation in BaseStorage class."""

    def test_validate_object(self, comprehensive_schema, valid_data):
        """Test the validate_object method."""
        # Create storage
        storage = BaseStorage(schema=comprehensive_schema)

        # Validate object
        validated = storage.validate_object(valid_data)

        # Verify object was validated
        assert validated is not None
        assert "id" in validated
        assert "title" in validated

    def test_validate_object_with_invalid_data(self, comprehensive_schema, valid_data):
        """Test validation with invalid data."""
        # Create storage
        storage = BaseStorage(schema=comprehensive_schema)

        # Create invalid data
        invalid_data = valid_data.copy()
        invalid_data["rating"] = "not a number"

        # Validation should fail
        with pytest.raises(ValueError) as exc_info:
            storage.validate_object(invalid_data)

        # Error message should mention validation failure
        assert "Validation failed" in str(exc_info.value)

    def test_preprocess_and_validate_objects_success(
        self, comprehensive_schema, valid_data
    ):
        """Test _preprocess_and_validate_objects with valid data."""
        # Create storage
        storage = BaseStorage(schema=comprehensive_schema)

        # Process objects
        objects = [valid_data]
        validated_objects = storage._preprocess_and_validate_objects(objects)

        # Verify objects were validated
        assert len(validated_objects) == 1
        assert "id" in validated_objects[0]
        assert "title" in validated_objects[0]

    def test_preprocess_and_validate_objects_fail(
        self, comprehensive_schema, valid_data
    ):
        """Test _preprocess_and_validate_objects with invalid data."""
        # Create storage
        storage = BaseStorage(schema=comprehensive_schema)

        # Create mix of valid and invalid data
        invalid_data = valid_data.copy()
        invalid_data["rating"] = "not a number"

        # Process should fail fast on first invalid object
        with pytest.raises(ValueError) as exc_info:
            storage._preprocess_and_validate_objects([invalid_data, valid_data])

        # Error message should mention validation failure
        assert "Validation failed" in str(exc_info.value)

    def test_write_one_validation(self, comprehensive_schema, valid_data):
        """Test that write_one validates objects."""
        # Create storage with mocked redis client
        client_mock = Mock()
        storage = BaseStorage(schema=comprehensive_schema, client=client_mock)

        # Mock hset to avoid actual Redis call
        client_mock.hset = Mock()

        # Call write_one
        storage.write_one(valid_data)

        # Verify hset was called
        client_mock.hset.assert_called_once()

    def test_write_one_validation_fail(self, comprehensive_schema, valid_data):
        """Test that write_one fails on invalid data."""
        # Create storage with mocked redis client
        client_mock = Mock()
        storage = BaseStorage(schema=comprehensive_schema, client=client_mock)

        # Create invalid data
        invalid_data = valid_data.copy()
        invalid_data["rating"] = "not a number"

        # Call write_one with invalid data
        with pytest.raises(ValueError) as exc_info:
            storage.write_one(invalid_data)

        # Verify error and that hset was not called
        assert "Validation failed" in str(exc_info.value)
        client_mock.hset.assert_not_called()

    def test_write_many_validation(self, comprehensive_schema, valid_data):
        """Test that write_many validates all objects."""
        # Create storage with mocked redis client
        client_mock = Mock()
        storage = BaseStorage(schema=comprehensive_schema, client=client_mock)

        # Mock pipeline to avoid actual Redis call
        pipeline_mock = Mock()
        client_mock.pipeline.return_value.__enter__.return_value = pipeline_mock

        # Call write_many with multiple valid objects
        storage.write_many([valid_data, valid_data.copy()])

        # Verify pipeline executed
        pipeline_mock.execute.assert_called_once()

    def test_write_many_validation_fail(self, comprehensive_schema, valid_data):
        """Test that write_many fails on invalid data."""
        # Create storage with mocked redis client
        client_mock = Mock()
        storage = BaseStorage(schema=comprehensive_schema, client=client_mock)

        # Mock pipeline to avoid actual Redis call
        pipeline_mock = Mock()
        client_mock.pipeline.return_value.__enter__.return_value = pipeline_mock

        # Create invalid data
        invalid_data = valid_data.copy()
        invalid_data["rating"] = "not a number"

        # Call write_many with invalid data
        with pytest.raises(ValueError) as exc_info:
            storage.write_many([valid_data, invalid_data])

        # Verify error and that execute was not called
        assert "Validation failed" in str(exc_info.value)
        pipeline_mock.execute.assert_not_called()


class TestJsonStorageValidation:
    """Tests for validation in JsonStorage class."""

    def test_validate_json_document(self, json_schema, valid_nested_data):
        """Test validating a JSON document."""
        # Create JSON storage
        storage = JsonStorage(schema=json_schema)

        # Validate object
        validated = storage.validate_object(valid_nested_data)

        # Verify object was validated and flattened
        assert validated is not None
        assert "id" in validated
        assert "user" in validated
        assert "title" in validated
        assert "rating" in validated

    def test_validate_json_missing_paths(self, json_schema):
        """Test validating JSON with missing paths."""
        # Create JSON storage
        storage = JsonStorage(schema=json_schema)

        # Create object with missing paths
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
        validated = storage.validate_object(partial_nested)

        # Verify validation succeeds with missing fields
        assert validated is not None
        assert "id" in validated
        assert "user" in validated
        assert "title" in validated

        # Missing fields should be absent
        assert "rating" not in validated
        assert "embedding" not in validated

    def test_validate_json_invalid_path(self, json_schema, valid_nested_data):
        """Test validating JSON with invalid path values."""
        # Create JSON storage
        storage = JsonStorage(schema=json_schema)

        # Create object with invalid data
        invalid_nested = valid_nested_data.copy()
        invalid_nested["metadata"]["rating"] = "not a number"

        # Validation should fail
        with pytest.raises(ValueError) as exc_info:
            storage.validate_object(invalid_nested)

        # Error message should mention validation failure
        assert "Validation failed" in str(exc_info.value)
        assert "rating" in str(exc_info.value)

    def test_write_json_document(self, json_schema, valid_nested_data):
        """Test writing a JSON document."""
        # Create storage with mocked redis client
        client_mock = Mock()
        storage = JsonStorage(schema=json_schema, client=client_mock)

        # Mock json.set to avoid actual Redis call
        client_mock.json.set = Mock()

        # Call write_one
        storage.write_one(valid_nested_data)

        # Verify json.set was called
        client_mock.json.set.assert_called_once()

    def test_write_json_validation_fail(self, json_schema, valid_nested_data):
        """Test that write fails on invalid JSON."""
        # Create storage with mocked redis client
        client_mock = Mock()
        storage = JsonStorage(schema=json_schema, client=client_mock)

        # Create invalid data
        invalid_nested = valid_nested_data.copy()
        invalid_nested["metadata"]["rating"] = "not a number"

        # Call write_one with invalid data
        with pytest.raises(ValueError) as exc_info:
            storage.write_one(invalid_nested)

        # Verify error and that json.set was not called
        assert "Validation failed" in str(exc_info.value)
        client_mock.json.set.assert_not_called()


@patch("redisvl.schema.validation.validate_object")
class TestValidationIntegration:
    """Tests for integration between storage and validation."""

    def test_validate_object_is_called(
        self, mock_validate, comprehensive_schema, valid_data
    ):
        """Test that validate_object is called from BaseStorage."""
        # Create storage
        storage = BaseStorage(schema=comprehensive_schema)

        # Set up mock to return the input data
        mock_validate.return_value = valid_data

        # Call validate_object
        storage.validate_object(valid_data)

        # Verify mock was called with correct args
        mock_validate.assert_called_once_with(comprehensive_schema, valid_data)

    def test_preprocess_calls_validate_for_each_object(
        self, mock_validate, comprehensive_schema, valid_data
    ):
        """Test that _preprocess_and_validate_objects calls validate for each object."""
        # Create storage
        storage = BaseStorage(schema=comprehensive_schema)

        # Set up mock to return the input data
        mock_validate.return_value = valid_data

        # Call _preprocess_and_validate_objects with multiple objects
        objects = [valid_data, valid_data.copy(), valid_data.copy()]
        storage._preprocess_and_validate_objects(objects)

        # Verify mock was called for each object
        assert mock_validate.call_count == len(objects)

    def test_preprocess_stops_on_first_validation_error(
        self, mock_validate, comprehensive_schema, valid_data
    ):
        """Test that processing stops on first validation error."""
        # Create storage
        storage = BaseStorage(schema=comprehensive_schema)

        # Set up mock to raise error on second call
        mock_validate.side_effect = [
            valid_data,
            ValueError("Validation failed for 2nd object"),
            valid_data,
        ]

        # Call _preprocess_and_validate_objects
        objects = [valid_data, valid_data.copy(), valid_data.copy()]
        with pytest.raises(ValueError) as exc_info:
            storage._preprocess_and_validate_objects(objects)

        # Verify error and that mock was called twice
        assert "Validation failed for 2nd object" in str(exc_info.value)
        assert mock_validate.call_count == 2
