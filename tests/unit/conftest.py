"""
Common test fixtures and utilities for RedisVL validation tests.
"""

from typing import Any, Dict

import pytest

from redisvl.schema import IndexSchema
from redisvl.schema.fields import VectorDataType, VectorDistanceMetric


@pytest.fixture
def comprehensive_schema():
    """Create a comprehensive schema with all field types for testing."""
    return IndexSchema.from_dict(
        {
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
                {
                    "name": "hnsw_vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                        "m": 16,
                        "ef_construction": 200,
                        "ef_runtime": 10,
                        "epsilon": 0.01,
                    },
                },
            ],
        }
    )


@pytest.fixture
def json_schema():
    """Create a schema with JSON storage and path fields."""
    return IndexSchema.from_dict(
        {
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
    )


@pytest.fixture
def valid_data():
    """Sample valid data for testing validation."""
    return {
        "id": "doc1",
        "title": "Test Document",
        "rating": 4.5,
        "location": "37.7749,-122.4194",
        "embedding": [0.1, 0.2, 0.3, 0.4],
        "int_vector": [1, 2, 3],
        "hnsw_vector": [0.1, 0.2, 0.3],
    }


@pytest.fixture
def valid_nested_data():
    """Sample valid nested data for testing JSON path validation."""
    return {
        "id": "doc1",
        "metadata": {"user": "user123", "rating": 4.5},
        "content": {"title": "Test Document", "embedding": [0.1, 0.2, 0.3, 0.4]},
    }


@pytest.fixture
def invalid_data_cases():
    """
    Test cases for invalid data.
    Each case contains:
    - field: name of the field
    - value: invalid value to test
    - error_text: text that should appear in error message
    """
    return [
        # Text field errors
        {"field": "title", "value": 123, "error_text": "must be a string"},
        # Numeric field errors
        {"field": "rating", "value": "high", "error_text": "must be a number"},
        {"field": "rating", "value": "123.45", "error_text": "must be a number"},
        # Tag field errors
        {"field": "id", "value": 123, "error_text": "must be a string"},
        # Geo field errors
        {
            "field": "location",
            "value": "invalid_geo",
            "error_text": "not a valid 'lat,lon' format",
        },
        {
            "field": "location",
            "value": "1000,-1000",
            "error_text": "not a valid 'lat,lon' format",
        },
        # Vector field errors - float32
        {"field": "embedding", "value": [0.1, 0.2, 0.3], "error_text": "dimensions"},
        {
            "field": "embedding",
            "value": [0.1, "string", 0.3, 0.4],
            "error_text": "numeric values",
        },
        {
            "field": "embedding",
            "value": "not_a_vector",
            "error_text": "must be a list or bytes",
        },
        # Vector field errors - int8
        {
            "field": "int_vector",
            "value": [0.1, 0.2, 0.3],
            "error_text": "integer values",
        },
        {"field": "int_vector", "value": [1, 2], "error_text": "dimensions"},
        {
            "field": "int_vector",
            "value": [1000, 2000, 3000],
            "error_text": "INT8 values must be between",
        },
        # HNSW Vector field errors
        {"field": "hnsw_vector", "value": [0.1, 0.2], "error_text": "dimensions"},
        {
            "field": "hnsw_vector",
            "value": ["a", "b", "c"],
            "error_text": "numeric values",
        },
    ]
