"""Unit tests for convert_index_info_to_schema function."""

import pytest

from redisvl.redis.connection import convert_index_info_to_schema


def test_convert_index_info_single_prefix():
    """Test converting index info with a single prefix.

    Single-element prefix lists are normalized to strings for backward compatibility.
    """
    index_info = {
        "index_name": "test_index",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["prefix_a"],
        ],
        "attributes": [],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_index"
    assert result["index"]["prefix"] == "prefix_a"  # normalized to string
    assert result["index"]["storage_type"] == "hash"


def test_convert_index_info_multiple_prefixes():
    """Test converting index info with multiple prefixes (issue #258)."""
    index_info = {
        "index_name": "test_index",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["prefix_a", "prefix_b", "prefix_c"],
        ],
        "attributes": [],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_index"
    assert result["index"]["prefix"] == ["prefix_a", "prefix_b", "prefix_c"]
    assert result["index"]["storage_type"] == "hash"


def test_convert_index_info_json_storage():
    """Test converting index info with JSON storage type.

    Single-element prefix lists are normalized to strings for backward compatibility.
    """
    index_info = {
        "index_name": "test_json_index",
        "index_definition": [
            "key_type",
            "JSON",
            "prefixes",
            ["json_prefix"],
        ],
        "attributes": [],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_json_index"
    assert result["index"]["prefix"] == "json_prefix"  # normalized to string
    assert result["index"]["storage_type"] == "json"


def test_convert_index_info_with_fields():
    """Test converting index info with field definitions."""
    index_info = {
        "index_name": "test_index",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["prefix_a", "prefix_b"],
        ],
        "attributes": [
            [
                "identifier",
                "user",
                "attribute",
                "user",
                "type",
                "TAG",
            ],
            [
                "identifier",
                "text",
                "attribute",
                "text",
                "type",
                "TEXT",
            ],
        ],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_index"
    assert result["index"]["prefix"] == ["prefix_a", "prefix_b"]
    assert len(result["fields"]) == 2
    assert result["fields"][0]["name"] == "user"
    assert result["fields"][0]["type"] == "tag"
    assert result["fields"][1]["name"] == "text"
    assert result["fields"][1]["type"] == "text"
