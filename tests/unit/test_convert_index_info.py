"""Unit tests for convert_index_info_to_schema function."""

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


def test_convert_index_info_stopwords_disabled():
    """Test converting index info with STOPWORDS 0 (disabled stopwords)."""
    index_info = {
        "index_name": "test_stopwords_disabled",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["test_sw:"],
        ],
        "attributes": [],
        "stopwords_list": [],  # STOPWORDS 0
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_stopwords_disabled"
    assert result["index"]["stopwords"] == []


def test_convert_index_info_custom_stopwords():
    """Test converting index info with custom stopwords list."""
    index_info = {
        "index_name": "test_custom_stopwords",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["test_csw:"],
        ],
        "attributes": [],
        "stopwords_list": [b"the", b"a", b"an"],  # Custom stopwords (as bytes)
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_custom_stopwords"
    assert result["index"]["stopwords"] == ["the", "a", "an"]


def test_convert_index_info_default_stopwords():
    """Test converting index info with default stopwords (no stopwords_list key).

    When no STOPWORDS clause is specified in FT.CREATE, Redis uses its default
    stopwords list, and FT.INFO does not include a stopwords_list key.
    """
    index_info = {
        "index_name": "test_default_stopwords",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["test_dsw:"],
        ],
        "attributes": [],
        # No stopwords_list key - indicates default behavior
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_default_stopwords"
    assert "stopwords" not in result["index"]  # Should not be present
