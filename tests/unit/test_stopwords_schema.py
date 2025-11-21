"""Unit tests for stopwords support in IndexSchema."""

import tempfile

import yaml

from redisvl.schema import IndexSchema


def test_index_schema_stopwords_none_default():
    """Test IndexSchema with no stopwords specified (default behavior)."""
    schema_dict = {
        "index": {
            "name": "test_index",
            "prefix": "test",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "title", "type": "text"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)

    assert schema.index.name == "test_index"
    assert schema.index.stopwords is None  # Default


def test_index_schema_stopwords_disabled():
    """Test IndexSchema with stopwords disabled (STOPWORDS 0)."""
    schema_dict = {
        "index": {
            "name": "test_index",
            "prefix": "test",
            "storage_type": "hash",
            "stopwords": [],  # Empty list = STOPWORDS 0
        },
        "fields": [
            {"name": "title", "type": "text"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)

    assert schema.index.name == "test_index"
    assert schema.index.stopwords == []


def test_index_schema_custom_stopwords():
    """Test IndexSchema with custom stopwords list."""
    schema_dict = {
        "index": {
            "name": "test_index",
            "prefix": "test",
            "storage_type": "hash",
            "stopwords": ["the", "a", "an"],
        },
        "fields": [
            {"name": "title", "type": "text"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)

    assert schema.index.name == "test_index"
    assert schema.index.stopwords == ["the", "a", "an"]


def test_index_schema_stopwords_from_yaml_disabled():
    """Test IndexSchema from YAML with stopwords disabled."""
    yaml_content = """
version: '0.1.0'

index:
    name: test_yaml_index
    prefix: test_yaml
    storage_type: hash
    stopwords: []

fields:
    - name: title
      type: text
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        schema = IndexSchema.from_yaml(yaml_path)
        assert schema.index.name == "test_yaml_index"
        assert schema.index.stopwords == []
    finally:
        import os

        os.unlink(yaml_path)


def test_index_schema_stopwords_from_yaml_custom():
    """Test IndexSchema from YAML with custom stopwords."""
    yaml_content = """
version: '0.1.0'

index:
    name: test_yaml_index
    prefix: test_yaml
    storage_type: hash
    stopwords:
        - the
        - a
        - an

fields:
    - name: title
      type: text
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        schema = IndexSchema.from_yaml(yaml_path)
        assert schema.index.name == "test_yaml_index"
        assert schema.index.stopwords == ["the", "a", "an"]
    finally:
        import os

        os.unlink(yaml_path)


def test_index_schema_to_dict_preserves_stopwords():
    """Test that to_dict() preserves stopwords configuration."""
    schema_dict = {
        "index": {
            "name": "test_index",
            "prefix": "test",
            "storage_type": "hash",
            "stopwords": ["the", "a"],
        },
        "fields": [
            {"name": "title", "type": "text"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)
    result_dict = schema.to_dict()

    assert result_dict["index"]["stopwords"] == ["the", "a"]


def test_index_schema_to_dict_omits_none_stopwords():
    """Test that to_dict() omits stopwords when None (default)."""
    schema_dict = {
        "index": {
            "name": "test_index",
            "prefix": "test",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "title", "type": "text"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)
    result_dict = schema.to_dict()

    # stopwords should not be in the dict when None (default behavior)
    assert "stopwords" not in result_dict["index"]


def test_index_schema_to_yaml_preserves_stopwords():
    """Test that to_yaml() preserves stopwords configuration."""
    schema_dict = {
        "index": {
            "name": "test_index",
            "prefix": "test",
            "storage_type": "hash",
            "stopwords": [],  # STOPWORDS 0
        },
        "fields": [
            {"name": "title", "type": "text"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name

    try:
        schema.to_yaml(yaml_path)

        # Read back and verify
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        assert yaml_data["index"]["stopwords"] == []
    finally:
        import os

        os.unlink(yaml_path)
