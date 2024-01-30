import os
import pathlib

import pytest

from redisvl.schema.fields import NumericField, TagField, TextField
from redisvl.schema.schema import IndexSchema, StorageType


def get_base_path():
    return pathlib.Path(__file__).parent.resolve()


# Sample data for testing
def create_sample_index_schema():
    sample_fields = [
        {"name": "example_text", "type": "text", "attrs": {"sortable": False}},
        {"name": "example_numeric", "type": "numeric", "attrs": {"sortable": True}},
    ]
    return IndexSchema.from_dict({"index": {"name": "test"}, "fields": sample_fields})


# Tests for IndexSchema


def test_initialization_with_default_params():
    """Test basic schema init with defaults."""
    default_schema = IndexSchema.from_dict({"index": {"name": "test"}})
    assert default_schema.version == "0.1.0"
    assert default_schema.index.name == "test"
    assert default_schema.index.prefix == "rvl"  # Default value
    assert default_schema.index.key_separator == ":"  # Default value
    assert default_schema.index.storage_type == StorageType.HASH  # Default value
    assert default_schema.fields == {}  # Default value


def test_initialization_with_custom_params():
    """Test custom schema params."""
    custom_schema = IndexSchema.from_dict(
        {
            "index": {
                "name": "custom_schema",
                "prefix": "custom",
                "key_separator": "|",
                "storage_type": "json",
            }
        }
    )
    assert custom_schema.index.name == "custom_schema"
    assert custom_schema.index.prefix == "custom"
    assert custom_schema.index.key_separator == "|"
    assert custom_schema.index.storage_type == StorageType.JSON


def test_add_field():
    """Test field addition."""
    index_schema = create_sample_index_schema()
    index_schema.add_field({"name": "new_text_field", "type": "text"})
    assert "new_text_field" in index_schema.fields
    assert isinstance(index_schema.fields["new_text_field"], TextField)


def test_add_fields():
    """Test multiple field addition."""
    index_schema = create_sample_index_schema()
    index_schema.add_fields(
        [
            {"name": "new_text_field", "type": "text"},
            {"name": "new_tag_field", "type": "tag"},
        ]
    )
    assert "new_text_field" in index_schema.fields
    assert isinstance(index_schema.fields["new_text_field"], TextField)
    assert "new_tag_field" in index_schema.fields
    assert isinstance(index_schema.fields["new_tag_field"], TagField)


def test_add_duplicate_field():
    """Test adding a duplicate field."""
    index_schema = create_sample_index_schema()
    with pytest.raises(ValueError):
        index_schema.add_field({"name": "example_text", "type": "text"})


def test_remove_field():
    """Test field removal."""
    index_schema = create_sample_index_schema()
    index_schema.remove_field("example_text")
    assert "example_text" not in index_schema.field_names


def test_schema_compare():
    """Test schema comparisons."""
    schema_1 = IndexSchema.from_dict({"index": {"name": "test"}})
    # manually add the same fields as the helper method provides below
    schema_1.add_fields(
        [
            {"name": "example_text", "type": "text", "attrs": {"sortable": False}},
            {"name": "example_numeric", "type": "numeric", "attrs": {"sortable": True}},
        ]
    )

    assert "example_text" in schema_1.fields
    assert "example_numeric" in schema_1.fields

    schema_2 = create_sample_index_schema()
    assert schema_1.fields == schema_2.fields
    assert schema_1.index.name == schema_2.index.name
    assert schema_1.to_dict() == schema_2.to_dict()


def test_generate_fields():
    """Test field generation."""
    sample = {"name": "John", "age": 30, "tags": ["test", "test2"]}
    index_schema = IndexSchema.from_dict({"index": {"name": "test"}})
    generated_fields = index_schema.generate_fields(sample)
    expected_field_names = sample.keys()
    for field in generated_fields:
        assert field["name"] in expected_field_names
        assert field["path"] == None


def test_to_dict():
    """Test schema to dict serialization."""
    index_schema = create_sample_index_schema()
    index_dict = index_schema.to_dict()
    assert index_dict["index"]["name"] == "test"
    assert isinstance(index_dict["fields"], list)
    assert len(index_dict["fields"]) == 2 == len(index_schema.fields)


def test_from_dict():
    """Test loading schema from a dictionary."""
    sample_fields = [
        {"name": "example_text", "type": "text", "attrs": {"sortable": False}},
        {"name": "example_numeric", "type": "tag", "attrs": {"sortable": True}},
    ]
    index_schema = IndexSchema.from_dict(
        {
            "index": {
                "name": "example_index",
                "prefix": "ex",
                "key_separator": "|",
                "storage_type": "json",
            },
            "fields": sample_fields,
        }
    )
    assert index_schema.index.name == "example_index"
    assert index_schema.index.key_separator == "|"
    assert index_schema.index.prefix == "ex"
    assert index_schema.index.storage_type == StorageType.JSON
    assert len(index_schema.fields) == 2


def test_hash_index_from_yaml():
    """Test loading from yaml."""
    index_schema = IndexSchema.from_yaml(
        str(get_base_path().joinpath("../../schemas/test_hash_schema.yaml"))
    )
    assert index_schema.index.name == "hash-test"
    assert index_schema.index.prefix == "hash"
    assert index_schema.index.storage_type == StorageType.HASH
    assert len(index_schema.fields) == 2


def test_json_index_from_yaml():
    """Test loading from yaml."""
    index_schema = IndexSchema.from_yaml(
        str(get_base_path().joinpath("../../schemas/test_json_schema.yaml"))
    )
    assert index_schema.index.name == "json-test"
    assert index_schema.index.prefix == "json"
    assert index_schema.index.storage_type == StorageType.JSON
    assert len(index_schema.fields) == 2


def test_to_yaml_and_reload():
    index_schema = create_sample_index_schema()
    index_schema.to_yaml("temp_test.yaml")

    assert os.path.exists("temp_test.yaml")

    new_schema = IndexSchema.from_yaml("temp_test.yaml")
    assert new_schema == index_schema
    assert new_schema.to_dict() == index_schema.to_dict()

    os.remove("temp_test.yaml")


def test_from_yaml_file_not_found():
    """Test loading from yaml with file not found."""
    with pytest.raises(FileNotFoundError):
        IndexSchema.from_yaml("nonexistent_file")
