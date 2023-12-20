import pathlib

import pytest

from redisvl.schema.fields import NumericField, TextField
from redisvl.schema.schema import IndexSchema, StorageType


def get_base_path():
    return pathlib.Path(__file__).parent.resolve()


# Sample data for testing
def create_sample_index_schema():
    sample_fields = {
        "text": [TextField(name="example_text", sortable=False)],
        "numeric": [NumericField(name="example_numeric", sortable=True)],
    }
    return IndexSchema(name="test", fields=sample_fields)


# Tests for IndexSchema


def test_initialization_with_default_params():
    """Test basic schema init with defaults."""
    default_schema = IndexSchema(name="test")
    assert default_schema.name == "test"
    assert default_schema.prefix == "rvl"  # Default value
    assert default_schema.key_separator == ":"  # Default value
    assert default_schema.storage_type == StorageType.HASH  # Default value
    assert default_schema.fields == {}  # Default value


def test_initialization_with_custom_params():
    """Test custom schema params."""
    custom_schema = IndexSchema(
        name="custom_schema",
        prefix="custom",
        key_separator="|",
        storage_type=StorageType.JSON,
    )
    assert custom_schema.name == "custom_schema"
    assert custom_schema.prefix == "custom"
    assert custom_schema.key_separator == "|"
    assert custom_schema.storage_type == StorageType.JSON


def test_add_field():
    """Test field addition."""
    index_schema = create_sample_index_schema()
    index_schema.add_field("text", name="new_text_field")
    assert "new_text_field" in index_schema.field_names


def test_add_fields():
    """Test multiple field addition."""
    index_schema = create_sample_index_schema()
    index_schema.add_fields(
        {"text": [{"name": "new_text_field"}], "tag": [{"name": "new_tag_field"}]}
    )
    assert "new_text_field" in index_schema.field_names
    assert "new_tag_field" in index_schema.field_names


def test_add_duplicate_field():
    """Test adding a duplicate field."""
    index_schema = create_sample_index_schema()
    with pytest.raises(ValueError):
        index_schema.add_field("text", name="example_text")


def test_remove_field():
    """Test field removal."""
    index_schema = create_sample_index_schema()
    index_schema.remove_field("text", "example_text")
    assert "example_text" not in index_schema.field_names


def test_remove_nonexistent_field():
    """Test failed remove of nonexistent field."""
    index_schema = create_sample_index_schema()
    with pytest.raises(ValueError):
        index_schema.remove_field("text", "nonexistent")


def test_schema_compare():
    """Test schema comparisons."""
    schema_1 = IndexSchema(name="test")
    # manually add the same fields as the helper method provides below
    schema_1.add_fields(
        {
            "text": [{"name": "example_text", "sortable": False}],
            "numeric": [{"name": "example_numeric", "sortable": True}],
        }
    )
    assert "example_text" in schema_1.field_names
    assert "example_numeric" in schema_1.field_names

    schema_2 = create_sample_index_schema()
    assert schema_1.field_names == schema_2.field_names
    assert schema_1.name == schema_2.name
    assert schema_1.to_dict() == schema_2.to_dict()


def test_generate_fields():
    """Test field generation."""
    data = {"name": "John", "age": 30, "tags": ["test", "test2"]}
    index_schema = IndexSchema(name="test")
    generated_fields = index_schema.generate_fields(data)
    assert "text" in generated_fields
    assert "numeric" in generated_fields
    assert "tag" in generated_fields


def test_to_dict():
    """Test schema to dict serialization"""
    index_schema = create_sample_index_schema()
    index_dict = index_schema.to_dict()
    assert index_dict["index"]["name"] == "test"


def test_from_dict():
    """Test loading schema from a dictionary"""
    sample_fields = {
        "text": [{"name": "example_text", "sortable": False}],
        "numeric": [{"name": "example_numeric", "sortable": True}],
    }
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
    assert index_schema.name == "example_index"
    assert index_schema.key_separator == "|"
    assert index_schema.prefix == "ex"
    assert index_schema.storage_type == StorageType.JSON
    assert len(index_schema.fields) == 2


def test_from_yaml():
    """Test loading from yaml"""
    index_schema = IndexSchema.from_yaml(
        str(get_base_path().joinpath("../sample_hash_schema.yaml"))
    )
    assert index_schema.name == "hash-test"
    assert index_schema.prefix == "hash"
    assert index_schema.storage_type == StorageType.HASH
    assert len(index_schema.fields) == 2
    assert "vector" in index_schema.fields
    assert "text" in index_schema.fields


def test_from_yaml_file_not_found():
    """Test loading from yaml with file not found"""
    with pytest.raises(FileNotFoundError):
        IndexSchema.from_yaml("nonexistent_file")
