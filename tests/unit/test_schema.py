import os
import pathlib

import pytest

from redisvl.schema.fields import TagField, TextField
from redisvl.schema.schema import IndexSchema, StorageType


def get_base_path():
    return pathlib.Path(__file__).parent.resolve()


# Sample data for testing
def create_sample_index_schema():
    sample_fields = [
        {"name": "example_text", "type": "text", "attrs": {"sortable": False}},
        {"name": "example_numeric", "type": "numeric", "attrs": {"sortable": True}},
        {"name": "example_tag", "type": "tag", "attrs": {"sortable": True}},
        {
            "name": "example_vector",
            "type": "vector",
            "attrs": {"dims": 1024, "algorithm": "flat"},
        },
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
    assert len(index_dict["fields"]) == 4 == len(index_schema.fields)


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


def test_schema_with_index_missing_and_empty_attributes():
    """Test schema creation and operations with INDEXMISSING and INDEXEMPTY attributes."""
    schema_dict = {
        "index": {
            "name": "test-missing-empty",
            "prefix": "test",
            "storage_type": "hash",
        },
        "fields": [
            {
                "name": "title",
                "type": "text",
                "attrs": {"index_missing": True, "index_empty": True, "sortable": True},
            },
            {
                "name": "tags",
                "type": "tag",
                "attrs": {"index_missing": True, "index_empty": True},
            },
            {
                "name": "price",
                "type": "numeric",
                "attrs": {"index_missing": True, "sortable": True},
            },
            {"name": "location", "type": "geo", "attrs": {"index_missing": True}},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 128,
                    "distance_metric": "cosine",
                    "index_missing": True,
                },
            },
        ],
    }

    # Test schema creation
    schema = IndexSchema.from_dict(schema_dict)

    # Verify field attributes are correctly set
    assert schema.fields["title"].attrs.index_missing == True
    assert schema.fields["title"].attrs.index_empty == True
    assert schema.fields["title"].attrs.sortable == True

    assert schema.fields["tags"].attrs.index_missing == True
    assert schema.fields["tags"].attrs.index_empty == True

    assert schema.fields["price"].attrs.index_missing == True
    assert schema.fields["price"].attrs.sortable == True

    assert schema.fields["location"].attrs.index_missing == True

    assert schema.fields["embedding"].attrs.index_missing == True
    assert schema.fields["embedding"].attrs.dims == 128

    # Test Redis field conversion
    redis_fields = schema.redis_fields
    assert len(redis_fields) == 5

    # Verify all fields can be converted to Redis fields successfully
    for field_name, field in schema.fields.items():
        redis_field = field.as_redis_field()
        assert redis_field.name == field_name


def test_schema_serialization_with_new_attributes():
    """Test schema creation and field attribute handling with INDEXMISSING and INDEXEMPTY attributes."""
    original_schema_dict = {
        "index": {
            "name": "test-serialization",
            "prefix": "ser",
            "storage_type": "hash",
        },
        "fields": [
            {
                "name": "description",
                "type": "text",
                "attrs": {"index_missing": True, "index_empty": True, "weight": 2.0},
            },
            {
                "name": "categories",
                "type": "tag",
                "attrs": {"index_missing": True, "index_empty": True, "separator": "|"},
            },
            {"name": "score", "type": "numeric", "attrs": {"index_missing": True}},
            {
                "name": "vector_field",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",
                    "dims": 256,
                    "index_missing": True,
                    "m": 24,
                },
            },
        ],
    }

    # Create schema from dict
    schema = IndexSchema.from_dict(original_schema_dict)

    # Verify field attributes are correctly set after creation
    assert schema.fields["description"].attrs.index_missing == True
    assert schema.fields["description"].attrs.index_empty == True
    assert schema.fields["description"].attrs.weight == 2.0

    assert schema.fields["categories"].attrs.index_missing == True
    assert schema.fields["categories"].attrs.index_empty == True
    assert schema.fields["categories"].attrs.separator == "|"

    assert schema.fields["score"].attrs.index_missing == True

    assert schema.fields["vector_field"].attrs.index_missing == True
    assert schema.fields["vector_field"].attrs.dims == 256
    assert schema.fields["vector_field"].attrs.m == 24

    # Test that Redis field conversion works with new attributes
    for field_name, field in schema.fields.items():
        redis_field = field.as_redis_field()
        assert redis_field.name == field_name

    # Test that the schema has the correct number of fields
    assert len(schema.fields) == 4
    assert schema.index.name == "test-serialization"
