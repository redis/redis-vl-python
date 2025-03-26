import pytest
from pydantic import ValidationError

from redisvl.exceptions import SchemaValidationError
from redisvl.index.storage import BaseStorage, HashStorage, JsonStorage
from redisvl.schema import IndexSchema


@pytest.fixture
def sample_hash_schema():
    """Create a sample schema with HASH storage for testing."""
    schema_dict = {
        "index": {
            "name": "test-hash-index",
            "prefix": "test",
            "key_separator": ":",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "test_id", "type": "tag"},
            {"name": "title", "type": "text"},
            {"name": "user", "type": "tag"},
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
        ],
    }
    return IndexSchema.from_dict(schema_dict)


@pytest.fixture
def sample_json_schema():
    """Create a sample schema with JSON storage for testing."""
    schema_dict = {
        "index": {
            "name": "test-json-index",
            "prefix": "test",
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {"name": "test_id", "type": "tag"},
            {"name": "user", "type": "tag"},
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
        ],
    }
    return IndexSchema.from_dict(schema_dict)


@pytest.fixture(params=[JsonStorage, HashStorage])
def storage_instance(request, sample_hash_schema, sample_json_schema):
    StorageClass = request.param
    if isinstance(StorageClass, JsonStorage):
        return StorageClass(index_schema=sample_json_schema)
    return StorageClass(index_schema=sample_hash_schema)


def test_key_formatting(storage_instance):
    key = "1234"
    generated_key = storage_instance._key(key, "", "")
    assert generated_key == key, "The generated key does not match the expected format."
    generated_key = storage_instance._key(key, "", ":")
    assert generated_key == key, "The generated key does not match the expected format."
    generated_key = storage_instance._key(key, "test", ":")
    assert (
        generated_key == f"test:{key}"
    ), "The generated key does not match the expected format."


def test_create_key(storage_instance):
    id_field = "id"
    obj = {id_field: "1234"}
    expected_key = f"{storage_instance.index_schema.index.prefix}{storage_instance.index_schema.index.key_separator}{obj[id_field]}"
    generated_key = storage_instance._create_key(obj, id_field)
    assert (
        generated_key == expected_key
    ), "The generated key does not match the expected format."


def test_validate_success(storage_instance):
    try:
        storage_instance._validate(
            {"test_id": "1234", "rating": 5, "user": "john", "title": "engineer"}
        )
    except Exception as e:
        pytest.fail(f"_validate should not raise an exception here, but raised {e}")


def test_validate_failure(storage_instance):
    data = {"title": 5}
    with pytest.raises(ValidationError):
        storage_instance._validate(data)

    data = {"user": [1]}
    with pytest.raises(ValidationError):
        storage_instance._validate(data)


def test_validate_preprocess_and_validate_failure(storage_instance):
    data = {"title": 5}
    data == storage_instance._preprocess_and_validate_objects(
        objects=[data], validate=False
    )
    with pytest.raises(SchemaValidationError):
        storage_instance._preprocess_and_validate_objects(objects=[data], validate=True)

    data = {"user": [1]}
    data == storage_instance._preprocess_and_validate_objects(
        objects=[data], validate=False
    )
    with pytest.raises(SchemaValidationError):
        storage_instance._preprocess_and_validate_objects(objects=[data], validate=True)


def test_preprocess(storage_instance):
    data = {"key": "value"}
    preprocessed_data = storage_instance._preprocess(obj=data, preprocess=None)
    assert preprocessed_data == data

    def fn(d):
        d["foo"] = "bar"
        return d

    preprocessed_data = storage_instance._preprocess(obj=data, preprocess=fn)
    assert "foo" in preprocessed_data
    assert preprocessed_data["foo"] == "bar"
