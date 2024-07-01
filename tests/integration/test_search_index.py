import pytest

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.redis.connection import RedisConnectionFactory, validate_modules
from redisvl.redis.utils import convert_bytes
from redisvl.schema import IndexSchema, StorageType

fields = [{"name": "test", "type": "tag"}]


@pytest.fixture
def index_schema():
    return IndexSchema.from_dict({"index": {"name": "my_index"}, "fields": fields})


@pytest.fixture
def index(index_schema):
    return SearchIndex(schema=index_schema)


@pytest.fixture
def index_from_dict():
    return SearchIndex.from_dict({"index": {"name": "my_index"}, "fields": fields})


@pytest.fixture
def index_from_yaml():
    return SearchIndex.from_yaml("schemas/test_json_schema.yaml")


def test_search_index_properties(index_schema, index):
    assert index.schema == index_schema
    # custom settings
    assert index.name == index_schema.index.name == "my_index"
    assert index.client == None
    # default settings
    assert index.prefix == index_schema.index.prefix == "rvl"
    assert index.key_separator == index_schema.index.key_separator == ":"
    assert index.storage_type == index_schema.index.storage_type == StorageType.HASH
    assert index.key("foo").startswith(index.prefix)


def test_search_index_from_yaml(index_from_yaml):
    assert index_from_yaml.name == "json-test"
    assert index_from_yaml.client == None
    assert index_from_yaml.prefix == "json"
    assert index_from_yaml.key_separator == ":"
    assert index_from_yaml.storage_type == StorageType.JSON
    assert index_from_yaml.key("foo").startswith(index_from_yaml.prefix)


def test_search_index_from_dict(index_from_dict):
    assert index_from_dict.name == "my_index"
    assert index_from_dict.client == None
    assert index_from_dict.prefix == "rvl"
    assert index_from_dict.key_separator == ":"
    assert index_from_dict.storage_type == StorageType.HASH
    assert len(index_from_dict.schema.fields) == len(fields)
    assert index_from_dict.key("foo").startswith(index_from_dict.prefix)


def test_search_index_from_existing(client, index):
    index.set_client(client)
    index.create(overwrite=True)

    try:
        index2 = SearchIndex.from_existing(index.name, redis_client=client)
    except Exception as e:
        pytest.skip(str(e))

    assert index2.schema == index.schema


def test_search_index_from_existing_complex(client):
    schema = {
        "index": {
            "name": "test",
            "prefix": "test",
            "storage_type": "json",
        },
        "fields": [
            {"name": "user", "type": "tag", "path": "$.user"},
            {"name": "credit_score", "type": "tag", "path": "$.metadata.credit_score"},
            {"name": "job", "type": "text", "path": "$.metadata.job"},
            {
                "name": "age",
                "type": "numeric",
                "path": "$.metadata.age",
                "attrs": {"sortable": False},
            },
            {
                "name": "user_embedding",
                "type": "vector",
                "attrs": {
                    "dims": 3,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                },
            },
        ],
    }
    index = SearchIndex.from_dict(schema, redis_client=client)
    index.create(overwrite=True)

    try:
        index2 = SearchIndex.from_existing(index.name, redis_client=client)
    except Exception as e:
        pytest.skip(str(e))

    assert index.schema == index2.schema


def test_search_index_no_prefix(index_schema):
    # specify an explicitly empty prefix...
    index_schema.index.prefix = ""
    index = SearchIndex(schema=index_schema)
    assert index.prefix == ""
    assert index.key("foo") == "foo"


def test_search_index_redis_url(redis_url, index_schema):
    index = SearchIndex(schema=index_schema, redis_url=redis_url)
    assert index.client

    index.disconnect()
    assert index.client == None


def test_search_index_client(client, index_schema):
    index = SearchIndex(schema=index_schema, redis_client=client)
    assert index.client == client


def test_search_index_set_client(async_client, client, index):
    index.set_client(client)
    assert index.client == client
    # should not be able to set the sync client here
    with pytest.raises(TypeError):
        index.set_client(async_client)

    index.disconnect()
    assert index.client == None


def test_search_index_create(client, index):
    index.set_client(client)
    index.create(overwrite=True, drop=True)
    assert index.exists()
    assert index.name in convert_bytes(index.client.execute_command("FT._LIST"))


def test_search_index_delete(client, index):
    index.set_client(client)
    index.create(overwrite=True, drop=True)
    index.delete(drop=True)
    assert not index.exists()
    assert index.name not in convert_bytes(index.client.execute_command("FT._LIST"))


def test_search_index_load_and_fetch(client, index):
    index.set_client(client)
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}]
    index.load(data, id_field="id")

    res = index.fetch("1")
    assert res["test"] == convert_bytes(client.hget("rvl:1", "test")) == "foo"

    index.delete(drop=True)
    assert not index.exists()
    assert not index.fetch("1")


def test_search_index_load_preprocess(client, index):
    index.set_client(client)
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}]

    def preprocess(record):
        record["test"] = "bar"
        return record

    index.load(data, id_field="id", preprocess=preprocess)
    res = index.fetch("1")
    assert res["test"] == convert_bytes(client.hget("rvl:1", "test")) == "bar"

    def bad_preprocess(record):
        return 1

    with pytest.raises(TypeError):
        index.load(data, id_field="id", preprocess=bad_preprocess)


def test_no_id_field(client, index):
    index.set_client(client)
    index.create(overwrite=True, drop=True)
    bad_data = [{"wrong_key": "1", "value": "test"}]

    # catch missing / invalid id_field
    with pytest.raises(ValueError):
        index.load(bad_data, id_field="key")


def test_check_index_exists_before_delete(client, index):
    index.set_client(client)
    index.create(overwrite=True, drop=True)
    index.delete(drop=True)
    with pytest.raises(RuntimeError):
        index.delete()


def test_check_index_exists_before_search(client, index):
    index.set_client(client)
    index.create(overwrite=True, drop=True)
    index.delete(drop=True)

    query = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        num_results=7,
    )
    with pytest.raises(RuntimeError):
        index.search(query.query, query_params=query.params)


def test_check_index_exists_before_info(client, index):
    index.set_client(client)
    index.create(overwrite=True, drop=True)
    index.delete(drop=True)

    with pytest.raises(RuntimeError):
        index.info()


def test_index_needs_valid_schema():
    with pytest.raises(ValueError, match=r"Must provide a valid IndexSchema object"):
        index = SearchIndex(schema="Not A Valid Schema")
