import warnings
from unittest import mock

import pytest
from redis import Redis

from redisvl.exceptions import QueryValidationError, RedisSearchError, RedisVLError
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.query import FilterQuery
from redisvl.redis.utils import convert_bytes
from redisvl.schema import IndexSchema, StorageType
from redisvl.schema.fields import VectorIndexAlgorithm

fields = [
    {"name": "test", "type": "tag"},
    {"name": "test_text", "type": "text"},
    {
        "name": "test_text_attrs",
        "type": "text",
        "attrs": {"no_stem": True, "sortable": True},
    },
    {"name": "test_tag", "type": "tag", "attrs": {"case_sensitive": True}},
    {"name": "test_numeric", "type": "numeric"},
    {"name": "test_numeric_attrs", "type": "numeric", "attrs": {"sortable": True}},
]


@pytest.fixture
def index_schema(worker_id):
    return IndexSchema.from_dict(
        {"index": {"name": f"my_index_{worker_id}"}, "fields": fields}
    )


@pytest.fixture
def index(index_schema, client):
    return SearchIndex(schema=index_schema, redis_client=client)


@pytest.fixture
def index_from_dict(worker_id):

    return SearchIndex.from_dict(
        {
            "index": {"name": f"my_index_{worker_id}", "prefix": f"rvl_{worker_id}"},
            "fields": fields,
        }
    )


@pytest.fixture
def index_from_yaml(worker_id):

    index = SearchIndex.from_yaml("schemas/test_json_schema.yaml")
    # Update the index name and prefix to include worker_id
    index.schema.index.name = f"{index.schema.index.name}_{worker_id}"
    index.schema.index.prefix = f"{index.schema.index.prefix}_{worker_id}"
    return index


def test_search_index_properties(index_schema, index):
    assert index.schema == index_schema
    # custom settings
    assert index.name == index_schema.index.name
    assert index.name.startswith("my_index_")

    # default settings
    assert index.prefix == index_schema.index.prefix == "rvl"
    assert index.key_separator == index_schema.index.key_separator == ":"
    assert index.storage_type == index_schema.index.storage_type == StorageType.HASH
    assert index.key("foo").startswith(index.prefix)


def test_search_index_from_yaml(index_from_yaml):
    assert index_from_yaml.name.startswith("json-test")
    assert index_from_yaml.client == None
    assert index_from_yaml.prefix.startswith("json_")
    assert index_from_yaml.key_separator == ":"
    assert index_from_yaml.storage_type == StorageType.JSON
    assert index_from_yaml.key("foo").startswith(index_from_yaml.prefix)


def test_search_index_from_dict(index_from_dict):
    assert index_from_dict.name.startswith("my_index")
    assert index_from_dict.client == None
    assert index_from_dict.prefix.startswith("rvl_")
    assert index_from_dict.key_separator == ":"
    assert index_from_dict.storage_type == StorageType.HASH
    assert len(index_from_dict.schema.fields) == len(fields)
    assert index_from_dict.key("foo").startswith(index_from_dict.prefix)


def test_search_index_from_existing(index, client):
    index.create(overwrite=True)

    try:
        index2 = SearchIndex.from_existing(index.name, redis_client=client)
    except Exception as e:
        pytest.skip(str(e))

    assert index2.schema == index.schema


def test_search_index_from_existing_url(redis_url, index):
    index.create(overwrite=True)

    try:
        index2 = SearchIndex.from_existing(index.name, redis_url=redis_url)
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
                "attrs": {"sortable": True},
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

    # Verify index metadata matches
    assert index2.schema.index.name == index.schema.index.name
    assert index2.schema.index.prefix == index.schema.index.prefix
    assert index2.schema.index.storage_type == index.schema.index.storage_type

    # Verify non-vector fields are present
    for field_name in ["user", "credit_score", "job", "age"]:
        assert field_name in index2.schema.fields
        assert (
            index2.schema.fields[field_name].type
            == index.schema.fields[field_name].type
        )

    # Vector field may not be present on older Redis versions
    if "user_embedding" in index2.schema.fields:
        assert index2.schema.fields["user_embedding"].type == "vector"


def test_search_index_from_existing_multiple_prefixes(client):
    """Test that from_existing correctly handles indices with multiple prefixes (issue #258)."""
    from redis.commands.search.field import TextField, VectorField

    index_name = "test_multi_prefix"

    # Create index manually using redis-py with multiple prefixes
    # This simulates an index created with: FT.CREATE index ON HASH PREFIX 3 prefix_a: prefix_b: prefix_c: ...
    try:
        # Clean up any existing index
        try:
            client.ft(index_name).dropindex(delete_documents=True)
        except Exception:
            pass

        # Create index using raw FT.CREATE command with multiple prefixes
        # FT.CREATE index ON HASH PREFIX 3 prefix_a: prefix_b: prefix_c: SCHEMA user TAG text TEXT ...
        client.execute_command(
            "FT.CREATE",
            index_name,
            "ON",
            "HASH",
            "PREFIX",
            "3",
            "prefix_a:",
            "prefix_b:",
            "prefix_c:",
            "SCHEMA",
            "user",
            "TAG",
            "text",
            "TEXT",
            "embedding",
            "VECTOR",
            "FLAT",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            "3",
            "DISTANCE_METRIC",
            "COSINE",
        )

        # Now test from_existing - this is where the bug was
        loaded_index = SearchIndex.from_existing(index_name, redis_client=client)

        # Verify all prefixes are preserved (this was failing before fix)
        # Before the fix, only "prefix_a:" would be returned
        assert loaded_index.schema.index.prefix == [
            "prefix_a:",
            "prefix_b:",
            "prefix_c:",
        ], "Multiple prefixes should be preserved when loading existing index"

        # Verify the index name and storage type
        assert loaded_index.schema.index.name == index_name
        assert loaded_index.schema.index.storage_type.value == "hash"

        # Verify TAG and TEXT fields are present
        assert "user" in loaded_index.schema.fields
        assert "text" in loaded_index.schema.fields

        # Verify vector field if present
        if "embedding" in loaded_index.schema.fields:
            assert loaded_index.schema.fields["embedding"].type == "vector"

    finally:
        # Cleanup
        try:
            client.ft(index_name).dropindex(delete_documents=True)
        except Exception:
            pass


def test_search_index_no_prefix(index_schema):
    # specify an explicitly empty prefix...
    index_schema.index.prefix = ""
    index = SearchIndex(schema=index_schema)
    assert index.prefix == ""
    assert index.key("foo") == "foo"


def test_search_index_redis_url(redis_url, index_schema):
    index = SearchIndex(schema=index_schema, redis_url=redis_url)
    # Client is not set until a command runs
    assert index.client is None

    index.create(overwrite=True)
    assert index.client

    index.disconnect()
    assert index.client is None


def test_search_index_client(client, index_schema):
    index = SearchIndex(schema=index_schema, redis_client=client)
    assert index.client == client


def test_search_index_set_client(async_client, redis_url, index_schema):
    index = SearchIndex(schema=index_schema, redis_url=redis_url)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        index.create(overwrite=True, drop=True)
        assert index.client
        # should not be able to set an async client here
        with pytest.raises(TypeError):
            index.set_client(async_client)
        assert index.client is not async_client

        index.disconnect()
        assert index.client is None


def test_search_index_connect(index, redis_url):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        index.connect(redis_url=redis_url)
    assert index.client


def test_search_index_create(index):
    index.create(overwrite=True, drop=True)
    assert index.exists()
    assert index.name in convert_bytes(index.client.execute_command("FT._LIST"))


def test_search_index_delete(index):
    index.create(overwrite=True, drop=True)
    index.delete(drop=True)
    assert not index.exists()
    assert index.name not in convert_bytes(index.client.execute_command("FT._LIST"))


def test_search_index_clear(index):
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}]
    index.load(data, id_field="id")

    count = index.clear()
    assert count == len(data)
    assert index.exists()


def test_search_index_drop_key(index):
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    keys = index.load(data, id_field="id")

    # test passing a single string key removes only that key
    dropped = index.drop_keys(keys[0])
    assert dropped == 1
    assert not index.fetch(keys[0])
    assert index.fetch(keys[1]) is not None  # still have all other entries


def test_search_index_drop_keys(index):
    index.create(overwrite=True, drop=True)
    data = [
        {"id": "1", "test": "foo"},
        {"id": "2", "test": "bar"},
        {"id": "3", "test": "baz"},
    ]
    keys = index.load(data, id_field="id")

    # test passing a list of keys selectively removes only those keys
    dropped = index.drop_keys(keys[0:2])
    assert dropped == 2
    assert not index.fetch(keys[0])
    assert not index.fetch(keys[1])
    assert index.fetch(keys[2]) is not None

    assert index.exists()


def test_search_index_drop_documents(index):
    index.create(overwrite=True, drop=True)
    data = [
        {"id": "1", "test": "foo"},
        {"id": "2", "test": "bar"},
        {"id": "3", "test": "baz"},
    ]
    index.load(data, id_field="id")

    # Test dropping a single document by ID
    dropped = index.drop_documents("1")
    assert dropped == 1
    assert not index.fetch("1")
    assert index.fetch("2") is not None
    assert index.fetch("3") is not None

    # Test dropping multiple documents by ID
    dropped = index.drop_documents(["2", "3"])
    assert dropped == 2
    assert not index.fetch("2")
    assert not index.fetch("3")

    # Test dropping with an empty list
    dropped = index.drop_documents([])
    assert dropped == 0

    # Ensure the index still exists
    assert index.exists()


def test_search_index_load_and_fetch(index):
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}]
    index.load(data, id_field="id")

    res = index.fetch("1")
    assert res["test"] == convert_bytes(index.client.hget("rvl:1", "test")) == "foo"

    index.delete(drop=True)
    assert not index.exists()
    assert not index.fetch("1")


def test_search_index_load_preprocess(index):
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}]

    def preprocess(record):
        record["test"] = "bar"
        return record

    index.load(data, id_field="id", preprocess=preprocess)
    res = index.fetch("1")
    assert res["test"] == convert_bytes(index.client.hget("rvl:1", "test")) == "bar"

    def bad_preprocess(record):
        return 1

    with pytest.raises(RedisVLError):
        index.load(data, id_field="id", preprocess=bad_preprocess)


def test_no_id_field(index):
    index.create(overwrite=True, drop=True)
    bad_data = [{"wrong_key": "1", "value": "test"}]

    # catch missing / invalid id_field
    with pytest.raises(RedisVLError):
        index.load(bad_data, id_field="key")


def test_check_index_exists_before_delete(index):
    index.create(overwrite=True, drop=True)
    index.delete(drop=True)
    with pytest.raises(RedisSearchError):
        index.delete()


def test_check_index_exists_before_search(index):
    index.create(overwrite=True, drop=True)
    index.delete(drop=True)

    query = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        num_results=7,
    )
    with pytest.raises(RedisSearchError):
        index.search(query.query, query_params=query.params)


def test_check_index_exists_before_info(index):
    index.create(overwrite=True, drop=True)
    index.delete(drop=True)

    with pytest.raises(RedisSearchError):
        index.info()


def test_index_needs_valid_schema():
    with pytest.raises(ValueError, match=r"Must provide a valid IndexSchema object"):
        SearchIndex(schema="Not A Valid Schema")  # type: ignore


def test_search_index_that_does_not_own_client_context_manager(index):
    with index:
        index.create(overwrite=True, drop=True)
        assert index.client
        client = index.client
    # Client should not have changed outside of the context manager
    assert index.client == client


def test_search_index_that_does_not_own_client_context_manager_with_exception(index):
    with pytest.raises(ValueError):
        with index:
            index.create(overwrite=True, drop=True)
            client = index.client
            raise ValueError("test")
    # Client should not have changed outside of the context manager
    assert index.client == client


def test_search_index_that_does_not_own_client_disconnect(index):
    index.create(overwrite=True, drop=True)
    client = index.client
    index.disconnect()
    # Client should not have changed after disconnecting
    assert index.client == client


def test_search_index_that_owns_client_context_manager(index_schema, redis_url):
    index = SearchIndex(schema=index_schema, redis_url=redis_url)
    with index:
        index.create(overwrite=True, drop=True)
        assert index.client
    assert index.client is None


def test_search_index_that_owns_client_context_manager_with_exception(
    index_schema, redis_url
):
    index = SearchIndex(schema=index_schema, redis_url=redis_url)
    with pytest.raises(ValueError):
        with index:
            index.create(overwrite=True, drop=True)
            raise ValueError("test")
    assert index.client is None


def test_search_index_that_owns_client_disconnect(index_schema, redis_url):
    index = SearchIndex(schema=index_schema, redis_url=redis_url)
    index.create(overwrite=True, drop=True)
    index.disconnect()
    assert index.client is None


def test_search_index_no_proactive_module_validation(redis_url):
    """
    Updated test for issue #370: SearchIndex should not validate modules proactively.
    Operations should fail naturally if modules are missing.
    """
    client = Redis.from_url(redis_url)
    with mock.patch(
        "redisvl.index.index.RedisConnectionFactory.validate_sync_redis"
    ) as mock_validate_sync_redis:
        # Create index - validation should only set lib name, not check modules
        index = SearchIndex(
            schema=IndexSchema.from_dict(
                {"index": {"name": "my_index"}, "fields": fields}
            ),
            redis_client=client,
        )

        # Access client to trigger lazy init
        _ = index._redis_client

        # validate_sync_redis might be called to set lib name, but won't raise module errors
        # The actual operation (create) will succeed if modules are present
        index.create(overwrite=True, drop=True)

        # Verify index was created successfully (modules are present in test env)
        assert index.exists()


def test_batch_search(index):
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    index.load(data, id_field="id")

    results = index.batch_search(["@test:{foo}", "@test:{bar}"])
    assert len(results) == 2
    assert results[0].total == 1
    assert results[0].docs[0]["id"] == "rvl:1"
    assert results[1].total == 1
    assert results[1].docs[0]["id"] == "rvl:2"


@pytest.mark.parametrize(
    "queries",
    [
        [
            [
                FilterQuery(filter_expression="@test:{foo}"),
                FilterQuery(filter_expression="@test:{bar}"),
            ],
            [
                FilterQuery(filter_expression="@test:{foo}"),
                FilterQuery(filter_expression="@test:{bar}"),
                FilterQuery(filter_expression="@test:{baz}"),
                FilterQuery(filter_expression="@test:{foo}"),
                FilterQuery(filter_expression="@test:{bar}"),
                FilterQuery(filter_expression="@test:{baz}"),
            ],
        ],
        [
            [
                "@test:{foo}",
                "@test:{bar}",
            ],
            [
                "@test:{foo}",
                "@test:{bar}",
                "@test:{baz}",
                "@test:{foo}",
                "@test:{bar}",
                "@test:{baz}",
            ],
        ],
    ],
)
def test_batch_search_with_multiple_batches(index, queries):
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    index.load(data, id_field="id")

    results = index.batch_search(queries[0])
    assert len(results) == 2
    assert results[0].total == 1
    assert results[0].docs[0]["id"] == "rvl:1"
    assert results[1].total == 1
    assert results[1].docs[0]["id"] == "rvl:2"

    results = index.batch_search(
        queries[1],
        batch_size=2,
    )
    assert len(results) == 6

    # First (and only) result for the first query
    assert results[0].docs[0]["id"] == "rvl:1"

    # Second (and only) result for the second query
    assert results[1].docs[0]["id"] == "rvl:2"

    # Third query should have zero results because there is no baz
    assert results[2].total == 0

    # Then the pattern repeats
    assert results[3].docs[0]["id"] == "rvl:1"
    assert results[4].docs[0]["id"] == "rvl:2"
    assert results[5].total == 0


def test_batch_query(index):
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    index.load(data, id_field="id")

    query = FilterQuery(filter_expression="@test:{foo}")
    results = index.batch_query([query])

    assert len(results) == 1
    assert results[0][0]["id"] == "rvl:1"


def test_batch_query_with_multiple_batches(index):
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    index.load(data, id_field="id")

    queries = [
        FilterQuery(filter_expression="@test:{foo}"),
        FilterQuery(filter_expression="@test:{bar}"),
    ]
    results = index.batch_query(queries, batch_size=1)
    assert len(results) == 2
    assert results[0][0]["id"] == "rvl:1"
    assert results[1][0]["id"] == "rvl:2"


def test_search_index_expire_keys(index):
    """Test that SearchIndex.expire_keys method properly sets expiration times on keys."""
    # Create the index and load some data
    index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    keys = index.load(data, id_field="id")

    # Set expiration on a single key
    index.expire_keys(keys[0], 60)
    ttl = index.client.ttl(keys[0])
    assert ttl > 0  # TTL should be positive
    assert ttl <= 60  # TTL should be 60 or less

    # Test no expiration on the other key
    ttl = index.client.ttl(keys[1])
    assert ttl == -1  # -1 means no expiration

    # Set expiration on multiple keys
    result = index.expire_keys(keys, 30)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(r == 1 for r in result)  # All operations should return 1 (success)

    # Verify TTLs are set
    for key in keys:
        ttl = index.client.ttl(key)
        assert ttl > 0
        assert ttl <= 30


def test_search_index_validates_query_with_flat_algorithm(flat_index, sample_data):
    assert (
        flat_index.schema.fields["user_embedding"].attrs.algorithm
        == VectorIndexAlgorithm.FLAT
    )
    query = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        num_results=7,
        ef_runtime=100,
    )
    with pytest.raises(QueryValidationError):
        flat_index.query(query)


def test_search_index_validates_query_with_hnsw_algorithm(hnsw_index, sample_data):
    assert (
        hnsw_index.schema.fields["user_embedding"].attrs.algorithm
        == VectorIndexAlgorithm.HNSW
    )
    query = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        num_results=7,
        ef_runtime=100,
    )
    # Should not raise
    hnsw_index.query(query)
