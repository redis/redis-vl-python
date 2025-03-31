import warnings
from unittest import mock

import pytest
from redis import Redis as SyncRedis
from redis.asyncio import Redis as AsyncRedis

from redisvl.exceptions import RedisModuleVersionError, RedisSearchError
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.query import FilterQuery
from redisvl.redis.utils import convert_bytes
from redisvl.schema import IndexSchema, StorageType

fields = [{"name": "test", "type": "tag"}]


@pytest.fixture
def index_schema():
    return IndexSchema.from_dict({"index": {"name": "my_index"}, "fields": fields})


@pytest.fixture
def async_index(index_schema, async_client):
    return AsyncSearchIndex(schema=index_schema, redis_client=async_client)


@pytest.fixture
def async_index_from_dict():
    return AsyncSearchIndex.from_dict({"index": {"name": "my_index"}, "fields": fields})


@pytest.fixture
def async_index_from_yaml():
    return AsyncSearchIndex.from_yaml("schemas/test_json_schema.yaml")


def test_search_index_properties(index_schema, async_index):
    assert async_index.schema == index_schema
    # custom settings
    assert async_index.name == index_schema.index.name == "my_index"
    assert async_index.client
    # default settings
    assert async_index.prefix == index_schema.index.prefix == "rvl"
    assert async_index.key_separator == index_schema.index.key_separator == ":"
    assert (
        async_index.storage_type == index_schema.index.storage_type == StorageType.HASH
    )
    assert async_index.key("foo").startswith(async_index.prefix)


def test_search_index_from_yaml(async_index_from_yaml):
    assert async_index_from_yaml.name == "json-test"
    assert async_index_from_yaml.client is None
    assert async_index_from_yaml.prefix == "json"
    assert async_index_from_yaml.key_separator == ":"
    assert async_index_from_yaml.storage_type == StorageType.JSON
    assert async_index_from_yaml.key("foo").startswith(async_index_from_yaml.prefix)


def test_search_index_from_dict(async_index_from_dict):
    assert async_index_from_dict.name == "my_index"
    assert async_index_from_dict.client is None
    assert async_index_from_dict.prefix == "rvl"
    assert async_index_from_dict.key_separator == ":"
    assert async_index_from_dict.storage_type == StorageType.HASH
    assert len(async_index_from_dict.schema.fields) == len(fields)
    assert async_index_from_dict.key("foo").startswith(async_index_from_dict.prefix)


@pytest.mark.asyncio
async def test_search_index_from_existing(async_client, async_index):
    await async_index.create(overwrite=True)

    try:
        async_index2 = await AsyncSearchIndex.from_existing(
            async_index.name, redis_client=async_client
        )
    except Exception as e:
        pytest.skip(str(e))

    assert async_index2.schema == async_index.schema


@pytest.mark.asyncio
async def test_search_index_from_existing_url(async_index, redis_url):
    await async_index.create(overwrite=True)

    try:
        async_index2 = await AsyncSearchIndex.from_existing(
            async_index.name, redis_url=redis_url
        )
    except Exception as e:
        pytest.skip(str(e))

    assert async_index2.schema == async_index.schema


@pytest.mark.asyncio
async def test_search_index_from_existing_complex(async_client):
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
    async_index = AsyncSearchIndex.from_dict(schema, redis_client=async_client)
    await async_index.create(overwrite=True)

    try:
        async_index2 = await AsyncSearchIndex.from_existing(
            async_index.name, redis_client=async_client
        )
    except Exception as e:
        pytest.skip(str(e))

    assert async_index2.schema == async_index.schema


def test_search_index_no_prefix(index_schema):
    # specify an explicitly empty prefix...
    index_schema.index.prefix = ""
    async_index = AsyncSearchIndex(schema=index_schema)
    assert async_index.prefix == ""
    assert async_index.key("foo") == "foo"


@pytest.mark.asyncio
async def test_search_index_redis_url(redis_url, index_schema):
    async_index = AsyncSearchIndex(schema=index_schema, redis_url=redis_url)
    # Client is None until a command is run
    assert async_index.client is None

    # Lazily create the client by running a command
    await async_index.create(overwrite=True, drop=True)
    assert async_index.client

    await async_index.disconnect()
    assert async_index.client is None


@pytest.mark.asyncio
async def test_search_index_client(async_client, index_schema):
    async_index = AsyncSearchIndex(schema=index_schema, redis_client=async_client)
    assert async_index.client == async_client


@pytest.mark.asyncio
async def test_search_index_set_client(client, redis_url, index_schema):
    async_index = AsyncSearchIndex(schema=index_schema, redis_url=redis_url)
    # Ignore deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        await async_index.create(overwrite=True, drop=True)
        assert isinstance(async_index.client, AsyncRedis)

        # Tests deprecated sync -> async conversion behavior
        assert isinstance(client, SyncRedis)
        await async_index.set_client(client)
        assert isinstance(async_index.client, AsyncRedis)

    await async_index.disconnect()
    assert async_index.client is None


@pytest.mark.asyncio
async def test_search_index_create(async_index):
    await async_index.create(overwrite=True, drop=True)
    assert await async_index.exists()
    assert async_index.name in convert_bytes(
        await async_index.client.execute_command("FT._LIST")
    )


@pytest.mark.asyncio
async def test_search_index_delete(async_index):
    await async_index.create(overwrite=True, drop=True)
    await async_index.delete(drop=True)
    assert not await async_index.exists()
    assert async_index.name not in convert_bytes(
        await async_index.client.execute_command("FT._LIST")
    )


@pytest.mark.asyncio
async def test_search_index_clear(async_index):
    await async_index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}]
    await async_index.load(data, id_field="id")

    count = await async_index.clear()
    assert count == len(data)
    assert await async_index.exists()


@pytest.mark.asyncio
async def test_search_index_drop_key(async_index):
    await async_index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    keys = await async_index.load(data, id_field="id")

    dropped = await async_index.drop_keys(keys[0])
    assert dropped == 1
    assert not await async_index.fetch(keys[0])
    assert await async_index.fetch(keys[1]) is not None


@pytest.mark.asyncio
async def test_search_index_drop_keys(async_index):
    await async_index.create(overwrite=True, drop=True)
    data = [
        {"id": "1", "test": "foo"},
        {"id": "2", "test": "bar"},
        {"id": "3", "test": "baz"},
    ]
    keys = await async_index.load(data, id_field="id")

    dropped = await async_index.drop_keys(keys[0:2])
    assert dropped == 2
    assert not await async_index.fetch(keys[0])
    assert not await async_index.fetch(keys[1])
    assert await async_index.fetch(keys[2]) is not None

    assert await async_index.exists()


@pytest.mark.asyncio
async def test_search_index_load_and_fetch(async_index):
    await async_index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}]
    await async_index.load(data, id_field="id")

    res = await async_index.fetch("1")
    assert (
        res["test"]
        == convert_bytes(await async_index.client.hget("rvl:1", "test"))
        == "foo"
    )

    await async_index.delete(drop=True)
    assert not await async_index.exists()
    assert not await async_index.fetch("1")


@pytest.mark.asyncio
async def test_search_index_load_preprocess(async_index):
    await async_index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}]

    async def preprocess(record):
        record["test"] = "bar"
        return record

    await async_index.load(data, id_field="id", preprocess=preprocess)
    res = await async_index.fetch("1")
    assert (
        res["test"]
        == convert_bytes(await async_index.client.hget("rvl:1", "test"))
        == "bar"
    )

    async def bad_preprocess(record):
        return 1

    with pytest.raises(TypeError):
        await async_index.load(data, id_field="id", preprocess=bad_preprocess)


@pytest.mark.asyncio
async def test_search_index_load_empty(async_index):
    await async_index.create(overwrite=True, drop=True)
    await async_index.load([])


@pytest.mark.asyncio
async def test_no_id_field(async_index):
    await async_index.create(overwrite=True, drop=True)
    bad_data = [{"wrong_key": "1", "value": "test"}]

    # catch missing / invalid id_field
    with pytest.raises(ValueError):
        await async_index.load(bad_data, id_field="key")


@pytest.mark.asyncio
async def test_check_index_exists_before_delete(async_index):
    await async_index.create(overwrite=True, drop=True)
    await async_index.delete(drop=True)
    with pytest.raises(RedisSearchError):
        await async_index.delete()


@pytest.mark.asyncio
async def test_check_index_exists_before_search(async_index):
    await async_index.create(overwrite=True, drop=True)
    await async_index.delete(drop=True)

    query = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        num_results=7,
    )
    with pytest.raises(RedisSearchError):
        await async_index.search(query.query, query_params=query.params)


@pytest.mark.asyncio
async def test_check_index_exists_before_info(async_index):
    await async_index.create(overwrite=True, drop=True)
    await async_index.delete(drop=True)

    with pytest.raises(RedisSearchError):
        await async_index.info()


@pytest.mark.asyncio
async def test_search_index_that_does_not_own_client_context_manager(async_index):
    async with async_index:
        await async_index.create(overwrite=True, drop=True)
        assert async_index._redis_client
        client = async_index._redis_client
    assert async_index._redis_client == client


@pytest.mark.asyncio
async def test_search_index_that_does_not_own_client_context_manager_with_exception(
    async_index,
):
    try:
        async with async_index:
            await async_index.create(overwrite=True, drop=True)
            client = async_index._redis_client
            raise ValueError("test")
    except ValueError:
        pass
    assert async_index._redis_client == client


@pytest.mark.asyncio
async def test_search_index_that_does_not_own_client_disconnect(async_index):
    await async_index.create(overwrite=True, drop=True)
    client = async_index._redis_client
    await async_index.disconnect()
    assert async_index._redis_client == client


@pytest.mark.asyncio
async def test_search_index_that_does_not_own_client_disconnect_sync(async_index):
    await async_index.create(overwrite=True, drop=True)
    client = async_index._redis_client
    async_index.disconnect_sync()
    assert async_index._redis_client == client


@pytest.mark.asyncio
async def test_search_index_that_owns_client_context_manager(index_schema, redis_url):
    async_index = AsyncSearchIndex(schema=index_schema, redis_url=redis_url)
    async with async_index:
        await async_index.create(overwrite=True, drop=True)
        assert async_index._redis_client
    assert async_index._redis_client is None


@pytest.mark.asyncio
async def test_search_index_that_owns_client_context_manager_with_exception(
    index_schema, redis_url
):
    async_index = AsyncSearchIndex(schema=index_schema, redis_url=redis_url)
    try:
        async with async_index:
            await async_index.create(overwrite=True, drop=True)
            raise ValueError("test")
    except ValueError:
        pass
    assert async_index._redis_client is None


@pytest.mark.asyncio
async def test_search_index_that_owns_client_disconnect(index_schema, redis_url):
    async_index = AsyncSearchIndex(schema=index_schema, redis_url=redis_url)
    await async_index.create(overwrite=True, drop=True)
    await async_index.disconnect()
    assert async_index._redis_client is None


@pytest.mark.asyncio
async def test_search_index_that_owns_client_disconnect_sync(index_schema, redis_url):
    async_index = AsyncSearchIndex(schema=index_schema, redis_url=redis_url)
    await async_index.create(overwrite=True, drop=True)
    await async_index.disconnect()
    assert async_index._redis_client is None


@pytest.mark.asyncio
async def test_async_search_index_validates_redis_modules(redis_url):
    """
    A regression test for RAAE-694: we should validate that a passed-in
    Redis client has the correct modules installed.
    """
    client = AsyncRedis.from_url(redis_url)
    with mock.patch(
        "redisvl.index.index.RedisConnectionFactory.validate_async_redis"
    ) as mock_validate_async_redis:
        mock_validate_async_redis.side_effect = RedisModuleVersionError(
            "Required modules not installed"
        )
        with pytest.raises(RedisModuleVersionError):
            index = AsyncSearchIndex(
                schema=IndexSchema.from_dict(
                    {"index": {"name": "my_index"}, "fields": fields}
                ),
                redis_client=client,
            )
            await index.create(overwrite=True, drop=True)

        mock_validate_async_redis.assert_called_once()


@pytest.mark.asyncio
async def test_batch_search(async_index):
    await async_index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    await async_index.load(data, id_field="id")

    results = await async_index.batch_search(["@test:{foo}", "@test:{bar}"])
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
@pytest.mark.asyncio
async def test_batch_search_with_multiple_batches(async_index, queries):
    await async_index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    await async_index.load(data, id_field="id")

    results = await async_index.batch_search(queries[0])
    assert len(results) == 2
    assert results[0].total == 1
    assert results[0].docs[0]["id"] == "rvl:1"
    assert results[1].total == 1
    assert results[1].docs[0]["id"] == "rvl:2"

    results = await async_index.batch_search(
        queries[1],
        batch_size=2,
    )
    assert len(results) == 6

    # First (and only) result for the first query
    assert results[0].total == 1
    assert results[0].docs[0]["id"] == "rvl:1"

    # Second (and only) result for the second query
    assert results[1].total == 1
    assert results[1].docs[0]["id"] == "rvl:2"

    # Third query should have zero results because there is no baz
    assert results[2].total == 0

    # Then the pattern repeats
    assert results[3].total == 1
    assert results[3].docs[0]["id"] == "rvl:1"
    assert results[4].total == 1
    assert results[4].docs[0]["id"] == "rvl:2"
    assert results[5].total == 0


@pytest.mark.asyncio
async def test_batch_query(async_index):
    await async_index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    await async_index.load(data, id_field="id")

    query = FilterQuery(filter_expression="@test:{foo}")
    results = await async_index.batch_query([query])

    assert len(results) == 1
    assert results[0][0]["id"] == "rvl:1"


@pytest.mark.asyncio
async def test_batch_query_with_multiple_batches(async_index):
    await async_index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}, {"id": "2", "test": "bar"}]
    await async_index.load(data, id_field="id")

    queries = [
        FilterQuery(filter_expression="@test:{foo}"),
        FilterQuery(filter_expression="@test:{bar}"),
    ]
    results = await async_index.batch_query(queries, batch_size=1)
    assert len(results) == 2
    assert results[0][0]["id"] == "rvl:1"
    assert results[1][0]["id"] == "rvl:2"
