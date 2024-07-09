import pytest

from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.redis.utils import convert_bytes
from redisvl.schema import IndexSchema, StorageType

fields = [{"name": "test", "type": "tag"}]


@pytest.fixture
def index_schema():
    return IndexSchema.from_dict({"index": {"name": "my_index"}, "fields": fields})


@pytest.fixture
def async_index(index_schema):
    return AsyncSearchIndex(schema=index_schema)


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
    assert async_index.client == None
    # default settings
    assert async_index.prefix == index_schema.index.prefix == "rvl"
    assert async_index.key_separator == index_schema.index.key_separator == ":"
    assert (
        async_index.storage_type == index_schema.index.storage_type == StorageType.HASH
    )
    assert async_index.key("foo").startswith(async_index.prefix)


def test_search_index_from_yaml(async_index_from_yaml):
    assert async_index_from_yaml.name == "json-test"
    assert async_index_from_yaml.client == None
    assert async_index_from_yaml.prefix == "json"
    assert async_index_from_yaml.key_separator == ":"
    assert async_index_from_yaml.storage_type == StorageType.JSON
    assert async_index_from_yaml.key("foo").startswith(async_index_from_yaml.prefix)


def test_search_index_from_dict(async_index_from_dict):
    assert async_index_from_dict.name == "my_index"
    assert async_index_from_dict.client == None
    assert async_index_from_dict.prefix == "rvl"
    assert async_index_from_dict.key_separator == ":"
    assert async_index_from_dict.storage_type == StorageType.HASH
    assert len(async_index_from_dict.schema.fields) == len(fields)
    assert async_index_from_dict.key("foo").startswith(async_index_from_dict.prefix)


@pytest.mark.asyncio
async def test_search_index_from_existing(async_client, async_index):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True)

    try:
        async_index2 = await AsyncSearchIndex.from_existing(
            async_index.name, redis_client=async_client
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


def test_search_index_redis_url(redis_url, index_schema):
    async_index = AsyncSearchIndex(schema=index_schema, redis_url=redis_url)
    assert async_index.client

    async_index.disconnect()
    assert async_index.client == None


def test_search_index_client(async_client, index_schema):
    async_index = AsyncSearchIndex(schema=index_schema, redis_client=async_client)
    assert async_index.client == async_client


def test_search_index_set_client(async_client, client, async_index):
    async_index.set_client(async_client)
    assert async_index.client == async_client
    # should not be able to set the sync client here
    with pytest.raises(TypeError):
        async_index.set_client(client)

    async_index.disconnect()
    assert async_index.client == None


@pytest.mark.asyncio
async def test_search_index_create(async_client, async_index):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True, drop=True)
    assert await async_index.exists()
    assert async_index.name in convert_bytes(
        await async_index.client.execute_command("FT._LIST")
    )


@pytest.mark.asyncio
async def test_search_index_delete(async_client, async_index):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True, drop=True)
    await async_index.delete(drop=True)
    assert not await async_index.exists()
    assert async_index.name not in convert_bytes(
        await async_index.client.execute_command("FT._LIST")
    )


@pytest.mark.asyncio
async def test_search_index_clear(async_client, async_index):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True, drop=True)
    data = [{"id": "1", "test": "foo"}]
    await async_index.load(data, id_field="id")

    count = await async_index.clear()
    assert count == len(data)
    assert await async_index.exists()


@pytest.mark.asyncio
async def test_search_index_load_and_fetch(async_client, async_index):
    async_index.set_client(async_client)
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
async def test_search_index_load_preprocess(async_client, async_index):
    async_index.set_client(async_client)
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
async def test_search_index_load_empty(async_client, async_index):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True, drop=True)
    await async_index.load([])


@pytest.mark.asyncio
async def test_no_id_field(async_client, async_index):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True, drop=True)
    bad_data = [{"wrong_key": "1", "value": "test"}]

    # catch missing / invalid id_field
    with pytest.raises(ValueError):
        await async_index.load(bad_data, id_field="key")


@pytest.mark.asyncio
async def test_check_index_exists_before_delete(async_client, async_index):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True, drop=True)
    await async_index.delete(drop=True)
    with pytest.raises(ValueError):
        await async_index.delete()


@pytest.mark.asyncio
async def test_check_index_exists_before_search(async_client, async_index):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True, drop=True)
    await async_index.delete(drop=True)

    query = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        num_results=7,
    )
    with pytest.raises(ValueError):
        await async_index.search(query.query, query_params=query.params)


@pytest.mark.asyncio
async def test_check_index_exists_before_info(async_client, async_index):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True, drop=True)
    await async_index.delete(drop=True)

    with pytest.raises(ValueError):
        await async_index.info()
