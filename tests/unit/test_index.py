import pytest
import redis

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.utils.utils import convert_bytes


@pytest.fixture
def search_index():
    return SearchIndex.from_dict(
        {
            "index": {
                "name": "my_index",
                "prefix": "rvl"
            },
            "fields": {
                "tag": [{"name": "test"}]
            }
        }
    )

@pytest.fixture
def async_index():
    return AsyncSearchIndex.from_dict(
        {
            "index": {
                "name": "my_index",
                "prefix": "rvl"
            },
            "fields": {
                "tag": [{"name": "test"}]
            }
        }
    )


def test_search_index_get_key(search_index):
    key = search_index.key("foo")
    assert key.startswith(search_index.prefix)
    assert "foo" in key
    key = search_index._storage._create_key({"id": "foo"})
    assert key.startswith(search_index.prefix)
    assert "foo" not in key


# def test_search_index_no_prefix():
#     # specify None as the prefix...
#     si = SearchIndex("my_index", prefix=None, fields=fields)
#     key = si.key("foo")
#     assert not si.prefix
#     assert key == "foo"


def test_search_index_client(search_index, client):
    search_index.set_client(client)
    assert search_index.client is not None


def test_search_index_create(search_index, client, redis_url):
    search_index.set_client(client)
    search_index.create(overwrite=True)
    assert search_index.exists()
    assert "my_index" in search_index.list_all()

    existing = SearchIndex.from_existing("my_index", url=redis_url)
    assert existing.info()["index_name"] == search_index.info()["index_name"]

    search_index.create(overwrite=False)
    assert search_index.exists()
    assert "my_index" in search_index.list_all()


def test_search_index_delete(search_index, client):
    search_index.set_client(client)
    search_index.create(overwrite=True)
    search_index.delete()
    assert not search_index.exists()
    assert "my_index" not in search_index.list_all()


def test_search_index_load(search_index, client):
    search_index.set_client(client)
    search_index.create(overwrite=True)
    data = [{"id": "1", "value": "test"}]
    search_index.load(data, key_field="id")

    assert convert_bytes(client.hget("rvl:1", "value")) == "test"


def test_search_index_load_preprocess(search_index, client):
    search_index.set_client(client)
    search_index.create(overwrite=True)
    data = [{"id": "1", "value": "test"}]

    def preprocess(record):
        record["test"] = "foo"
        return record

    search_index.load(data, key_field="id", preprocess=preprocess)
    assert convert_bytes(client.hget("rvl:1", "test")) == "foo"

    def bad_preprocess(record):
        return 1

    with pytest.raises(TypeError):
        search_index.load(data, key_field="id", preprocess=bad_preprocess)


@pytest.mark.asyncio
async def test_async_search_index_creation(async_index, async_client):
    async_index.set_client(async_client)
    assert async_index.client == async_client


@pytest.mark.asyncio
async def test_async_search_index_create(async_index, async_client):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True)
    assert await async_index.exists()
    assert "my_index" in await async_index.list_all()


@pytest.mark.asyncio
async def test_async_search_index_delete(async_index, async_client):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True)
    await async_index.delete()

    assert not await async_index.exists()
    assert "my_index" not in await async_index.list_all()


@pytest.mark.asyncio
async def test_async_search_index_load(async_index, async_client):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True)
    data = [{"id": "1", "value": "test"}]
    await async_index.load(data, key_field="id")
    result = await async_client.hget("rvl:1", "value")
    assert convert_bytes(result) == "test"
    await async_index.delete()


# --- Index Errors ----


def test_search_index_delete_nonexistent(search_index, client):
    search_index.set_client(client)
    with pytest.raises(ValueError):
        search_index.delete()


@pytest.mark.asyncio
async def test_async_search_index_delete_nonexistent(async_index, async_client):
    async_index.set_client(async_client)
    with pytest.raises(redis.exceptions.ResponseError):
        await async_index.delete()


# --- Data Errors ----


def test_no_key_field(search_index, client):
    search_index.set_client(client)
    search_index.create(overwrite=True)
    bad_data = [{"wrong_key": "1", "value": "test"}]

    # TODO make a better error
    with pytest.raises(ValueError):
        search_index.load(bad_data, key_field="key")


@pytest.mark.asyncio
async def test_async_search_index_load_bad_data(async_index, async_client):
    async_index.set_client(async_client)
    await async_index.create(overwrite=True)

    # dictionary not list of dictionaries
    bad_data = {"wrong_key": "1", "value": "test"}
    with pytest.raises(TypeError):
        await async_index.load(bad_data, key_field="id")
