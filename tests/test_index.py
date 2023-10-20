import pytest
import redis
from redis.commands.search.field import TagField

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.utils.utils import convert_bytes

fields = [TagField("test")]


def test_search_index_get_key():
    si = SearchIndex("my_index", fields=fields)
    key = si.key("foo")
    assert key.startswith(si._prefix)
    assert "foo" in key
    key = si._create_key({"id": "foo"})
    assert key.startswith(si._prefix)
    assert "foo" not in key


def test_search_index_no_prefix():
    # specify None as the prefix...
    si = SearchIndex("my_index", prefix=None, fields=fields)
    key = si.key("foo")
    assert not si._prefix
    assert key == "foo"


def test_search_index_client(client):
    si = SearchIndex("my_index", fields=fields)
    si.set_client(client)

    assert si.client is not None


def test_search_index_create(client, redis_url):
    si = SearchIndex("my_index", fields=fields)
    si.set_client(client)
    si.create(overwrite=True)
    assert si.exists()
    assert "my_index" in convert_bytes(si.client.execute_command("FT._LIST"))

    s1_2 = SearchIndex.from_existing("my_index", url=redis_url)
    assert s1_2.info()["index_name"] == si.info()["index_name"]


def test_search_index_delete(client):
    si = SearchIndex("my_index", fields=fields)
    si.set_client(client)
    si.create(overwrite=True)
    si.delete()
    assert "my_index" not in convert_bytes(si.client.execute_command("FT._LIST"))


def test_search_index_load(client):
    si = SearchIndex("my_index", fields=fields)
    si.set_client(client)
    si.create(overwrite=True)
    data = [{"id": "1", "value": "test"}]
    si.load(data, key_field="id")

    assert convert_bytes(client.hget("rvl:1", "value")) == "test"


def test_search_index_load_preprocess(client):
    si = SearchIndex("my_index", fields=fields)
    si.set_client(client)
    si.create(overwrite=True)
    data = [{"id": "1", "value": "test"}]

    def preprocess(record):
        record["test"] = "foo"
        return record

    si.load(data, key_field="id", preprocess=preprocess)
    assert convert_bytes(client.hget("rvl:1", "test")) == "foo"

    def bad_preprocess(record):
        return 1

    with pytest.raises(TypeError):
        si.load(data, key_field="id", preprocess=bad_preprocess)


@pytest.mark.asyncio
async def test_async_search_index_creation(async_client):
    asi = AsyncSearchIndex("my_index", fields=fields)
    asi.set_client(async_client)

    assert asi.client == async_client


@pytest.mark.asyncio
async def test_async_search_index_create(async_client):
    asi = AsyncSearchIndex("my_index", fields=fields)
    asi.set_client(async_client)
    await asi.create(overwrite=True)

    indices = await asi.client.execute_command("FT._LIST")
    assert "my_index" in convert_bytes(indices)


@pytest.mark.asyncio
async def test_async_search_index_delete(async_client):
    asi = AsyncSearchIndex("my_index", fields=fields)
    asi.set_client(async_client)
    await asi.create(overwrite=True)
    await asi.delete()

    indices = await asi.client.execute_command("FT._LIST")
    assert "my_index" not in convert_bytes(indices)


@pytest.mark.asyncio
async def test_async_search_index_load(async_client):
    asi = AsyncSearchIndex("my_index", fields=fields)
    asi.set_client(async_client)
    await asi.create(overwrite=True)
    data = [{"id": "1", "value": "test"}]
    await asi.load(data, key_field="id")
    result = await async_client.hget("rvl:1", "value")
    assert convert_bytes(result) == "test"
    await asi.delete()


# --- Index Errors ----


def test_search_index_delete_nonexistent(client):
    si = SearchIndex("my_index", fields=fields)
    si.set_client(client)
    with pytest.raises(redis.exceptions.ResponseError):
        si.delete()


@pytest.mark.asyncio
async def test_async_search_index_delete_nonexistent(async_client):
    asi = AsyncSearchIndex("my_index", fields=fields)
    asi.set_client(async_client)
    with pytest.raises(redis.exceptions.ResponseError):
        await asi.delete()


# --- Data Errors ----


def test_no_key_field(client):
    si = SearchIndex("my_index", fields=fields)
    si.set_client(client)
    si.create(overwrite=True)
    bad_data = [{"wrong_key": "1", "value": "test"}]

    # TODO make a better error
    with pytest.raises(ValueError):
        si.load(bad_data, key_field="key")


@pytest.mark.asyncio
async def test_async_search_index_load_bad_data(async_client):
    asi = AsyncSearchIndex("my_index", fields=fields)
    asi.set_client(async_client)
    await asi.create(overwrite=True)

    # dictionary not list of dictionaries
    bad_data = {"wrong_key": "1", "value": "test"}
    with pytest.raises(TypeError):
        await asi.load(bad_data, key_field="id")
