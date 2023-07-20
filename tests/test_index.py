import pytest
import redis
from redis.commands.search.field import TagField

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.utils.utils import convert_bytes

fields = [TagField("test")]


def test_search_index_client(client):
    si = SearchIndex("my_index", fields=fields)
    si.set_client(client)

    assert si.client is not None


def test_search_index_create(client):
    si = SearchIndex("my_index", fields=fields)
    si.set_client(client)
    si.create(overwrite=True)

    assert "my_index" in convert_bytes(si.client.execute_command("FT._LIST"))

    s1_2 = SearchIndex.from_existing(client, "my_index")
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
    si.load(data)

    assert convert_bytes(client.hget(":1", "value")) == "test"


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
    await asi.load(data)
    result = await async_client.hget(":1", "value")
    assert convert_bytes(result) == "test"
    await asi.delete()


# --- Index Errors ----


def test_search_index_delete_nonexistent(client):
    si = SearchIndex("my_index")
    si.set_client(client)
    with pytest.raises(redis.exceptions.ResponseError):
        si.delete()


@pytest.mark.asyncio
async def test_async_search_index_delete_nonexistent(async_client):
    asi = AsyncSearchIndex("my_index")
    asi.set_client(async_client)
    with pytest.raises(redis.exceptions.ResponseError):
        await asi.delete()


# --- Data Errors ----


def test_no_key_field(client):
    si = SearchIndex("my_index", fields=fields, key_field="key")
    si.set_client(client)
    si.create(overwrite=True)
    bad_data = [{"wrong_key": "1", "value": "test"}]

    # TODO make a better error
    with pytest.raises(KeyError):
        si.load(bad_data)


@pytest.mark.asyncio
async def test_async_search_index_load_bad_data(async_client):
    asi = AsyncSearchIndex("my_index", fields=fields)
    asi.set_client(async_client)
    await asi.create(overwrite=True)

    # dictionary not list of dictionaries
    bad_data = {"wrong_key": "1", "value": "test"}
    with pytest.raises(TypeError):
        await asi.load(bad_data)
