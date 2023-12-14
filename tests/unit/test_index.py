import pytest

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.schema.fields import TagField
from redisvl.utils.utils import convert_bytes


fields = {"tag": [TagField(name="test")]}


@pytest.fixture
def index_schema():
    return IndexSchema(name="my_index", fields=fields)

@pytest.fixture
def index(index_schema):
    return SearchIndex(schema=index_schema)


def test_search_index_get_key(index):
    si = index
    key = si.key("foo")
    assert key.startswith(si.prefix)
    assert "foo" in key
    key = si._storage._create_key({"id": "foo"})
    assert key.startswith(si.prefix)
    assert "foo" not in key


def test_search_index_no_prefix(index_schema):
    # specify None as the prefix...
    si = index_schema.prefix = ""
    si = SearchIndex(schema=index_schema)
    key = si.key("foo")
    assert not si.prefix
    assert key == "foo"


def test_search_index_client(client, index_schema):
    si = index_schema.prefix = ""
    si = SearchIndex(schema=index_schema)
    si.set_client(client)

    assert si.client == client
    assert si.aclient == client


def test_search_index_create(client, index, index_schema):
    si = index
    si.set_client(client)
    si.create(overwrite=True)
    assert si.exists()
    assert "my_index" in convert_bytes(si.client.execute_command("FT._LIST"))

    s1_2 = SearchIndex(schema=index_schema)
    assert s1_2.info()["index_name"] == si.info()["index_name"]

    si.create(overwrite=False)
    assert si.exists()
    assert "my_index" in convert_bytes(si.client.execute_command("FT._LIST"))


def test_search_index_delete(client, index):
    si = index
    si.set_client(client)
    si.create(overwrite=True)
    si.delete()
    assert "my_index" not in convert_bytes(si.client.execute_command("FT._LIST"))


def test_search_index_load(client, index):
    si = index
    si.set_client(client)
    si.create(overwrite=True)
    data = [{"id": "1", "value": "test"}]
    si.load(data, key_field="id")

    assert convert_bytes(client.hget("rvl:1", "value")) == "test"


# def test_search_index_load_preprocess(client, index_schema):
#     si = SearchIndex("my_index", fields=fields)
#     si.set_client(client)
#     si.create(overwrite=True)
#     data = [{"id": "1", "value": "test"}]

#     def preprocess(record):
#         record["test"] = "foo"
#         return record

#     si.load(data, key_field="id", preprocess=preprocess)
#     assert convert_bytes(client.hget("rvl:1", "test")) == "foo"

#     def bad_preprocess(record):
#         return 1

#     with pytest.raises(TypeError):
#         si.load(data, key_field="id", preprocess=bad_preprocess)


@pytest.mark.asyncio
async def test_async_search_index_creation(async_client, index):
    asi = index
    asi.set_client(async_client)

    assert asi.aclient == async_client


@pytest.mark.asyncio
async def test_async_search_index_create(async_client, index):
    asi = index
    asi.set_client(async_client)
    await asi.acreate(overwrite=True)

    indices = await asi.aclient.execute_command("FT._LIST")
    assert "my_index" in convert_bytes(indices)


@pytest.mark.asyncio
async def test_async_search_index_delete(async_client, index):
    asi = index
    asi.set_client(async_client)
    await asi.acreate(overwrite=True)
    await asi.adelete()

    indices = await asi.aclient.execute_command("FT._LIST")
    assert "my_index" not in convert_bytes(indices)


# @pytest.mark.asyncio
# async def test_async_search_index_load(async_client):
#     asi = SearchIndex("my_index", fields=fields)
#     asi.set_client(async_client)
#     await asi.acreate(overwrite=True)
#     data = [{"id": "1", "value": "test"}]
#     await asi.aload(data, key_field="id")
#     result = await async_client.hget("rvl:1", "value")
#     assert convert_bytes(result) == "test"
#     await asi.adelete()


# # --- Index Errors ----


# def test_search_index_delete_nonexistent(client):
#     si = SearchIndex("my_index", fields=fields)
#     si.set_client(client)
#     with pytest.raises(ValueError):
#         si.delete()


# @pytest.mark.asyncio
# async def test_async_search_index_delete_nonexistent(async_client):
#     asi = SearchIndex("my_index", fields=fields)
#     asi.set_client(async_client)
#     with pytest.raises(ValueError):
#         await asi.adelete()


# # --- Data Errors ----


# def test_no_key_field(client):
#     si = SearchIndex("my_index", fields=fields)
#     si.set_client(client)
#     si.create(overwrite=True)
#     bad_data = [{"wrong_key": "1", "value": "test"}]

#     # TODO make a better error
#     with pytest.raises(ValueError):
#         si.load(bad_data, key_field="key")


# @pytest.mark.asyncio
# async def test_async_search_index_load_bad_data(async_client):
#     asi = SearchIndex("my_index", fields=fields)
#     asi.set_client(async_client)
#     await asi.acreate(overwrite=True)

#     # dictionary not list of dictionaries
#     bad_data = {"wrong_key": "1", "value": "test"}
#     with pytest.raises(TypeError):
#         await asi.aload(bad_data, key_field="id")
