"""Explicit load keys and data may be single-pass iterables."""

import pytest

from redisvl.exceptions import RedisVLError
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.schema import IndexSchema


@pytest.fixture(params=["hash", "json"])
def load_schema(request, redis_test_name):
    return IndexSchema.from_dict(
        {
            "index": {
                "name": redis_test_name("iterable_load"),
                "prefix": redis_test_name("iterable_load"),
                "storage_type": request.param,
            },
            "fields": [{"name": "title", "type": "text"}],
        }
    )


@pytest.mark.parametrize("data_iterator", [False, True])
@pytest.mark.parametrize("keys_iterator", [False, True])
def test_load_with_iterable_keys(client, load_schema, data_iterator, keys_iterator):
    index = SearchIndex(load_schema, redis_client=client)
    records = [{"title": "first"}, {"title": "second"}]
    keys = [index.key(str(i)) for i in range(len(records))]
    try:
        loaded = index.load(
            iter(records) if data_iterator else records,
            keys=iter(keys) if keys_iterator else keys,
            batch_size=1,
        )
        assert loaded == keys
        assert [index.fetch(str(i)) for i in range(len(records))] == records
    finally:
        client.delete(*keys)


@pytest.mark.parametrize("data_iterator", [False, True])
@pytest.mark.parametrize("keys_iterator", [False, True])
async def test_async_load_with_iterable_keys(
    async_client, load_schema, data_iterator, keys_iterator
):
    index = AsyncSearchIndex(load_schema, redis_client=async_client)
    records = [{"title": "first"}, {"title": "second"}]
    keys = [index.key(str(i)) for i in range(len(records))]
    try:
        loaded = await index.load(
            iter(records) if data_iterator else records,
            keys=iter(keys) if keys_iterator else keys,
            batch_size=1,
        )
        assert loaded == keys
        assert [await index.fetch(str(i)) for i in range(len(records))] == records
    finally:
        await async_client.delete(*keys)


@pytest.mark.parametrize("key_count", [0, 1, 3])
def test_load_rejects_mismatched_keys_before_writing(client, load_schema, key_count):
    index = SearchIndex(load_schema, redis_client=client)
    keys = [index.key(str(i)) for i in range(key_count)]
    try:
        with pytest.raises(RedisVLError, match="Length of keys does not match"):
            index.load(
                iter([{"title": "first"}, {"title": "second"}]),
                keys=iter(keys) if keys else [],
                batch_size=1,
            )
        assert list(client.scan_iter(f"{index.prefix}:*")) == []
    finally:
        stored_keys = list(client.scan_iter(f"{index.prefix}:*"))
        if stored_keys:
            client.delete(*stored_keys)


@pytest.mark.parametrize("key_count", [0, 1, 3])
async def test_async_load_rejects_mismatched_keys_before_writing(
    async_client, load_schema, key_count
):
    index = AsyncSearchIndex(load_schema, redis_client=async_client)
    keys = [index.key(str(i)) for i in range(key_count)]
    try:
        with pytest.raises(RedisVLError, match="Length of keys does not match"):
            await index.load(
                iter([{"title": "first"}, {"title": "second"}]),
                keys=iter(keys) if keys else [],
                batch_size=1,
            )
        assert [key async for key in async_client.scan_iter(f"{index.prefix}:*")] == []
    finally:
        stored_keys = [key async for key in async_client.scan_iter(f"{index.prefix}:*")]
        if stored_keys:
            await async_client.delete(*stored_keys)
