"""Client ownership semantics against a real Redis.

Verifies with live connections that a client handed to an index via the
deprecated `set_client()` is never closed by that index, while a client the
index creates for itself (via the deprecated `connect()`) still is.

See tests/unit/test_index_client_ownership.py for the mocked-client coverage.
"""

import asyncio
import gc
import warnings
import weakref

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex

fields = [{"name": "tag", "type": "tag"}, {"name": "num", "type": "numeric"}]


def collect():
    for _ in range(3):
        gc.collect()


@pytest.fixture
def schema_dict(redis_test_name):
    name = redis_test_name("ownership")
    return {
        "index": {"name": name, "prefix": name, "storage_type": "hash"},
        "fields": fields,
    }


def pool_sockets_closed(sync_client) -> bool:
    pool = sync_client.connection_pool
    conns = list(pool._available_connections) + list(pool._in_use_connections)
    return all(getattr(conn, "_sock", None) is None for conn in conns)


class TestSetClientLeavesCallerClientOpen:
    def test_sync_caller_client_usable_after_index_collected(
        self, redis_url, schema_dict, client
    ):
        # Index owns a client of its own first, then the caller swaps theirs in.
        index = SearchIndex.from_dict(schema_dict, redis_url=redis_url)
        assert index.exists() is False
        index_own_client = index.client

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index.set_client(client)

        assert index._owns_redis_client is False
        # The index's own client was released when it was replaced.
        assert pool_sockets_closed(index_own_client)

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None, "index was not collected"
        assert client.ping() is True, "caller's client was closed by the index"

    def test_sync_disconnect_leaves_caller_client_open(
        self, redis_url, schema_dict, client
    ):
        index = SearchIndex.from_dict(schema_dict, redis_url=redis_url)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index.set_client(client)

        index.disconnect()

        assert client.ping() is True, "disconnect() closed the caller's client"

    async def test_async_caller_client_usable_after_index_collected(
        self, redis_url, schema_dict, async_client
    ):
        index = AsyncSearchIndex.from_dict(schema_dict, redis_url=redis_url)
        assert await index.exists() is False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await index.set_client(async_client)

        assert index._owns_redis_client is False

        ref = weakref.ref(index)
        del index
        collect()
        # Give any (incorrectly) scheduled close a chance to run.
        await asyncio.sleep(0)

        assert ref() is None, "async index was not collected"
        assert (
            await async_client.ping() is True
        ), "caller's async client was closed by the index"

    async def test_async_disconnect_leaves_caller_client_open(
        self, redis_url, schema_dict, async_client
    ):
        index = AsyncSearchIndex.from_dict(schema_dict, redis_url=redis_url)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await index.set_client(async_client)

        await index.disconnect()

        assert (
            await async_client.ping() is True
        ), "disconnect() closed the caller's async client"


class TestConnectStillOwnsItsClient:
    """The deprecated connect() creates the client, so the index must still
    close it. This guards against over-correcting the ownership fix."""

    def test_sync_connect_created_client_closed_on_collection(
        self, redis_url, schema_dict
    ):
        index = SearchIndex.from_dict(schema_dict)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index.connect(redis_url=redis_url)

        assert index._owns_redis_client is True
        created = index.client
        assert created is not None
        assert created.ping() is True

        del index
        collect()

        assert pool_sockets_closed(
            created
        ), "client created by connect() was not closed on collection"

    def test_async_connect_created_client_closed_on_collection(
        self, redis_url, schema_dict
    ):
        aclose_calls = []

        async def build():
            index = AsyncSearchIndex.from_dict(schema_dict)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                await index.connect(redis_url=redis_url)
            assert index._owns_redis_client is True
            created = index.client
            assert await created.ping() is True

            original = created.aclose

            async def recording_aclose():
                aclose_calls.append(1)
                await original()

            created.aclose = recording_aclose
            return index

        index = asyncio.run(build())
        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None
        assert aclose_calls == [
            1
        ], "client created by connect() was not closed exactly once"


class TestInjectedClientAtConstruction:
    """Baseline: __init__ already got this right. Kept so the three entry
    points (constructor, set_client, connect) are covered together."""

    def test_sync_constructor_injected_client_survives(self, schema_dict, client):
        from redisvl.schema import IndexSchema

        index = SearchIndex(IndexSchema.from_dict(schema_dict), redis_client=client)
        assert index._owns_redis_client is False
        assert index.exists() is False

        del index
        collect()

        assert client.ping() is True
