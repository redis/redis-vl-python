"""Integration tests for SearchIndex / AsyncSearchIndex garbage collection.

Runs against a real Redis (testcontainers). Verifies, with live connections,
that:

1. Index instances that own their client are collectable once dropped, and
   collection closes the owned client's connections.
2. Index instances built around a caller-provided client never close it.
3. Building an index per operation (the reported production pattern) does not
   accumulate live instances.

See tests/unit/test_index_gc_finalizer.py for the mocked-client unit coverage
of the same contract.
"""

import asyncio
import gc
import weakref

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex

fields = [
    {"name": "tag", "type": "tag"},
    {"name": "num", "type": "numeric"},
    {
        "name": "vec",
        "type": "vector",
        "attrs": {
            "dims": 8,
            "algorithm": "flat",
            "distance_metric": "cosine",
            "datatype": "float32",
        },
    },
]


def collect():
    for _ in range(3):
        gc.collect()


@pytest.fixture
def schema_dict(redis_test_name):
    name = redis_test_name("gc_finalizer")
    return {
        "index": {"name": name, "prefix": name, "storage_type": "hash"},
        "fields": fields,
    }


def pool_sockets_closed(sync_client) -> bool:
    """True when every connection in the client's pool has dropped its socket."""
    pool = sync_client.connection_pool
    conns = list(pool._available_connections) + list(pool._in_use_connections)
    return all(getattr(conn, "_sock", None) is None for conn in conns)


class TestOwnedClientLifecycle:
    def test_sync_index_collected_and_owned_client_closed(self, redis_url, schema_dict):
        index = SearchIndex.from_dict(schema_dict, redis_url=redis_url)
        # Force lazy client creation with a real round trip.
        assert index.exists() is False
        client = index.client
        assert client is not None

        close_calls = []
        original_close = client.close

        def recording_close():
            close_calls.append(1)
            original_close()

        client.close = recording_close

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None, "owned sync index was not garbage collected"
        assert close_calls == [1], "owned client was not closed exactly once"
        assert pool_sockets_closed(
            client
        ), "owned client still holds open sockets after index collection"

    def test_async_index_collected_and_owned_client_closed(
        self, redis_url, schema_dict
    ):
        async def build():
            index = AsyncSearchIndex.from_dict(schema_dict, redis_url=redis_url)
            assert await index.exists() is False
            return index, index.client

        # Build inside a private loop, then drop the index outside any running
        # loop so the finalizer's sync_wrapper can run the async close.
        index, client = asyncio.run(build())
        assert client is not None

        aclose_calls = []
        original_aclose = client.aclose

        async def recording_aclose():
            aclose_calls.append(1)
            await original_aclose()

        client.aclose = recording_aclose

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None, "owned async index was not garbage collected"
        assert aclose_calls == [1], "owned async client was not closed exactly once"

    def test_sync_explicit_disconnect_then_collection_closes_once(
        self, redis_url, schema_dict
    ):
        index = SearchIndex.from_dict(schema_dict, redis_url=redis_url)
        assert index.exists() is False
        client = index.client

        close_calls = []
        original_close = client.close

        def recording_close():
            close_calls.append(1)
            original_close()

        client.close = recording_close

        index.disconnect()
        assert close_calls == [1]

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None
        assert close_calls == [1], "finalizer double-closed after disconnect()"


class TestUnownedClientLifecycle:
    def test_sync_injected_client_survives_index_collection(self, client, schema_dict):
        from redisvl.schema import IndexSchema

        index = SearchIndex(IndexSchema.from_dict(schema_dict), redis_client=client)
        assert index.exists() is False

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None, "index with injected client was not collected"
        assert client.ping() is True, "injected client was closed by the index"

    async def test_async_injected_client_survives_index_collection(
        self, async_client, schema_dict
    ):
        from redisvl.schema import IndexSchema

        index = AsyncSearchIndex(
            IndexSchema.from_dict(schema_dict), redis_client=async_client
        )
        assert await index.exists() is False

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None, "async index with injected client was not collected"
        assert (
            await async_client.ping() is True
        ), "injected async client was closed by the index"


class TestIndexPerRequestPattern:
    def test_repeated_sync_construction_does_not_accumulate(
        self, redis_url, schema_dict
    ):
        """The reported production pattern: one SearchIndex per request."""
        collect()
        baseline = sum(1 for o in gc.get_objects() if type(o) is SearchIndex)

        for _ in range(20):
            index = SearchIndex.from_dict(schema_dict, redis_url=redis_url)
            assert index.exists() is False
            del index

        collect()
        live = sum(1 for o in gc.get_objects() if type(o) is SearchIndex)
        assert live == baseline, (
            f"leaked {live - baseline} SearchIndex instances across 20 "
            "construct/use/drop cycles"
        )
