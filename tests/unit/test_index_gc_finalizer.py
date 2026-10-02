"""Regression tests for the SearchIndex / AsyncSearchIndex finalizer memory leak.

SearchIndex.__init__ historically registered ``weakref.finalize(self,
self.disconnect)``. A bound method holds a strong reference to ``self``, and
``weakref.finalize`` keeps its callback alive in a module-level registry until
the finalizer fires. The finalizer can only fire once the instance is
unreachable, but the instance can never become unreachable while the callback
references it, so every index that owned its Redis client was retained for the
lifetime of the process.

These tests assert the two halves of the contract:

1. Index instances must be garbage collectable after they go out of scope.
2. The cleanup the finalizer exists to perform (closing an owned client) must
   still happen when the index is collected, and must not touch clients the
   index does not own.

No Redis server is required here; client construction is mocked. Integration
coverage against a real Redis lives in
``tests/integration/test_index_gc_finalizer_integration.py``.
"""

import asyncio
import gc
import weakref
from unittest import mock

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.schema import IndexSchema

SCHEMA_DICT = {
    "index": {"name": "gc-probe", "prefix": "gc", "storage_type": "hash"},
    "fields": [
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
    ],
}


def collect():
    """Run a couple of GC passes so collection is not order-dependent."""
    for _ in range(3):
        gc.collect()


def count_live(cls) -> int:
    return sum(1 for obj in gc.get_objects() if type(obj) is cls)


class TestInstancesAreCollectable:
    def test_sync_index_is_garbage_collected(self):
        index = SearchIndex.from_dict(SCHEMA_DICT)
        ref = weakref.ref(index)
        del index
        collect()
        assert ref() is None, (
            "SearchIndex instance survived del + gc.collect(); the finalizer "
            "callback must not hold a reference to the instance"
        )

    def test_async_index_is_garbage_collected(self):
        index = AsyncSearchIndex.from_dict(SCHEMA_DICT)
        ref = weakref.ref(index)
        del index
        collect()
        assert ref() is None, (
            "AsyncSearchIndex instance survived del + gc.collect(); the "
            "finalizer callback must not hold a reference to the instance"
        )

    def test_instances_do_not_accumulate_across_many_constructions(self):
        """Mirror of the reported reproduction: build N indexes, drop them,
        and require that none remain alive."""
        for cls in (SearchIndex, AsyncSearchIndex):
            collect()
            baseline = count_live(cls)
            for _ in range(50):
                index = cls.from_dict(SCHEMA_DICT)
                del index
            collect()
            assert count_live(cls) == baseline, (
                f"{cls.__name__} leaked instances across repeated "
                "construction and deletion"
            )

    def test_sync_index_with_injected_client_is_garbage_collected(self):
        schema = IndexSchema.from_dict(SCHEMA_DICT)
        fake_client = mock.MagicMock(name="injected_sync_client")
        index = SearchIndex(schema, redis_client=fake_client)
        ref = weakref.ref(index)
        del index
        collect()
        assert ref() is None


class TestOwnedClientIsClosedOnCollection:
    def test_sync_lazily_created_client_closed_when_index_collected(self):
        fake_client = mock.MagicMock(name="owned_sync_client")
        with mock.patch(
            "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
            return_value=fake_client,
        ):
            index = SearchIndex.from_dict(SCHEMA_DICT, redis_url="redis://fake:6379")
            # Trigger lazy client creation through the internal accessor.
            assert index._redis_client is fake_client

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None
        fake_client.close.assert_called_once()

    def test_async_lazily_created_client_closed_when_index_collected(self):
        fake_client = mock.MagicMock(name="owned_async_client")
        fake_client.aclose = mock.AsyncMock()

        async def build():
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
                new=mock.AsyncMock(return_value=fake_client),
            ):
                index = AsyncSearchIndex.from_dict(
                    SCHEMA_DICT, redis_url="redis://fake:6379"
                )
                assert await index._get_client() is fake_client
            return index

        # Build inside a loop, then drop the reference outside of any running
        # loop so the finalizer's sync_wrapper can create its own loop.
        index = asyncio.run(build())
        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None
        fake_client.aclose.assert_awaited_once()

    def test_sync_index_that_never_created_a_client_collects_cleanly(self):
        index = SearchIndex.from_dict(SCHEMA_DICT, redis_url="redis://fake:6379")
        ref = weakref.ref(index)
        del index
        collect()
        assert ref() is None


class TestExplicitDisconnect:
    def test_sync_disconnect_closes_once_and_detaches_finalizer(self):
        fake_client = mock.MagicMock(name="owned_sync_client")
        with mock.patch(
            "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
            return_value=fake_client,
        ):
            index = SearchIndex.from_dict(SCHEMA_DICT, redis_url="redis://fake:6379")
            assert index._redis_client is fake_client

        index.disconnect()
        fake_client.close.assert_called_once()

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None
        # The finalizer must not fire a second close after disconnect().
        fake_client.close.assert_called_once()

    def test_async_disconnect_closes_once_and_detaches_finalizer(self):
        fake_client = mock.MagicMock(name="owned_async_client")
        fake_client.aclose = mock.AsyncMock()

        async def build_and_disconnect():
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
                new=mock.AsyncMock(return_value=fake_client),
            ):
                index = AsyncSearchIndex.from_dict(
                    SCHEMA_DICT, redis_url="redis://fake:6379"
                )
                assert await index._get_client() is fake_client
            await index.disconnect()
            return index

        index = asyncio.run(build_and_disconnect())
        fake_client.aclose.assert_awaited_once()

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None
        fake_client.aclose.assert_awaited_once()


class TestUnownedClientIsNeverClosed:
    def test_sync_injected_client_not_closed_on_collection(self):
        schema = IndexSchema.from_dict(SCHEMA_DICT)
        fake_client = mock.MagicMock(name="injected_sync_client")
        index = SearchIndex(schema, redis_client=fake_client)

        del index
        collect()

        fake_client.close.assert_not_called()

    def test_async_injected_client_not_closed_on_collection(self):
        schema = IndexSchema.from_dict(SCHEMA_DICT)
        fake_client = mock.MagicMock(name="injected_async_client")
        fake_client.aclose = mock.AsyncMock()
        index = AsyncSearchIndex(schema, redis_client=fake_client)

        del index
        collect()

        fake_client.aclose.assert_not_awaited()


class TestOwnsClientHandover:
    """``owns_client`` overrides who closes the client.

    By default an index closes only a client it created itself. These tests
    cover the two explicit overrides at construction, which is now the only
    place ownership can be stated: the removed ``set_client()`` inherited
    whatever ownership the index already had, and so could be handed a
    caller's client and then close it.
    """

    def test_sync_injected_client_closed_when_ownership_handed_over(self):
        schema = IndexSchema.from_dict(SCHEMA_DICT)
        fake_client = mock.MagicMock(name="handed_over_sync_client")
        index = SearchIndex(schema, redis_client=fake_client, owns_client=True)

        del index
        collect()

        fake_client.close.assert_called_once()

    def test_async_injected_client_closed_when_ownership_handed_over(self):
        schema = IndexSchema.from_dict(SCHEMA_DICT)
        fake_client = mock.MagicMock(name="handed_over_async_client")
        fake_client.aclose = mock.AsyncMock()
        index = AsyncSearchIndex(schema, redis_client=fake_client, owns_client=True)

        del index
        collect()

        fake_client.aclose.assert_awaited_once()

    def test_sync_lazily_created_client_kept_when_ownership_declined(self):
        """``owns_client=False`` keeps a client the index would have owned.

        Only covered on the sync class: the flag is read by shared
        ``__init__`` code, and the per-flavour close paths are covered above.
        """
        schema = IndexSchema.from_dict(SCHEMA_DICT)
        fake_client = mock.MagicMock(name="lazily_created_sync_client")

        with mock.patch(
            "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
            return_value=fake_client,
        ):
            index = SearchIndex(
                schema, redis_url="redis://fake:6379", owns_client=False
            )
            assert index._redis_client is fake_client

        del index
        collect()

        fake_client.close.assert_not_called()
