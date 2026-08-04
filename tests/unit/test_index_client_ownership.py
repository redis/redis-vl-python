"""Ownership semantics for clients handed to an index after construction.

`__init__` already treats a caller-provided client as not owned, and never
closes it. The deprecated `set_client()` did not follow that rule: an index
created with `redis_url` keeps `_owns_redis_client=True`, so after
`set_client(caller_client)` the index would close a client it never created.
Since 0.25.0 the client finalizer actually fires, which made that observable
as the caller's client being closed when the index is garbage collected.

The deprecated `connect()` is the mirror case and must keep working: it
creates the client itself, so the index does own it and must still close it.
"""

import asyncio
import gc
import warnings
import weakref
from unittest import mock

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.schema import IndexSchema

SCHEMA_DICT = {
    "index": {"name": "ownership-probe", "prefix": "own", "storage_type": "hash"},
    "fields": [{"name": "tag", "type": "tag"}],
}


def collect():
    for _ in range(3):
        gc.collect()


def sync_index_owning_client(created_client=None):
    """Index built from a URL, so it owns whatever client it creates."""
    index = SearchIndex.from_dict(SCHEMA_DICT, redis_url="redis://fake:6379")
    if created_client is not None:
        with mock.patch(
            "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
            return_value=created_client,
        ):
            assert index._redis_client is created_client
    return index


def set_client_sync(index, client):
    with mock.patch(
        "redisvl.index.index.RedisConnectionFactory.validate_sync_redis",
        return_value=None,
    ):
        return index.set_client(client)


async def set_client_async(index, client):
    with mock.patch.object(
        AsyncSearchIndex, "_validate_client", new=mock.AsyncMock(return_value=client)
    ):
        return await index.set_client(client)


class TestCallerProvidedClientIsNotOwned:
    def test_sync_set_client_marks_client_unowned(self):
        index = sync_index_owning_client()
        caller_client = mock.MagicMock(name="caller_client")
        set_client_sync(index, caller_client)
        assert index._owns_redis_client is False

    def test_sync_set_client_client_survives_gc(self):
        index = sync_index_owning_client()
        caller_client = mock.MagicMock(name="caller_client")
        set_client_sync(index, caller_client)

        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None, "index was not collected"
        caller_client.close.assert_not_called()

    def test_sync_set_client_client_survives_disconnect(self):
        index = sync_index_owning_client()
        caller_client = mock.MagicMock(name="caller_client")
        set_client_sync(index, caller_client)

        index.disconnect()

        caller_client.close.assert_not_called()

    def test_async_set_client_marks_client_unowned(self):
        async def run():
            index = AsyncSearchIndex.from_dict(
                SCHEMA_DICT, redis_url="redis://fake:6379"
            )
            caller_client = mock.MagicMock(name="caller_async_client")
            caller_client.aclose = mock.AsyncMock()
            await set_client_async(index, caller_client)
            return index

        index = asyncio.run(run())
        assert index._owns_redis_client is False

    def test_async_set_client_client_survives_gc(self):
        caller_client = mock.MagicMock(name="caller_async_client")
        caller_client.aclose = mock.AsyncMock()

        async def run():
            index = AsyncSearchIndex.from_dict(
                SCHEMA_DICT, redis_url="redis://fake:6379"
            )
            await set_client_async(index, caller_client)
            return index

        index = asyncio.run(run())
        ref = weakref.ref(index)
        del index
        collect()

        assert ref() is None, "index was not collected"
        caller_client.aclose.assert_not_awaited()

    def test_async_set_client_client_survives_disconnect(self):
        caller_client = mock.MagicMock(name="caller_async_client")
        caller_client.aclose = mock.AsyncMock()

        async def run():
            index = AsyncSearchIndex.from_dict(
                SCHEMA_DICT, redis_url="redis://fake:6379"
            )
            await set_client_async(index, caller_client)
            await index.disconnect()

        asyncio.run(run())
        caller_client.aclose.assert_not_awaited()


class TestIndexCreatedClientIsStillOwned:
    """Regression guard: the deprecated connect() creates its own client, so
    the index must keep ownership and still close it."""

    def test_sync_connect_keeps_ownership_and_closes_on_gc(self):
        created = mock.MagicMock(name="connect_created_client")
        index = SearchIndex.from_dict(SCHEMA_DICT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
                return_value=created,
            ):
                index.connect(redis_url="redis://fake:6379")

        assert index._owns_redis_client is True
        del index
        collect()
        created.close.assert_called_once()

    def test_async_connect_keeps_ownership_and_closes_on_gc(self):
        created = mock.MagicMock(name="aconnect_created_client")
        created.aclose = mock.AsyncMock()

        async def run():
            index = AsyncSearchIndex.from_dict(SCHEMA_DICT)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with mock.patch(
                    "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
                    new=mock.AsyncMock(return_value=created),
                ):
                    with mock.patch.object(
                        AsyncSearchIndex,
                        "_validate_client",
                        new=mock.AsyncMock(return_value=created),
                    ):
                        await index.connect(redis_url="redis://fake:6379")
            return index

        index = asyncio.run(run())
        assert (
            index._owns_redis_client is True
        ), "connect() created the client, so the index must still own it"
        del index
        collect()
        created.aclose.assert_awaited_once()


class TestConnectTakesOwnershipFromAnUnownedState:
    """connect() creates the client itself, so the index must own it even when
    the index was previously holding a caller-provided (unowned) client. Async
    gets this right via _swap_client(owns=True); sync must match, otherwise the
    client it just created is never closed."""

    def test_sync_connect_takes_ownership_over_a_caller_client(self):
        caller_client = mock.MagicMock(name="caller_client")
        created = mock.MagicMock(name="connect_created_client")
        schema = IndexSchema.from_dict(SCHEMA_DICT)
        index = SearchIndex(schema, redis_client=caller_client)
        assert index._owns_redis_client is False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
                return_value=created,
            ):
                index.connect(redis_url="redis://fake:6379")

        assert (
            index._owns_redis_client is True
        ), "connect() created this client, so the index must own it"
        del index
        collect()
        created.close.assert_called_once()
        caller_client.close.assert_not_called()

    def test_sync_connect_takes_ownership_after_set_client(self):
        caller_client = mock.MagicMock(name="caller_client")
        created = mock.MagicMock(name="connect_created_client")
        index = sync_index_owning_client()
        set_client_sync(index, caller_client)
        assert index._owns_redis_client is False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
                return_value=created,
            ):
                index.connect(redis_url="redis://fake:6379")

        assert index._owns_redis_client is True
        del index
        collect()
        created.close.assert_called_once()
        caller_client.close.assert_not_called()

    def test_async_connect_takes_ownership_after_set_client(self):
        caller_client = mock.MagicMock(name="caller_async_client")
        caller_client.aclose = mock.AsyncMock()
        created = mock.MagicMock(name="aconnect_created_client")
        created.aclose = mock.AsyncMock()

        async def run():
            index = AsyncSearchIndex.from_dict(
                SCHEMA_DICT, redis_url="redis://fake:6379"
            )
            await set_client_async(index, caller_client)
            assert index._owns_redis_client is False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with mock.patch(
                    "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
                    new=mock.AsyncMock(return_value=created),
                ):
                    with mock.patch.object(
                        AsyncSearchIndex,
                        "_validate_client",
                        new=mock.AsyncMock(return_value=created),
                    ):
                        await index.connect(redis_url="redis://fake:6379")
            return index

        index = asyncio.run(run())
        assert index._owns_redis_client is True
        del index
        collect()
        created.aclose.assert_awaited_once()
        caller_client.aclose.assert_not_awaited()


class TestConnectReleasesThePreviousOwnedClient:
    """connect() replaces the active client. When the index owned the old one,
    it must be closed: registering a finalizer for the new client detaches the
    old client's finalizer, so nothing else would ever close it."""

    def test_sync_repeated_connect_closes_the_first_client(self):
        first = mock.MagicMock(name="first_client")
        second = mock.MagicMock(name="second_client")
        index = SearchIndex.from_dict(SCHEMA_DICT)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
                side_effect=[first, second],
            ):
                index.connect(redis_url="redis://fake:6379")
                index.connect(redis_url="redis://fake:6380")

        first.close.assert_called_once()
        assert index.client is second

        del index
        collect()
        second.close.assert_called_once()

    def test_sync_connect_closes_a_lazily_created_client(self):
        lazy = mock.MagicMock(name="lazy_client")
        connected = mock.MagicMock(name="connect_client")
        index = SearchIndex.from_dict(SCHEMA_DICT, redis_url="redis://fake:6379")

        with mock.patch(
            "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
            return_value=lazy,
        ):
            assert index._redis_client is lazy

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
                return_value=connected,
            ):
                index.connect(redis_url="redis://fake:6380")

        lazy.close.assert_called_once()

    def test_async_repeated_connect_closes_the_first_client(self):
        first = mock.MagicMock(name="first_async_client")
        first.aclose = mock.AsyncMock()
        second = mock.MagicMock(name="second_async_client")
        second.aclose = mock.AsyncMock()

        async def run():
            index = AsyncSearchIndex.from_dict(SCHEMA_DICT)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with mock.patch(
                    "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
                    new=mock.AsyncMock(side_effect=[first, second]),
                ):
                    with mock.patch.object(
                        AsyncSearchIndex,
                        "_validate_client",
                        new=mock.AsyncMock(side_effect=lambda c: c),
                    ):
                        await index.connect(redis_url="redis://fake:6379")
                        await index.connect(redis_url="redis://fake:6380")
            return index

        index = asyncio.run(run())
        first.aclose.assert_awaited_once()
        assert index.client is second


class TestSwappingInTheSameClient:
    """Handing back the client the index already holds must not close it.
    Releasing the old client unconditionally would close the very instance
    being installed, leaving the index holding a closed client."""

    def test_sync_set_client_with_the_current_client_does_not_close_it(self):
        owned = mock.MagicMock(name="owned_client")
        index = sync_index_owning_client(created_client=owned)
        assert index.client is owned

        set_client_sync(index, owned)

        owned.close.assert_not_called()
        assert index.client is owned
        # Ownership tracks who created the object, so handing the same client
        # back does not change it. Flipping to unowned here would strand this
        # client: a later swap would not close it and its finalizer is gone.
        assert index._owns_redis_client is True

    def test_sync_reinstalled_client_is_still_closed_by_a_later_swap(self):
        owned = mock.MagicMock(name="owned_client")
        other = mock.MagicMock(name="other_client")
        index = sync_index_owning_client(created_client=owned)

        set_client_sync(index, owned)
        set_client_sync(index, other)

        owned.close.assert_called_once()
        other.close.assert_not_called()

    def test_sync_set_client_with_an_unowned_current_client_stays_unowned(self):
        caller_client = mock.MagicMock(name="caller_client")
        schema = IndexSchema.from_dict(SCHEMA_DICT)
        index = SearchIndex(schema, redis_client=caller_client)

        set_client_sync(index, caller_client)

        assert index._owns_redis_client is False
        caller_client.close.assert_not_called()

    def test_sync_ownership_is_cleared_before_the_caller_client_is_installed(self):
        """`disconnect()` does not take the lock, so another thread must never
        be able to observe a caller-provided client while ownership still says
        the index owns it: it would close a client it does not own."""

        class RecordingIndex(SearchIndex):
            def __init__(self, *args, **kwargs):
                self.attr_writes: list[str] = []
                super().__init__(*args, **kwargs)

            def __setattr__(self, name, value):
                if name in ("_SearchIndex__redis_client", "_owns_redis_client"):
                    self.attr_writes.append(name)
                super().__setattr__(name, value)

        owned = mock.MagicMock(name="owned_client")
        index = RecordingIndex(
            IndexSchema.from_dict(SCHEMA_DICT), redis_url="redis://fake:6379"
        )
        with mock.patch(
            "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
            return_value=owned,
        ):
            assert index._redis_client is owned

        index.attr_writes.clear()
        set_client_sync(index, mock.MagicMock(name="caller_client"))

        last_owns = max(
            i for i, n in enumerate(index.attr_writes) if n == "_owns_redis_client"
        )
        last_client = max(
            i
            for i, n in enumerate(index.attr_writes)
            if n == "_SearchIndex__redis_client"
        )
        assert last_owns < last_client, (
            "ownership must be cleared before the caller's client is installed, "
            f"got write order {index.attr_writes}"
        )

    def test_async_set_client_with_the_current_client_does_not_close_it(self):
        owned = mock.MagicMock(name="owned_async_client")
        owned.aclose = mock.AsyncMock()

        async def run():
            index = AsyncSearchIndex.from_dict(
                SCHEMA_DICT, redis_url="redis://fake:6379"
            )
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
                new=mock.AsyncMock(return_value=owned),
            ):
                assert await index._get_client() is owned
            await set_client_async(index, owned)
            return index

        index = asyncio.run(run())
        owned.aclose.assert_not_awaited()
        assert index.client is owned
        # Same rule as the sync case: reinstalling the active client changes
        # nothing about who created it, so ownership is left alone.
        assert index._owns_redis_client is True


class TestSwapResetsValidation:
    """A replacement client has not had this index's lib name applied, so the
    validated flag must clear or the new client keeps the old one's state."""

    def test_sync_set_client_clears_validated_flag(self):
        index = SearchIndex.from_dict(
            SCHEMA_DICT, redis_url="redis://fake:6379", lib_name="probe"
        )
        index._validated_client = True
        set_client_sync(index, mock.MagicMock(name="caller_client"))
        assert index._validated_client is False

    def test_async_set_client_clears_validated_flag(self):
        async def run():
            index = AsyncSearchIndex.from_dict(
                SCHEMA_DICT, redis_url="redis://fake:6379", lib_name="probe"
            )
            index._validated_client = True
            caller = mock.MagicMock(name="caller_async_client")
            caller.aclose = mock.AsyncMock()
            await set_client_async(index, caller)
            return index

        index = asyncio.run(run())
        assert index._validated_client is False


class TestSwapSurvivesAFailingClose:
    """Closing the outgoing client must not abort the swap. If it did, the
    replacement would never be installed and nothing would close it, and the
    index would be left holding a client it had already closed."""

    def test_sync_set_client_completes_when_closing_the_old_client_raises(self):
        old = mock.MagicMock(name="old_client")
        old.close.side_effect = RuntimeError  # fresh instance per call
        index = sync_index_owning_client(created_client=old)
        caller_client = mock.MagicMock(name="caller_client")

        set_client_sync(index, caller_client)

        assert index.client is caller_client
        assert index._owns_redis_client is False

    def test_sync_connect_completes_and_owns_when_closing_the_old_client_raises(self):
        old = mock.MagicMock(name="old_client")
        old.close.side_effect = RuntimeError  # fresh instance per call
        created = mock.MagicMock(name="connect_created_client")
        index = sync_index_owning_client(created_client=old)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
                return_value=created,
            ):
                index.connect(redis_url="redis://fake:6379")

        assert index.client is created
        assert index._owns_redis_client is True
        # The replacement is still tracked, so it is not leaked.
        del index
        collect()
        created.close.assert_called_once()

    def test_async_set_client_completes_when_closing_the_old_client_raises(self):
        old = mock.MagicMock(name="old_async_client")
        old.aclose = mock.AsyncMock(side_effect=RuntimeError)
        caller_client = mock.MagicMock(name="caller_async_client")
        caller_client.aclose = mock.AsyncMock()

        async def run():
            index = AsyncSearchIndex.from_dict(
                SCHEMA_DICT, redis_url="redis://fake:6379"
            )
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
                new=mock.AsyncMock(return_value=old),
            ):
                assert await index._get_client() is old
            await set_client_async(index, caller_client)
            return index

        index = asyncio.run(run())
        assert index.client is caller_client
        assert index._owns_redis_client is False


class TestPreviouslyOwnedClientIsReleased:
    """Swapping in a caller's client must not silently abandon a client the
    index created for itself."""

    def test_sync_set_client_closes_previously_owned_client(self):
        owned = mock.MagicMock(name="index_owned_client")
        index = sync_index_owning_client(created_client=owned)

        caller_client = mock.MagicMock(name="caller_client")
        set_client_sync(index, caller_client)

        owned.close.assert_called_once()
        caller_client.close.assert_not_called()

    def test_async_set_client_closes_previously_owned_client(self):
        owned = mock.MagicMock(name="index_owned_async_client")
        owned.aclose = mock.AsyncMock()
        caller_client = mock.MagicMock(name="caller_async_client")
        caller_client.aclose = mock.AsyncMock()

        async def run():
            index = AsyncSearchIndex.from_dict(
                SCHEMA_DICT, redis_url="redis://fake:6379"
            )
            with mock.patch(
                "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
                new=mock.AsyncMock(return_value=owned),
            ):
                assert await index._get_client() is owned
            await set_client_async(index, caller_client)

        asyncio.run(run())
        owned.aclose.assert_awaited_once()
        caller_client.aclose.assert_not_awaited()
