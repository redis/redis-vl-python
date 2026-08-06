"""Unit tests for BaseCache.clear/aclear key enumeration, especially on cluster.

On a cluster client, ``scan`` is broadcast to every primary and replies with a
``{node_name: cursor}`` mapping. Those cursors are node-local: they can neither
be fed back as a single cursor nor broadcast to the other primaries. The clear
loop used to leave its cursor at 0 in that case, so it re-issued ``SCAN 0``
forever and never made progress once the first page stopped yielding matches.

``clear`` now delegates enumeration to redis-py's ``scan_iter``, which drives
each primary on its own cursor via ``target_nodes``. The fakes below bind the
real upstream ``scan_iter`` onto themselves, so these tests exercise the actual
library loop rather than a reimplementation of it, and assert that every
follow-up ``SCAN`` carries the cursor the previous reply returned for that node.
"""

import pytest
from redis.asyncio.client import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.client import Redis
from redis.cluster import RedisCluster

from redisvl.extensions.cache.base import CLEAR_BATCH_SIZE, BaseCache
from redisvl.extensions.cache.embeddings import EmbeddingsCache

PREFIX = "clear_cursor_test"
MATCH = f"{PREFIX}:*"


class MockClusterClient:
    """Stand-in for RedisCluster SCAN/DEL, driven by a per-node key layout.

    Keys live on named primaries and each primary pages through its own keys on
    its own cursor, which is what makes the cluster contract testable. ``scan``
    raises if a node-local cursor is ever broadcast, and if the call count runs
    away -- so a loop that fails to advance fails the test loudly instead of
    hanging the suite (``pytest-timeout`` is not installed).
    """

    # Exercise the real upstream loop instead of imitating it.
    scan_iter = RedisCluster.scan_iter

    def __init__(self, keys_by_node, page=2, missing_nodes=frozenset(), max_calls=200):
        self.keys_by_node = {n: list(k) for n, k in keys_by_node.items()}
        self.page = page
        self.missing_nodes = frozenset(missing_nodes)
        self.max_calls = max_calls
        self.scan_calls = []
        self.deleted = []

    def _page(self, node, cursor):
        keys = self.keys_by_node[node]
        nxt = cursor + self.page
        return (0 if nxt >= len(keys) else nxt), keys[cursor : cursor + self.page]

    def _next_scan(self, cursor, match, count, target_nodes):
        self.scan_calls.append(
            {"cursor": cursor, "match": match, "target_nodes": target_nodes}
        )
        if len(self.scan_calls) > self.max_calls:
            raise AssertionError(
                f"{len(self.scan_calls)} SCAN calls -- the cursor is not "
                f"advancing: {self.scan_calls[:6]}..."
            )
        if target_nodes is None:
            # Broadcast to all primaries. Only cursor 0 may be broadcast; a
            # node-local cursor sent here would resume other nodes mid-keyspace.
            assert cursor == 0, f"node-local cursor {cursor!r} broadcast to primaries"
            pages = {n: self._page(n, 0) for n in self.keys_by_node}
            return (
                {n: c for n, (c, _) in pages.items()},
                [k for _, ks in pages.values() for k in ks],
            )
        node = str(target_nodes).removeprefix("node-object:")
        nxt, keys = self._page(node, cursor)
        return {node: nxt}, keys

    def scan(self, cursor=0, match=None, count=None, target_nodes=None, **kwargs):
        return self._next_scan(cursor, match, count, target_nodes)

    def delete(self, *keys):
        self.deleted.extend(keys)
        return len(keys)

    def get_node(self, host=None, port=None, node_name=None):
        if node_name in self.missing_nodes:
            return None
        return f"node-object:{node_name}"


class MockAsyncClusterClient(MockClusterClient):
    """Async variant, bound to the async cluster client's ``scan_iter``."""

    scan_iter = AsyncRedisCluster.scan_iter

    async def scan(self, cursor=0, match=None, count=None, target_nodes=None, **kwargs):
        return self._next_scan(cursor, match, count, target_nodes)

    async def delete(self, *keys):
        self.deleted.extend(keys)
        return len(keys)


class MockStandaloneClient(MockClusterClient):
    """Standalone client: one integer cursor, no per-node mapping."""

    scan_iter = Redis.scan_iter

    def _next_scan(self, cursor, match, count, target_nodes):
        self.scan_calls.append(
            {"cursor": cursor, "match": match, "target_nodes": target_nodes}
        )
        if len(self.scan_calls) > self.max_calls:
            raise AssertionError(f"{len(self.scan_calls)} SCAN calls -- not advancing")
        keys = self.keys_by_node["single"]
        cursor = int(cursor)
        nxt = cursor + self.page
        return (0 if nxt >= len(keys) else nxt), keys[cursor : cursor + self.page]


class MockAsyncStandaloneClient(MockStandaloneClient):
    """Async standalone client."""

    scan_iter = AsyncRedis.scan_iter

    async def scan(self, cursor=0, match=None, count=None, target_nodes=None, **kwargs):
        return self._next_scan(cursor, match, count, target_nodes)

    async def delete(self, *keys):
        self.deleted.extend(keys)
        return len(keys)


def _layout():
    """Three primaries with uneven scan depth, one done after the broadcast."""
    return {
        "node-1": [f"{PREFIX}:a{i}" for i in range(6)],  # 3 rounds
        "node-2": [f"{PREFIX}:b{i}" for i in range(2)],  # done on the broadcast
        "node-3": [f"{PREFIX}:c{i}" for i in range(4)],  # 2 rounds
    }


def _all_keys(layout):
    return sorted(k for ks in layout.values() for k in ks)


def _assert_cursors_advanced(client):
    """Every targeted SCAN must carry a non-zero, node-local cursor."""
    targeted = [c for c in client.scan_calls if c["target_nodes"] is not None]
    assert targeted, "no per-node continuation happened; test proves nothing"
    assert all(
        c["cursor"] != 0 for c in targeted
    ), f"a continuation restarted at cursor 0: {targeted}"
    assert all(c["match"] == MATCH for c in client.scan_calls)


class TestClearOnCluster:
    """clear() must drain every primary and terminate."""

    def test_drains_all_primaries_with_uneven_depth(self):
        layout = _layout()
        client = MockClusterClient(layout)
        cache = BaseCache(name=PREFIX, redis_client=client)

        cache.clear()

        assert sorted(client.deleted) == _all_keys(layout)
        _assert_cursors_advanced(client)
        # node-2 finished on the broadcast and is never targeted again.
        assert not any(
            c["target_nodes"] == "node-object:node-2" for c in client.scan_calls
        )

    def test_single_node_cluster_is_drained(self):
        layout = {"node-1": [f"{PREFIX}:{i}" for i in range(5)]}
        client = MockClusterClient(layout)

        BaseCache(name=PREFIX, redis_client=client).clear()

        assert sorted(client.deleted) == _all_keys(layout)
        _assert_cursors_advanced(client)

    def test_empty_cache_issues_no_delete(self):
        client = MockClusterClient({"node-1": [], "node-2": []})

        BaseCache(name=PREFIX, redis_client=client).clear()

        assert client.deleted == []
        assert len(client.scan_calls) == 1

    def test_tolerates_duplicate_keys_across_pages(self):
        # SCAN may return the same key more than once. DEL is idempotent, so
        # clear() must not choke -- and nothing here counts keys.
        dup = f"{PREFIX}:dup"
        client = MockClusterClient({"node-1": [dup, dup, f"{PREFIX}:other", dup]})

        BaseCache(name=PREFIX, redis_client=client).clear()

        assert sorted(set(client.deleted)) == [dup, f"{PREFIX}:other"]

    def test_deletes_are_batched(self):
        n = CLEAR_BATCH_SIZE * 2 + 7
        layout = {"node-1": [f"{PREFIX}:{i}" for i in range(n)]}
        client = MockClusterClient(layout, page=CLEAR_BATCH_SIZE)
        calls = []

        real_delete = client.delete
        client.delete = lambda *keys: (calls.append(len(keys)), real_delete(*keys))[1]

        BaseCache(name=PREFIX, redis_client=client).clear()

        assert sorted(client.deleted) == _all_keys(layout)
        assert calls == [CLEAR_BATCH_SIZE, CLEAR_BATCH_SIZE, 7]


class TestAsyncClearOnCluster:
    """aclear() must match clear() behavior exactly."""

    @pytest.mark.asyncio
    async def test_drains_all_primaries_with_uneven_depth(self):
        layout = _layout()
        client = MockAsyncClusterClient(layout)
        cache = BaseCache(name=PREFIX, async_redis_client=client)

        await cache.aclear()

        assert sorted(client.deleted) == _all_keys(layout)
        _assert_cursors_advanced(client)

    @pytest.mark.asyncio
    async def test_empty_cache_issues_no_delete(self):
        client = MockAsyncClusterClient({"node-1": [], "node-2": []})

        await BaseCache(name=PREFIX, async_redis_client=client).aclear()

        assert client.deleted == []

    @pytest.mark.asyncio
    async def test_tolerates_duplicate_keys_across_pages(self):
        dup = f"{PREFIX}:dup"
        client = MockAsyncClusterClient({"node-1": [dup, dup, f"{PREFIX}:other", dup]})

        await BaseCache(name=PREFIX, async_redis_client=client).aclear()

        assert sorted(set(client.deleted)) == [dup, f"{PREFIX}:other"]


class TestClearOnStandalone:
    """The standalone path must keep working unchanged."""

    def test_advances_single_cursor(self):
        keys = [f"{PREFIX}:{i}" for i in range(7)]
        client = MockStandaloneClient({"single": keys})

        BaseCache(name=PREFIX, redis_client=client).clear()

        assert sorted(client.deleted) == sorted(keys)
        cursors = [c["cursor"] for c in client.scan_calls]
        assert cursors[0] in (0, "0")
        assert [int(c) for c in cursors] == [0, 2, 4, 6]

    @pytest.mark.asyncio
    async def test_async_advances_single_cursor(self):
        keys = [f"{PREFIX}:{i}" for i in range(7)]
        client = MockAsyncStandaloneClient({"single": keys})

        await BaseCache(name=PREFIX, async_redis_client=client).aclear()

        assert sorted(client.deleted) == sorted(keys)
        assert [int(c["cursor"]) for c in client.scan_calls] == [0, 2, 4, 6]


class TestConcreteCachesClearOnCluster:
    """The fix must hold through the cache classes users actually instantiate.

    Also pins that each subclass's effective key prefix really is ``<name>:*``,
    which clear() depends on and nothing else covers.
    """

    def test_embeddings_cache_drains_cluster(self):
        layout = _layout()
        client = MockClusterClient(layout)

        EmbeddingsCache(name=PREFIX, redis_client=client).clear()

        assert sorted(client.deleted) == _all_keys(layout)
        assert all(c["match"] == MATCH for c in client.scan_calls)

    @pytest.mark.asyncio
    async def test_embeddings_cache_drains_cluster_async(self):
        layout = _layout()
        client = MockAsyncClusterClient(layout)

        await EmbeddingsCache(name=PREFIX, async_redis_client=client).aclear()

        assert sorted(client.deleted) == _all_keys(layout)
        assert all(c["match"] == MATCH for c in client.scan_calls)
