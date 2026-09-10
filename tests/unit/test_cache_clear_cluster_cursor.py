"""Unit tests for BaseCache.clear/aclear enumeration on Redis Cluster.

On a cluster client, ``scan`` is broadcast to every primary and replies with a
``{node_name: cursor}`` mapping of node-local cursors. The clear loop used to
leave its cursor at 0 in that case, so it re-issued ``SCAN 0`` forever and made
no progress once the first page stopped yielding matches.

``clear`` now delegates enumeration to redis-py's ``scan_iter``, so the cursor
walk itself is upstream's code and not worth re-asserting here. What these tests
pin is our part: every primary gets drained, the match pattern stays scoped to
the cache prefix, and deletes are batched rather than issued per page or all at
once. The fakes bind the real upstream ``scan_iter`` so the drain runs through
the actual library loop.

Standalone clear/aclear is covered end-to-end against a real Redis by
tests/integration/test_llmcache.py; the cluster path is covered against a real
cluster by tests/integration/test_redis_cluster_support.py, which only runs
under --run-cluster-tests.
"""

import pytest
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster

from redisvl.extensions.cache import base as cache_base
from redisvl.extensions.cache.base import BaseCache

PREFIX = "clear_cursor_test"
MATCH = f"{PREFIX}:*"


class MockClusterClient:
    """Stand-in for RedisCluster SCAN/DEL, driven by a per-node key layout.

    Keys live on named primaries and each primary pages through its own keys on
    its own cursor, which is what makes the cluster contract testable.

    Two built-in guards, so failures are loud rather than silent:
    ``scan`` raises if a node-local cursor is ever broadcast or if the call
    count runs away (a non-advancing cursor would otherwise hang the suite --
    pytest-timeout is not installed), and ``delete`` raises on a zero-argument
    call, which real Redis rejects with "wrong number of arguments".
    """

    # Exercise the real upstream loop instead of imitating it.
    scan_iter = RedisCluster.scan_iter

    def __init__(self, keys_by_node, page=2, max_calls=200):
        self.keys_by_node = {n: list(k) for n, k in keys_by_node.items()}
        self.page = page
        self.max_calls = max_calls
        self.scan_calls = []
        self.delete_batches = []

    @property
    def deleted(self):
        return [key for batch in self.delete_batches for key in batch]

    def _page(self, node, cursor):
        keys = self.keys_by_node[node]
        nxt = cursor + self.page
        return (0 if nxt >= len(keys) else nxt), keys[cursor : cursor + self.page]

    def _next_scan(self, cursor, match, target_nodes):
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

    def _record_delete(self, keys):
        if not keys:
            raise AssertionError("DEL issued with no keys; real Redis rejects this")
        self.delete_batches.append(list(keys))
        return len(keys)

    def scan(self, cursor=0, match=None, count=None, target_nodes=None, _type=None):
        return self._next_scan(cursor, match, target_nodes)

    def delete(self, *keys):
        return self._record_delete(keys)

    def get_node(self, host=None, port=None, node_name=None):
        return f"node-object:{node_name}"


class MockAsyncClusterClient(MockClusterClient):
    """Async variant, bound to the async cluster client's ``scan_iter``."""

    scan_iter = AsyncRedisCluster.scan_iter

    async def scan(
        self, cursor=0, match=None, count=None, target_nodes=None, _type=None
    ):
        return self._next_scan(cursor, match, target_nodes)

    async def delete(self, *keys):
        return self._record_delete(keys)


def _layout():
    """Three primaries with uneven scan depth, one done after the broadcast."""
    return {
        "node-1": [f"{PREFIX}:a{i}" for i in range(6)],  # 3 rounds
        "node-2": [f"{PREFIX}:b{i}" for i in range(2)],  # done on the broadcast
        "node-3": [f"{PREFIX}:c{i}" for i in range(4)],  # 2 rounds
    }


def _assert_drained(client, layout):
    """Every primary emptied, nothing scanned outside the cache prefix."""
    assert sorted(client.deleted) == sorted(k for ks in layout.values() for k in ks)
    # Scoping the match pattern is what keeps clear() from wiping the whole DB.
    assert all(c["match"] == MATCH for c in client.scan_calls)
    # Anti-vacuity: a single broadcast page would prove nothing about paging.
    assert any(
        c["target_nodes"] is not None for c in client.scan_calls
    ), "no per-node continuation happened; test proves nothing"
    # node-2 finished on the broadcast and must not be revisited.
    assert not any(c["target_nodes"] == "node-object:node-2" for c in client.scan_calls)


class TestClearOnCluster:
    def test_drains_every_primary(self):
        layout = _layout()
        client = MockClusterClient(layout)

        BaseCache(name=PREFIX, redis_client=client).clear()

        _assert_drained(client, layout)

    def test_empty_cache_issues_no_delete(self):
        client = MockClusterClient({"node-1": [], "node-2": []})

        BaseCache(name=PREFIX, redis_client=client).clear()

        assert client.delete_batches == []

    def test_deletes_are_batched(self, monkeypatch):
        # Patch the batch size rather than asserting against its real value, so
        # tuning CLEAR_BATCH_SIZE in production doesn't touch this test.
        monkeypatch.setattr(cache_base, "CLEAR_BATCH_SIZE", 3)
        keys = [f"{PREFIX}:{i}" for i in range(8)]
        client = MockClusterClient({"node-1": keys}, page=3)

        BaseCache(name=PREFIX, redis_client=client).clear()

        # Bounded batches, and the trailing remainder is still flushed.
        assert [len(b) for b in client.delete_batches] == [3, 3, 2]
        assert sorted(client.deleted) == sorted(keys)


class TestAsyncClearOnCluster:
    """aclear is written separately from clear, so it needs its own coverage."""

    @pytest.mark.asyncio
    async def test_drains_every_primary(self):
        layout = _layout()
        client = MockAsyncClusterClient(layout)

        await BaseCache(name=PREFIX, async_redis_client=client).aclear()

        _assert_drained(client, layout)

    @pytest.mark.asyncio
    async def test_empty_cache_issues_no_delete(self):
        client = MockAsyncClusterClient({"node-1": [], "node-2": []})

        await BaseCache(name=PREFIX, async_redis_client=client).aclear()

        assert client.delete_batches == []
