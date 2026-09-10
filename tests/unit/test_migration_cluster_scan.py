"""Regression tests for migration SCAN key enumeration on Redis Cluster.

Six migration call sites used to feed the previous reply's cursor straight back
into ``client.scan(cursor=...)``. On a cluster client that reply is a
``{node_name: cursor}`` mapping, so ``cursor == 0`` is never true and the second
iteration passes a dict as the cursor -- redis-py rejects it with ``DataError``.
That is a hard crash on any cluster, not a silent under-count.

The fake below reproduces exactly that: it replies with per-node cursors like a
real cluster and raises ``DataError`` if handed a non-integer cursor, so the old
loop fails here the same way it fails against real Redis.

Covers the four always-reachable sites (validator key counts, planner key
sampling; sync and async). The two executor sites
(``_enumerate_with_scan``) are the same one-line ``scan_iter`` pattern but sit
behind index-info mocking, and they only run on fallback paths; they are left to
the migration integration tests.
"""

import pytest
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
from redis.exceptions import DataError

from redisvl.migration.async_planner import AsyncMigrationPlanner
from redisvl.migration.async_validation import AsyncMigrationValidator
from redisvl.migration.planner import MigrationPlanner
from redisvl.migration.validation import MigrationValidator

PREFIX = "doc"
NODES = {
    "node-1": [f"{PREFIX}:a{i}".encode() for i in range(6)],
    "node-2": [f"{PREFIX}:b{i}".encode() for i in range(2)],
    "node-3": [f"{PREFIX}:c{i}".encode() for i in range(4)],
}
ALL_KEYS = sorted(k.decode() for ks in NODES.values() for k in ks)


class MockClusterClient:
    """Cluster-shaped SCAN: per-node cursors, and DataError on a dict cursor."""

    scan_iter = RedisCluster.scan_iter

    def __init__(self, keys_by_node=None, page=2, max_calls=200):
        self.keys_by_node = {n: list(k) for n, k in (keys_by_node or NODES).items()}
        self.page = page
        self.max_calls = max_calls
        self.scan_calls = 0

    def _page(self, node, cursor):
        keys = self.keys_by_node[node]
        nxt = cursor + self.page
        return (0 if nxt >= len(keys) else nxt), keys[cursor : cursor + self.page]

    def _next_scan(self, cursor, target_nodes):
        self.scan_calls += 1
        if self.scan_calls > self.max_calls:
            raise AssertionError(f"{self.scan_calls} SCAN calls -- not advancing")
        # What redis-py's encoder does when the old loop hands back the mapping.
        if not isinstance(cursor, (int, str, bytes)):
            raise DataError(
                f"Invalid input of type: {type(cursor).__name__!r}. "
                "Convert to a bytes, string, int or float first."
            )
        cursor = int(cursor)
        if target_nodes is None:
            assert cursor == 0, f"node-local cursor {cursor!r} broadcast to primaries"
            pages = {n: self._page(n, 0) for n in self.keys_by_node}
            return (
                {n: c for n, (c, _) in pages.items()},
                [k for _, ks in pages.values() for k in ks],
            )
        node = str(target_nodes).removeprefix("node-object:")
        nxt, keys = self._page(node, cursor)
        return {node: nxt}, keys

    def scan(self, cursor=0, match=None, count=None, target_nodes=None, _type=None):
        return self._next_scan(cursor, target_nodes)

    def get_node(self, host=None, port=None, node_name=None):
        return f"node-object:{node_name}"


class MockAsyncClusterClient(MockClusterClient):
    scan_iter = AsyncRedisCluster.scan_iter

    async def scan(
        self, cursor=0, match=None, count=None, target_nodes=None, _type=None
    ):
        return self._next_scan(cursor, target_nodes)


class _Index:
    """Minimal SearchIndex stand-in for the validators' key-count path."""

    def __init__(self, client):
        self.client = client
        self.schema = type(
            "S",
            (),
            {"index": type("I", (), {"prefix": PREFIX, "key_separator": ":"})()},
        )()


class TestValidatorCountsKeysOnCluster:
    """_count_index_keys runs on every validate, so this is the hottest path."""

    def test_counts_across_all_primaries(self):
        client = MockClusterClient()
        validator = MigrationValidator.__new__(MigrationValidator)

        assert validator._count_index_keys(_Index(client)) == len(ALL_KEYS)

    @pytest.mark.asyncio
    async def test_counts_across_all_primaries_async(self):
        client = MockAsyncClusterClient()
        validator = AsyncMigrationValidator.__new__(AsyncMigrationValidator)

        assert await validator._count_index_keys(_Index(client)) == len(ALL_KEYS)


class TestPlannerSamplesKeysOnCluster:
    """_sample_keys returns early at the sample limit, mid-page."""

    def test_samples_across_primaries(self):
        client = MockClusterClient()
        planner = MigrationPlanner.__new__(MigrationPlanner)
        planner.key_sample_limit = len(ALL_KEYS)

        sample = planner._sample_keys(
            client=client, prefixes=[PREFIX], key_separator=":"
        )

        assert sorted(sample) == ALL_KEYS

    @pytest.mark.asyncio
    async def test_samples_across_primaries_async(self):
        client = MockAsyncClusterClient()
        planner = AsyncMigrationPlanner.__new__(AsyncMigrationPlanner)
        planner.key_sample_limit = len(ALL_KEYS)

        sample = await planner._async_sample_keys(
            client=client, prefixes=[PREFIX], key_separator=":"
        )

        assert sorted(sample) == ALL_KEYS
