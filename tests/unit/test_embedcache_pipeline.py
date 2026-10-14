"""Unit tests for how EmbeddingsCache drives async Redis pipelines.

Queueing a command on a redis-py async pipeline is synchronous and returns the
pipeline itself, so a queued command must never be awaited. On
``redis.asyncio.cluster.ClusterPipeline`` awaiting it calls ``initialize()``,
which clears the queued commands -- batched writes then vanish without an error.

The fake below reproduces exactly those two behaviours, so these tests fail if
an ``await`` is ever reintroduced. They need no Redis, so unlike the cluster
tests in ``tests/integration/test_redis_cluster_support.py`` they run by default.
"""

import pytest

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
from redisvl.extensions.cache.embeddings.schema import CacheEntry


class FakeAsyncClusterPipeline:
    """Async pipeline that drops its queue when awaited, like ClusterPipeline.

    HGETALL replies come from ``store``; every other command replies 1.
    """

    def __init__(self, store=None):
        self.store = store or {}
        self.queued: list[tuple[str, str]] = []
        self.executed: list[list[tuple[str, str]]] = []

    def __await__(self):
        async def initialize():
            self.queued.clear()
            return self

        return initialize().__await__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def hset(self, name, mapping):
        self.queued.append(("hset", name))
        return self

    def expire(self, key, ttl):
        self.queued.append(("expire", key))
        return self

    def exists(self, key):
        self.queued.append(("exists", key))
        return self

    def hgetall(self, key):
        self.queued.append(("hgetall", key))
        return self

    def _run(self):
        self.executed.append(list(self.queued))
        results = [
            self.store.get(key, {}) if command == "hgetall" else 1
            for command, key in self.queued
        ]
        self.queued.clear()
        return results

    async def execute(self):
        return self._run()


class FakeSyncPipeline(FakeAsyncClusterPipeline):
    """The same recorder behind redis-py's sync pipeline interface."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self):
        return self._run()


class FakeAsyncClusterClient:
    """Minimal async client that hands out FakeAsyncClusterPipeline instances."""

    pipeline_class = FakeAsyncClusterPipeline

    def __init__(self, store=None):
        self.store = store
        self.pipelines: list[FakeAsyncClusterPipeline] = []
        self.transactions: list[bool] = []

    def pipeline(self, transaction=True):
        self.transactions.append(transaction)
        pipeline = self.pipeline_class(self.store)
        self.pipelines.append(pipeline)
        return pipeline


class FakeSyncClient(FakeAsyncClusterClient):
    pipeline_class = FakeSyncPipeline


def make_cache(client, ttl=None):
    return EmbeddingsCache(name="embedcache", ttl=ttl, async_redis_client=client)


@pytest.mark.parametrize(
    "ttl, expected",
    [
        (None, [("hset", "embedcache:a"), ("hset", "embedcache:b")]),
        (
            60,
            [
                ("hset", "embedcache:a"),
                ("expire", "embedcache:a"),
                ("hset", "embedcache:b"),
                ("expire", "embedcache:b"),
            ],
        ),
    ],
    ids=["no_ttl", "with_ttl"],
)
@pytest.mark.asyncio
async def test_amset_sends_every_command_in_one_pipeline(monkeypatch, ttl, expected):
    """amset must queue each write, and its expiry, into a single execute()."""
    monkeypatch.setattr(
        EmbeddingsCache, "_make_entry_id", lambda self, content, model_name: content
    )
    client = FakeAsyncClusterClient()
    cache = make_cache(client, ttl=ttl)
    items = [
        {"content": name, "model_name": "m", "embedding": [0.1, 0.2]}
        for name in ("a", "b")
    ]

    keys = await cache.amset(items)

    assert keys == ["embedcache:a", "embedcache:b"]
    # A dropped queue would show up as a single empty execute().
    assert [pipeline.executed for pipeline in client.pipelines] == [[expected]]


@pytest.mark.asyncio
async def test_amexists_by_keys_returns_one_result_per_key():
    """A dropped queue made this return [] rather than a bool per key."""
    client = FakeAsyncClusterClient()
    cache = make_cache(client)

    assert await cache.amexists_by_keys(["k1", "k2", "k3"]) == [True, True, True]


HIT = CacheEntry(entry_id="a", content="a", model_name="m", embedding=[0.1])


@pytest.mark.parametrize(
    "ttl, expected",
    [
        (None, [("hgetall", "embedcache:a"), ("hgetall", "embedcache:b")]),
        (
            60,
            [
                ("hgetall", "embedcache:a"),
                ("expire", "embedcache:a"),
                ("hgetall", "embedcache:b"),
                ("expire", "embedcache:b"),
            ],
        ),
    ],
    ids=["no_ttl", "with_ttl"],
)
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_mget_by_keys_refreshes_ttls_in_the_lookup_pipeline(
    use_async, ttl, expected
):
    """Lookups and TTL refreshes share one non-transactional execute().

    A transactional pipeline would raise CROSSSLOT on a Redis Cluster, and a
    refresh outside the pipeline costs one extra round trip per hit.
    """
    store = {"embedcache:a": HIT.to_dict()}
    if use_async:
        client = FakeAsyncClusterClient(store)
        cache = make_cache(client, ttl=ttl)
        results = await cache.amget_by_keys(["embedcache:a", "embedcache:b"])
    else:
        client = FakeSyncClient(store)
        cache = EmbeddingsCache(name="embedcache", ttl=ttl, redis_client=client)
        results = cache.mget_by_keys(["embedcache:a", "embedcache:b"])

    assert results == [HIT.model_dump(exclude_none=True), None]
    assert client.transactions == [False]
    assert [pipeline.executed for pipeline in client.pipelines] == [[expected]]
