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


class FakeAsyncClusterPipeline:
    """Async pipeline that drops its queue when awaited, like ClusterPipeline."""

    def __init__(self):
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

    async def execute(self):
        self.executed.append(list(self.queued))
        results = [1] * len(self.queued)
        self.queued.clear()
        return results


class FakeAsyncClusterClient:
    """Minimal async client that hands out FakeAsyncClusterPipeline instances."""

    def __init__(self):
        self.pipelines: list[FakeAsyncClusterPipeline] = []

    def pipeline(self, transaction=False):
        pipeline = FakeAsyncClusterPipeline()
        self.pipelines.append(pipeline)
        return pipeline


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
