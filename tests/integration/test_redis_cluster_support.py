"""Tests for Redis Cluster support in RedisVL."""

import asyncio

import pytest
from redis import Redis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
from redisvl.extensions.router.semantic import Route, SemanticRouter
from redisvl.index import SearchIndex
from redisvl.index.index import AsyncSearchIndex
from redisvl.query.query import TextQuery
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.schema import IndexSchema


@pytest.mark.requires_cluster
def test_sync_client_validation(redis_url, redis_cluster_url):
    """Test validation of sync Redis client types."""
    # Test regular Redis client
    redis_client = Redis.from_url(redis_url)
    RedisConnectionFactory.validate_sync_redis(redis_client)

    # Test with RedisCluster client type
    cluster_client = RedisCluster.from_url(redis_cluster_url)
    RedisConnectionFactory.validate_sync_redis(cluster_client)


@pytest.mark.requires_cluster
@pytest.mark.asyncio
async def test_async_client_validation(redis_cluster_url):
    """Test validation of async Redis client types."""
    async_cluster_client = await RedisConnectionFactory._get_aredis_connection(
        redis_cluster_url
    )
    await RedisConnectionFactory.validate_async_redis(async_cluster_client)


@pytest.mark.requires_cluster
@pytest.mark.asyncio
async def test_sync_to_async_conversion_rejects_cluster_client(redis_cluster_url):
    """Test that sync-to-async conversion rejects RedisCluster clients."""
    cluster_client = RedisCluster.from_url(redis_cluster_url)
    with pytest.raises(
        ValueError, match="RedisCluster is not supported for sync-to-async conversion."
    ):
        RedisConnectionFactory.sync_to_async_redis(cluster_client)


@pytest.mark.requires_cluster
def test_search_index_cluster_client(redis_cluster_url, redis_test_name):
    """Test that SearchIndex correctly accepts RedisCluster clients."""
    # Create a simple schema
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": redis_test_name("test_cluster_index"),
                "prefix": redis_test_name("test_cluster"),
            },
            "fields": [
                {"name": "name", "type": "text"},
                {"name": "age", "type": "numeric"},
            ],
        }
    )

    cluster_client = RedisCluster.from_url(redis_cluster_url)
    index = SearchIndex(schema=schema, redis_client=cluster_client)
    try:
        index.create(overwrite=True, drop=True)
        index.load([{"name": "test1", "age": 30}])
        results = index.query(TextQuery("test1", "name"))
        assert results[0]["name"] == "test1"
    finally:
        index.delete(drop=True)


@pytest.mark.requires_cluster
def test_search_index_cluster_info(redis_cluster_url, redis_test_name):
    """Test .info() method on SearchIndex with RedisCluster client."""
    index_name = redis_test_name("test_cluster_info")
    schema = IndexSchema.from_dict(
        {
            "index": {"name": index_name, "prefix": redis_test_name("test_info")},
            "fields": [{"name": "name", "type": "text"}],
        }
    )
    client = RedisCluster.from_url(redis_cluster_url)
    index = SearchIndex(schema=schema, redis_client=client)
    try:
        index.create(overwrite=True, drop=True)
        info = index.info()
        assert isinstance(info, dict)
        assert info.get("index_name", None) == index_name
    finally:
        index.delete(drop=True)


@pytest.mark.requires_cluster
@pytest.mark.asyncio
async def test_async_search_index_cluster_info(redis_cluster_url, redis_test_name):
    """Test .info() method on AsyncSearchIndex with AsyncRedisCluster client."""
    index_name = redis_test_name("async_cluster_info")
    schema = IndexSchema.from_dict(
        {
            "index": {"name": index_name, "prefix": redis_test_name("async_info")},
            "fields": [{"name": "name", "type": "text"}],
        }
    )
    client = AsyncRedisCluster.from_url(redis_cluster_url)
    index = AsyncSearchIndex(schema=schema, redis_client=client)
    try:
        await index.create(overwrite=True, drop=True)
        info = await index.info()
        assert isinstance(info, dict)
        assert info.get("index_name", None) == index_name
    finally:
        await index.delete(drop=True)
        await client.aclose()


@pytest.mark.requires_cluster
@pytest.mark.asyncio
async def test_async_search_index_client(redis_cluster_url, redis_test_name):
    """Test that AsyncSearchIndex correctly handles AsyncRedis clients."""
    # Create a simple schema
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": redis_test_name("async_test_index"),
                "prefix": redis_test_name("async_test"),
            },
            "fields": [
                {"name": "name", "type": "text"},
                {"name": "age", "type": "numeric"},
            ],
        }
    )

    # Test with AsyncRedis client
    cluster_client = AsyncRedisCluster.from_url(redis_cluster_url)
    index = AsyncSearchIndex(schema=schema, redis_client=cluster_client)
    try:
        await index.create(overwrite=True, drop=True)
        await index.load([{"name": "async_test", "age": 25}])
        results = await index.query(TextQuery("async_test", "name"))
        assert results[0]["name"] == "async_test"
        await index.delete(drop=True)
    finally:
        # Manually close the cluster client to prevent connection leaks
        await cluster_client.aclose()


@pytest.mark.requires_cluster
@pytest.mark.asyncio
async def test_embeddings_cache_cluster_async(redis_cluster_url, redis_test_name):
    """Test that EmbeddingsCache correctly handles AsyncRedisCluster clients."""
    cluster_client = RedisConnectionFactory.get_async_redis_cluster_connection(
        redis_cluster_url
    )
    cache = EmbeddingsCache(
        name=redis_test_name("embedcache"), async_redis_client=cluster_client
    )

    try:
        await cache.aset(
            content="hey",
            model_name="test",
            embedding=[1, 2, 3],
        )
        result = await cache.aget("hey", "test")
        assert result is not None
        assert result["embedding"] == [1, 2, 3]
        await cache.aclear()
        assert await cache.aget("hey", "test") is None
    finally:
        # Manually close the cluster client to prevent connection leaks
        await cluster_client.aclose()


@pytest.mark.requires_cluster
def test_embeddings_cache_cluster_sync(redis_cluster_url, redis_test_name):
    """Test that EmbeddingsCache correctly handles RedisCluster clients."""
    cluster_client = RedisCluster.from_url(redis_cluster_url)
    cache = EmbeddingsCache(
        name=redis_test_name("embedcache"), redis_client=cluster_client
    )

    for i in range(100):
        cache.set(
            content=f"hey_{i}",
            model_name="test",
            embedding=[1, 2, 3],
        )
    result = cache.get("hey_0", "test")
    assert result is not None
    assert result["embedding"] == [1, 2, 3]
    cache.clear()
    assert cache.get("hey_0", "test") is None

    cache.mset(
        [
            {"content": "hey_0", "model_name": "test", "embedding": [1, 2, 3]},
            {"content": "hey_1", "model_name": "test", "embedding": [1, 2, 3]},
        ]
    )
    result = cache.mget(["hey_0", "hey_1"], "test")
    assert result[0] is not None
    assert result[1] is not None
    assert result[0]["embedding"] == [1, 2, 3]
    assert result[1]["embedding"] == [1, 2, 3]
    cache.clear()
    assert cache.mget(["hey_0", "hey_1"], "test") == [None, None]


@pytest.mark.requires_cluster
def test_semantic_router_cluster_client(
    redis_cluster_url, hf_vectorizer, redis_test_name
):
    """Test that SemanticRouter works correctly with RedisCluster clients."""
    routes = [
        Route(
            name="General Inquiry",
            references=["What are your hours?", "Tell me about your services."],
        ),
        Route(
            name="Technical Support",
            references=[
                "I have an issue with my account.",
                "My product is broken.",
            ],
        ),
    ]
    client = RedisCluster.from_url(redis_cluster_url)

    router_name = redis_test_name("test_cluster_router")
    router = SemanticRouter(
        name=router_name,
        routes=routes,
        vectorizer=hf_vectorizer,
        redis_client=client,
        overwrite=True,
    )

    query_text = "I need help with my login."
    matched_route = router(query_text)

    assert matched_route is not None
    assert matched_route.name == "Technical Support"

    if router._index and router._index.exists():
        router._index.delete(drop=True)


# =============================================================================
# BaseCache.clear/aclear on a cluster keyspace larger than one SCAN page.
#
# On a cluster, SCAN is broadcast to all primaries and replies with a
# {node_name: cursor} mapping of node-local cursors. The clear loop used to
# leave its cursor at 0, so it re-issued SCAN 0 forever. It accidentally made
# progress when every key in the DB matched the cache prefix -- deleting the
# first page shrank the keyspace -- which is why small-cache tests passed. The
# genuine hang needs unrelated keys in the DB, the normal case for redisvl,
# where index docs and caches share a keyspace: then a SCAN 0 page can match
# nothing, nothing gets deleted, and the loop spins with zero progress forever.
#
# These tests therefore seed BOTH unrelated keys and a multi-page cache, and
# bound the SCAN count during clear() so a regression fails loudly instead of
# hanging the suite (pytest-timeout is not installed).
#
# Note: count keys with scan_iter, never KEYS or DBSIZE. Those are routed to a
# single node on a cluster, so they silently report roughly one shard's worth.
# =============================================================================

CLEAR_NOISE_KEYS = 2000
CLEAR_CACHE_KEYS = 600
MAX_CLEAR_SCANS = 400


def _count_keys(client, pattern):
    return sum(1 for _ in client.scan_iter(match=pattern, count=500))


async def _acount_keys(client, pattern):
    total = 0
    async for _ in client.scan_iter(match=pattern, count=500):
        total += 1
    return total


def _drop_keys(client, pattern):
    keys = list(client.scan_iter(match=pattern, count=500))
    if keys:
        client.delete(*keys)


async def _adrop_keys(client, pattern):
    keys = [k async for k in client.scan_iter(match=pattern, count=500)]
    if keys:
        await client.delete(*keys)


@pytest.mark.requires_cluster
def test_embeddings_cache_clear_multipage_cluster(redis_cluster_url, redis_test_name):
    """clear() empties a multi-page cache on a cluster and spares other keys."""
    cluster_client = RedisCluster.from_url(redis_cluster_url)
    name = redis_test_name("clear_multipage")
    noise_prefix = redis_test_name("clear_noise")
    cache = EmbeddingsCache(name=name, redis_client=cluster_client)

    try:
        # Unrelated keys: without these the buggy loop accidentally terminates.
        pipe = cluster_client.pipeline()
        for i in range(CLEAR_NOISE_KEYS):
            pipe.set(f"{noise_prefix}:{i}", "keep")
        pipe.execute()

        cache.mset(
            [
                {
                    "content": f"content-{i}",
                    "model_name": "test",
                    "embedding": [0.1, 0.2, 0.3],
                }
                for i in range(CLEAR_CACHE_KEYS)
            ]
        )
        assert _count_keys(cluster_client, f"{name}:*") == CLEAR_CACHE_KEYS

        # Bound the SCAN calls clear() itself issues: the pre-fix loop never
        # terminates, and pytest-timeout is not installed.
        real_scan = cluster_client.scan
        calls: list = []

        def counting_scan(*args, **kwargs):
            calls.append(kwargs.get("cursor", args[0] if args else None))
            if len(calls) > MAX_CLEAR_SCANS:
                raise AssertionError(
                    f"clear() issued {len(calls)} SCAN calls without finishing; "
                    f"first cursors: {calls[:8]}"
                )
            return real_scan(*args, **kwargs)

        cluster_client.scan = counting_scan
        try:
            cache.clear()
        finally:
            cluster_client.scan = real_scan

        # The real-world bug: keys past the first page survived.
        assert _count_keys(cluster_client, f"{name}:*") == 0
        # clear() must not touch anything outside its own prefix.
        assert _count_keys(cluster_client, f"{noise_prefix}:*") == CLEAR_NOISE_KEYS
        # Guard against a vacuous pass: paging must actually have happened.
        assert len(calls) > 1, "one SCAN sufficed; raise CLEAR_CACHE_KEYS"
    finally:
        _drop_keys(cluster_client, f"{name}:*")
        _drop_keys(cluster_client, f"{noise_prefix}:*")
        cluster_client.close()


@pytest.mark.requires_cluster
@pytest.mark.asyncio
async def test_embeddings_cache_aclear_multipage_cluster(
    redis_cluster_url, redis_test_name
):
    """aclear() must match clear(): aclear is a separate code path."""
    cluster_client = RedisConnectionFactory.get_async_redis_cluster_connection(
        redis_cluster_url
    )
    name = redis_test_name("aclear_multipage")
    noise_prefix = redis_test_name("aclear_noise")
    cache = EmbeddingsCache(name=name, async_redis_client=cluster_client)

    try:
        pipe = cluster_client.pipeline()
        for i in range(CLEAR_NOISE_KEYS):
            pipe.set(f"{noise_prefix}:{i}", "keep")
        await pipe.execute()

        # Seeded with aset, not amset: amset silently writes nothing on an
        # async cluster client (it awaits the pipeline returned by the queueing
        # call, which drains the queue). That is a separate bug; this test is
        # about aclear, so don't let it depend on the broken path.
        await asyncio.gather(
            *(
                cache.aset(
                    content=f"content-{i}",
                    model_name="test",
                    embedding=[0.1, 0.2, 0.3],
                )
                for i in range(CLEAR_CACHE_KEYS)
            )
        )
        assert await _acount_keys(cluster_client, f"{name}:*") == CLEAR_CACHE_KEYS

        real_scan = cluster_client.scan
        calls: list = []

        async def counting_scan(*args, **kwargs):
            calls.append(kwargs.get("cursor", args[0] if args else None))
            if len(calls) > MAX_CLEAR_SCANS:
                raise AssertionError(
                    f"aclear() issued {len(calls)} SCAN calls without finishing; "
                    f"first cursors: {calls[:8]}"
                )
            return await real_scan(*args, **kwargs)

        cluster_client.scan = counting_scan
        try:
            await cache.aclear()
        finally:
            cluster_client.scan = real_scan

        assert await _acount_keys(cluster_client, f"{name}:*") == 0
        assert (
            await _acount_keys(cluster_client, f"{noise_prefix}:*") == CLEAR_NOISE_KEYS
        )
        assert len(calls) > 1, "one SCAN sufficed; raise CLEAR_CACHE_KEYS"
    finally:
        await _adrop_keys(cluster_client, f"{name}:*")
        await _adrop_keys(cluster_client, f"{noise_prefix}:*")
        await cluster_client.aclose()
