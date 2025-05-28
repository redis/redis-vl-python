"""Tests for Redis Cluster support in RedisVL."""

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
def test_search_index_cluster_client(redis_cluster_url):
    """Test that SearchIndex correctly accepts RedisCluster clients."""
    # Create a simple schema
    schema = IndexSchema.from_dict(
        {
            "index": {"name": "test_cluster_index", "prefix": "test_cluster"},
            "fields": [
                {"name": "name", "type": "text"},
                {"name": "age", "type": "numeric"},
            ],
        }
    )

    cluster_client = RedisCluster.from_url(redis_cluster_url)
    index = SearchIndex(schema=schema, redis_client=cluster_client)
    index.create(overwrite=True)
    index.load([{"name": "test1", "age": 30}])
    results = index.query(TextQuery("test1", "name"))
    assert results[0]["name"] == "test1"
    index.delete(drop=True)


@pytest.mark.requires_cluster
@pytest.mark.asyncio
async def test_async_search_index_client(redis_cluster_url):
    """Test that AsyncSearchIndex correctly handles AsyncRedis clients."""
    # Create a simple schema
    schema = IndexSchema.from_dict(
        {
            "index": {"name": "async_test_index", "prefix": "async_test"},
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
        await index.create(overwrite=True)
        await index.load([{"name": "async_test", "age": 25}])
        results = await index.query(TextQuery("async_test", "name"))
        assert results[0]["name"] == "async_test"
        await index.delete(drop=True)
    finally:
        # Manually close the cluster client to prevent connection leaks
        await cluster_client.aclose()


@pytest.mark.requires_cluster
@pytest.mark.asyncio
async def test_embeddings_cache_cluster_async(redis_cluster_url):
    """Test that EmbeddingsCache correctly handles AsyncRedisCluster clients."""
    cluster_client = RedisConnectionFactory.get_async_redis_cluster_connection(
        redis_cluster_url
    )
    cache = EmbeddingsCache(async_redis_client=cluster_client)

    try:
        await cache.aset(
            text="hey",
            model_name="test",
            embedding=[1, 2, 3],
        )
        result = await cache.aget("hey", "test")
        assert result is not None
        assert result["embedding"] == [1, 2, 3]
        await cache.aclear()
    finally:
        # Manually close the cluster client to prevent connection leaks
        await cluster_client.aclose()


@pytest.mark.requires_cluster
def test_embeddings_cache_cluster_sync(redis_cluster_url):
    """Test that EmbeddingsCache correctly handles RedisCluster clients."""
    cluster_client = RedisCluster.from_url(redis_cluster_url)
    cache = EmbeddingsCache(redis_client=cluster_client)

    for i in range(100):
        cache.set(
            text=f"hey_{i}",
            model_name="test",
            embedding=[1, 2, 3],
        )
    result = cache.get("hey_0", "test")
    assert result is not None
    assert result["embedding"] == [1, 2, 3]
    cache.clear()

    cache.mset(
        [
            {"text": "hey_0", "model_name": "test", "embedding": [1, 2, 3]},
            {"text": "hey_1", "model_name": "test", "embedding": [1, 2, 3]},
        ]
    )
    result = cache.mget(["hey_0", "hey_1"], "test")
    assert result[0] is not None
    assert result[1] is not None
    assert result[0]["embedding"] == [1, 2, 3]
    assert result[1]["embedding"] == [1, 2, 3]
    cache.clear()


@pytest.mark.requires_cluster
def test_semantic_router_cluster_client(redis_cluster_url, hf_vectorizer):
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

    router_name = "test_cluster_router"
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
