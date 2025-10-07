"""Integration tests for AsyncRedisCluster connection with cluster parameter (issue #346)."""

import pytest
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster

from redisvl.redis.connection import RedisConnectionFactory


@pytest.mark.asyncio
class TestAsyncClusterConnection:
    """Test AsyncRedisCluster connections with cluster parameter."""

    async def test_get_aredis_connection_with_cluster_url(self, redis_url):
        """Test _get_aredis_connection handles cluster parameter in URL."""
        # Add cluster=true to the URL (even though it's not actually a cluster)
        # This simulates the issue where cluster=true is passed but not accepted
        cluster_url = (
            f"{redis_url}?cluster=false"  # Use false since we don't have a real cluster
        )

        # This should not raise a TypeError
        client = await RedisConnectionFactory._get_aredis_connection(cluster_url)

        assert client is not None
        assert isinstance(client, (AsyncRedis, AsyncRedisCluster))

        await client.aclose()

    async def test_get_aredis_connection_with_cluster_kwargs(self, redis_url):
        """Test _get_aredis_connection handles cluster parameter in kwargs."""
        # This should not raise a TypeError even with cluster in kwargs
        client = await RedisConnectionFactory._get_aredis_connection(
            redis_url, cluster=False  # Use false since we don't have a real cluster
        )

        assert client is not None
        assert isinstance(client, (AsyncRedis, AsyncRedisCluster))

        await client.aclose()

    @pytest.mark.requires_cluster
    async def test_get_async_redis_cluster_connection_with_params(
        self, redis_cluster_url
    ):
        """Test get_async_redis_cluster_connection with cluster parameter."""
        # Add cluster=true to the URL
        cluster_url_with_param = f"{redis_cluster_url}?cluster=true"

        # This should not raise a TypeError
        client = RedisConnectionFactory.get_async_redis_cluster_connection(
            cluster_url_with_param
        )

        assert client is not None
        assert isinstance(client, AsyncRedisCluster)

        await client.aclose()

    @pytest.mark.requires_cluster
    async def test_get_async_redis_cluster_connection_with_kwargs(
        self, redis_cluster_url
    ):
        """Test get_async_redis_cluster_connection with cluster in kwargs."""
        # This should not raise a TypeError
        client = RedisConnectionFactory.get_async_redis_cluster_connection(
            redis_cluster_url, cluster=True
        )

        assert client is not None
        assert isinstance(client, AsyncRedisCluster)

        await client.aclose()

    async def test_get_async_redis_connection_deprecated_with_cluster(self, redis_url):
        """Test deprecated get_async_redis_connection handles cluster parameter."""
        # Add cluster=false to the URL
        cluster_url = f"{redis_url}?cluster=false"

        with pytest.warns(DeprecationWarning):
            # This should not raise a TypeError
            client = RedisConnectionFactory.get_async_redis_connection(cluster_url)

        assert client is not None
        assert isinstance(client, (AsyncRedis, AsyncRedisCluster))

        await client.aclose()
