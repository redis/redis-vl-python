from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redisvl.redis.connection import RedisConnectionFactory


def test_sync_connection_defaults_to_resp2():
    client = MagicMock()

    with patch(
        "redisvl.redis.connection.Redis.from_url", return_value=client
    ) as from_url:
        RedisConnectionFactory.get_redis_connection("redis://localhost:6379")

    from_url.assert_called_once_with("redis://localhost:6379", protocol=2)


def test_sync_connection_preserves_explicit_protocol():
    client = MagicMock()

    with patch(
        "redisvl.redis.connection.Redis.from_url", return_value=client
    ) as from_url:
        RedisConnectionFactory.get_redis_connection(
            "redis://localhost:6379", protocol=3
        )

    from_url.assert_called_once_with("redis://localhost:6379", protocol=3)


def test_sync_cluster_connection_defaults_to_resp2():
    with patch("redisvl.redis.connection.RedisCluster.from_url") as from_url:
        RedisConnectionFactory.get_redis_cluster_connection("redis://localhost:6379")

    from_url.assert_called_once_with("redis://localhost:6379", protocol=2)


def test_sync_cluster_connection_preserves_explicit_protocol():
    with patch("redisvl.redis.connection.RedisCluster.from_url") as from_url:
        RedisConnectionFactory.get_redis_cluster_connection(
            "redis://localhost:6379", protocol=3
        )

    from_url.assert_called_once_with("redis://localhost:6379", protocol=3)


@pytest.mark.asyncio
async def test_async_connection_defaults_to_resp2():
    client = AsyncMock()

    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=client
    ) as from_url:
        await RedisConnectionFactory._get_aredis_connection("redis://localhost:6379")

    from_url.assert_called_once_with("redis://localhost:6379", protocol=2)


@pytest.mark.asyncio
async def test_async_connection_preserves_explicit_protocol():
    client = AsyncMock()

    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=client
    ) as from_url:
        await RedisConnectionFactory._get_aredis_connection(
            "redis://localhost:6379", protocol=3
        )

    from_url.assert_called_once_with("redis://localhost:6379", protocol=3)


def test_deprecated_async_connection_defaults_to_resp2():
    with (
        pytest.warns(DeprecationWarning),
        patch("redisvl.redis.connection.AsyncRedis.from_url") as from_url,
    ):
        RedisConnectionFactory.get_async_redis_connection("redis://localhost:6379")

    from_url.assert_called_once_with("redis://localhost:6379", protocol=2)


def test_deprecated_async_connection_preserves_explicit_protocol():
    with (
        pytest.warns(DeprecationWarning),
        patch("redisvl.redis.connection.AsyncRedis.from_url") as from_url,
    ):
        RedisConnectionFactory.get_async_redis_connection(
            "redis://localhost:6379", protocol=3
        )

    from_url.assert_called_once_with("redis://localhost:6379", protocol=3)


def test_async_cluster_connection_defaults_to_resp2():
    with patch("redisvl.redis.connection.AsyncRedisCluster.from_url") as from_url:
        RedisConnectionFactory.get_async_redis_cluster_connection(
            "redis://localhost:6379"
        )

    from_url.assert_called_once_with("redis://localhost:6379", protocol=2)


def test_async_cluster_connection_preserves_explicit_protocol():
    with patch("redisvl.redis.connection.AsyncRedisCluster.from_url") as from_url:
        RedisConnectionFactory.get_async_redis_cluster_connection(
            "redis://localhost:6379", protocol=3
        )

    from_url.assert_called_once_with("redis://localhost:6379", protocol=3)
