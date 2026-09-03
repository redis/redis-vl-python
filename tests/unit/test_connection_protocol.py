from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redisvl.redis.connection import RedisConnectionFactory
from redisvl.utils.utils import assert_no_warnings


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


@pytest.mark.asyncio
async def test_aredis_connection_emits_no_warning_on_the_modern_spelling():
    """The main async factory must stay warning-free.

    Carried over from the deleted url-deprecation tests, and load-bearing:
    BaseCache builds its async client through this factory specifically to
    avoid warning users about an API they never called.
    """
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=AsyncMock()
    ):
        with assert_no_warnings():
            await RedisConnectionFactory._get_aredis_connection(
                redis_url="redis://localhost:6379"
            )


@pytest.mark.parametrize(
    "call",
    [
        lambda: RedisConnectionFactory.get_redis_connection(url="redis://x:6379"),
        lambda: RedisConnectionFactory.get_async_redis_connection(url="redis://x:6379"),
    ],
    ids=["sync", "async"],
)
def test_factories_reject_the_removed_url_keyword(call):
    """url= now names its replacement instead of failing obscurely.

    Left unguarded it reaches is_cluster_url, which takes url positionally,
    and raises "got multiple values for argument 'url'" — an error naming
    neither RedisVL nor the rename.
    """
    with pytest.raises(TypeError) as excinfo:
        call()

    assert "redis_url" in str(excinfo.value)
