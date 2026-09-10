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
    "factory",
    [
        RedisConnectionFactory.get_redis_connection,
        RedisConnectionFactory.get_async_redis_connection,
        RedisConnectionFactory.get_redis_cluster_connection,
        RedisConnectionFactory.get_async_redis_cluster_connection,
    ],
    ids=["sync", "async", "sync-cluster", "async-cluster"],
)
def test_factories_reject_the_removed_url_keyword(factory):
    """url= now names its replacement instead of failing obscurely.

    Left unguarded it reaches is_cluster_url or from_url, both of which take
    url positionally, and raises "got multiple values for argument 'url'" --
    an error naming neither RedisVL nor the rename.

    A Redis URL routinely embeds a password, so this also pins that the
    message names the keyword without echoing the value.
    """
    with pytest.raises(TypeError) as excinfo:
        factory(url="redis://user:s3cr3t-do-not-log@x:6379")

    message = str(excinfo.value)
    assert "url" in message and "redis_url" in message
    assert "s3cr3t-do-not-log" not in message


@pytest.mark.asyncio
async def test_private_async_factory_rejects_the_removed_url_keyword():
    """The guard on the factory every internal async path actually uses.

    AsyncSearchIndex and BaseCache both build their client through this one,
    so a regression here is the one that would reach users.
    """
    with pytest.raises(TypeError) as excinfo:
        await RedisConnectionFactory._get_aredis_connection(url="redis://x:6379")

    assert "redis_url" in str(excinfo.value)


def test_get_async_redis_connection_still_warns_about_becoming_async():
    """The signature-change notice is this function's only user-facing signal.

    Its previous assertion went out with the url-deprecation tests, so the
    warn() could have been deleted or reworded with the suite still green.
    """
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=MagicMock()
    ):
        with pytest.warns(DeprecationWarning, match="will become a coroutine"):
            RedisConnectionFactory.get_async_redis_connection(
                redis_url="redis://localhost:6379"
            )
