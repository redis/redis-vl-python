from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ResponseError

from redisvl.redis.connection import RedisConnectionFactory


def test_get_redis_connection_acl_fallback_denied():
    """Connection setup must survive an ACL that denies CLIENT SETINFO and ECHO."""
    mock_client = MagicMock()
    mock_client.client_setinfo.side_effect = ResponseError(
        "NOPERM client setinfo denied"
    )
    mock_client.echo.side_effect = ResponseError("NOPERM echo denied")

    with patch("redisvl.redis.connection.Redis.from_url", return_value=mock_client):
        client = RedisConnectionFactory.get_redis_connection("redis://localhost:6379")

    assert client == mock_client
    mock_client.client_setinfo.assert_called_once()
    mock_client.echo.assert_called_once()


@pytest.mark.asyncio
async def test_get_aredis_connection_acl_fallback_denied():
    """The async factory must behave identically to the sync one."""
    mock_client = MagicMock()
    mock_client.client_setinfo = AsyncMock(
        side_effect=ResponseError("NOPERM client setinfo denied")
    )
    mock_client.echo = AsyncMock(side_effect=ResponseError("NOPERM echo denied"))

    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=mock_client
    ):
        client = await RedisConnectionFactory._get_aredis_connection(
            "redis://localhost:6379"
        )

    assert client == mock_client
    mock_client.client_setinfo.assert_called_once()
    mock_client.echo.assert_called_once()


def test_validate_sync_redis_acl_fallback_denied():
    """validate_sync_redis must not propagate a denied ECHO fallback.

    A real ``Redis`` instance is used because ``validate_sync_redis`` gates on
    ``issubclass(type(redis_client), ...)``, which a ``MagicMock(spec=Redis)``
    does not satisfy. Construction does not open a socket, and both commands are
    patched, so no server is contacted.
    """
    client = Redis(host="localhost", port=6379)

    with (
        patch.object(
            client, "client_setinfo", side_effect=ResponseError("NOPERM setinfo denied")
        ) as setinfo,
        patch.object(
            client, "echo", side_effect=ResponseError("NOPERM echo denied")
        ) as echo,
    ):
        RedisConnectionFactory.validate_sync_redis(client)

    setinfo.assert_called_once()
    echo.assert_called_once()


@pytest.mark.asyncio
async def test_validate_async_redis_acl_fallback_denied():
    """validate_async_redis must not propagate a denied ECHO fallback."""
    client = AsyncRedis(host="localhost", port=6379)

    with (
        patch.object(
            client,
            "client_setinfo",
            AsyncMock(side_effect=ResponseError("NOPERM setinfo denied")),
        ) as setinfo,
        patch.object(
            client, "echo", AsyncMock(side_effect=ResponseError("NOPERM echo denied"))
        ) as echo,
    ):
        await RedisConnectionFactory.validate_async_redis(client)

    setinfo.assert_called_once()
    echo.assert_called_once()
