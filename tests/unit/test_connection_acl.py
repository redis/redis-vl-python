import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from redis.exceptions import ResponseError

from redisvl.redis.connection import RedisConnectionFactory


def test_get_redis_connection_acl_fallback_denied():
    mock_client = MagicMock()
    mock_client.client_setinfo.side_effect = ResponseError("NOPERM client setinfo denied")
    mock_client.echo.side_effect = ResponseError("NOPERM echo denied")

    with patch("redisvl.redis.connection.Redis.from_url", return_value=mock_client):
        # Should not raise exception even when both client_setinfo and echo fail with ACL ResponseError
        client = RedisConnectionFactory.get_redis_connection("redis://localhost:6379")
        assert client == mock_client
        mock_client.client_setinfo.assert_called_once()
        mock_client.echo.assert_called_once()


@pytest.mark.asyncio
async def test_get_aredis_connection_acl_fallback_denied():
    mock_client = MagicMock()
    mock_client.client_setinfo = AsyncMock(side_effect=ResponseError("NOPERM client setinfo denied"))
    mock_client.echo = AsyncMock(side_effect=ResponseError("NOPERM echo denied"))

    with patch("redisvl.redis.connection.AsyncRedis.from_url", return_value=mock_client):
        client = await RedisConnectionFactory._get_aredis_connection("redis://localhost:6379")
        assert client == mock_client
        mock_client.client_setinfo.assert_called_once()
        mock_client.echo.assert_called_once()


def test_validate_sync_redis_acl_fallback_denied():
    mock_client = MagicMock()
    # Ensure issubclass check passes by mocking type or using Redis subclass
    from redis import Redis
    class MockRedis(Redis):
        pass

    mock_client = MagicMock(spec=MockRedis)
    mock_client.client_setinfo.side_effect = ResponseError("NOPERM client setinfo denied")
    mock_client.echo.side_effect = ResponseError("NOPERM echo denied")

    RedisConnectionFactory.validate_sync_redis(mock_client)
    mock_client.client_setinfo.assert_called_once()
    mock_client.echo.assert_called_once()


@pytest.mark.asyncio
async def test_validate_async_redis_acl_fallback_denied():
    from redis.asyncio import Redis as AsyncRedis
    class MockAsyncRedis(AsyncRedis):
        pass

    mock_client = MagicMock(spec=MockAsyncRedis)
    mock_client.client_setinfo = AsyncMock(side_effect=ResponseError("NOPERM client setinfo denied"))
    mock_client.echo = AsyncMock(side_effect=ResponseError("NOPERM echo denied"))

    await RedisConnectionFactory.validate_async_redis(mock_client)
    mock_client.client_setinfo.assert_called_once()
    mock_client.echo.assert_called_once()
