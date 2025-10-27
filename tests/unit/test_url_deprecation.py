import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redisvl.redis.connection import RedisConnectionFactory


class DummyAsyncClient:
    async def client_setinfo(self, *args, **kwargs):
        return None

    async def echo(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test__get_aredis_connection_deprecates_url_kwarg_only(caplog):
    # Patch AsyncRedis.from_url to avoid real network calls
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=DummyAsyncClient()
    ):
        caplog.set_level(logging.WARNING, logger="redisvl.redis.connection")
        await RedisConnectionFactory._get_aredis_connection(
            url="redis://localhost:6379"
        )

    assert (
        "The `url` parameter is deprecated. Please use `redis_url` instead."
        in caplog.text
    )


@pytest.mark.asyncio
async def test__get_aredis_connection_no_deprecation_with_redis_url(caplog):
    # Patch AsyncRedis.from_url to avoid real network calls
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=DummyAsyncClient()
    ):
        caplog.set_level(logging.WARNING, logger="redisvl.redis.connection")
        await RedisConnectionFactory._get_aredis_connection(
            redis_url="redis://localhost:6379"
        )

    assert (
        "The `url` parameter is deprecated. Please use `redis_url` instead."
        not in caplog.text
    )


def test_get_async_redis_connection_deprecates_url_kwarg_only(caplog):
    # Patch AsyncRedis.from_url to avoid real network calls
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=MagicMock()
    ):
        caplog.set_level(logging.WARNING, logger="redisvl.redis.connection")
        with pytest.warns(DeprecationWarning):
            RedisConnectionFactory.get_async_redis_connection(
                url="redis://localhost:6379"
            )

    assert (
        "The `url` parameter is deprecated. Please use `redis_url` instead."
        in caplog.text
    )


def test_get_async_redis_connection_no_deprecation_with_redis_url(caplog):
    # Patch AsyncRedis.from_url to avoid real network calls
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=MagicMock()
    ):
        caplog.set_level(logging.WARNING, logger="redisvl.redis.connection")
        with pytest.warns(DeprecationWarning):
            RedisConnectionFactory.get_async_redis_connection(
                redis_url="redis://localhost:6379"
            )

    assert (
        "The `url` parameter is deprecated. Please use `redis_url` instead."
        not in caplog.text
    )
