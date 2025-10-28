import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redisvl.redis.connection import RedisConnectionFactory
from redisvl.utils.utils import assert_no_warnings


class DummyAsyncClient:
    async def client_setinfo(self, *args, **kwargs):
        return None

    async def echo(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test__get_aredis_connection_deprecates_url_kwarg_only():
    # Patch AsyncRedis.from_url to avoid real network calls
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=DummyAsyncClient()
    ):
        with pytest.warns(DeprecationWarning) as record:
            await RedisConnectionFactory._get_aredis_connection(
                url="redis://localhost:6379"
            )

    assert any(
        str(w.message)
        == (
            "Argument url is deprecated and will be removed in the next major release. "
            "Use redis_url instead."
        )
        for w in record
    )


@pytest.mark.asyncio
async def test__get_aredis_connection_no_deprecation_with_redis_url():
    # Patch AsyncRedis.from_url to avoid real network calls
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=DummyAsyncClient()
    ):
        with assert_no_warnings():
            await RedisConnectionFactory._get_aredis_connection(
                redis_url="redis://localhost:6379"
            )


def test_get_async_redis_connection_deprecates_url_kwarg_only():
    # Patch AsyncRedis.from_url to avoid real network calls
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=MagicMock()
    ):
        with pytest.warns(DeprecationWarning):
            RedisConnectionFactory.get_async_redis_connection(
                url="redis://localhost:6379"
            )


def test_get_async_redis_connection_no_deprecation_with_redis_url():
    # Patch AsyncRedis.from_url to avoid real network calls
    with patch(
        "redisvl.redis.connection.AsyncRedis.from_url", return_value=MagicMock()
    ):
        with pytest.warns(DeprecationWarning):
            RedisConnectionFactory.get_async_redis_connection(
                redis_url="redis://localhost:6379"
            )
