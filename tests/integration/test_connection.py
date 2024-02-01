import os

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ConnectionError

from redisvl.redis.connection import RedisConnectionFactory, get_address_from_env


def test_get_address_from_env(redis_url):
    assert get_address_from_env() == redis_url


def test_sync_redis_connection(redis_url):
    client = RedisConnectionFactory.connect(redis_url)
    assert client is not None
    assert isinstance(client, Redis)
    # Perform a simple operation
    assert client.ping()


@pytest.mark.asyncio
async def test_async_redis_connection(redis_url):
    client = RedisConnectionFactory.connect(redis_url, use_async=True)
    assert client is not None
    assert isinstance(client, AsyncRedis)
    # Perform a simple operation
    assert await client.ping()


def test_missing_env_var():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        del os.environ["REDIS_URL"]
        with pytest.raises(ValueError):
            RedisConnectionFactory.connect()
        os.environ["REDIS_URL"] = redis_url


def test_invalid_url_format():
    with pytest.raises(ValueError):
        RedisConnectionFactory.connect(redis_url="invalid_url_format")


def test_unknown_redis():
    bad_client = RedisConnectionFactory.connect(redis_url="redis://fake:1234")
    with pytest.raises(ConnectionError):
        bad_client.ping()


def test_required_modules(client):
    RedisConnectionFactory.validate_redis_modules(client)
    RedisConnectionFactory.validate_async_redis_modules(client)
