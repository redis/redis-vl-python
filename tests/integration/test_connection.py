import os

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ConnectionError

from redisvl.redis.connection import RedisConnectionFactory, get_address_from_env
from redisvl.version import __version__

EXPECTED_LIB_NAME = f"redis-py(redisvl_v{__version__})"


def compare_versions(version1, version2):
    """
    Compare two Redis version strings numerically.

    Parameters:
    version1 (str): The first version string (e.g., "7.2.4").
    version2 (str): The second version string (e.g., "6.2.1").

    Returns:
    int: -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2.
    """
    v1_parts = list(map(int, version1.split(".")))
    v2_parts = list(map(int, version2.split(".")))

    for v1, v2 in zip(v1_parts, v2_parts):
        if v1 < v2:
            return False
        elif v1 > v2:
            return True

    # If the versions are equal so far, compare the lengths of the version parts
    if len(v1_parts) < len(v2_parts):
        return False
    elif len(v1_parts) > len(v2_parts):
        return True

    return True


def test_get_address_from_env(redis_url):
    assert get_address_from_env() == redis_url


def test_sync_redis_connect(redis_url):
    client = RedisConnectionFactory.connect(redis_url)
    assert client is not None
    assert isinstance(client, Redis)
    # Perform a simple operation
    assert client.ping()


@pytest.mark.asyncio
async def test_async_redis_connect(redis_url):
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


def test_validate_redis(client):
    redis_version = client.info()["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")
    RedisConnectionFactory.validate_redis(client)
    lib_name = client.client_info()
    assert lib_name["lib-name"] == EXPECTED_LIB_NAME


def test_validate_redis_custom_lib_name(client):
    redis_version = client.info()["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")
    RedisConnectionFactory.validate_redis(client, "langchain_v0.1.0")
    lib_name = client.client_info()
    assert lib_name["lib-name"] == f"redis-py(redisvl_v{__version__};langchain_v0.1.0)"
