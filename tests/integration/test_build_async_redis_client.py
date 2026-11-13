"""Integration tests for RedisConnectionFactory.build_async_redis_client.

It is the synchronous, non-warning way to get an async client from a URL, for
callers such as synchronous constructors that cannot await. These tests run
against a real Redis and pin the contract it shares with the other async
factories: RESP2 by default, the REDIS_URL fallback, stripping of the
``cluster`` parameter, and the TypeError for removed keywords.
"""

import asyncio
import warnings

import pytest
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster

from redisvl.redis.connection import RedisConnectionFactory
from redisvl.version import __version__

# Every test here builds through the non-warning path, so any
# DeprecationWarning -- at build time or when a connection first opens --
# is a failure.
pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


def _build_without_warnings(*args, **kwargs):
    """Build a client, failing if any warning at all is emitted."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return RedisConnectionFactory.build_async_redis_client(*args, **kwargs)


async def _resp_version(client) -> str:
    info = await client.client_info()
    return str(info["resp"])


async def test_client_works_against_redis_and_emits_no_warning(redis_url):
    client = _build_without_warnings(redis_url)
    try:
        assert isinstance(client, AsyncRedis)
        await client.set("build_async_redis_client:roundtrip", "value")
        assert await client.get("build_async_redis_client:roundtrip") == b"value"
    finally:
        await client.delete("build_async_redis_client:roundtrip")
        await client.aclose()


async def test_client_speaks_resp2_by_default(redis_url):
    client = _build_without_warnings(redis_url)
    try:
        assert await _resp_version(client) == "2"
    finally:
        await client.aclose()


async def test_explicit_protocol_is_honoured(redis_url):
    client = _build_without_warnings(redis_url, protocol=3)
    try:
        assert await _resp_version(client) == "3"
    finally:
        await client.aclose()


async def test_falls_back_to_redis_url_env_var(redis_url, monkeypatch):
    monkeypatch.setenv("REDIS_URL", redis_url)
    client = _build_without_warnings()
    try:
        assert await client.ping()
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "url_suffix, extra_kwargs",
    [("?cluster=false", {}), ("", {"cluster": False})],
    ids=["cluster-in-url", "cluster-keyword"],
)
async def test_cluster_parameter_is_stripped_for_a_standalone_server(
    redis_url, url_suffix, extra_kwargs
):
    """Neither async client accepts 'cluster'; left in, connecting fails."""
    client = _build_without_warnings(f"{redis_url}{url_suffix}", **extra_kwargs)
    try:
        assert isinstance(client, AsyncRedis)
        assert await client.ping()
    finally:
        await client.aclose()


def test_client_can_be_built_where_no_event_loop_is_running(redis_url):
    """The case it exists for: a synchronous constructor, used later in a loop."""

    class Wrapper:
        def __init__(self, url: str) -> None:
            self.client = _build_without_warnings(url)

    wrapper = Wrapper(redis_url)

    async def use() -> bool:
        try:
            return await wrapper.client.ping()
        finally:
            await wrapper.client.aclose()

    assert asyncio.run(use())


async def test_client_is_not_identified_until_validated(redis_url):
    """It cannot await CLIENT SETINFO, so identifying it is the caller's step."""
    client = _build_without_warnings(redis_url)
    try:
        info = await client.client_info()
        assert info["lib-name"] == "redis-py"

        await RedisConnectionFactory.validate_async_redis(client, "mylib_v1.0")

        info = await client.client_info()
        assert info["lib-name"] == f"redis-py(redisvl_v{__version__};mylib_v1.0)"
    finally:
        await client.aclose()


async def test_lib_name_keyword_causes_no_warning_when_connecting(redis_url):
    """redis-py 8.1+ deprecates lib_name, warning when a connection opens.

    The builder drops it, so neither building nor the first command warns,
    and the connection reports plain redis-py until validate_async_redis.
    """
    client = _build_without_warnings(redis_url, lib_name="mylib_v1.0")
    try:
        info = await client.client_info()
        assert info["lib-name"] == "redis-py"
    finally:
        await client.aclose()


@pytest.mark.requires_cluster
async def test_cluster_url_gives_a_cluster_client(redis_cluster_url):
    client = _build_without_warnings(f"{redis_cluster_url}?cluster=true")
    try:
        assert isinstance(client, AsyncRedisCluster)
        assert await client.ping()
    finally:
        await client.aclose()
