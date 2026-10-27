"""Unit tests for RedisConnectionFactory.sync_to_async_redis.

No server is needed: every failure the conversion can produce happens when the
async pool builds its first connection, so ``make_connection()`` is enough.
"""

import pytest
from redis import Redis
from redis.asyncio.connection import Connection as AsyncConnection
from redis.asyncio.connection import SSLConnection as AsyncSSLConnection
from redis.asyncio.connection import (
    UnixDomainSocketConnection as AsyncUnixDomainSocketConnection,
)
from redis.connection import ConnectionPool, SSLConnection
from redis.sentinel import Sentinel

from redisvl.redis.connection import RedisConnectionFactory


def _first_connection(async_client):
    return async_client.connection_pool.make_connection()


def test_plain_client_keeps_its_settings():
    client = Redis(host="example.com", port=6380, db=3, decode_responses=True)

    async_client = RedisConnectionFactory.sync_to_async_redis(client)
    conn = _first_connection(async_client)

    assert isinstance(conn, AsyncConnection)
    assert (conn.host, conn.port, conn.db) == ("example.com", 6380, 3)
    assert async_client.connection_pool.connection_kwargs["decode_responses"] is True


def test_tls_keyword_client_converts_and_keeps_tls():
    client = Redis(
        host="example.com",
        ssl=True,
        password="secret",
        ssl_cert_reqs="required",
        ssl_check_hostname=True,
    )

    async_client = RedisConnectionFactory.sync_to_async_redis(client)
    conn = _first_connection(async_client)

    assert isinstance(conn, AsyncSSLConnection)
    assert conn.host == "example.com"
    assert conn.ssl_context.check_hostname is True
    assert async_client.connection_pool.connection_kwargs["password"] == "secret"


def test_tls_url_client_still_converts():
    client = Redis.from_url("rediss://:secret@example.com:6380/1")

    conn = _first_connection(RedisConnectionFactory.sync_to_async_redis(client))

    assert isinstance(conn, AsyncSSLConnection)
    assert conn.db == 1


def test_ssl_connection_subclass_gets_a_tls_async_connection():
    class CustomSSLConnection(SSLConnection):
        pass

    client = Redis(
        connection_pool=ConnectionPool(
            connection_class=CustomSSLConnection, host="example.com", password="pw"
        )
    )

    conn = _first_connection(RedisConnectionFactory.sync_to_async_redis(client))

    assert isinstance(conn, AsyncSSLConnection)


def test_unix_socket_client_converts():
    client = Redis(unix_socket_path="/tmp/redis.sock", db=2)

    conn = _first_connection(RedisConnectionFactory.sync_to_async_redis(client))

    assert isinstance(conn, AsyncUnixDomainSocketConnection)
    assert conn.path == "/tmp/redis.sock"
    assert conn.db == 2


def test_sentinel_managed_client_is_rejected_clearly():
    client = Sentinel([("localhost", 26379)]).master_for("mymaster")

    with pytest.raises(ValueError, match="Sentinel-managed"):
        RedisConnectionFactory.sync_to_async_redis(client)


def test_sync_only_setting_that_is_set_is_not_silently_dropped():
    client = Redis(host="example.com", ssl=True, ssl_validate_ocsp=True)

    with pytest.raises(ValueError, match="ssl_validate_ocsp"):
        RedisConnectionFactory.sync_to_async_redis(client)
