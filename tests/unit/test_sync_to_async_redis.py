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
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import ExponentialBackoff
from redis.connection import Connection, ConnectionPool, SSLConnection
from redis.exceptions import ConnectionError
from redis.retry import Retry
from redis.sentinel import Sentinel

from redisvl.redis.connection import (
    RedisConnectionFactory,
    _async_connection_kwargs,
    _init_parameters,
)


class _TLS(SSLConnection):
    pass


@pytest.mark.parametrize(
    "client, expected, settings",
    [
        (
            Redis(host="h", port=6380, db=3, decode_responses=True),
            AsyncConnection,
            {"host": "h", "port": 6380, "db": 3, "decode_responses": True},
        ),
        # RESP3 makes the sync pool inject maintenance-notification plumbing
        # that older async classes don't accept (redis-py 7.4 and 8.0).
        (
            Redis(host="h", protocol=3),
            AsyncConnection,
            {"host": "h", "protocol": 3},
        ),
        (
            Redis(host="h", ssl=True, password="pw", ssl_check_hostname=True),
            AsyncSSLConnection,
            {"host": "h", "password": "pw", "ssl_check_hostname": True},
        ),
        (
            Redis(connection_pool=ConnectionPool(connection_class=_TLS, host="h")),
            AsyncSSLConnection,
            {"host": "h"},
        ),
        (
            Redis(unix_socket_path="/tmp/redis.sock", db=2),
            AsyncUnixDomainSocketConnection,
            {"path": "/tmp/redis.sock", "db": 2},
        ),
        # The URL form sets connection_class rather than unix_socket_path.
        (
            Redis.from_url("unix:///tmp/redis.sock"),
            AsyncUnixDomainSocketConnection,
            {"path": "/tmp/redis.sock"},
        ),
    ],
    ids=["plain", "resp3", "tls", "tls-subclass", "unix", "unix-url"],
)
def test_converts_to_the_matching_async_connection(client, expected, settings):
    pool = RedisConnectionFactory.sync_to_async_redis(client).connection_pool

    assert type(pool.make_connection()) is expected
    assert settings.items() <= pool.connection_kwargs.items()


@pytest.mark.parametrize(
    "client, match",
    [
        (Sentinel([("localhost", 26379)]).master_for("mymaster"), "Sentinel-managed"),
        (Redis(host="h", ssl=True, ssl_validate_ocsp=True), "ssl_validate_ocsp"),
    ],
    ids=["sentinel", "sync-only-setting"],
)
def test_unconvertible_client_raises(client, match):
    with pytest.raises(ValueError, match=match):
        RedisConnectionFactory.sync_to_async_redis(client)


def test_sync_only_keyword_is_unset_only_at_its_own_default():
    # ssl_validate_ocsp_stapled defaults to False on the sync class; 0 is a
    # value the user chose, not the default, so it is not silently dropped.
    assert "ssl_validate_ocsp_stapled" not in _init_parameters(AsyncSSLConnection)

    kwargs = _async_connection_kwargs(
        {"host": "h", "ssl_validate_ocsp_stapled": False},
        AsyncSSLConnection,
        SSLConnection,
    )
    assert kwargs == {"host": "h"}

    with pytest.raises(ValueError, match="ssl_validate_ocsp_stapled"):
        _async_connection_kwargs(
            {"host": "h", "ssl_validate_ocsp_stapled": 0},
            AsyncSSLConnection,
            SSLConnection,
        )


def test_explicitly_enabled_maintenance_notifications_are_not_dropped():
    maint = pytest.importorskip("redis.maint_notifications")
    config = maint.MaintNotificationsConfig(enabled=True)
    client = Redis(host="h", protocol=3, maint_notifications_config=config)

    if "maint_notifications_config" in _init_parameters(AsyncConnection):
        pool = RedisConnectionFactory.sync_to_async_redis(client).connection_pool
        assert pool.connection_kwargs["maint_notifications_config"] is config
    else:
        with pytest.raises(ValueError, match="maint_notifications_config"):
            RedisConnectionFactory.sync_to_async_redis(client)


def test_sync_retry_is_rebuilt_as_the_async_retry():
    # The async connection awaits call_with_retry, so a sync Retry would hand
    # it an unawaited coroutine and the policy would never run.
    client = Redis(
        host="h",
        retry=Retry(ExponentialBackoff(cap=2, base=0.1), 3, (ConnectionError, OSError)),
    )
    pool = RedisConnectionFactory.sync_to_async_redis(client).connection_pool

    retry = pool.make_connection().retry
    assert type(retry) is AsyncRetry
    assert retry == AsyncRetry(
        ExponentialBackoff(cap=2, base=0.1), 3, (ConnectionError, OSError)
    )


def test_redis_py_default_retry_is_converted_too():
    # Since redis-py 6 every Redis() carries a sync Retry unless told otherwise.
    pool = RedisConnectionFactory.sync_to_async_redis(Redis(host="h")).connection_pool
    assert type(pool.make_connection().retry) is AsyncRetry


def test_missing_retry_is_left_alone():
    client = Redis(host="h", retry=None)
    pool = RedisConnectionFactory.sync_to_async_redis(client).connection_pool
    assert pool.connection_kwargs["retry"] is None

    assert _async_connection_kwargs({"host": "h"}, AsyncConnection, Connection) == {
        "host": "h"
    }
