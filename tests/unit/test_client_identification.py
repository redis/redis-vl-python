"""
Unit tests for client identification tolerance.

RedisVL announces itself with ``CLIENT SETINFO LIB-NAME`` when a connection is
made, so that adoption metrics can attribute traffic to the library and to any
wrapper above it. The command is cosmetic -- it only populates the ``lib-name``
field of ``CLIENT LIST`` and ``CLIENT INFO`` -- so a refusal must never stop a
connection from opening. Two refusals matter: a credential that grants neither
`@connection` nor the command itself, and a server predating Redis 7.2, where
the command does not exist.
"""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ConnectionError, NoPermissionError, ResponseError

from redisvl.redis import connection as connection_module
from redisvl.redis.connection import RedisConnectionFactory, make_lib_name

PLAIN_URL = "redis://localhost:6379"
CLUSTER_URL = "redis://localhost:6379?cluster=true"
SENTINEL_URL = "redis+sentinel://localhost:26379/mymaster"

# Both must be tolerated, and only the first is a permission problem. The second
# is the case that lets the ECHO fallback go: redis-py's own handshake already
# absorbs it, so RedisVL does not need a workaround for old servers.
REFUSALS = [
    (
        NoPermissionError,
        "User acl_user has no permissions to run the 'client|setinfo' command",
    ),
    (
        ResponseError,
        "ERR Unknown subcommand or wrong number of arguments for 'setinfo'",
    ),
]
REFUSAL_IDS = ["denied-by-acl", "unsupported-by-server"]

UNREACHABLE = "Error 111 connecting to localhost:6379. Connection refused."


def _sync_client(setinfo_error=None):
    """A real Redis instance -- no socket is opened -- with stubbed commands.

    The type has to survive ``issubclass`` in ``validate_sync_redis``, so this
    cannot be a bare ``MagicMock``. Returns the stubs so assertions do not have
    to reach back through the client.
    """
    client = Redis.from_url(PLAIN_URL)
    setinfo, echo = Mock(side_effect=setinfo_error), Mock()
    client.client_setinfo, client.echo = setinfo, echo
    return client, setinfo, echo


def _async_client(setinfo_error=None):
    client = AsyncRedis.from_url(PLAIN_URL)
    setinfo, echo = AsyncMock(side_effect=setinfo_error), AsyncMock()
    client.client_setinfo, client.echo = setinfo, echo
    return client, setinfo, echo


class TestConnectionFactoryIdentification:
    """get_redis_connection() must survive a refused identification."""

    @pytest.mark.parametrize("exc_type,message", REFUSALS, ids=REFUSAL_IDS)
    def test_refused_identification_still_returns_a_client(self, exc_type, message):
        client, _, echo = _sync_client(setinfo_error=exc_type(message))
        with patch.object(connection_module.Redis, "from_url", return_value=client):
            returned = RedisConnectionFactory.get_redis_connection(redis_url=PLAIN_URL)
        assert returned is client
        # ECHO used to carry the library name when SETINFO was refused. It is
        # denied by the same ACL rule and reaches nothing that reads lib-name,
        # so the fallback is gone and must not come back.
        echo.assert_not_called()

    def test_connection_failure_is_not_swallowed(self):
        # SETINFO is the first command on a freshly created connection, which
        # makes it the de-facto connectivity check. Widening the except clause
        # would defer a real failure to some later, more confusing command.
        client, _, _ = _sync_client(setinfo_error=ConnectionError(UNREACHABLE))
        with patch.object(connection_module.Redis, "from_url", return_value=client):
            with pytest.raises(ConnectionError):
                RedisConnectionFactory.get_redis_connection(redis_url=PLAIN_URL)

    @pytest.mark.parametrize(
        "url",
        [PLAIN_URL, CLUSTER_URL, SENTINEL_URL],
        ids=["standalone", "cluster", "sentinel"],
    )
    def test_identification_reaches_every_url_shape(self, url):
        # Identification sits after the sentinel/cluster/standalone fan-out.
        # Only the standalone branch has live coverage elsewhere -- cluster
        # integration tests need --run-cluster-tests and never run in CI -- so
        # this is the only guard against the call sliding into one branch.
        client, setinfo, _ = _sync_client()
        if url == SENTINEL_URL:
            target = patch.object(
                RedisConnectionFactory, "_redis_sentinel_client", return_value=client
            )
        elif url == CLUSTER_URL:
            target = patch.object(
                connection_module.RedisCluster, "from_url", return_value=client
            )
        else:
            target = patch.object(
                connection_module.Redis, "from_url", return_value=client
            )
        with target:
            RedisConnectionFactory.get_redis_connection(redis_url=url)
        setinfo.assert_called_once_with("LIB-NAME", make_lib_name(None))

    def test_wrapper_lib_name_is_reported(self):
        # The composed string is the reason the explicit call is kept at all:
        # redis-py's handshake reports its own name, not this one.
        client, setinfo, _ = _sync_client()
        with patch.object(connection_module.Redis, "from_url", return_value=client):
            RedisConnectionFactory.get_redis_connection(
                redis_url=PLAIN_URL, lib_name="langchain-redis_v1.0.0"
            )
        reported = setinfo.call_args.args[1]
        assert "redisvl_v" in reported and "langchain-redis_v1.0.0" in reported

    def test_refusal_is_logged(self, caplog):
        # A refusal is otherwise invisible to the caller, so the log line is the
        # only affordance for "why is lib-name empty?".
        client, _, _ = _sync_client(setinfo_error=NoPermissionError(REFUSALS[0][1]))
        with caplog.at_level(logging.DEBUG, logger="redisvl.redis.connection"):
            with patch.object(connection_module.Redis, "from_url", return_value=client):
                RedisConnectionFactory.get_redis_connection(redis_url=PLAIN_URL)
        assert "CLIENT SETINFO" in caplog.text


class TestAsyncConnectionFactoryIdentification:
    """The async twin is a hand-maintained copy, so each branch needs a witness."""

    @pytest.mark.parametrize("exc_type,message", REFUSALS, ids=REFUSAL_IDS)
    @pytest.mark.asyncio
    async def test_refused_identification_still_returns_a_client(
        self, exc_type, message
    ):
        client, _, echo = _async_client(setinfo_error=exc_type(message))
        with patch.object(
            connection_module.AsyncRedis, "from_url", return_value=client
        ):
            returned = await RedisConnectionFactory._get_aredis_connection(
                redis_url=PLAIN_URL
            )
        assert returned is client
        # assert_not_called, not assert_not_awaited: the latter passes for a
        # reintroduced fallback that forgot its await, which is exactly the slip
        # a hand-maintained twin invites.
        echo.assert_not_called()

    @pytest.mark.asyncio
    async def test_connection_failure_is_not_swallowed(self):
        client, _, _ = _async_client(setinfo_error=ConnectionError(UNREACHABLE))
        with patch.object(
            connection_module.AsyncRedis, "from_url", return_value=client
        ):
            with pytest.raises(ConnectionError):
                await RedisConnectionFactory._get_aredis_connection(redis_url=PLAIN_URL)

    @pytest.mark.asyncio
    async def test_wrapper_lib_name_is_reported(self):
        client, setinfo, _ = _async_client()
        with patch.object(
            connection_module.AsyncRedis, "from_url", return_value=client
        ):
            await RedisConnectionFactory._get_aredis_connection(
                redis_url=PLAIN_URL, lib_name="langchain-redis_v1.0.0"
            )
        reported = setinfo.call_args.args[1]
        assert "redisvl_v" in reported and "langchain-redis_v1.0.0" in reported


class TestValidateSyncRedis:
    """The user-supplied-client path shares the same tolerance."""

    def test_refused_identification_is_tolerated(self):
        client, setinfo, echo = _sync_client(
            setinfo_error=NoPermissionError(REFUSALS[0][1])
        )
        RedisConnectionFactory.validate_sync_redis(client)
        setinfo.assert_called_once_with("LIB-NAME", make_lib_name(None))
        echo.assert_not_called()

    def test_client_type_is_still_validated(self):
        with pytest.raises(TypeError):
            RedisConnectionFactory.validate_sync_redis("not a client")


class TestValidateAsyncRedis:
    @pytest.mark.asyncio
    async def test_client_type_is_still_validated(self):
        # A sync client is the mistake this guard exists to catch.
        with pytest.raises(TypeError):
            await RedisConnectionFactory.validate_async_redis(Redis.from_url(PLAIN_URL))

    @pytest.mark.asyncio
    async def test_refused_identification_is_tolerated(self):
        client, setinfo, echo = _async_client(
            setinfo_error=NoPermissionError(REFUSALS[0][1])
        )
        await RedisConnectionFactory.validate_async_redis(client)
        setinfo.assert_called_once_with("LIB-NAME", make_lib_name(None))
        echo.assert_not_called()
