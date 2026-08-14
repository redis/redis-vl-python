"""SQL queries over a RESP3 connection fail inside sql-redis, unexplained.

sql-redis reads FT.SEARCH and FT.AGGREGATE replies positionally, which only
holds under RESP2. Under RESP3 the replies are maps, so those reads raise
``KeyError`` from inside the dependency with nothing pointing at the
connection. RedisVL catches that and says what to change.

The context is added on failure rather than refused up front, because
sql-redis is expected to gain RESP3 support and a check on the connection
alone cannot tell a fixed dependency from a broken one. The
``resp3_success_is_not_intercepted`` test below is the one that pins that
property: a working sql-redis must never be turned away.

These tests drive ``index.query(SQLQuery(...))`` so that removing the handler
fails the suite.
"""

from unittest.mock import patch

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from redisvl.exceptions import RedisSearchError
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query.sql import SQLQuery

pytest.importorskip(
    "sql_redis", reason="the handler only matters when sql-redis is installed"
)

SCHEMA = {
    "index": {"name": "sql_gate", "prefix": "sql_gate", "storage_type": "hash"},
    "fields": [{"name": "cat", "type": "tag"}],
}
QUERY = SQLQuery("SELECT cat FROM sql_gate")

# What sql-redis 0.7.1 actually raises, measured against Redis 8.4.6: the
# search path fails on KeyError(2) and the aggregate path on a slice key.
RESP3_PARSE_FAILURES = [
    pytest.param(KeyError(2), id="search-path"),
    pytest.param(KeyError(slice(1, None, None)), id="aggregate-path"),
]


def _client(protocol):
    """A real client, not a mock.

    redis-py's ``get_protocol_version`` is ``isinstance``-gated and reads
    ``connection_pool``, which ``__init__`` sets, so a ``Mock`` fails the type
    check and ``Mock(spec=Redis)`` has no ``connection_pool``. Constructing a
    client opens no socket.
    """
    kwargs = {} if protocol is None else {"protocol": protocol}
    return Redis.from_url("redis://localhost:6379", **kwargs)


def _async_client(protocol):
    kwargs = {} if protocol is None else {"protocol": protocol}
    return AsyncRedis.from_url("redis://localhost:6379", **kwargs)


@pytest.mark.parametrize("failure", RESP3_PARSE_FAILURES)
def test_a_resp3_parse_failure_names_the_connection(failure):
    index = SearchIndex.from_dict(SCHEMA, redis_client=_client(3))

    with patch("sql_redis.create_executor") as create_executor:
        create_executor.return_value.execute.side_effect = failure
        with pytest.raises(RedisSearchError) as excinfo:
            index.query(QUERY)

    message = str(excinfo.value)
    # Name the real cause and the way out, or a user reads "SQL is broken" and
    # goes looking in RedisVL rather than at their connection.
    assert "sql-redis" in message
    assert "protocol=2" in message
    # Keep the original reachable; it is the only record of where it broke.
    assert isinstance(excinfo.value.__cause__, KeyError)


@pytest.mark.asyncio
async def test_a_resp3_parse_failure_names_the_connection_async():
    index = AsyncSearchIndex.from_dict(SCHEMA, redis_client=_async_client(3))

    async def _boom(*_args, **_kwargs):
        raise KeyError(2)

    with patch("sql_redis.create_async_executor") as create_executor:
        executor = create_executor.return_value
        executor.execute = _boom
        create_executor.return_value = executor

        with pytest.raises(RedisSearchError):
            await index.query(QUERY)


def test_resp3_success_is_not_intercepted():
    """A sql-redis that can read RESP3 must be left alone.

    This is the property a pre-flight check on the connection could not hold:
    it would refuse this query despite the dependency handling it correctly.
    """
    index = SearchIndex.from_dict(SCHEMA, redis_client=_client(3))

    with patch("sql_redis.create_executor") as create_executor:
        create_executor.return_value.execute.return_value.rows = [{"cat": "a"}]
        assert index.query(QUERY) == [{"cat": "a"}]


def test_a_keyerror_on_resp2_is_not_reinterpreted():
    """On RESP2 a KeyError is a real bug, not a protocol mismatch."""
    index = SearchIndex.from_dict(SCHEMA, redis_client=_client(2))

    with patch("sql_redis.create_executor") as create_executor:
        create_executor.return_value.execute.side_effect = KeyError("something else")
        with pytest.raises(KeyError):
            index.query(QUERY)


def test_a_missing_dependency_is_reported_before_the_connection_is_touched():
    """``ImportError`` outranks everything: reading the client can connect."""
    index = SearchIndex.from_dict(SCHEMA, redis_client=_client(3))

    with patch.dict("sys.modules", {"sql_redis": None}):
        with pytest.raises(ImportError) as excinfo:
            index.query(QUERY)

    assert "redisvl[sql-redis]" in str(excinfo.value)
