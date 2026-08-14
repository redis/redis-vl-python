"""Detect the day sql-redis learns to read RESP3 replies.

``_explain_sql_query_failure`` exists only because sql-redis reads FT.SEARCH
and FT.AGGREGATE replies positionally, which breaks against RESP3's
map-shaped replies. It wraps the resulting ``KeyError`` rather than refusing
the query, so it is already safe when the dependency is fixed: the query
simply succeeds and the handler is never reached.

What it cannot do is tell anyone the workaround is obsolete.
``pyproject.toml`` declares ``sql-redis>=0.7.1`` with no upper bound, so the
fix will arrive silently and the handler would linger indefinitely.

This test drives sql-redis directly and is expected to fail. ``strict=True``
turns the day it starts passing into a build failure, at which point: raise
the ``sql-redis`` floor in both places it appears in ``pyproject.toml`` to
name the fixed release, delete ``_explain_sql_query_failure`` and its tests,
and delete this file.

Dependency metadata cannot express this instead. PEP 508 markers reference
only the twelve environment variables (``python_version``, ``sys_platform``
and so on); a package's version is not among them, so "require sql-redis >= X
when redis-py >= 8" is not declarable. It also would not be needed: the fixed
release reads both protocols, so an unconditional floor is the right end
state.
"""

import pytest

from redisvl.index import SearchIndex

pytest.importorskip("sql_redis")

SCHEMA = {
    "index": {
        "name": "sql_tripwire",
        "prefix": "sql_tripwire",
        "storage_type": "hash",
    },
    "fields": [{"name": "cat", "type": "tag"}],
}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "sql-redis 0.7.1 cannot parse RESP3 FT.SEARCH replies. When this "
        "xpasses, sql-redis has been fixed: raise the sql-redis floor in "
        "pyproject.toml, remove _explain_sql_query_failure, delete this file."
    ),
)
def test_sql_redis_can_read_a_resp3_search_reply(redis_url, worker_id):
    """Ask sql-redis to run a query over a RESP3 connection.

    Uses its own executor rather than ``index.query`` so the assertion is
    about the dependency alone, unmediated by RedisVL's error handling.
    """
    from redis import Redis
    from sql_redis import create_executor

    name = f"sql_tripwire_{worker_id}"
    schema = {**SCHEMA, "index": {**SCHEMA["index"], "name": name, "prefix": name}}

    # RESP2 for setup, so loading is unaffected by the protocol under test.
    setup_client = Redis.from_url(redis_url, protocol=2)
    index = SearchIndex.from_dict(schema, redis_client=setup_client)
    index.create(overwrite=True, drop=True)
    try:
        index.load([{"id": "1", "cat": "a"}], id_field="id")

        resp3_client = Redis.from_url(redis_url, protocol=3)
        executor = create_executor(resp3_client)
        result = executor.execute(f"SELECT cat FROM {name} WHERE cat = 'a'")

        assert [dict(row) for row in result.rows] == [{"cat": "a"}]
    finally:
        index.delete(drop=True)
        setup_client.close()
