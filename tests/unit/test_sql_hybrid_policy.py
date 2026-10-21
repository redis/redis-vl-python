"""Policy forwarding through SQL translation and sync/async index execution."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import SQLQuery

SCHEMA = {"index": {"name": "idx", "prefix": "idx:"}, "fields": []}
SQL = "SELECT title FROM idx"


@pytest.mark.parametrize("policy", ["ADHOC_BF", "BATCHES"])
def test_policy_forwarded_for_translation(policy):
    client = MagicMock()
    executor = MagicMock()
    executor._translator.translate.return_value.to_command_string.return_value = "cmd"
    factory = MagicMock(return_value=executor)
    query = SQLQuery(SQL, hybrid_policy=policy)

    with patch.dict(
        "sys.modules", {"sql_redis": SimpleNamespace(create_executor=factory)}
    ):
        assert query.redis_query_string(redis_client=client) == "cmd"

    factory.assert_called_once_with(
        client, schema_cache_strategy="lazy", hybrid_policy=policy
    )
    executor._translator.translate.assert_called_once_with(SQL)


def test_explicit_policy_overrides_options_without_mutating_them():
    options = {"hybrid_policy": "BATCHES", "schema_cache_strategy": "load_all"}
    query = SQLQuery(SQL, hybrid_policy="ADHOC_BF", sql_redis_options=options)

    assert query.sql_redis_options == {
        "hybrid_policy": "ADHOC_BF",
        "schema_cache_strategy": "load_all",
    }
    assert options == {"hybrid_policy": "BATCHES", "schema_cache_strategy": "load_all"}


@pytest.mark.parametrize("options", [{}, {"hybrid_policy": "BATCHES"}])
def test_none_preserves_existing_options(options):
    query = SQLQuery(SQL, hybrid_policy=None, sql_redis_options=options)
    assert query.sql_redis_options == options
    assert query.sql_redis_options is not options


def test_omitted_policy_not_forwarded_to_legacy_executor():
    client = MagicMock()
    executor = MagicMock()
    factory = MagicMock(return_value=executor)
    with patch.dict(
        "sys.modules", {"sql_redis": SimpleNamespace(create_executor=factory)}
    ):
        SQLQuery(SQL).redis_query_string(redis_client=client)
    factory.assert_called_once_with(client, schema_cache_strategy="lazy")


def test_sync_query_keeps_policy_executor_caches_separate():
    client = MagicMock()
    index = SearchIndex.from_dict(SCHEMA, redis_client=client)
    result = SimpleNamespace(rows=[{"title": "Redis"}])
    executors = [MagicMock(), MagicMock(), MagicMock()]
    for executor in executors:
        executor.execute.return_value = result
    factory = MagicMock(side_effect=executors)
    with patch.dict(
        "sys.modules", {"sql_redis": SimpleNamespace(create_executor=factory)}
    ):
        for policy in [None, "ADHOC_BF", "BATCHES", "ADHOC_BF", None]:
            query = SQLQuery(SQL, params={"id": 1}, hybrid_policy=policy)
            assert index.query(query) == result.rows

    assert factory.call_args_list == [
        call(client, schema_cache_strategy="lazy"),
        call(client, schema_cache_strategy="lazy", hybrid_policy="ADHOC_BF"),
        call(client, schema_cache_strategy="lazy", hybrid_policy="BATCHES"),
    ]
    assert [e.execute.call_count for e in executors] == [2, 2, 1]
    for executor in executors:
        executor.execute.assert_called_with(SQL, params={"id": 1})


@pytest.mark.asyncio
async def test_async_query_keeps_policy_executor_caches_separate():
    client = MagicMock()
    index = AsyncSearchIndex.from_dict(SCHEMA)
    result = SimpleNamespace(rows=[{"title": "Redis"}])
    executors = [
        SimpleNamespace(execute=AsyncMock(return_value=result)) for _ in range(3)
    ]
    factory = AsyncMock(side_effect=executors)
    with (
        patch.dict(
            "sys.modules", {"sql_redis": SimpleNamespace(create_async_executor=factory)}
        ),
        patch.object(index, "_get_client", new=AsyncMock(return_value=client)),
    ):
        for policy in [None, "ADHOC_BF", "BATCHES", "ADHOC_BF", None]:
            query = SQLQuery(SQL, params={"id": 1}, hybrid_policy=policy)
            assert await index.query(query) == result.rows

    assert factory.await_args_list == [
        call(client, schema_cache_strategy="lazy"),
        call(client, schema_cache_strategy="lazy", hybrid_policy="ADHOC_BF"),
        call(client, schema_cache_strategy="lazy", hybrid_policy="BATCHES"),
    ]
    assert [e.execute.await_count for e in executors] == [2, 2, 1]
    for executor in executors:
        executor.execute.assert_awaited_with(SQL, params={"id": 1})
