"""Migration key enumeration across Redis wire response formats."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from redis.asyncio.client import Redis as AsyncRedis
from redis.client import Redis
from redis.exceptions import ResponseError

from redisvl.migration import AsyncMigrationExecutor, MigrationExecutor


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def migration(request):
    client = MagicMock()
    executor = MigrationExecutor()
    if request.param:
        executor = AsyncMigrationExecutor()
        client.ft.return_value.info = AsyncMock()
        client.execute_command = AsyncMock()
        client.scan = AsyncMock()
    redis_class = AsyncRedis if request.param else Redis
    client.scan_iter = redis_class.scan_iter.__get__(client)
    return executor, client


async def _collect(executor, client):
    keys = executor._enumerate_indexed_keys(client, "source", batch_size=2)
    if isinstance(executor, AsyncMigrationExecutor):
        return [key async for key in keys]
    return list(keys)


def _wire(value, decode_responses):
    if isinstance(value, str):
        return value if decode_responses else value.encode()
    if isinstance(value, dict):
        return {
            _wire(key, decode_responses): _wire(item, decode_responses)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_wire(item, decode_responses) for item in value]
    return value


def _aggregate_page(keys, cursor, protocol, decode_responses):
    if protocol == 2:
        # The leading metadata value need not equal the number of rows.
        rows = [1, *[["__key", key] for key in keys]]
    else:
        rows = {
            "attributes": [],
            "format": "STRING",
            "results": [
                {"extra_attributes": {"__key": key}, "values": []} for key in keys
            ],
            "total_results": len(keys),
            "warning": [],
        }
    return [_wire(rows, decode_responses), cursor]


@pytest.mark.parametrize("protocol", [2, 3])
@pytest.mark.parametrize("decode_responses", [False, True])
@pytest.mark.asyncio
async def test_enumerate_aggregate_cursor_pages(migration, protocol, decode_responses):
    executor, client = migration
    client.ft.return_value.info.return_value = _wire(
        {"hash_indexing_failures": 0, "percent_indexed": 1.0}, decode_responses
    )
    client.execute_command.side_effect = [
        _aggregate_page(["doc:1", "doc:中文"], 17, protocol, decode_responses),
        _aggregate_page([], 17, protocol, decode_responses),
        _aggregate_page(["doc:3"], 0, protocol, decode_responses),
    ]

    assert await _collect(executor, client) == ["doc:1", "doc:中文", "doc:3"]
    assert client.execute_command.call_args_list == [
        call(
            "FT.AGGREGATE",
            "source",
            "*",
            "LOAD",
            "1",
            "__key",
            "WITHCURSOR",
            "COUNT",
            "2",
            "MAXIDLE",
            "300000",
        ),
        call("FT.CURSOR", "READ", "source", "17", "COUNT", "2"),
        call("FT.CURSOR", "READ", "source", "17", "COUNT", "2"),
    ]
    client.scan.assert_not_called()


@pytest.mark.parametrize(
    "decode_responses, readiness",
    [
        (False, {"hash_indexing_failures": 2, "percent_indexed": 1.0}),
        (True, {"hash_indexing_failures": 2, "percent_indexed": 1.0}),
        (True, {"hash_indexing_failures": 0, "percent_indexed": 0.5}),
        (True, {"hash_indexing_failures": 0, "percent_indexed": 0.0}),
    ],
    ids=["byte-keys", "failed-documents", "partial-index", "zero-progress"],
)
@pytest.mark.asyncio
async def test_incomplete_index_scans_its_prefixes(
    migration, decode_responses, readiness
):
    executor, client = migration
    client.ft.return_value.info.return_value = _wire(
        {**readiness, "index_definition": {"prefixes": ["doc:", "archive:"]}},
        decode_responses,
    )
    client.scan.side_effect = [
        (5, _wire(["archive:1"], decode_responses)),
        (0, _wire(["archive:1", "archive:failed"], decode_responses)),
        (0, _wire(["doc:pending"], decode_responses)),
    ]
    assert await _collect(executor, client) == [
        "archive:1",
        "archive:failed",
        "doc:pending",
    ]
    assert client.scan.call_args_list == [
        call(cursor="0", match="archive:*", count=2, _type=None),
        call(cursor=5, match="archive:*", count=2, _type=None),
        call(cursor="0", match="doc:*", count=2, _type=None),
    ]
    # The fast path would omit failed/pending documents, even if it did not crash.
    client.execute_command.assert_not_called()


@pytest.mark.parametrize("decode_responses", [False, True])
@pytest.mark.asyncio
async def test_aggregate_error_preserves_scan_prefix(migration, decode_responses):
    executor, client = migration
    client.ft.return_value.info.return_value = _wire(
        {
            "hash_indexing_failures": 0,
            "percent_indexed": 1.0,
            "index_definition": {"prefixes": ["doc:"]},
        },
        decode_responses,
    )
    client.execute_command.side_effect = ResponseError("aggregate unavailable")
    client.scan.return_value = (0, [b"doc:1"])

    assert await _collect(executor, client) == ["doc:1"]
    client.scan.assert_called_once_with(cursor="0", match="doc:*", count=2, _type=None)


@pytest.mark.asyncio
async def test_closing_enumeration_releases_cursor(migration):
    executor, client = migration
    client.execute_command.return_value = _aggregate_page(["doc:1"], 17, 3, False)
    keys = executor._enumerate_with_aggregate(client, "source", batch_size=2)
    if isinstance(executor, AsyncMigrationExecutor):
        try:
            assert await anext(keys) == "doc:1"
        finally:
            await keys.aclose()
    else:
        try:
            assert next(keys) == "doc:1"
        finally:
            keys.close()

    assert client.execute_command.call_count == 2
    client.execute_command.assert_called_with("FT.CURSOR", "DEL", "source", "17")
