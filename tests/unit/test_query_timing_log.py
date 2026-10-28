"""``query()`` emits a debug log with the query type, elapsed time and result count.

Canned ``FT.SEARCH`` replies are served from a fake ``index.search`` so the real
``query`` -> ``_query`` -> ``process_results`` chain runs without Redis.
"""

import logging

from redis.commands.search.result import Result

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import CountQuery, VectorQuery
from redisvl.schema import IndexSchema

LOGGER_NAME = "redisvl.index.index"

SECRET_TEXT = "do-not-log-me"


def _schema():
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "timing_test",
                "prefix": "timing_test",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "user_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 4,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
                {"name": "brand", "type": "tag"},
            ],
        }
    )


def _query():
    return VectorQuery(
        vector=[0.1, 0.1, 0.5, 0.15],
        vector_field_name="user_embedding",
        return_fields=["brand"],
        filter_expression=f"@brand:{{{SECRET_TEXT}}}",
    )


def _reply():
    return Result(
        [
            2,
            "timing_test:1",
            ["vector_distance", "0.1", "brand", "Nike"],
            "timing_test:2",
            ["vector_distance", "0.2", "brand", "Adidas"],
        ],
        True,
    )


def _timing_records(caplog):
    return [
        r
        for r in caplog.records
        if r.name == LOGGER_NAME and r.levelno == logging.DEBUG
    ]


def _assert_timing_record(caplog):
    records = _timing_records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert message.startswith("Index timing_test executed VectorQuery in ")
    assert message.endswith(" ms (2 results)")
    # The query string and its params must never reach the log.
    assert SECRET_TEXT not in message


def test_query_logs_timing_at_debug(caplog):
    index = SearchIndex(_schema())
    index.search = lambda *args, **kwargs: _reply()  # type: ignore[method-assign]

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        results = index.query(_query())

    assert [doc["id"] for doc in results] == ["timing_test:1", "timing_test:2"]
    _assert_timing_record(caplog)


def test_count_query_logs_its_count_at_debug(caplog):
    # CountQuery resolves to an int, not a list, so the log must not call len().
    index = SearchIndex(_schema())
    index.search = lambda *args, **kwargs: Result([7], True)  # type: ignore[method-assign]

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert index.query(CountQuery()) == 7

    records = _timing_records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert message.startswith("Index timing_test executed CountQuery in ")
    assert message.endswith(" ms (7 results)")


def test_query_does_not_log_timing_above_debug(caplog):
    index = SearchIndex(_schema())
    index.search = lambda *args, **kwargs: _reply()  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        index.query(_query())

    assert _timing_records(caplog) == []


async def test_async_query_logs_timing_at_debug(caplog):
    index = AsyncSearchIndex(_schema())

    async def _search(*args, **kwargs):
        return _reply()

    index.search = _search  # type: ignore[method-assign]

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        results = await index.query(_query())

    assert [doc["id"] for doc in results] == ["timing_test:1", "timing_test:2"]
    _assert_timing_record(caplog)
