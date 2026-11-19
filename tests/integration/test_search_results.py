import os

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag


@pytest.fixture
def filter_query():
    return FilterQuery(
        return_fields=None,
        filter_expression=Tag("credit_score") == "high",
    )


@pytest.fixture
def index(sample_data, redis_url, redis_test_name):
    fields_spec = [
        {"name": "credit_score", "type": "tag"},
        {"name": "user", "type": "tag"},
        {"name": "job", "type": "text"},
        {"name": "age", "type": "numeric"},
        {
            "name": "user_embedding",
            "type": "vector",
            "attrs": {
                "dims": 3,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
    ]

    json_schema = {
        "index": {
            "name": redis_test_name("user_index_json"),
            "prefix": redis_test_name("users_json"),
            "storage_type": "json",
        },
        "fields": fields_spec,
    }

    # construct a search index from the schema
    index = SearchIndex.from_dict(json_schema, redis_url=redis_url)

    # create the index (no data yet); drop any stale docs left by an
    # interrupted earlier run sharing this worker's Redis database
    index.create(overwrite=True, drop=True)

    # Prepare and load the data
    index.load(sample_data)

    # run the test
    yield index

    # clean up
    index.delete(drop=True)


def test_process_results_unpacks_json_properly(index, filter_query):
    results = index.query(filter_query)
    assert len(results) == 4


ID_CASES = [
    (storage, return_fields)
    for storage in ("hash", "json")
    for return_fields in (None, ["status"])
]


def _id_schema(redis_test_name, storage):
    return {
        "index": {
            "name": redis_test_name(f"id_is_key_{storage}"),
            "prefix": redis_test_name(f"id_is_key_{storage}"),
            "storage_type": storage,
        },
        "fields": [{"name": "status", "type": "tag"}],
    }


def test_result_id_is_redis_key(redis_url, redis_test_name):
    # A document's own "id" field must never shadow the Redis key, whatever
    # the storage type or return_fields.
    for storage, return_fields in ID_CASES:
        index = SearchIndex.from_dict(
            _id_schema(redis_test_name, storage), redis_url=redis_url
        )
        index.create(overwrite=True, drop=True)
        try:
            [key] = index.load([{"id": "t1", "status": "open"}], id_field="id")
            query = FilterQuery(
                filter_expression=Tag("status") == "open",
                return_fields=return_fields,
            )
            assert index.query(query) == [{"id": key, "status": "open"}], (
                storage,
                return_fields,
            )
        finally:
            index.delete(drop=True)


async def test_async_result_id_is_redis_key(redis_url, redis_test_name):
    for storage, return_fields in ID_CASES:
        index = AsyncSearchIndex.from_dict(
            _id_schema(redis_test_name, storage), redis_url=redis_url
        )
        await index.create(overwrite=True, drop=True)
        try:
            [key] = await index.load([{"id": "t1", "status": "open"}], id_field="id")
            query = FilterQuery(
                filter_expression=Tag("status") == "open",
                return_fields=return_fields,
            )
            assert await index.query(query) == [{"id": key, "status": "open"}], (
                storage,
                return_fields,
            )
        finally:
            await index.delete(drop=True)
            await index.disconnect()
