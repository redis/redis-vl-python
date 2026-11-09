"""
Unit tests for how SearchIndex.search() and AsyncSearchIndex.search() return a
RESP3 FT.SEARCH reply on a cluster client.

A client on RESP3 gets the reply back as a dict rather than a Result, and
query() reads it as a Result. The integration suite covers standalone clients
on every response format; these tests cover the cluster branch, which the
integration suite runs only with --run-cluster-tests and CI does not run.
"""

from unittest.mock import MagicMock

import pytest
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
from redis.commands.search import AsyncSearch, Search

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import CountQuery, FilterQuery
from redisvl.query.filter import FilterExpression
from redisvl.schema import IndexSchema

INDEX_NAME = "my_index"

# An FT.SEARCH reply for one document, as a RESP3 client returns it.
RESP3_REPLY = {
    "attributes": [],
    "format": "STRING",
    "results": [{"id": "test:1", "extra_attributes": {"t": "foo"}, "values": []}],
    "total_results": 1,
    "warning": [],
}


def _resp3_cluster_index(index_cls, client_cls, search_cls):
    """Build an index on a RESP3 cluster client whose FT.SEARCH returns RESP3_REPLY.

    The client is mocked, but ft() returns a real Search or AsyncSearch, so the
    reply goes through cluster_search() and redis-py's own parsers.
    """
    schema = IndexSchema.from_dict(
        {
            "index": {"name": INDEX_NAME, "prefix": "test"},
            "fields": [{"name": "t", "type": "tag"}],
        }
    )
    client = MagicMock(spec=client_cls)
    client.nodes_manager = MagicMock(connection_kwargs={"protocol": 3})
    client.get_default_node.return_value = MagicMock()
    client.execute_command.return_value = RESP3_REPLY
    client.ft.return_value = search_cls(client, index_name=INDEX_NAME)
    return index_cls(schema=schema, redis_client=client)


def test_cluster_query_on_resp3():
    index = _resp3_cluster_index(SearchIndex, RedisCluster, Search)

    assert index.query(FilterQuery(filter_expression="@t:{foo}")) == [
        {"id": "test:1", "t": "foo"}
    ]
    assert index.query(CountQuery(FilterExpression("*"))) == 1


@pytest.mark.asyncio
async def test_async_cluster_query_on_resp3():
    index = _resp3_cluster_index(AsyncSearchIndex, AsyncRedisCluster, AsyncSearch)

    assert await index.query(FilterQuery(filter_expression="@t:{foo}")) == [
        {"id": "test:1", "t": "foo"}
    ]
    assert await index.query(CountQuery(FilterExpression("*"))) == 1
