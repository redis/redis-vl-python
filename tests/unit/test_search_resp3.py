"""
Unit tests for how SearchIndex.search() and AsyncSearchIndex.search() return a
RESP3 FT.SEARCH reply.

A client on RESP3 gets the reply back as a dict rather than a Result, and
query(), paginate() and clear() all read it as a Result. The integration suite
covers standalone clients on every response format; these tests cover the
cluster branch, which the integration suite runs only with --run-cluster-tests
and CI does not run.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
from redis.commands.search import AsyncSearch, Search

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import CountQuery, FilterQuery
from redisvl.query.filter import FilterExpression, Tag
from redisvl.schema import IndexSchema

INDEX_NAME = "my_index"


def _schema():
    return IndexSchema.from_dict(
        {
            "index": {"name": INDEX_NAME, "prefix": "test"},
            "fields": [{"name": "t", "type": "tag"}],
        }
    )


def _resp3_reply():
    """An FT.SEARCH reply for one document, as a RESP3 client returns it."""
    return {
        "attributes": [],
        "format": "STRING",
        "results": [{"id": "test:1", "extra_attributes": {"t": "foo"}, "values": []}],
        "total_results": 1,
        "warning": [],
    }


class TestSearchOnResp3Cluster:
    """On a cluster, search() goes through cluster_search(), not ft().search()."""

    def _index(self):
        client = MagicMock(spec=RedisCluster)
        # A real Search, so the reply is converted by redis-py's own parsers.
        client.ft.return_value = Search(Redis(protocol=3), index_name=INDEX_NAME)
        return SearchIndex(schema=_schema(), redis_client=client)

    @patch("redisvl.index.index.cluster_search")
    def test_query_returns_documents(self, cluster_search):
        cluster_search.return_value = _resp3_reply()
        index = self._index()

        assert index.query(FilterQuery(Tag("t") == "foo")) == [
            {"id": "test:1", "t": "foo"}
        ]

    @patch("redisvl.index.index.cluster_search")
    def test_count_query_returns_total(self, cluster_search):
        cluster_search.return_value = _resp3_reply()
        index = self._index()

        assert index.query(CountQuery(FilterExpression("*"))) == 1

    @pytest.mark.asyncio
    @patch("redisvl.index.index.async_cluster_search", new_callable=AsyncMock)
    async def test_async_query_returns_documents(self, async_cluster_search):
        async_cluster_search.return_value = _resp3_reply()
        client = MagicMock()
        client.__class__ = AsyncRedisCluster  # isinstance drives the branch
        client.ft.return_value = AsyncSearch(
            AsyncRedis(protocol=3), index_name=INDEX_NAME
        )
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        assert await index.query(FilterQuery(Tag("t") == "foo")) == [
            {"id": "test:1", "t": "foo"}
        ]
        assert await index.query(CountQuery(FilterExpression("*"))) == 1
