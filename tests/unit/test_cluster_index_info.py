"""
Unit tests for the Redis Cluster branch of SearchIndex._info() and
AsyncSearchIndex._info().

On a cluster, _info() sends a raw FT.INFO to one node, so redis-py's search
parser never sees the reply and its shape depends on the wire protocol: a flat
list on RESP2, a map on RESP3 (the redis-py 8 default). Both must produce the
same index information.

_info() must also ask the node that FT.CREATE went to, the default node: a
replica learns of a new index only once replication catches up, so a replica
can answer "no such index" straight after create(). The stand-in clients below
model that with a replica that never has the index.

The async cluster client also discovers its nodes lazily, on its first command,
so a freshly built client has no node to pick. The live cluster suite covers
these cases but only runs with --run-cluster-tests, which CI never passes.
"""

from unittest.mock import create_autospec

import pytest
from redis.asyncio.cluster import ClusterNode as AsyncClusterNode
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import ClusterNode, RedisCluster
from redis.exceptions import ResponseError

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.redis.connection import convert_index_info_to_schema
from redisvl.schema import IndexSchema

INDEX_NAME = "my_index"

# The same index as FT.INFO returns it on each protocol, trimmed to the
# sections from_existing() reads.
RESP2_REPLY = [
    b"index_name",
    INDEX_NAME.encode(),
    b"index_definition",
    [b"key_type", b"HASH", b"prefixes", [b"doc"]],
    b"attributes",
    [[b"identifier", b"title", b"attribute", b"title", b"type", b"TEXT"]],
    b"num_docs",
    b"0",
]
RESP3_REPLY = {
    b"index_name": INDEX_NAME.encode(),
    b"index_definition": {b"key_type": b"HASH", b"prefixes": [b"doc"]},
    b"attributes": [
        {
            b"identifier": b"title",
            b"attribute": b"title",
            b"type": b"TEXT",
            b"flags": [],
        }
    ],
    b"num_docs": 0,
}
REPLIES = pytest.mark.parametrize(
    "reply", [RESP2_REPLY, RESP3_REPLY], ids=["resp2", "resp3"]
)


def _schema():
    return IndexSchema.from_dict(
        {
            "index": {"name": INDEX_NAME, "prefix": "doc"},
            "fields": [{"name": "title", "type": "text"}],
        }
    )


def _answer_from_primary(primary, reply):
    """FT.INFO that only the primary can answer: the replica has not caught up."""

    def execute_command(*args, target_nodes=None):
        if target_nodes not in (primary, [primary]):
            raise ResponseError(f"{INDEX_NAME}: no such index")
        return reply

    return execute_command


def _sync_cluster(reply):
    client = create_autospec(RedisCluster, instance=True)
    primary = create_autospec(ClusterNode, instance=True)
    client.get_default_node.return_value = primary
    client.get_random_node.return_value = create_autospec(ClusterNode, instance=True)
    client.execute_command.side_effect = _answer_from_primary(primary, reply)
    return client


def _fresh_async_cluster(reply=None):
    """An async cluster client that, like a real one, knows no nodes until
    initialize() has run."""
    client = create_autospec(AsyncRedisCluster, instance=True)
    primary = create_autospec(AsyncClusterNode, instance=True)
    replica = create_autospec(AsyncClusterNode, instance=True)
    initialized = False

    async def initialize():
        nonlocal initialized
        initialized = True
        return client

    def get_random_node():
        if not initialized:
            raise IndexError("Cannot choose from an empty sequence")
        return replica

    client.initialize.side_effect = initialize
    client.get_random_node.side_effect = get_random_node
    client.get_default_node.side_effect = lambda: primary if initialized else None
    client.execute_command.side_effect = _answer_from_primary(primary, reply)
    primary.execute_command.return_value = [INDEX_NAME.encode()]
    replica.execute_command.return_value = []
    return client, primary


class TestClusterInfoReplyShape:
    @REPLIES
    def test_sync_info_reads_either_protocol(self, reply):
        info = SearchIndex._info(INDEX_NAME, _sync_cluster(reply))

        assert info["index_name"] == INDEX_NAME
        assert convert_index_info_to_schema(info) == convert_index_info_to_schema(
            SearchIndex._info(INDEX_NAME, _sync_cluster(RESP2_REPLY))
        )

    @REPLIES
    @pytest.mark.asyncio
    async def test_async_info_reads_either_protocol(self, reply):
        client, _ = _fresh_async_cluster(reply)
        reference, _ = _fresh_async_cluster(RESP2_REPLY)

        info = await AsyncSearchIndex._info(INDEX_NAME, client)

        assert info["index_name"] == INDEX_NAME
        assert convert_index_info_to_schema(info) == convert_index_info_to_schema(
            await AsyncSearchIndex._info(INDEX_NAME, reference)
        )


class TestFreshAsyncClusterClient:
    @pytest.mark.asyncio
    async def test_delete_targets_a_node_not_none(self):
        client, node = _fresh_async_cluster()
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        await index.delete(drop=False)

        client.execute_command.assert_awaited_once_with(
            "FT.DROPINDEX", INDEX_NAME, target_nodes=[node]
        )


class TestListallOnCluster:
    @pytest.mark.asyncio
    async def test_listall_sees_a_new_index(self):
        client, _ = _fresh_async_cluster()
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        assert await index.listall() == [INDEX_NAME]

    def test_sync_listall_sees_a_new_index(self):
        client = _sync_cluster([INDEX_NAME.encode()])
        index = SearchIndex(schema=_schema(), redis_client=client)

        assert index.listall() == [INDEX_NAME]
