"""
Tests ClusterPipeline
"""

import pytest
from redis.cluster import RedisCluster
from redis.commands.helpers import get_protocol_version

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema


@pytest.fixture
def cluster_index_url(redis_cluster_url):
    """A cluster URL that makes SearchIndex build a RedisCluster client.

    RedisVL only builds a cluster client when the URL (or kwargs) says so; a
    bare single-node URL gets a plain Redis client that raises MovedError as
    soon as a key hashes to another shard.
    """
    return f"{redis_cluster_url}?cluster=true"


def _create_cluster_index(schema, url):
    index = SearchIndex(schema, redis_url=url)
    index.create(overwrite=True, drop=True)
    # The client is built lazily, so it only exists once create() has run.
    assert isinstance(index.client, RedisCluster)
    return index


@pytest.mark.requires_cluster
def test_real_cluster_pipeline_get_protocol_version(redis_cluster_url):
    """
    Test that get_protocol_version works with ClusterPipeline
    """
    # Create REAL Redis Cluster client
    cluster_client = RedisCluster.from_url(redis_cluster_url)

    # Create REAL pipeline from cluster
    pipeline = cluster_client.pipeline()

    # This is the actual line that was failing in issue #365
    # If our fix works, this should NOT raise AttributeError
    protocol = get_protocol_version(pipeline)

    # Protocol should be a string ("2" or "3") or None
    assert protocol in [None, "2", "3", 2, 3], f"Unexpected protocol: {protocol}"

    # Clean up
    cluster_client.close()


@pytest.mark.requires_cluster
def test_real_searchindex_with_cluster_batch_operations(
    cluster_index_url, redis_test_name
):
    """
    Test SearchIndex.load() with Redis Cluster.
    """
    # Create schema like the user had
    index_prefix = redis_test_name("doc")
    schema_dict = {
        "index": {
            "name": redis_test_name("test-real-365"),
            "prefix": index_prefix,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "id", "type": "tag"},
            {"name": "text", "type": "text"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)

    # Create SearchIndex with REAL cluster URL
    index = _create_cluster_index(schema, cluster_index_url)

    try:
        # Test data like user had
        test_data = [{"id": f"item{i}", "text": f"Document {i}"} for i in range(10)]

        # See issue #365
        # index.load() with batch_size triggers pipeline operations internally
        keys = index.load(
            data=test_data,
            id_field="id",
            batch_size=3,  # Forces multiple pipeline operations
        )

        assert len(keys) == 10
        assert all(k.startswith(f"{index_prefix}:") for k in keys)

    finally:
        # Clean up
        index.delete(drop=True)


@pytest.mark.requires_cluster
@pytest.mark.parametrize(
    "protocol, expected",
    [(None, [None, "2", "3", 2, 3]), (2, [2, "2", None]), (3, [3, "3", None])],
)
def test_cluster_pipeline_protocol_version_directly(
    redis_cluster_url, protocol, expected
):
    """
    Test get_protocol_version with various cluster configurations.
    """
    kwargs = {} if protocol is None else {"protocol": protocol}
    cluster = RedisCluster.from_url(redis_cluster_url, **kwargs)
    try:
        # This should work without AttributeError
        assert get_protocol_version(cluster.pipeline()) in expected
    finally:
        cluster.close()


@pytest.mark.requires_cluster
def test_batch_search_with_real_cluster(cluster_index_url, redis_test_name):
    """
    Test batch_search which uses get_protocol_version internally.
    """
    from redisvl.query import FilterQuery

    schema_dict = {
        "index": {
            "name": redis_test_name("test-batch-365"),
            "prefix": redis_test_name("batch"),
            "storage_type": "json",
        },
        "fields": [
            {"name": "id", "type": "tag"},
            {"name": "category", "type": "tag"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)
    index = _create_cluster_index(schema, cluster_index_url)

    try:
        # Load test data
        data = [{"id": f"doc{i}", "category": f"cat{i % 3}"} for i in range(15)]
        index.load(data=data, id_field="id")

        # Create multiple queries
        queries = [
            FilterQuery(filter_expression=f"@category:{{cat{i}}}") for i in range(3)
        ]

        # batch_search internally uses get_protocol_version on pipelines
        results = index.batch_search(
            [(q.query, q.params) for q in queries], batch_size=2
        )

        assert len(results) == 3

    finally:
        index.delete(drop=True)


@pytest.mark.requires_cluster
@pytest.mark.parametrize("ttl", [None, 30])
def test_cluster_load_with_ttl(cluster_index_url, ttl, redis_test_name):
    """
    Test that TTL is correctly set on keys when using load() with ttl parameter on cluster.
    """
    schema_dict = {
        "index": {
            "name": redis_test_name("test-ttl-cluster"),
            "prefix": redis_test_name("ttl"),
            "storage_type": "hash",
        },
        "fields": [
            {"name": "id", "type": "tag"},
            {"name": "text", "type": "text"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)
    index = _create_cluster_index(schema, cluster_index_url)

    try:
        # Load test data with TTL parameter
        data = [{"id": "1", "text": "foo"}]
        keys = index.load(data, id_field="id", ttl=ttl)

        # Check TTL on the loaded key
        key_ttl = index.client.ttl(keys[0])

        if ttl is None:
            # No TTL set, should return -1
            assert key_ttl == -1
        else:
            # TTL should be set and close to the expected value
            assert key_ttl > 0
            assert abs(key_ttl - ttl) <= 5

    finally:
        index.delete(drop=True)
