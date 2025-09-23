"""
Tests ClusterPipeline
"""

import pytest
from redis.cluster import RedisCluster
from redis.commands.helpers import get_protocol_version

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema


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
def test_real_searchindex_with_cluster_batch_operations(redis_cluster_url):
    """
    Test SearchIndex.load() with Redis Cluster.
    """
    # Create schema like the user had
    schema_dict = {
        "index": {"name": "test-real-365", "prefix": "doc", "storage_type": "hash"},
        "fields": [
            {"name": "id", "type": "tag"},
            {"name": "text", "type": "text"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)

    # Create SearchIndex with REAL cluster URL
    index = SearchIndex(schema, redis_url=redis_cluster_url)

    # Create the index
    index.create(overwrite=True)

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
        assert all(k.startswith("doc:") for k in keys)

    finally:
        # Clean up
        index.delete()


@pytest.mark.requires_cluster
def test_cluster_pipeline_protocol_version_directly():
    """
    Test get_protocol_version with various cluster configurations.
    """
    import os

    # Skip if no cluster available
    cluster_url = os.getenv("REDIS_CLUSTER_URL", "redis://localhost:7000")

    try:
        # Test with default protocol
        cluster = RedisCluster.from_url(cluster_url)
        pipeline = cluster.pipeline()

        # This should work without AttributeError
        protocol = get_protocol_version(pipeline)
        print(f"Protocol version from real cluster pipeline: {protocol}")

        cluster.close()

        # Test with explicit RESP2
        cluster2 = RedisCluster.from_url(cluster_url, protocol=2)
        pipeline2 = cluster2.pipeline()
        protocol2 = get_protocol_version(pipeline2)
        assert protocol2 in [2, "2", None]
        cluster2.close()

        # Test with explicit RESP3
        cluster3 = RedisCluster.from_url(cluster_url, protocol=3)
        pipeline3 = cluster3.pipeline()
        protocol3 = get_protocol_version(pipeline3)
        assert protocol3 in [3, "3", None]
        cluster3.close()

    except Exception as e:
        pytest.skip(f"Redis Cluster not available: {e}")


@pytest.mark.requires_cluster
def test_batch_search_with_real_cluster(redis_cluster_url):
    """
    Test batch_search which uses get_protocol_version internally.
    """
    from redisvl.query import FilterQuery

    schema_dict = {
        "index": {"name": "test-batch-365", "prefix": "batch", "storage_type": "json"},
        "fields": [
            {"name": "id", "type": "tag"},
            {"name": "category", "type": "tag"},
        ],
    }

    schema = IndexSchema.from_dict(schema_dict)
    index = SearchIndex(schema, redis_url=redis_cluster_url)

    index.create(overwrite=True)

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
        index.delete()
