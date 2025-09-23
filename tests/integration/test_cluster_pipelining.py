"""
Integration test for issue #365: ClusterPipeline AttributeError fix
https://github.com/redis/redis-vl-python/issues/365

This test verifies that the safe_get_protocol_version fix prevents the
AttributeError: 'ClusterPipeline' object has no attribute 'nodes_manager'
"""

from unittest.mock import Mock

import pytest
from redis.asyncio.cluster import ClusterPipeline as AsyncClusterPipeline
from redis.cluster import ClusterPipeline

from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.schema import IndexSchema
from redisvl.utils.redis_protocol import get_protocol_version


def test_pipeline_operations_no_nodes_manager_error(redis_url):
    """
    Test that pipeline operations don't fail with nodes_manager AttributeError.

    Before the fix, operations that use get_protocol_version() internally would fail
    with AttributeError when using ClusterPipeline. This test ensures those operations
    now work without that specific error.
    """
    # Create a simple schema
    schema_dict = {
        "index": {"name": "test-365-fix", "prefix": "doc", "storage_type": "hash"},
        "fields": [{"name": "id", "type": "tag"}, {"name": "text", "type": "text"}],
    }

    schema = IndexSchema.from_dict(schema_dict)
    index = SearchIndex(schema, redis_url=redis_url)

    # Create the index
    index.create(overwrite=True)

    try:
        # Test 1: Load with batching (uses pipelines internally)
        test_data = [{"id": f"item{i}", "text": f"Document {i}"} for i in range(10)]

        # This would fail with AttributeError before the fix
        keys = index.load(
            data=test_data,
            id_field="id",
            batch_size=3,  # Force multiple pipeline operations
        )

        assert len(keys) == 10

        # Test 2: Batch search (uses safe_get_protocol_version internally)
        queries = [FilterQuery(filter_expression=f"@id:{{item{i}}}") for i in range(3)]

        try:
            # The critical test: no AttributeError about nodes_manager
            results = index.batch_search(queries, batch_size=2)
            assert len(results) == 3
        except Exception as e:
            # If there's an error, it must NOT be the nodes_manager AttributeError
            assert "nodes_manager" not in str(
                e
            ), f"Got nodes_manager error which indicates fix isn't working: {e}"

        # Test 3: TTL operations
        try:
            index.expire_keys(keys[:3], 3600)
        except Exception as e:
            # Again, ensure no nodes_manager error
            assert "nodes_manager" not in str(e)

    finally:
        index.delete()


def test_json_storage_no_error(redis_url):
    """Test with JSON storage type."""
    schema_dict = {
        "index": {"name": "test-365-json", "prefix": "json", "storage_type": "json"},
        "fields": [{"name": "id", "type": "tag"}, {"name": "data", "type": "text"}],
    }

    schema = IndexSchema.from_dict(schema_dict)
    index = SearchIndex(schema, redis_url=redis_url)

    index.create(overwrite=True)

    try:
        # Load test data
        test_data = [{"id": f"doc{i}", "data": f"Document {i}"} for i in range(5)]

        # Should work without nodes_manager AttributeError
        keys = index.load(data=test_data, id_field="id", batch_size=2)

        assert len(keys) == 5

    finally:
        index.delete()


def test_clusterpipeline_with_valid_redis_cluster_attribute():
    """
    Test get_protocol_version when ClusterPipeline has _redis_cluster attribute.
    """
    # Create mock ClusterPipeline with _redis_cluster attribute
    mock_pipeline = Mock(spec=ClusterPipeline)
    mock_cluster = Mock()
    mock_cluster.nodes_manager.connection_kwargs.get.return_value = "3"
    mock_pipeline._redis_cluster = mock_cluster

    # Should successfully get protocol from _redis_cluster
    result = get_protocol_version(mock_pipeline)
    assert result == "3"


def test_clusterpipeline_with_none_redis_cluster():
    """
    Test get_protocol_version when _redis_cluster is None.
    """
    mock_pipeline = Mock(spec=ClusterPipeline)
    mock_pipeline._redis_cluster = None

    # Should fallback to "3"
    result = get_protocol_version(mock_pipeline)
    assert result == "3"


def test_async_clusterpipeline_without_nodes_manager():
    """
    Test get_protocol_version with AsyncClusterPipeline missing nodes_manager.
    """
    mock_pipeline = Mock(spec=AsyncClusterPipeline)
    # Ensure no nodes_manager attribute
    if hasattr(mock_pipeline, "nodes_manager"):
        delattr(mock_pipeline, "nodes_manager")
    mock_pipeline._redis_cluster = None

    # Should fallback to "3" without error
    result = get_protocol_version(mock_pipeline)
    assert result == "3"
