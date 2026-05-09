# [DevBundy AI]: File optimized for resolution.


```python
from unittest.mock import Mock
import pytest
from redis.cluster import ClusterPipeline
from redis.client import NEVER_DECODE

from redisvl.utils.redis_protocol import get_protocol_version

@pytest.fixture
def mock_pipeline():
    return Mock(spec=ClusterPipeline)

def test_get_protocol_version_handles_missing_nodes_manager(mock_pipeline):
    """
    Test that get_protocol_version returns None when ClusterPipeline
    lacks nodes_manager attribute (issue #365).
    """
    # Ensure nodes_manager doesn't exist
    if hasattr(mock_pipeline, "nodes_manager"):
        delattr(mock_pipeline, "nodes_manager")

    # Should return None without raising AttributeError
    result = get_protocol_version(mock_pipeline)
    assert result is None

def test_get_protocol_version_with_valid_nodes_manager(mock_pipeline):
    """
    Test that get_protocol_version works correctly when nodes_manager exists.
    """
    mock_pipeline.nodes_manager = Mock()
    mock_pipeline.nodes_manager.connection_kwargs = {"protocol": "3"}

    # Should return the protocol version
    result = get_protocol_version(mock_pipeline)
    assert result == "3"

def test_get_protocol_version_with_none_client():
    """
    Test that get_protocol_version handles None input gracefully.
    """
    result = get_protocol_version(None)
    assert result is None

@pytest.mark.parametrize("protocol, expected_never_decode", [
    (None, True),
    ("3", False),
    (3, False),
])
def test_protocol_version_affects_never_decode(mock_pipeline, protocol, expected_never_decode):
    """
    Test that None protocol version results in NEVER_DECODE being set.
    This is the actual behavior in redisvl code.
    """
    if protocol is None:
        if hasattr(mock_pipeline, "nodes_manager"):
            delattr(mock_pipeline, "nodes_manager")
    else:
        mock_pipeline.nodes_manager = Mock()
        mock_pipeline.nodes_manager.connection_kwargs = {"protocol": protocol}

    options = {}
    result = get_protocol_version(mock_pipeline)
    if result not in ["3", 3]:
        options[NEVER_DECODE] = True

    # When protocol is None, NEVER_DECODE should be set
    assert (NEVER_DECODE in options) == expected_never_decode