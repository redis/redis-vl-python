"""
Wrapper for redis-py's get_protocol_version to handle edge cases.

This fixes issue #365 where ClusterPipeline objects may not have nodes_manager attribute.
"""

from typing import Optional, Union

from redis.asyncio.cluster import ClusterPipeline as AsyncClusterPipeline
from redis.cluster import ClusterPipeline
from redis.commands.helpers import get_protocol_version as redis_get_protocol_version


def get_protocol_version(client) -> Optional[str]:
    """
    Safe wrapper for redis-py's get_protocol_version that handles edge cases.

    The main issue is that ClusterPipeline objects may not always have a
    nodes_manager attribute properly set, causing AttributeError.

    Args:
        client: Redis client, pipeline, or cluster pipeline object

    Returns:
        Protocol version string ("2" or "3") or None if unable to determine
    """
    try:
        # Use redis-py's function - it returns None for unknown types
        result = redis_get_protocol_version(client)
        return result
    except AttributeError:
        # This happens when ClusterPipeline doesn't have nodes_manager
        # Return None to let the caller decide what to do
        return None
