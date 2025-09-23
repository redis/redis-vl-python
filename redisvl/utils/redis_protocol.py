"""
Utilities for handling Redis protocol version detection safely across different client types.

This module provides safe wrappers around redis-py's get_protocol_version function
to handle edge cases with Redis Cluster pipelines.
"""

from typing import Union

from redis.asyncio.cluster import ClusterPipeline as AsyncClusterPipeline
from redis.cluster import ClusterPipeline
from redis.commands.helpers import get_protocol_version as redis_get_protocol_version

from redisvl.utils.log import get_logger

logger = get_logger(__name__)


def get_protocol_version(client) -> str:
    """
    Wrapper for redis-py's get_protocol_version that handles ClusterPipeline.

    ClusterPipeline doesn't have nodes_manager attribute, so we need to
    handle this case specially to avoid AttributeError.

    Args:
        client: Redis client, pipeline, or cluster pipeline object

    Returns:
        str: Protocol version ("2" or "3")

    Note:
        This function addresses issue #365 where get_protocol_version() fails
        with ClusterPipeline objects due to missing nodes_manager attribute.
    """
    # Handle sync ClusterPipeline
    if isinstance(client, ClusterPipeline):
        try:
            # Try to get protocol from the underlying cluster client
            if hasattr(client, "_redis_cluster") and client._redis_cluster:
                try:
                    result = redis_get_protocol_version(client._redis_cluster)
                    if result is not None:
                        return result
                except (AttributeError, Exception):
                    # If anything fails, fall back to default
                    pass

            logger.debug(
                "ClusterPipeline without valid _redis_cluster, defaulting to protocol 3"
            )
            return "3"
        except AttributeError as e:
            logger.debug(
                f"Failed to get protocol version from ClusterPipeline: {e}, defaulting to protocol 3"
            )
            return "3"

    # Handle async ClusterPipeline
    if isinstance(client, AsyncClusterPipeline):
        try:
            # Try to get protocol from the underlying cluster client
            if hasattr(client, "_redis_cluster") and client._redis_cluster:
                try:
                    result = redis_get_protocol_version(client._redis_cluster)
                    if result is not None:
                        return result
                except (AttributeError, Exception):
                    # If anything fails, fall back to default
                    pass

            logger.debug(
                "AsyncClusterPipeline without valid _redis_cluster, defaulting to protocol 3"
            )
            return "3"
        except AttributeError as e:
            logger.debug(
                f"Failed to get protocol version from AsyncClusterPipeline: {e}, defaulting to protocol 3"
            )
            return "3"

    # For all other client types, use the standard function
    try:
        result = redis_get_protocol_version(client)
        if result is None:
            logger.warning(
                f"get_protocol_version returned None for client {type(client)}, defaulting to protocol 3"
            )
            return "3"
        return result
    except AttributeError as e:
        logger.warning(
            f"Failed to get protocol version from client {type(client)}: {e}, defaulting to protocol 3"
        )
        return "3"
