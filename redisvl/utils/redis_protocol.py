"""Helpers for asking what RESP protocol a Redis client speaks.

There are two distinct questions here, and answering one with the other is a
bug. ``get_protocol_version`` reports what the caller *configured*;
``effective_protocol`` reports what the connection will *actually* speak.
"""

from redis.commands.helpers import get_protocol_version as redis_get_protocol_version
from redis.connection import DEFAULT_RESP_VERSION


def get_protocol_version(client) -> str | int | None:
    """Return the ``protocol`` connection kwarg the caller configured.

    Wraps redis-py's helper of the same name, which raises ``AttributeError``
    for a ``ClusterPipeline`` whose ``nodes_manager`` is unset (issue #365).

    The return type is deliberately loose. redis-py only coerces ``protocol``
    out of a URL query string from 8.0 onward, so ``redis://host?protocol=3``
    yields the string ``"3"`` on 6.x and 7.x but the integer ``3`` on 8.x.
    Callers comparing against a number must coerce; see ``effective_protocol``.

    Args:
        client: Redis client, cluster client, or pipeline.

    Returns:
        The configured protocol, or ``None`` when the caller did not set one.
    """
    try:
        return redis_get_protocol_version(client)
    except AttributeError:
        # ClusterPipeline without nodes_manager. Let the caller decide.
        return None


def effective_protocol(client) -> int:
    """Return the RESP version ``client`` will actually speak.

    An unset ``protocol`` kwarg is not the same as RESP2: it means "whatever
    this redis-py defaults to", which changed from 2 to 3 in redis-py 8.0. A
    client from ``Redis.from_url()`` therefore reports ``None`` while speaking
    RESP3, so ``get_protocol_version`` alone cannot answer questions about
    reply shape.

    Resolve the unset case from redis-py's own ``DEFAULT_RESP_VERSION`` rather
    than from its version number, so this stays correct if the default moves
    again.

    Args:
        client: Redis client, cluster client, or pipeline.

    Returns:
        ``2`` or ``3``.
    """
    protocol = get_protocol_version(client)
    if protocol is not None:
        return int(protocol)
    return DEFAULT_RESP_VERSION
