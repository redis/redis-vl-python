"""
RedisVL Exception Classes

This module defines all custom exceptions used throughout the RedisVL library.
"""

from redis.exceptions import ResponseError


class RedisVLError(Exception):
    """Base exception for all RedisVL errors."""

    pass


class RedisSearchError(RedisVLError):
    """Error raised for Redis Search specific operations."""

    pass


class PartialDeletionError(RedisVLError):
    """Raised when some keys could not be removed from Redis.

    On Redis Cluster, keys are removed one at a time and a per-key failure is
    logged rather than raised, so one unreachable slot does not abort the whole
    batch. That is fine for callers who only want a best-effort count, but not for
    ones that mirror the keyspace in their own state — a ``SemanticRouter``, for
    instance, persists the list of references it holds. Those callers reconcile
    their state against the keys that actually went away, persist it, and *then*
    raise this, so the partial deletion cannot pass unnoticed and a retry targets
    exactly what is left.

    Attributes:
        failed_keys (List[str]): The keys still present in Redis.
        deleted (int): How many keys were removed.
    """

    def __init__(self, message: str, failed_keys: list[str], deleted: int = 0):
        super().__init__(message)
        self.failed_keys = list(failed_keys)
        self.deleted = deleted


class SchemaValidationError(RedisVLError):
    """Error when validating data against a schema."""

    def __init__(self, message, index=None):
        # Only add index prefix if the message doesn't already contain detailed validation info
        if index is not None and not message.startswith("Schema validation failed"):
            message = f"Validation failed for object at index {index}: {message}"
        super().__init__(message)


class QueryValidationError(RedisVLError):
    """Error when validating a query."""

    pass


class RedisModuleVersionError(RedisVLError):
    """Error when Redis or module versions are incompatible with requested features."""

    @classmethod
    def for_svs_vamana(cls, min_redis_version: str):
        """Create error for unsupported SVS-VAMANA.

        Args:
            min_redis_version: Minimum required Redis version

        Returns:
            RedisModuleVersionError with formatted message
        """
        message = (
            f"SVS-VAMANA requires Redis >= {min_redis_version} with Redis Search >= 2.8.10. "
            f"Options: 1) Upgrade Redis to a version with Redis Search >= 2.8.10, "
            f"2) Use algorithm='hnsw' or 'flat', "
            f"3) Remove compression parameters"
        )
        return cls(message)


def _is_missing_index_error(exc: RedisSearchError) -> bool:
    """Check whether a failed ``FT.INFO`` means the index does not exist.

    Redis Search reports an absent index as a plain error reply whose wording
    changes between versions, so the reply is classified by type, not text.
    redis-py raises a bare :class:`~redis.exceptions.ResponseError` for an
    ordinary error reply and a more specific exception for the replies that
    mean the check itself failed (``NOPERM``, ``NOAUTH``, ``LOADING``,
    ``READONLY``, ``CLUSTERDOWN``, ``TRYAGAIN`` and others, mapped by prefix in
    redis-py's parser); connection and timeout failures are not
    ``ResponseError`` at all. ``FT.INFO`` takes a single index name, so a bare
    error reply to it is, in practice, the missing-index reply.

    The other known bare reply is ``unknown command``, from a server without
    Redis Search. It is read as "missing" too. Either way a misread yields
    ``False``, never ``True``, so :meth:`~redisvl.index.SearchIndex.create`
    goes on to run ``FT.CREATE``, which raises the server's real error rather
    than dropping anything.

    Args:
        exc: A :class:`RedisSearchError` wrapping the ``redis-py`` exception
            that Redis raised.

    Returns:
        bool: True if the error indicates a missing index, False otherwise.
    """
    return type(exc.__cause__) is ResponseError
