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


# redis-py raises a ResponseError subclass for most error codes that mean the
# check itself failed, but leaves BUSY (a script or function is running)
# unmapped, so it arrives as a bare ResponseError with the code still leading
# the message.
_UNMAPPED_TRANSIENT_ERROR_CODES = frozenset({"BUSY"})


def _is_missing_index_error(exc: RedisSearchError) -> bool:
    """Check whether a failed ``FT.INFO`` means the index does not exist.

    An absent index is a bare :class:`~redis.exceptions.ResponseError` whose
    wording changes between Redis versions, so the reply is classified by type
    and leading error code, never by the wording after the code. Error codes
    redis-py maps to a subclass (``NOPERM``, ``LOADING``, ``CLUSTERDOWN`` and
    others), connection failures and an unmapped ``BUSY`` reply all mean the
    check failed.

    Args:
        exc: A :class:`RedisSearchError` wrapping the ``redis-py`` exception
            that Redis raised.

    Returns:
        bool: True if the error indicates a missing index, False otherwise.
    """
    cause = exc.__cause__
    if type(cause) is not ResponseError:
        return False
    return str(cause).split(" ", 1)[0] not in _UNMAPPED_TRANSIENT_ERROR_CODES
