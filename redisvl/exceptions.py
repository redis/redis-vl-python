"""
RedisVL Exception Classes

This module defines all custom exceptions used throughout the RedisVL library.
"""


class RedisVLError(Exception):
    """Base exception for all RedisVL errors."""

    pass


class RedisSearchError(RedisVLError):
    """Error raised for Redis Search specific operations."""

    pass


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


# Redis Search reports an absent index as an ordinary error reply whose wording
# depends on the server version, so matching the message is the only option.
# Every known wording lives here rather than at each call site. See the "Telling
# 'the index is missing' apart from other failures" section of docs/api/exceptions.rst.
_MISSING_INDEX_ERROR_FRAGMENTS = (
    "unknown index name",
    "no such index",
    "search_index_not_found",
    "index not found",
)


def _is_missing_index_error(exc: RedisSearchError) -> bool:
    """Check whether a Redis Search error means the index does not exist.

    Different Redis Search versions phrase the error differently, so every
    known wording is matched. The check reads the underlying Redis error rather
    than the :class:`RedisSearchError` wrapping it, because the wrapper's
    message interpolates the index name and would otherwise match for an index
    named after one of the wordings.

    Args:
        exc: A :class:`RedisSearchError` wrapping the ``redis-py`` exception
            that Redis raised. Wrappers raised without a cause are matched on
            their own message.

    Returns:
        bool: True if the error indicates a missing index, False otherwise.
    """
    cause = exc.__cause__ if exc.__cause__ is not None else exc
    message = str(cause).lower()
    return any(fragment in message for fragment in _MISSING_INDEX_ERROR_FRAGMENTS)
