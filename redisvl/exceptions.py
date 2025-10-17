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
            f"SVS-VAMANA requires Redis >= {min_redis_version} with RediSearch >= 2.8.10. "
            f"Options: 1) Upgrade Redis Stack, "
            f"2) Use algorithm='hnsw' or 'flat', "
            f"3) Remove compression parameters"
        )
        return cls(message)
