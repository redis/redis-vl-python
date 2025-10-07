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
