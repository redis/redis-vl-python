"""
RedisVL Exception Classes

This module defines all custom exceptions used throughout the RedisVL library.
"""


class RedisVLError(Exception):
    """Base exception for all RedisVL errors."""

    pass


class RedisModuleVersionError(RedisVLError):
    """Error raised when required Redis modules are missing or have incompatible versions."""

    pass


class RedisSearchError(RedisVLError):
    """Error raised for Redis Search specific operations."""

    pass


class SchemaValidationError(RedisVLError):
    """Error when validating data against a schema."""

    def __init__(self, message, index=None):
        if index is not None:
            message = f"Validation failed for object at index {index}: {message}"
        super().__init__(message)
