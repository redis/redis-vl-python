import asyncio
from enum import Enum
from typing import Any

from pydantic import ValidationError
from redis.exceptions import RedisError

from redisvl.exceptions import RedisSearchError, RedisVLError, SchemaValidationError


class MCPErrorCode(str, Enum):
    """Stable internal error codes exposed by the MCP framework."""

    INVALID_REQUEST = "invalid_request"
    INVALID_FILTER = "invalid_filter"
    FORBIDDEN = "forbidden"
    DEPENDENCY_MISSING = "dependency_missing"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    INTERNAL_ERROR = "internal_error"


class RedisVLMCPError(Exception):
    """Framework-facing exception carrying a stable MCP error contract."""

    def __init__(
        self,
        message: str,
        *,
        code: MCPErrorCode,
        retryable: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.metadata = metadata or {}


def map_exception(exc: Exception) -> RedisVLMCPError:
    """Map framework exceptions into deterministic MCP-facing exceptions."""
    if isinstance(exc, RedisVLMCPError):
        return exc

    if isinstance(
        exc, (ValidationError, SchemaValidationError, ValueError, FileNotFoundError)
    ):
        return RedisVLMCPError(
            str(exc),
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )

    if isinstance(exc, ImportError):
        return RedisVLMCPError(
            str(exc),
            code=MCPErrorCode.DEPENDENCY_MISSING,
            retryable=False,
        )

    if isinstance(
        exc, (TimeoutError, asyncio.TimeoutError, RedisSearchError, RedisError)
    ):
        return RedisVLMCPError(
            str(exc),
            code=MCPErrorCode.BACKEND_UNAVAILABLE,
            retryable=True,
        )

    # Index methods such as ``load`` wrap whatever went wrong in a plain
    # ``RedisVLError``. Classify by the cause, so a lost connection stays retryable,
    # but keep the wrapper's message, which says which operation failed.
    if isinstance(exc, RedisVLError) and isinstance(exc.__cause__, Exception):
        cause = map_exception(exc.__cause__)
        return RedisVLMCPError(str(exc), code=cause.code, retryable=cause.retryable)

    return RedisVLMCPError(
        str(exc),
        code=MCPErrorCode.INTERNAL_ERROR,
        retryable=False,
    )
