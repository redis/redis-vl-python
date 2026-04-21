import asyncio
from enum import Enum
from typing import Any

from pydantic import ValidationError
from redis.exceptions import RedisError

from redisvl.exceptions import RedisSearchError


class MCPErrorCode(str, Enum):
    """Stable internal error codes exposed by the MCP framework."""

    INVALID_REQUEST = "invalid_request"
    INVALID_FILTER = "invalid_filter"
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

    if isinstance(exc, (ValidationError, ValueError, FileNotFoundError)):
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

    return RedisVLMCPError(
        str(exc),
        code=MCPErrorCode.INTERNAL_ERROR,
        retryable=False,
    )
