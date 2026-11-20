import pytest
from pydantic import BaseModel, ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from redisvl.exceptions import RedisSearchError, RedisVLError, SchemaValidationError
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError, map_exception


class SampleModel(BaseModel):
    value: int


def test_validation_errors_map_to_invalid_request():
    try:
        SampleModel.model_validate({"value": "bad"})
    except ValidationError as exc:
        mapped = map_exception(exc)

    assert mapped.code == MCPErrorCode.INVALID_REQUEST
    assert mapped.retryable is False


def test_import_error_maps_to_dependency_missing():
    mapped = map_exception(ImportError("missing package"))

    assert mapped.code == MCPErrorCode.DEPENDENCY_MISSING
    assert mapped.retryable is False


def test_filter_error_is_preserved():
    original = RedisVLMCPError(
        "bad filter",
        code=MCPErrorCode.INVALID_FILTER,
        retryable=False,
    )

    mapped = map_exception(original)

    assert mapped is original


def test_redis_errors_map_to_backend_unavailable():
    mapped = map_exception(RedisSearchError("redis unavailable"))

    assert mapped.code == MCPErrorCode.BACKEND_UNAVAILABLE
    assert mapped.retryable is True


def test_redis_connection_errors_map_to_backend_unavailable():
    mapped = map_exception(RedisConnectionError("boom"))

    assert mapped.code == MCPErrorCode.BACKEND_UNAVAILABLE
    assert mapped.retryable is True


def test_timeout_error_maps_to_backend_unavailable():
    mapped = map_exception(TimeoutError("timed out"))

    assert mapped.code == MCPErrorCode.BACKEND_UNAVAILABLE
    assert mapped.retryable is True


@pytest.mark.parametrize(
    "cause", [RedisConnectionError("down"), RedisTimeoutError("slow")]
)
def test_wrapped_redis_errors_map_by_their_cause(cause):
    try:
        raise RedisVLError("Failed to load data") from cause
    except RedisVLError as exc:
        mapped = map_exception(exc)

    assert mapped.code == MCPErrorCode.BACKEND_UNAVAILABLE
    assert mapped.retryable is True
    assert str(mapped) == "Failed to load data"


def test_wrapped_redis_reply_errors_stay_non_retryable():
    try:
        raise RedisVLError("Failed to load data") from ResponseError("WRONGTYPE")
    except RedisVLError as exc:
        mapped = map_exception(exc)

    assert mapped.code == MCPErrorCode.INTERNAL_ERROR
    assert mapped.retryable is False


def test_schema_validation_error_maps_to_invalid_request():
    mapped = map_exception(SchemaValidationError("bad record", index=0))

    assert mapped.code == MCPErrorCode.INVALID_REQUEST
    assert mapped.retryable is False


def test_unknown_errors_map_to_internal_error():
    mapped = map_exception(RuntimeError("unexpected"))

    assert mapped.code == MCPErrorCode.INTERNAL_ERROR
    assert mapped.retryable is False


def test_existing_framework_error_is_preserved():
    original = RedisVLMCPError(
        "already mapped",
        code=MCPErrorCode.INVALID_REQUEST,
        retryable=False,
    )

    mapped = map_exception(original)

    assert mapped is original
