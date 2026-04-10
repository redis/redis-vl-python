import pytest

from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.filters import parse_filter
from redisvl.query.filter import FilterExpression
from redisvl.schema import IndexSchema


def _schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "docs-index",
                "prefix": "doc",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "category", "type": "tag"},
                {"name": "rating", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


def _render_filter(value):
    if isinstance(value, FilterExpression):
        return str(value)
    return value


def test_parse_filter_passes_through_raw_string():
    raw = "@category:{science} @rating:[4 +inf]"

    parsed = parse_filter(raw, _schema())

    assert parsed == raw


def test_parse_filter_builds_atomic_expression():
    parsed = parse_filter(
        {"field": "category", "op": "eq", "value": "science"},
        _schema(),
    )

    assert isinstance(parsed, FilterExpression)
    assert str(parsed) == "@category:{science}"


def test_parse_filter_builds_nested_logical_expression():
    parsed = parse_filter(
        {
            "and": [
                {"field": "category", "op": "eq", "value": "science"},
                {
                    "or": [
                        {"field": "rating", "op": "gte", "value": 4.5},
                        {"field": "content", "op": "like", "value": "quant*"},
                    ]
                },
            ]
        },
        _schema(),
    )

    assert isinstance(parsed, FilterExpression)
    assert (
        str(parsed) == "(@category:{science} (@rating:[4.5 +inf] | @content:(quant*)))"
    )


def test_parse_filter_builds_not_expression():
    parsed = parse_filter(
        {
            "not": {"field": "category", "op": "eq", "value": "science"},
        },
        _schema(),
    )

    assert _render_filter(parsed) == "(-(@category:{science}))"


def test_parse_filter_builds_exists_expression():
    parsed = parse_filter(
        {"field": "content", "op": "exists"},
        _schema(),
    )

    assert _render_filter(parsed) == "(-ismissing(@content))"


def test_parse_filter_rejects_unknown_field():
    with pytest.raises(RedisVLMCPError) as exc_info:
        parse_filter({"field": "missing", "op": "eq", "value": "science"}, _schema())

    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER


def test_parse_filter_rejects_unknown_operator():
    with pytest.raises(RedisVLMCPError) as exc_info:
        parse_filter(
            {"field": "category", "op": "contains", "value": "science"}, _schema()
        )

    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER


def test_parse_filter_rejects_type_mismatch():
    with pytest.raises(RedisVLMCPError) as exc_info:
        parse_filter({"field": "rating", "op": "gte", "value": "high"}, _schema())

    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER


def test_parse_filter_rejects_empty_logical_array():
    with pytest.raises(RedisVLMCPError) as exc_info:
        parse_filter({"and": []}, _schema())

    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER


def test_parse_filter_rejects_malformed_payload():
    with pytest.raises(RedisVLMCPError) as exc_info:
        parse_filter({"field": "category", "value": "science"}, _schema())

    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER
