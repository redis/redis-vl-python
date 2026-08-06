import pytest
from conftest import _schema

from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.filters import parse_filter
from redisvl.query.filter import FilterExpression


def _render_filter(value):
    if isinstance(value, FilterExpression):
        return str(value)
    return value


def _strip_escapes(rendered: str) -> str:
    """Drop backslash-escaped pairs, leaving only unescaped query syntax."""
    out, index = [], 0
    while index < len(rendered):
        if rendered[index] == "\\":
            index += 2
            continue
        out.append(rendered[index])
        index += 1
    return "".join(out)


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


@pytest.mark.parametrize(
    ("op", "operand"),
    [
        ("eq", 'alpha") | (-@category:{secret}'),
        ("ne", 'alpha") | (-@category:{secret}'),
        ("like", "alpha) | (-@category:{secret}"),
        ("in", ['alpha") | (-@category:{secret}']),
    ],
)
def test_parse_filter_escapes_text_values_so_they_cannot_leave_their_clause(
    op, operand
):
    parsed = parse_filter({"field": "content", "op": op, "value": operand}, _schema())

    # Text operator templates interpolate the value into `@field:("...")` or
    # `@field:(...)`, so an unescaped quote or paren would close the clause and
    # let the rest of the value inject query syntax -- including a `|` that
    # escapes an enclosing AND. Stripping escape pairs leaves the structural
    # skeleton: the payload must contribute no syntax to it.
    skeleton = _strip_escapes(_render_filter(parsed))
    # The clause boundary survives: quotes and parens from the payload are
    # escaped, so its `|` stays scoped inside this field's own query instead of
    # splitting the whole expression into a union.
    assert skeleton.count("(") == skeleton.count(")")
    assert skeleton.count('"') % 2 == 0


def test_parse_filter_preserves_wildcards_in_text_like_patterns():
    parsed = parse_filter(
        {"field": "content", "op": "like", "value": "quant*"}, _schema()
    )

    # `like` is the pattern operator, so escaping must not neuter `*`.
    assert _render_filter(parsed) == "@content:(quant*)"
