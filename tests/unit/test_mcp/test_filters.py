import pytest
from conftest import _schema

from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.filters import parse_filter
from redisvl.query.filter import FilterExpression, Text


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


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        # Each metacharacter that gives a `like` pattern its meaning. Escaping any
        # of these does not fail loudly -- it turns the pattern into a literal that
        # matches nothing -- so each is pinned separately.
        ("quant*", "@content:(quant*)"),
        ("qu?nt", "@content:(qu?nt)"),
        # A space is the implicit AND between terms, not padding.
        ("foo bar", "@content:(foo bar)"),
        # `%` is fuzzy matching.
        ("%foo%", "@content:(%foo%)"),
    ],
)
def test_parse_filter_preserves_like_pattern_metacharacters(value, rendered):
    parsed = parse_filter({"field": "content", "op": "like", "value": value}, _schema())

    assert _render_filter(parsed) == rendered


def test_parse_filter_like_matches_library_semantics_for_patterns():
    """MCP `like` must not diverge from `Text.__mod__` for ordinary patterns."""
    for value in ["quant*", "foo bar", "%foo%", "qu?nt"]:
        mcp = _render_filter(
            parse_filter({"field": "content", "op": "like", "value": value}, _schema())
        )
        assert mcp == str(Text("content") % value), f"diverged for {value!r}"


def test_parse_filter_like_still_escapes_clause_delimiters():
    """Preserving pattern metacharacters must not cost containment.

    Containment comes from the delimiters, not from `%`/space: with `(` and `)`
    escaped the value cannot close its own `@field:(...)`, so a `|` it carries
    stays scoped to this field. `|` is deliberately checked because no escaper in
    RedisVL touches it.
    """
    parsed = parse_filter(
        {"field": "content", "op": "like", "value": "alpha) | (-@category:{secret}"},
        _schema(),
    )
    rendered = _render_filter(parsed)

    # The payload's own parens are escaped...
    assert "\\)" in rendered and "\\(" in rendered
    # ...so stripping escape pairs leaves a balanced skeleton: nothing the payload
    # contributed can terminate the clause early.
    skeleton = _strip_escapes(rendered)
    assert skeleton.count("(") == skeleton.count(")")
    # The `|` survives unescaped but is inside the field clause, so it unions
    # within `content` rather than splitting the whole expression.
    assert skeleton.index("|") > skeleton.index("(")
    assert skeleton.index("|") < skeleton.rindex(")")
