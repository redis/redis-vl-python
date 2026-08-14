import pytest
from conftest import _schema

from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.filters import parse_filter
from redisvl.mcp.tools.search import merge_locked_filter
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


# --------------------------------------------------------------------------
# Caller shapes merged under a locked filter
#
# The merge is the security boundary for custom tool profiles: whatever the
# caller sends must end up ANDed *under* the locked clause rather than beside or
# instead of it. The caller shapes below are enumerated because they reach `&` as
# different renderings -- a single clause, a parenthesized AND, a parenthesized
# OR chain, a negation, a missing-field check -- and only the compound ones could
# hoist themselves out of the AND.
#
# The locked side is a single shape rather than a second axis. Nothing in the
# merge inspects it: the guard is `_reject_escapable_filter(caller)`, which takes
# only the caller's rendering, so the locked shape cannot change whether a caller
# shape is accepted. A compound AND is the one kept because it is the locked
# shape most likely to be damaged by a careless merge -- it arrives already
# parenthesized, so a merge that concatenated instead of nesting would show here.
# --------------------------------------------------------------------------

_LOCKED_SHAPE = {
    "and": [
        {"field": "category", "op": "eq", "value": "resolved"},
        {"field": "rating", "op": "gte", "value": 3},
    ]
}

_CALLER_SHAPES = {
    "not": {"not": {"field": "category", "op": "eq", "value": "sports"}},
    "and": {
        "and": [
            {"field": "rating", "op": "gte", "value": 4},
            {"field": "content", "op": "like", "value": "jam*"},
        ]
    },
    # Text and numeric `in` expand to an OR chain, so they reach `&` already
    # parenthesized -- a different shape from the single-clause cases above.
    "text_in": {"field": "content", "op": "in", "value": ["jam", "jelly"]},
    "numeric_in": {"field": "rating", "op": "in", "value": [3, 5]},
    "exists": {"field": "content", "op": "exists"},
}


@pytest.mark.parametrize("caller_shape", sorted(_CALLER_SHAPES))
def test_merge_locked_filter_ands_every_caller_shape_under_the_locked_filter(
    caller_shape,
):
    locked = parse_filter(_LOCKED_SHAPE, _schema())
    caller = parse_filter(_CALLER_SHAPES[caller_shape], _schema())

    merged = merge_locked_filter(locked, caller)

    # Never a bare string: a string cannot be safely combined further, and a
    # profile that received one back would have lost the composability the lock
    # depends on.
    assert isinstance(merged, FilterExpression)
    # The merge is exactly the AND of the two sides as each renders on its own.
    # That pins both halves of the guarantee at once: the locked clause is
    # unchanged and still at the top level of the AND, and the caller's side is
    # reproduced verbatim beside it -- fully parenthesized when compound, so
    # neither its `|` nor its `-` can reach the top level and widen the scope.
    assert str(merged) == f"({locked} {caller})"


def test_merge_locked_filter_ands_a_multi_value_tag_in_caller_filter():
    locked = parse_filter(
        {"field": "category", "op": "eq", "value": "resolved"}, _schema()
    )
    caller = parse_filter(
        {"field": "category", "op": "in", "value": ["sports", "health"]}, _schema()
    )

    # Tag `in` is the one caller shape that collapses to a SINGLE clause,
    # `@category:{sports|health}`, instead of expanding to an OR chain the way
    # text and numeric `in` do. The `|` is scoped by the tag braces and cannot
    # widen past an enclosing AND, so this is a legitimate narrowing filter and
    # should merge like any other clause.
    merged = merge_locked_filter(locked, caller)

    assert isinstance(merged, FilterExpression)
    assert str(merged) == "(@category:{resolved} @category:{sports|health})"


# --------------------------------------------------------------------------
# The escape backstop -- what it refuses, not just what it lets through
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "rendered"),
    [
        ("stray close paren", "@content:(a) | (-@category:{secret})"),
        ("unclosed group", "@content:(a( | (-@category:{secret})"),
        ("stray close brace", "@category:{a} | @category:{secret}"),
        # A literal backslash does NOT escape what follows it, so this `|` is a
        # real union. Looking back one character would misread it as escaped.
        ("literal backslash then pipe", "@category:{a}\\\\|@category:{secret}"),
        # The `[...]` span is skipped so an exclusive bound's `(` is not counted
        # as a group. Skipping the span wholesale would also skip a `|` inside it,
        # letting a union hide behind brackets.
        ("pipe hidden inside a range span", "@rating:[4 | @category:{secret}]"),
    ],
)
def test_merge_locked_filter_refuses_a_caller_rendering_that_could_escape(
    label, rendered
):
    del label
    locked = parse_filter(
        {"field": "category", "op": "eq", "value": "science"}, _schema()
    )

    # Simulates a field type that renders a value unescaped. Text values are
    # escaped today, so this path is unreachable through the DSL -- which is
    # exactly why it needs an explicit test: without one, deleting the guard
    # leaves the whole suite green.
    with pytest.raises(RedisVLMCPError) as exc_info:
        merge_locked_filter(locked, FilterExpression(rendered))

    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER


@pytest.mark.parametrize(
    ("label", "caller_filter"),
    [
        ("exclusive lower bound", {"field": "rating", "op": "gt", "value": 4}),
        ("exclusive upper bound", {"field": "rating", "op": "lt", "value": 4}),
        ("trailing backslash", {"field": "category", "op": "eq", "value": "ns\\\\"}),
        (
            "windows path",
            {"field": "content", "op": "like", "value": "C:\\\\Users\\\\"},
        ),
        (
            "compound with exclusive bound",
            {
                "and": [
                    {"field": "category", "op": "eq", "value": "science"},
                    {"field": "rating", "op": "gt", "value": 4},
                ]
            },
        ),
    ],
)
def test_merge_locked_filter_accepts_legitimate_narrowing_shapes(label, caller_filter):
    del label
    locked = parse_filter(
        {"field": "category", "op": "eq", "value": "science"}, _schema()
    )
    caller = parse_filter(caller_filter, _schema())

    # An exclusive numeric bound renders `[(4 +inf]` and an escaped value can end
    # in a backslash. Neither is an escape, and refusing them would block the most
    # ordinary narrowing a caller can ask for.
    merged = merge_locked_filter(locked, caller)

    assert isinstance(merged, FilterExpression)
    assert "@category:{science}" in str(merged)


def test_merge_locked_filter_rejects_a_raw_string_caller_filter_against_a_lock():
    locked = parse_filter(_LOCKED_SHAPE, _schema())

    with pytest.raises(RedisVLMCPError) as exc_info:
        merge_locked_filter(locked, "@category:{open}")

    # A string has no safe composition with an expression: concatenating could
    # close the locked group and escape the scope entirely.
    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER
    assert exc_info.value.retryable is False


def test_merge_locked_filter_returns_each_side_unchanged_when_the_other_is_absent():
    caller = parse_filter({"field": "category", "op": "eq", "value": "open"}, _schema())
    locked = parse_filter(_LOCKED_SHAPE, _schema())

    # No lock means the built-in's own behavior, including the raw-string form
    # that a locked filter has to refuse.
    assert merge_locked_filter(None, caller) is caller
    assert merge_locked_filter(None, "@category:{open}") == "@category:{open}"
    # A lock with no caller filter applies on its own.
    assert merge_locked_filter(locked, None) is locked


@pytest.mark.parametrize(
    ("label", "rendered"),
    [
        # The reason the span is skipped at all: `(` here is an exclusive-bound
        # marker, not a group, so counting it would reject a legitimate range.
        ("exclusive lower bound", "@rating:[(5 +inf]"),
        ("exclusive upper bound", "@rating:[-inf (5]"),
        ("plain inclusive range", "@rating:[4 +inf]"),
    ],
)
def test_merge_locked_filter_still_accepts_legitimate_range_bounds(label, rendered):
    """Narrowing the range skip to parens must not start rejecting real ranges."""
    del label
    locked = parse_filter(
        {"field": "category", "op": "eq", "value": "science"}, _schema()
    )

    merged = merge_locked_filter(locked, FilterExpression(rendered))

    assert isinstance(merged, FilterExpression)
    assert "@category:{science}" in str(merged)
