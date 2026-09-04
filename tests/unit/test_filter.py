import calendar
import operator
import time as time_module
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pytest

from redisvl.query.filter import (
    FilterExpression,
    FilterOperator,
    Geo,
    GeoRadius,
    Num,
    Tag,
    Text,
    Timestamp,
)


# Test cases for various scenarios of tag usage, combinations, and their string representations.
@pytest.mark.parametrize(
    "operation,tags,expected",
    [
        # Testing single tags
        ("==", "simpletag", "@tag_field:{simpletag}"),
        (
            "==",
            "tag with space",
            "@tag_field:{tag\\ with\\ space}",
        ),  # Escaping spaces within quotes
        (
            "==",
            "special$char",
            "@tag_field:{special\\$char}",
        ),  # Escaping a special character
        ("!=", "negated", "(-@tag_field:{negated})"),
        # Testing multiple tags
        ("==", ["tag1", "tag2"], "@tag_field:{tag1|tag2}"),
        (
            "==",
            ["alpha", "beta with space", "gamma$special"],
            "@tag_field:{alpha|beta\\ with\\ space|gamma\\$special}",
        ),  # Multiple tags with spaces and special chars
        ("!=", ["tagA", "tagB"], "(-@tag_field:{tagA|tagB})"),
        # Complex tag scenarios with special characters
        ("==", "weird:tag", "@tag_field:{weird\\:tag}"),  # Tags with colon
        ("==", "tag&another", "@tag_field:{tag\\&another}"),  # Tags with ampersand
        # Escaping various special characters within tags
        ("==", "tag/with/slashes", "@tag_field:{tag\\/with\\/slashes}"),
        (
            "==",
            ["hyphen-tag", "under_score", "dot.tag"],
            "@tag_field:{hyphen\\-tag|under_score|dot\\.tag}",
        ),
        # ...additional unique cases as desired...
    ],
)
def test_tag_filter_varied(operation, tags, expected):
    if operation == "==":
        tf = Tag("tag_field") == tags
    elif operation == "!=":
        tf = Tag("tag_field") != tags
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    # Verify the string representation matches the expected Redis Search query part
    assert str(tf) == expected


@pytest.mark.parametrize(
    "pattern,expected",
    [
        # Basic prefix wildcard
        ("tech*", "@tag_field:{tech*}"),
        # Multiple patterns via list
        (["tech*", "soft*"], "@tag_field:{tech*|soft*}"),
        # Wildcard with special chars that still get escaped
        ("tech*-pro", "@tag_field:{tech*\\-pro}"),
        # Prefix with space (space escaped, wildcard preserved)
        ("hello w*", "@tag_field:{hello\\ w*}"),
        # Multiple wildcards in same pattern
        ("*test*", "@tag_field:{*test*}"),
        # Empty pattern returns wildcard match-all
        ("", "*"),
        ([], "*"),
        (None, "*"),
        # Pattern with special characters
        ("cat$*", "@tag_field:{cat\\$*}"),
    ],
    ids=[
        "prefix_wildcard",
        "multiple_patterns",
        "wildcard_with_special_char",
        "prefix_with_space",
        "multiple_wildcards",
        "empty_string",
        "empty_list",
        "none",
        "special_char_with_wildcard",
    ],
)
def test_tag_wildcard_filter(pattern, expected):
    """Test Tag % operator for wildcard/prefix matching."""
    tf = Tag("tag_field") % pattern
    assert str(tf) == expected


def test_tag_wildcard_preserves_asterisk():
    """Verify that * is not escaped when using % operator."""
    # With == operator, * should be escaped
    tf_eq = Tag("tag_field") == "tech*"
    assert str(tf_eq) == "@tag_field:{tech\\*}"

    # With % operator, * should NOT be escaped
    tf_like = Tag("tag_field") % "tech*"
    assert str(tf_like) == "@tag_field:{tech*}"


def test_tag_equality_escapes_pipe_but_list_still_unions():
    """A `|` inside one tag value is literal; a list of values is still a union."""
    # Unescaped, this value would widen its clause into a union across tenants.
    assert str(Tag("tenant_id") == "acme|victim") == "@tenant_id:{acme\\|victim}"
    assert str(Tag("tenant_id") != "acme|victim") == "(-@tenant_id:{acme\\|victim})"

    # Values are escaped before being joined, so the list form is unaffected.
    assert str(Tag("tenant_id") == ["acme", "victim"]) == "@tenant_id:{acme|victim}"

    # The % operator documents `|` as a union between wildcard patterns.
    assert str(Tag("category") % "elec*|*soft") == "@category:{elec*|*soft}"


def test_text_query_pipe_is_still_a_union():
    """Escaping `|` is scoped to tags; a text query joins its own terms with it."""
    from redisvl.query import TextQuery

    query = TextQuery(text="engineer|doctor", text_field_name="job")
    assert "@job:(engineer|doctor)" in str(query)


def test_tag_wildcard_combined_with_exact_match():
    """Test combining wildcard and exact match Tag filters in the same query."""
    # Create filters with different operators
    exact_match = Tag("brand") == "nike"
    wildcard_match = Tag("category") % "tech*"

    # Verify individual filters work correctly
    assert str(exact_match) == "@brand:{nike}"
    assert str(wildcard_match) == "@category:{tech*}"

    # Combine with AND - wildcard should be preserved, exact match should not have *
    combined_and = exact_match & wildcard_match
    assert str(combined_and) == "(@brand:{nike} @category:{tech*})"

    # Combine with OR
    combined_or = exact_match | wildcard_match
    assert str(combined_or) == "(@brand:{nike} | @category:{tech*})"

    # More complex: mix of exact, wildcard, and exact with * in value
    exact_with_asterisk = Tag("status") == "active*"  # * should be escaped
    complex_filter = exact_match & wildcard_match & exact_with_asterisk
    assert "@brand:{nike}" in str(complex_filter)
    assert "@category:{tech*}" in str(complex_filter)  # wildcard preserved
    assert "@status:{active\\*}" in str(complex_filter)  # asterisk escaped


@pytest.mark.parametrize(
    "operation, value, expected",
    [
        ("__eq__", None, "*"),
        ("__eq__", [], "*"),
        ("__eq__", "", "*"),
        ("__eq__", [None], "*"),
        ("__eq__", [None, "tag"], "@tag_field:{tag}"),
        # Tag.__str__ short-circuits to "*" before consulting OPERATOR_MAP, so
        # one falsy row covers every falsy value on the negated operator too.
        ("__ne__", None, "*"),
        ("__ne__", [None, "tag"], "(-@tag_field:{tag})"),
    ],
    ids=[
        "none",
        "empty_list",
        "empty_string",
        "list_with_none",
        "list_with_none_and_tag",
        "ne_none",
        "ne_list_with_none_and_tag",
    ],
)
def test_nullable(operation, value, expected):
    tag = Tag("tag_field")
    assert str(getattr(tag, operation)(value)) == expected


@pytest.mark.parametrize(
    "operation, value, expected",
    [
        ("__eq__", 5, "@numeric_field:[5 5]"),
        ("__ne__", 5, "(-@numeric_field:[5 5])"),
        ("__gt__", 5, "@numeric_field:[(5 +inf]"),
        ("__ge__", 5, "@numeric_field:[5 +inf]"),
        ("__lt__", 5, "@numeric_field:[-inf (5]"),
        ("__le__", 5, "@numeric_field:[-inf 5]"),
        ("__le__", None, "*"),
        ("__eq__", None, "*"),
        ("__ne__", None, "*"),
    ],
    ids=["eq", "ne", "gt", "ge", "lt", "le", "le_none", "eq_none", "ne_none"],
)
def test_numeric_filter(operation, value, expected):
    nf = Num("numeric_field")
    assert str(getattr(nf, operation)(value)) == expected


@pytest.mark.parametrize(
    "inclusive, expected",
    [
        ("both", "@numeric_field:[2 5]"),
        ("neither", "@numeric_field:[(2 (5]"),
        ("left", "@numeric_field:[2 (5]"),
        ("right", "@numeric_field:[(2 5]"),
    ],
)
def test_numeric_between(inclusive, expected):
    assert str(Num("numeric_field").between(2, 5, inclusive=inclusive)) == expected


class _StrOverridingInt(int):
    """A numeric type that satisfies isinstance and injects when formatted."""

    def __str__(self) -> str:
        return "5] | @secret:{leaked} @numeric_field:[-inf +inf"


class _StrOverridingFloat(float):
    """The same, on the float branch of the coercion."""

    def __str__(self) -> str:
        return "5.5] | @secret:{leaked} @numeric_field:[-inf +inf"


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        (np.int64(5), "@numeric_field:[2 5]"),
        (_StrOverridingInt(5), "@numeric_field:[2 5]"),
        (_StrOverridingFloat(5.5), "@numeric_field:[2 5.5]"),
    ],
    ids=["numpy_scalar_is_real_but_not_int", "subclass_int_str", "subclass_float_str"],
)
def test_numeric_between_coerces_a_real_endpoint(endpoint, expected):
    """The first row proves `numbers.Real` is wide enough, the rest why it is not enough.

    numpy integers are not `int` subclasses, so a concrete `(int, float)` check
    would reject them. And the type check alone lets a subclass through, since
    the endpoint is formatted into the query string -- coercion is the guard, on
    both the Integral and the Real branch.
    """
    assert str(Num("numeric_field").between(2, endpoint)) == expected


def test_numeric_comparison_coerces_a_formattable_value():
    """Coercion covers every operator, not only between()."""
    assert (
        str(Num("numeric_field") <= _StrOverridingInt(5)) == "@numeric_field:[-inf 5]"
    )


@pytest.mark.parametrize(
    "call, expected_error",
    [
        (
            lambda: Num("numeric_field").between(
                4, "5] | @secret:{leaked} @r:[-inf +inf"
            ),
            TypeError,
        ),
        # `@field:[nan ...]` is a query RediSearch refuses.
        (lambda: Num("numeric_field") == float("nan"), ValueError),
        # `tuple` outlived the BETWEEN branch that unpacked it, and nothing ever
        # validated the elements.
        (lambda: Num("numeric_field") == ("5] | @secret:{leaked}", 1), TypeError),
    ],
    ids=["between_string_endpoint", "nan", "tuple"],
)
def test_numeric_refuses_an_unrenderable_value(call, expected_error):
    with pytest.raises(expected_error):
        call()


@pytest.mark.parametrize(
    "operation, value, expected",
    [
        ("__eq__", "text", '@text_field:("text")'),
        ("__ne__", "text", '(-@text_field:"text")'),
        ("__eq__", "", "*"),
        ("__ne__", "", "*"),
        ("__eq__", None, "*"),
        ("__ne__", None, "*"),
        ("__mod__", "text", "@text_field:(text)"),
        ("__mod__", "tex*", "@text_field:(tex*)"),
        ("__mod__", "%text%", "@text_field:(%text%)"),
        ("__mod__", "", "*"),
        ("__mod__", None, "*"),
        # The quote is the only character that can terminate a quoted value, so
        # it goes and everything it carried stays inside the phrase.
        (
            "__eq__",
            'hello") | (@secret:{leaked} @v:"nothing',
            '@text_field:("hello ) | (@secret:{leaked} @v: nothing")',
        ),
        (
            "__ne__",
            'hello") | (@secret:{leaked} @v:"nothing',
            '(-@text_field:"hello ) | (@secret:{leaked} @v: nothing")',
        ),
        # A value that neutralizes down to whitespace still renders as a phrase
        # rather than collapsing to `*`. An empty phrase matches no document, so
        # `==` fails closed; collapsing would invert that to matching every one,
        # and `format_expression` drops a `*` operand, so it would also delete
        # this clause from a surrounding AND.
        ("__eq__", '"', '@text_field:(" ")'),
        ("__ne__", '""', '(-@text_field:"  ")'),
        ("__eq__", "   ", '@text_field:("   ")'),
        # Everything else survives untouched, the backslash included: escaping is
        # symmetric, so a document written with one indexes the joined term and
        # only an equally escaped query finds it. Replacing any of these would ask
        # for a term no document stored -- a silent zero-hit failure. These rows
        # fail if a single extra character joins the pattern.
        ("__eq__", "trailing\\", '@text_field:("trailing\\")'),
        (
            "__eq__",
            "e-mail, 50% off (C++) @ user@x.com O'Brien 3.5",
            '@text_field:("e-mail, 50% off (C++) @ user@x.com O\'Brien 3.5")',
        ),
        # `%` is the raw pattern operator by design, so its value is untouched.
        (
            "__mod__",
            'hello") | (@secret:{leaked}',
            '@text_field:(hello") | (@secret:{leaked})',
        ),
    ],
    ids=[
        "eq",
        "ne",
        "eq-empty",
        "ne-empty",
        "eq-none",
        "ne-none",
        "like",
        "like_wildcard",
        "like_full",
        "like_empty",
        "like_none",
        "eq_quote_neutralized",
        "ne_quote_neutralized",
        "eq_quotes_only",
        "ne_quotes_only",
        "eq_whitespace_only",
        "eq_backslash_untouched",
        "eq_punctuation_preserved",
        "like_raw_by_design",
    ],
)
def test_text_filter(operation, value, expected):
    txt_f = getattr(Text("text_field"), operation)(value)
    assert str(txt_f) == expected


def test_text_subclass_inherits_containment_for_a_new_quoted_operator():
    """Listing only the raw operator means a new one is contained by default.

    A subclass adding a quoted template is the case a set of *quoted* operators
    would miss, because it would be computed from the base class's own map.
    """

    class Phrase(Text):
        OPERATOR_MAP = {
            **Text.OPERATOR_MAP,
            FilterOperator.IN: '@%s:("%s")=>{$slop: 2}',
        }
        OPERATORS = {**Text.OPERATORS, FilterOperator.IN: "within"}

        def within(self, other):
            self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.IN)
            return FilterExpression(str(self))

    rendered = str(Phrase("t").within('a") | (@secret:{leaked}'))

    assert rendered == '@t:("a ) | (@secret:{leaked}")=>{$slop: 2}'


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("__eq__", "@geo_field:[1.0 2.0 3 km]"),
        ("__ne__", "(-@geo_field:[1.0 2.0 3 km])"),
    ],
    ids=["eq", "ne"],
)
def test_geo_filter(operation, expected):
    geo_radius = GeoRadius(1.0, 2.0, 3, "km")
    geo_f = Geo("geo_field")
    assert str(getattr(geo_f, operation)(geo_radius)) == expected


def test_filters_combination():
    tf1 = Tag("tag_field") == ["tag1", "tag2"]
    tf2 = Tag("tag_field") == "tag3"
    combined = tf1 & tf2
    assert str(combined) == "(@tag_field:{tag1|tag2} @tag_field:{tag3})"

    combined = tf1 | tf2
    assert str(combined) == "(@tag_field:{tag1|tag2} | @tag_field:{tag3})"

    tf1 = Tag("tag_field") == []
    assert str(tf1) == "*"
    assert str(tf1 & tf2) == str(tf2)
    assert str(tf1 | tf2) == str(tf2)

    # test combining filters with None values and empty strings
    tf1 = Tag("tag_field") == None
    tf2 = Tag("tag_field") == ""
    assert str(tf1 & tf2) == "*"

    tf1 = Tag("tag_field") == None
    tf2 = Tag("tag_field") == "tag"
    assert str(tf1 & tf2) == str(tf2)

    tf1 = Tag("tag_field") == None
    tf2 = Tag("tag_field") == ["tag1", "tag2"]
    assert str(tf1 & tf2) == str(tf2)

    tf1 = Tag("tag_field") == None
    tf2 = Tag("tag_field") != None
    assert str(tf1 & tf2) == "*"

    tf1 = Tag("tag_field") == ""
    tf2 = Tag("tag_field") == "tag"
    tf3 = Tag("tag_field") == ["tag1", "tag2"]
    assert str(tf1 & tf2 & tf3) == str(tf2 & tf3)

    # test none filters for Tag Num Text and Geo
    tf1 = Tag("tag_field") == None
    tf2 = Num("num_field") == None
    tf3 = Text("text_field") == None
    tf4 = Geo("geo_field") == None
    assert str(tf1 & tf2 & tf3 & tf4) == "*"

    tf1 = Tag("tag_field") != None
    tf2 = Num("num_field") != None
    tf3 = Text("text_field") != None
    tf4 = Geo("geo_field") != None
    assert str(tf1 & tf2 & tf3 & tf4) == "*"

    # test combinations of real and None filters across tag
    # text and geo filters
    tf1 = Tag("tag_field") == "tag"
    tf2 = Num("num_field") == None
    tf3 = Text("text_field") == None
    tf4 = Geo("geo_field") == GeoRadius(1.0, 2.0, 3, "km")
    assert str(tf1 & tf2 & tf3 & tf4) == str(tf1 & tf4)


def test_num_filter_zero():
    num_filter = Num("chunk_number") == 0
    assert (
        str(num_filter) == "@chunk_number:[0 0]"
    ), "Num filter should handle zero correctly"


def test_timestamp_datetime():
    """Test Timestamp filter with datetime objects."""
    # Test with timezone-aware datetime
    dt = datetime(2023, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
    ts = Timestamp("created_at") == dt
    # Expected timestamp would be the Unix timestamp for the datetime
    expected_ts = dt.timestamp()
    assert str(ts) == f"@created_at:[{expected_ts} {expected_ts}]"

    # Test with timezone-naive datetime (should convert to UTC)
    dt = datetime(2023, 3, 17, 14, 30, 0)
    ts = Timestamp("created_at") == dt
    expected_ts = dt.replace(tzinfo=timezone.utc).timestamp()
    assert str(ts) == f"@created_at:[{expected_ts} {expected_ts}]"


def test_timestamp_date():
    """Test Timestamp filter with date objects (should match full day in UTC)."""
    d = date(2023, 3, 17)
    ts = Timestamp("created_at") == d

    expected_ts_start = datetime.combine(d, time.min, tzinfo=timezone.utc).timestamp()
    expected_ts_end = datetime.combine(d, time.max, tzinfo=timezone.utc).timestamp()

    assert str(ts) == f"@created_at:[{expected_ts_start} {expected_ts_end}]"

    # Independent ground truth: 2023-03-17T00:00:00Z / T23:59:59.999999Z.
    # Hard-coded so this test cannot drift along with the implementation.
    assert str(ts) == "@created_at:[1679011200.0 1679097599.999999]"


def test_timestamp_not_equal_date():
    """Test Timestamp != with date objects (should exclude the full day in UTC)."""
    d = date(2023, 3, 17)
    ts = Timestamp("created_at") != d

    expected_ts_start = datetime.combine(d, time.min, tzinfo=timezone.utc).timestamp()
    expected_ts_end = datetime.combine(d, time.max, tzinfo=timezone.utc).timestamp()

    assert str(ts) == f"(-@created_at:[{expected_ts_start} {expected_ts_end}])"

    # != must be the exact negation of == for the same date
    eq = Timestamp("created_at") == d
    assert str(ts) == f"(-{eq!s})"

    # A date-only ISO string takes the same path as a date object
    assert str(Timestamp("created_at") != "2023-03-17") == str(ts)


@pytest.mark.skipif(not hasattr(time_module, "tzset"), reason="tzset() is POSIX-only")
@pytest.mark.parametrize(
    "d",
    [
        date(2023, 3, 17),  # ordinary date
        date(2023, 3, 12),  # US DST spring-forward
        date(2023, 11, 5),  # US DST fall-back
        date(1970, 1, 1),  # Unix epoch
        date(2038, 1, 20),  # past the signed 32-bit rollover
    ],
)
def test_timestamp_date_bounds_are_utc_regardless_of_local_timezone(d, monkeypatch):
    """Date filters resolve to the UTC day, not the host's local day.

    CI runs in UTC, where the correct and the local-time conversions agree, so
    this is the only test that would catch a regression to local-day bounds.
    """
    # calendar.timegm is a timezone-independent oracle that shares no code with
    # the conversion path under test.
    start = float(calendar.timegm(datetime.combine(d, time.min).timetuple()))
    end = float(calendar.timegm(datetime.combine(d, time.max).timetuple())) + 0.999999
    expected_eq = f"@created_at:[{start} {end}]"

    try:
        # Includes zones with non-hour offsets (+05:45, +12:45/+13:45), which an
        # hour-granularity mistake would pass.
        for tz in (
            "UTC",
            "America/New_York",
            "Asia/Tokyo",
            "Asia/Kathmandu",
            "Pacific/Chatham",
        ):
            monkeypatch.setenv("TZ", tz)
            time_module.tzset()

            assert str(Timestamp("created_at") == d) == expected_eq
            assert str(Timestamp("created_at") != d) == f"(-{expected_eq})"
            # Date-only ISO strings take the same branch
            assert str(Timestamp("created_at") == d.isoformat()) == expected_eq
    finally:
        # Restore TZ and re-read it, so later tests see the original zone
        monkeypatch.undo()
        time_module.tzset()


def test_timestamp_iso_string():
    """Test Timestamp filter with ISO format strings."""
    # Date-only ISO string
    ts = Timestamp("created_at") == "2023-03-17"
    d = date(2023, 3, 17)
    expected_ts_start = datetime.combine(d, time.min, tzinfo=timezone.utc).timestamp()
    expected_ts_end = datetime.combine(d, time.max, tzinfo=timezone.utc).timestamp()
    assert str(ts) == f"@created_at:[{expected_ts_start} {expected_ts_end}]"

    # Full ISO datetime string
    dt_str = "2023-03-17T14:30:00+00:00"
    ts = Timestamp("created_at") == dt_str
    dt = datetime.fromisoformat(dt_str)
    expected_ts = dt.timestamp()
    assert str(ts) == f"@created_at:[{expected_ts} {expected_ts}]"


def test_timestamp_unix():
    """Test Timestamp filter with Unix timestamps."""
    # Integer timestamp
    ts = Timestamp("created_at") == 1679062200  # 2023-03-17T14:30:00+00:00
    assert str(ts) == "@created_at:[1679062200.0 1679062200.0]"

    # Float timestamp
    ts = Timestamp("created_at") == 1679062200.5
    assert str(ts) == "@created_at:[1679062200.5 1679062200.5]"


def test_timestamp_operators():
    """Test all comparison operators for Timestamp filter."""
    dt = datetime(2023, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
    ts_value = dt.timestamp()

    # Equal
    ts = Timestamp("created_at") == dt
    assert str(ts) == f"@created_at:[{ts_value} {ts_value}]"

    # Not equal
    ts = Timestamp("created_at") != dt
    assert str(ts) == f"(-@created_at:[{ts_value} {ts_value}])"

    # Greater than
    ts = Timestamp("created_at") > dt
    assert str(ts) == f"@created_at:[({ts_value} +inf]"

    # Less than
    ts = Timestamp("created_at") < dt
    assert str(ts) == f"@created_at:[-inf ({ts_value}]"

    # Greater than or equal
    ts = Timestamp("created_at") >= dt
    assert str(ts) == f"@created_at:[{ts_value} +inf]"

    # Less than or equal
    ts = Timestamp("created_at") <= dt
    assert str(ts) == f"@created_at:[-inf {ts_value}]"

    td = timedelta(days=5)
    dt2 = dt + td
    ts_value2 = dt2.timestamp()

    ts = Timestamp("created_at").between(dt, dt2)
    assert str(ts) == f"@created_at:[{ts_value} {ts_value2}]"

    ts = Timestamp("created_at").between(dt, dt2, inclusive="neither")
    assert str(ts) == f"@created_at:[({ts_value} ({ts_value2}]"

    ts = Timestamp("created_at").between(dt, dt2, inclusive="left")
    assert str(ts) == f"@created_at:[{ts_value} ({ts_value2}]"

    ts = Timestamp("created_at").between(dt, dt2, inclusive="right")
    assert str(ts) == f"@created_at:[({ts_value} {ts_value2}]"


# The four comparison operators, keyed by the symbol used in failure messages.
TIMESTAMP_COMPARISONS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


def test_timestamp_comparison_operators_with_date():
    """A bare date bounds the whole UTC day, so > and <= sit at its end.

    Hard-coded against 2023-03-17T00:00:00Z / T23:59:59.999999Z so the test
    cannot drift along with the implementation.
    """
    expected = {
        ">": "@created_at:[(1679097599.999999 +inf]",
        "<": "@created_at:[-inf (1679011200.0]",
        ">=": "@created_at:[1679011200.0 +inf]",
        "<=": "@created_at:[-inf 1679097599.999999]",
    }

    for symbol, op in TIMESTAMP_COMPARISONS.items():
        # Bare dates and date-only ISO strings take the same branch
        for value in (date(2023, 3, 17), "2023-03-17"):
            assert (
                str(op(Timestamp("created_at"), value)) == expected[symbol]
            ), f"{symbol} {value!r}"


def test_timestamp_between_date_only_string_matches_date_object():
    """A date-only string endpoint spans the same whole day a date object does.

    `end` is the interesting one: it goes through `end_date=True`, which only
    reaches the end of the day if the string was coerced to a date first.
    """
    by_date = str(Timestamp("created_at").between(date(2023, 3, 1), date(2023, 3, 17)))
    by_string = str(Timestamp("created_at").between("2023-03-01", "2023-03-17"))

    assert by_string == by_date
    assert by_date == "@created_at:[1677628800.0 1679097599.999999]"


@pytest.mark.parametrize("symbol", list(TIMESTAMP_COMPARISONS))
@pytest.mark.parametrize("value", ["2023-02-30", "2023-13-45", "0000-00-00"])
def test_timestamp_date_shaped_but_invalid_string(symbol, value):
    """Every operator rejects a date-shaped non-date the same way.

    `_is_date_only` matches on the digit pattern alone, so these reach the
    coercion and have to fall through to one shared error message.
    """
    with pytest.raises(ValueError, match=f"must be in ISO format: {value}"):
        TIMESTAMP_COMPARISONS[symbol](Timestamp("created_at"), value)


@pytest.mark.skipif(not hasattr(time_module, "tzset"), reason="tzset() is POSIX-only")
@pytest.mark.parametrize(
    "d",
    [
        date(2023, 3, 17),  # ordinary date
        date(2023, 3, 12),  # US DST spring-forward
        date(2023, 11, 5),  # US DST fall-back
        date(1970, 1, 1),  # Unix epoch
        date(2038, 1, 20),  # past the signed 32-bit rollover
    ],
)
def test_timestamp_comparison_date_bounds_are_utc_days(d, monkeypatch):
    """Comparison operators bound the UTC day, whatever the host's local zone.

    CI runs in UTC, where the correct and the local-time conversions agree, so
    this is the only test that would catch a regression to local-day bounds.
    """
    # calendar.timegm is a timezone-independent oracle that shares no code with
    # the conversion path under test. Caveat for anyone extending the list above:
    # adding the microseconds back on is float-exact only for non-negative
    # timestamps, so a pre-epoch date such as 1969-12-31 fails here spuriously
    # (oracle -1.0000000000287557e-06 vs. a correct -1e-06). Assert those against
    # datetime.combine(d, time.max, tzinfo=timezone.utc).timestamp() instead.
    start = float(calendar.timegm(datetime.combine(d, time.min).timetuple()))
    end = float(calendar.timegm(datetime.combine(d, time.max).timetuple())) + 0.999999

    # > and <= exclude/include the whole day, so they anchor to its end; >= and <
    # anchor to its start.
    expected = {
        ">": f"@created_at:[({end} +inf]",
        "<": f"@created_at:[-inf ({start}]",
        ">=": f"@created_at:[{start} +inf]",
        "<=": f"@created_at:[-inf {end}]",
    }

    try:
        # Includes zones with non-hour offsets (+05:45, +12:45/+13:45), which an
        # hour-granularity mistake would pass.
        for tz in (
            "UTC",
            "America/New_York",
            "Asia/Tokyo",
            "Asia/Kathmandu",
            "Pacific/Chatham",
        ):
            monkeypatch.setenv("TZ", tz)
            time_module.tzset()

            for symbol, op in TIMESTAMP_COMPARISONS.items():
                for value in (d, d.isoformat()):
                    assert (
                        str(op(Timestamp("created_at"), value)) == expected[symbol]
                    ), f"{symbol} {value!r} in TZ={tz}"
    finally:
        # Restore TZ and re-read it, so later tests see the original zone
        monkeypatch.undo()
        time_module.tzset()


def test_timestamp_between():
    """Test the between method for date ranges."""
    start = datetime(2023, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc)

    ts = Timestamp("created_at").between(start, end)

    start_ts = start.timestamp()
    end_ts = end.timestamp()

    assert str(ts) == f"@created_at:[{start_ts} {end_ts}]"

    # Test with dates (should expand to full days)
    start_date = date(2023, 3, 1)
    end_date = date(2023, 3, 31)

    ts = Timestamp("created_at").between(start_date, end_date)

    # Start should be beginning of day
    expected_start = datetime.combine(start_date, datetime.min.time())
    expected_start = expected_start.replace(tzinfo=timezone.utc)

    # End should be end of day
    expected_end = datetime.combine(end_date, datetime.max.time())
    expected_end = expected_end.replace(tzinfo=timezone.utc)

    expected_start_ts = expected_start.timestamp()
    expected_end_ts = expected_end.timestamp()

    assert str(ts) == f"@created_at:[{expected_start_ts} {expected_end_ts}]"


def test_timestamp_none():
    """Test handling of None values."""
    ts = Timestamp("created_at") == None
    assert str(ts) == "*"

    ts = Timestamp("created_at") != None
    assert str(ts) == "*"

    ts = Timestamp("created_at") > None
    assert str(ts) == "*"


def test_timestamp_invalid_input():
    """Test error handling for invalid inputs."""
    # Invalid ISO format
    with pytest.raises(ValueError):
        Timestamp("created_at") == "not-a-date"

    # Unsupported type
    with pytest.raises(TypeError):
        Timestamp("created_at") == object()


def test_timestamp_filter_combination():
    """Test combining timestamp filters with other filters."""
    ts = Timestamp("created_at") > datetime(2023, 3, 1)
    num = Num("age") > 30
    tag = Tag("status") == "active"

    combined = ts & num & tag

    # The exact string depends on the timestamp value, but we can check structure
    assert str(combined).startswith("((@created_at:")
    assert "@age:[(30 +inf]" in str(combined)
    assert "@status:{active}" in str(combined)


def test_is_missing_filter_methods():
    """Test the new is_missing() method for all filter types."""
    # Test all filter types
    tag_missing = Tag("brand").is_missing()
    text_missing = Text("title").is_missing()
    num_missing = Num("price").is_missing()
    geo_missing = Geo("location").is_missing()
    timestamp_missing = Timestamp("created_at").is_missing()

    # Check that they generate the correct query strings
    assert str(tag_missing) == "ismissing(@brand)"
    assert str(text_missing) == "ismissing(@title)"
    assert str(num_missing) == "ismissing(@price)"
    assert str(geo_missing) == "ismissing(@location)"
    assert str(timestamp_missing) == "ismissing(@created_at)"


def test_is_missing_filter_combinations():
    """Test combining is_missing filters with other filters."""
    # Test combining is_missing with regular filters
    missing_brand = Tag("brand").is_missing()
    has_price = Num("price") > 100
    has_tag = Tag("category") == "electronics"

    # Test AND combinations
    combined_and = missing_brand & has_price
    combined_str = str(combined_and)
    assert "ismissing(@brand)" in combined_str
    assert "@price:[(100 +inf]" in combined_str

    # Test OR combinations
    combined_or = missing_brand | has_tag
    combined_str = str(combined_or)
    assert "ismissing(@brand)" in combined_str
    assert "@category:{electronics}" in combined_str
    assert " | " in combined_str

    # Test complex combinations
    complex_filter = (missing_brand & has_price) | has_tag
    complex_str = str(complex_filter)
    assert "ismissing(@brand)" in complex_str
    assert "@price:[(100 +inf]" in complex_str
    assert "@category:{electronics}" in complex_str
