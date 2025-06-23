from datetime import date, datetime, time, timedelta, timezone

import pytest

from redisvl.query.filter import Geo, GeoRadius, Num, Tag, Text, Timestamp


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

    # Verify the string representation matches the expected RediSearch query part
    assert str(tf) == expected


def test_nullable():
    tag = Tag("tag_field") == None
    assert str(tag) == "*"

    tag = Tag("tag_field") != None
    assert str(tag) == "*"

    tag = Tag("tag_field") == []
    assert str(tag) == "*"

    tag = Tag("tag_field") != []
    assert str(tag) == "*"

    tag = Tag("tag_field") == ""
    assert str(tag) == "*"

    tag = Tag("tag_field") != ""
    assert str(tag) == "*"

    tag = Tag("tag_field") == [None]
    assert str(tag) == "*"

    tag = Tag("tag_field") == [None, "tag"]
    assert str(tag) == "@tag_field:{tag}"


def test_numeric_filter():
    nf = Num("numeric_field") == 5
    assert str(nf) == "@numeric_field:[5 5]"

    nf = Num("numeric_field") != 5
    assert str(nf) == "(-@numeric_field:[5 5])"

    nf = Num("numeric_field") > 5
    assert str(nf) == "@numeric_field:[(5 +inf]"

    nf = Num("numeric_field") >= 5
    assert str(nf) == "@numeric_field:[5 +inf]"

    nf = Num("numeric_field") < 5
    assert str(nf) == "@numeric_field:[-inf (5]"

    nf = Num("numeric_field") <= 5
    assert str(nf) == "@numeric_field:[-inf 5]"

    nf = Num("numeric_field") > 5.5
    assert str(nf) == "@numeric_field:[-inf 5.5]"

    nf = Num("numeric_field") <= None
    assert str(nf) == "*"

    nf = Num("numeric_field") == None
    assert str(nf) == "*"

    nf = Num("numeric_field") != None
    assert str(nf) == "*"

    nf = Num("numeric_field").between(2, 5)
    assert str(nf) == "@numeric_field:[2 5]"

    nf = Num("numeric_field").between(2, 5, inclusive="neither")
    assert str(nf) == "@numeric_field:[2 5]"

    nf = Num("numeric_field").between(2, 5, inclusive="left")
    assert str(nf) == "@numeric_field:[2 (5]"

    nf = Num("numeric_field").between(2, 5, inclusive="right")
    assert str(nf) == "@numeric_field:[(2 5]"


def test_text_filter():
    txt_f = Text("text_field") == "text"
    assert str(txt_f) == '@text_field:("text")'

    txt_f = Text("text_field") != "text"
    assert str(txt_f) == '(-@text_field:"text")'

    txt_f = Text("text_field") % "text"
    assert str(txt_f) == "@text_field:(text)"

    txt_f = Text("text_field") % "tex*"
    assert str(txt_f) == "@text_field:(tex*)"

    txt_f = Text("text_field") % "%text%"
    assert str(txt_f) == "@text_field:(%text%)"

    txt_f = Text("text_field") % ""
    assert str(txt_f) == "*"


def test_geo_filter():
    geo_f = Geo("geo_field") == GeoRadius(1.0, 2.0, 3, "km")
    assert str(geo_f) == "@geo_field:[1.0 2.0 3 km]"

    geo_f = Geo("geo_field") != GeoRadius(1.0, 2.0, 3, "km")
    assert str(geo_f) != "(-@geo_field:[1.0 2.0 3 m])"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "*"),
        ([], "*"),
        ("", "*"),
        ([None], "*"),
        ([None, "tag"], "@tag_field:{tag}"),
    ],
    ids=[
        "none",
        "empty_list",
        "empty_string",
        "list_with_none",
        "list_with_none_and_tag",
    ],
)
def test_nullable(value, expected):
    tag = Tag("tag_field")
    assert str(tag == value) == expected


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
    ],
)
def test_text_filter(operation, value, expected):
    txt_f = getattr(Text("text_field"), operation)(value)
    assert str(txt_f) == expected


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
    """Test Timestamp filter with date objects (should match full day)."""
    d = date(2023, 3, 17)
    ts = Timestamp("created_at") == d

    expected_ts_start = (
        datetime.combine(d, time.min).astimezone(timezone.utc).timestamp()
    )
    expected_ts_end = datetime.combine(d, time.max).astimezone(timezone.utc).timestamp()

    assert str(ts) == f"@created_at:[{expected_ts_start} {expected_ts_end}]"


def test_timestamp_iso_string():
    """Test Timestamp filter with ISO format strings."""
    # Date-only ISO string
    ts = Timestamp("created_at") == "2023-03-17"
    d = date(2023, 3, 17)
    expected_ts_start = (
        datetime.combine(d, time.min).astimezone(timezone.utc).timestamp()
    )
    expected_ts_end = datetime.combine(d, time.max).astimezone(timezone.utc).timestamp()
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
