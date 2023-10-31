import pytest

from redisvl.query.filter import Geo, GeoRadius, Num, Tag, Text


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
            ["hypen-tag", "under_score", "dot.tag"],
            "@tag_field:{hypen\\-tag|under_score|dot\\.tag}",
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

    with pytest.raises(TypeError):
        nf = Num("numeric_field") == None


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
    assert str(geo_f) == "@geo_field:[1.000000 2.000000 3 km]"

    geo_f = Geo("geo_field") != GeoRadius(1.0, 2.0, 3, "km")
    assert str(geo_f) != "(-@geo_field:[1.000000 2.000000 3 m])"


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