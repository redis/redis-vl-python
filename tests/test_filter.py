import pytest

from redisvl.query.filter import Geo, GeoRadius, Num, Tag, Text


def test_tag_filter():
    tf = Tag("tag_field") == ["tag1", "tag2"]
    assert str(tf) == "@tag_field:{tag1|tag2}"

    tf = Tag("tag_field") == "tag1"
    assert str(tf) == "@tag_field:{tag1}"

    tf = Tag("tag_field") != ["tag1", "tag2"]
    assert str(tf) == "(-@tag_field:{tag1|tag2})"


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


def test_text_filter():
    txt_f = Text("text_field") == "text"
    assert str(txt_f) == '@text_field:"text"'

    txt_f = Text("text_field") != "text"
    assert str(txt_f) == '(-@text_field:"text")'

    txt_f = Text("text_field") % "text"
    assert str(txt_f) == "@text_field:text"


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
