import pytest

from redisvl.query import (
    Filter,
    GeoFilter,
    NumericFilter,
    TagFilter,
    TextFilter,
    VectorQuery,
)
from redisvl.utils.utils import TokenEscaper


class TestFilters:
    def test_tag_filter(self):
        tf = TagFilter("tag_field", ["tag1", "tag2"])
        assert tf.to_string() == "@tag_field:{tag1 | tag2}"

    def test_numeric_filter(self):
        nf = NumericFilter(
            "numeric_field", 1, 10, min_exclusive=True, max_exclusive=True
        )
        assert nf.to_string() == "@numeric_field:[(1 (10]"

    def test_numeric_filter_2(self):
        nf = NumericFilter(
            "numeric_field", 1, 10, min_exclusive=False, max_exclusive=False
        )
        assert nf.to_string() == "@numeric_field:[1 10]"

    def test_text_filter(self):
        txt_f = TextFilter("text_field", "text")
        assert txt_f.to_string() == "@text_field:text"

    def test_geo_filter(self):
        geo_f = GeoFilter("geo_field", 1, 2, 3)
        assert geo_f.to_string() == "@geo_field:[1 2 3 km]"

        geo_f = GeoFilter("geo_field", 1, 2, 3, unit="m")
        assert geo_f.to_string() == "@geo_field:[1 2 3 m]"

    def test_filters_combination(self):
        tf1 = TagFilter("tag_field", ["tag1", "tag2"])
        tf2 = TagFilter("tag_field", ["tag3"])
        tf1 += tf2
        assert str(tf1) == "(@tag_field:{tag1 | tag2} @tag_field:{tag3})"
        tf1 &= tf2
        assert (
            str(tf1)
            == "(@tag_field:{tag1 | tag2} @tag_field:{tag3} | @tag_field:{tag3})"
        )
        tf1 -= tf2
        assert (
            str(tf1)
            == "(@tag_field:{tag1 | tag2} @tag_field:{tag3} | @tag_field:{tag3} -@tag_field:{tag3})"
        )
        tf1 ^= tf2
        assert (
            str(tf1)
            == "(@tag_field:{tag1 | tag2} @tag_field:{tag3} | @tag_field:{tag3} -@tag_field:{tag3} ~@tag_field:{tag3})"
        )

    def test_filter_raise_not_implemented_error(self):
        with pytest.raises(NotImplementedError):
            Filter("field").to_string()
