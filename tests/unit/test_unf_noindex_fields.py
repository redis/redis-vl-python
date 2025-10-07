"""Unit tests for UNF and NOINDEX field attributes."""

import pytest

from redisvl.schema.fields import (
    FieldFactory,
    GeoField,
    NumericField,
    TagField,
    TextField,
)


class TestTextFieldAttributes:
    """Test TextField support for no_index and unf attributes."""

    def test_no_index_attribute_with_sortable(self):
        """Test TextField with no_index=True and sortable=True."""
        field = TextField(name="title", attrs={"no_index": True, "sortable": True})

        assert field.attrs.no_index is True
        assert field.attrs.sortable is True

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "NOINDEX" in args
        assert "SORTABLE" in args

    def test_no_index_default_value(self):
        """Test TextField no_index defaults to False."""
        field = TextField(name="content")
        assert field.attrs.no_index is False

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "NOINDEX" not in args

    def test_unf_attribute_with_sortable(self):
        """Test TextField with unf=True and sortable=True."""
        field = TextField(name="title", attrs={"unf": True, "sortable": True})

        assert field.attrs.unf is True
        assert field.attrs.sortable is True

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "UNF" in args
        assert "SORTABLE" in args

    def test_unf_without_sortable(self):
        """Test that UNF is not added when field is not sortable."""
        field = TextField(name="description", attrs={"unf": True, "sortable": False})

        assert field.attrs.unf is True
        assert field.attrs.sortable is False

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "UNF" not in args
        assert "SORTABLE" not in args

    def test_unf_default_value(self):
        """Test TextField unf defaults to False."""
        field = TextField(name="content")
        assert field.attrs.unf is False


class TestNumericFieldAttributes:
    """Test NumericField support for no_index and unf attributes."""

    def test_no_index_attribute_with_sortable(self):
        """Test NumericField with no_index=True and sortable=True."""
        field = NumericField(name="price", attrs={"no_index": True, "sortable": True})

        assert field.attrs.no_index is True
        assert field.attrs.sortable is True

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "NOINDEX" in args
        assert "SORTABLE" in args

    def test_no_index_default_value(self):
        """Test NumericField no_index defaults to False."""
        field = NumericField(name="quantity")
        assert field.attrs.no_index is False

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "NOINDEX" not in args

    def test_unf_attribute_with_sortable(self):
        """Test NumericField with unf=True and sortable=True."""
        field = NumericField(name="score", attrs={"unf": True, "sortable": True})

        assert field.attrs.unf is True
        assert field.attrs.sortable is True

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "UNF" in args
        assert "SORTABLE" in args

    def test_unf_without_sortable(self):
        """Test that UNF is not added when field is not sortable."""
        field = NumericField(name="count", attrs={"unf": True, "sortable": False})

        assert field.attrs.unf is True
        assert field.attrs.sortable is False

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "UNF" not in args
        assert "SORTABLE" not in args

    def test_unf_default_value(self):
        """Test NumericField unf defaults to False."""
        field = NumericField(name="rating")
        assert field.attrs.unf is False


class TestTagFieldNoIndex:
    """Test TagField support for no_index attribute."""

    def test_no_index_attribute_with_sortable(self):
        """Test TagField with no_index=True and sortable=True."""
        field = TagField(name="tags", attrs={"no_index": True, "sortable": True})

        assert field.attrs.no_index is True
        assert field.attrs.sortable is True

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "NOINDEX" in args
        assert "SORTABLE" in args

    def test_no_index_default_value(self):
        """Test TagField no_index defaults to False."""
        field = TagField(name="categories")
        assert field.attrs.no_index is False

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "NOINDEX" not in args


class TestGeoFieldNoIndex:
    """Test GeoField support for no_index attribute."""

    def test_no_index_attribute_with_sortable(self):
        """Test GeoField with no_index=True and sortable=True."""
        field = GeoField(name="location", attrs={"no_index": True, "sortable": True})

        assert field.attrs.no_index is True
        assert field.attrs.sortable is True

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "NOINDEX" in args
        assert "SORTABLE" in args

    def test_no_index_default_value(self):
        """Test GeoField no_index defaults to False."""
        field = GeoField(name="coordinates")
        assert field.attrs.no_index is False

        redis_field = field.as_redis_field()
        args = redis_field.redis_args()
        assert "NOINDEX" not in args


class TestFieldFactoryWithNewAttributes:
    """Test FieldFactory creating fields with new attributes."""

    def test_create_text_field_with_unf_and_noindex(self):
        """Test creating TextField with unf and no_index via FieldFactory."""
        field = FieldFactory.create_field(
            type="text",
            name="title",
            attrs={"unf": True, "no_index": True, "sortable": True},
        )

        assert isinstance(field, TextField)
        assert field.attrs.unf is True
        assert field.attrs.no_index is True
        assert field.attrs.sortable is True

    def test_create_numeric_field_with_unf_and_noindex(self):
        """Test creating NumericField with unf and no_index via FieldFactory."""
        field = FieldFactory.create_field(
            type="numeric",
            name="score",
            attrs={"unf": True, "no_index": True, "sortable": True},
        )

        assert isinstance(field, NumericField)
        assert field.attrs.unf is True
        assert field.attrs.no_index is True
        assert field.attrs.sortable is True

    def test_create_tag_field_with_noindex(self):
        """Test creating TagField with no_index via FieldFactory."""
        field = FieldFactory.create_field(
            type="tag", name="tags", attrs={"no_index": True, "sortable": True}
        )

        assert isinstance(field, TagField)
        assert field.attrs.no_index is True
        assert field.attrs.sortable is True

    def test_create_geo_field_with_noindex(self):
        """Test creating GeoField with no_index via FieldFactory."""
        field = FieldFactory.create_field(
            type="geo", name="location", attrs={"no_index": True, "sortable": True}
        )

        assert isinstance(field, GeoField)
        assert field.attrs.no_index is True
        assert field.attrs.sortable is True


class TestBackwardCompatibility:
    """Test that new attributes don't break backward compatibility."""

    def test_text_field_without_new_attributes(self):
        """Test TextField works without specifying new attributes."""
        field = TextField(name="content", attrs={"weight": 2.0})

        assert field.attrs.weight == 2.0
        assert field.attrs.unf is False
        assert field.attrs.no_index is False

    def test_numeric_field_without_new_attributes(self):
        """Test NumericField works without specifying new attributes."""
        field = NumericField(name="price", attrs={"sortable": True})

        assert field.attrs.sortable is True
        assert field.attrs.unf is False
        assert field.attrs.no_index is False

    def test_tag_field_without_new_attributes(self):
        """Test TagField works without specifying new attributes."""
        field = TagField(name="tags", attrs={"separator": "|"})

        assert field.attrs.separator == "|"
        assert field.attrs.no_index is False

    def test_geo_field_without_new_attributes(self):
        """Test GeoField works without specifying new attributes."""
        field = GeoField(name="location", attrs={"sortable": True})

        assert field.attrs.sortable is True
        assert field.attrs.no_index is False
