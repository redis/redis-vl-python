"""
Unit tests for field modifier ordering fix.

Tests verify that field modifiers are generated in the correct order
to satisfy RediSearch parser requirements. The canonical order is:
    [INDEXEMPTY] [INDEXMISSING] [SORTABLE [UNF]] [NOINDEX]

This is required because RediSearch has a parser limitation
where INDEXEMPTY/INDEXMISSING must appear BEFORE SORTABLE in field definitions.
"""

import pytest
from redis.commands.search.field import TextField as RedisTextField

from redisvl.schema.fields import (
    GeoField,
    NumericField,
    TagField,
    TextField,
    _normalize_field_modifiers,
)


class TestTextFieldModifierOrdering:
    """Test TextField generates modifiers in correct order."""

    def test_sortable_and_index_missing_order(self):
        """Test that INDEXMISSING comes before SORTABLE."""
        field = TextField(name="title", attrs={"sortable": True, "index_missing": True})
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        # INDEXMISSING must come before SORTABLE
        assert "INDEXMISSING" in suffix
        assert "SORTABLE" in suffix
        assert suffix.index("INDEXMISSING") < suffix.index("SORTABLE")

    def test_all_modifiers_order(self):
        """Test canonical order with all modifiers."""
        field = TextField(
            name="content",
            attrs={
                "index_empty": True,
                "index_missing": True,
                "sortable": True,
                "unf": True,
                "no_index": False,
            },
        )
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        # Verify canonical order: INDEXEMPTY, INDEXMISSING, SORTABLE, UNF
        assert suffix == ["INDEXEMPTY", "INDEXMISSING", "SORTABLE", "UNF"]

    def test_unf_only_with_sortable(self):
        """Test that UNF only appears when sortable=True."""
        field = TextField(name="title", attrs={"sortable": True, "unf": True})
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        assert "UNF" in suffix
        assert "SORTABLE" in suffix
        assert suffix.index("SORTABLE") < suffix.index("UNF")

    def test_unf_ignored_without_sortable(self):
        """Test that UNF is ignored when sortable=False."""
        field = TextField(name="description", attrs={"sortable": False, "unf": True})
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        assert "UNF" not in suffix
        assert "SORTABLE" not in suffix


class TestNumericFieldModifierOrdering:
    """Test NumericField generates modifiers in correct order."""

    def test_sortable_and_index_missing_order(self):
        """Test that INDEXMISSING comes before SORTABLE."""
        field = NumericField(
            name="price", attrs={"sortable": True, "index_missing": True}
        )
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        # INDEXMISSING must come before SORTABLE
        assert "INDEXMISSING" in suffix
        assert "SORTABLE" in suffix
        assert suffix.index("INDEXMISSING") < suffix.index("SORTABLE")

    def test_all_modifiers_order(self):
        """Test canonical order with all modifiers."""
        field = NumericField(
            name="score",
            attrs={
                "index_missing": True,
                "sortable": True,
                "unf": True,
                "no_index": False,
            },
        )
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        # Verify canonical order: INDEXMISSING, SORTABLE, UNF
        assert suffix == ["INDEXMISSING", "SORTABLE", "UNF"]

    def test_unf_only_with_sortable(self):
        """Test that UNF only appears when sortable=True."""
        field = NumericField(name="rating", attrs={"sortable": True, "unf": True})
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        assert "UNF" in suffix
        assert "SORTABLE" in suffix
        assert suffix.index("SORTABLE") < suffix.index("UNF")


class TestTagFieldModifierOrdering:
    """Test TagField generates modifiers in correct order."""

    def test_sortable_and_index_missing_order(self):
        """Test that INDEXMISSING comes before SORTABLE."""
        field = TagField(name="tags", attrs={"sortable": True, "index_missing": True})
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        # INDEXMISSING must come before SORTABLE
        assert "INDEXMISSING" in suffix
        assert "SORTABLE" in suffix
        assert suffix.index("INDEXMISSING") < suffix.index("SORTABLE")

    def test_all_modifiers_order(self):
        """Test canonical order with all modifiers."""
        field = TagField(
            name="categories",
            attrs={
                "index_empty": True,
                "index_missing": True,
                "sortable": True,
                "no_index": False,
            },
        )
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        # Verify canonical order: INDEXEMPTY, INDEXMISSING, SORTABLE
        assert suffix == ["INDEXEMPTY", "INDEXMISSING", "SORTABLE"]

    def test_index_empty_before_index_missing(self):
        """Test that INDEXEMPTY comes before INDEXMISSING."""
        field = TagField(
            name="status", attrs={"index_empty": True, "index_missing": True}
        )
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        assert "INDEXEMPTY" in suffix
        assert "INDEXMISSING" in suffix
        assert suffix.index("INDEXEMPTY") < suffix.index("INDEXMISSING")


class TestGeoFieldModifierOrdering:
    """Test GeoField generates modifiers in correct order."""

    def test_sortable_and_index_missing_order(self):
        """Test that INDEXMISSING comes before SORTABLE."""
        field = GeoField(
            name="location", attrs={"sortable": True, "index_missing": True}
        )
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        # INDEXMISSING must come before SORTABLE
        assert "INDEXMISSING" in suffix
        assert "SORTABLE" in suffix
        assert suffix.index("INDEXMISSING") < suffix.index("SORTABLE")

    def test_all_modifiers_order(self):
        """Test canonical order with all modifiers."""
        field = GeoField(
            name="coordinates",
            attrs={
                "index_missing": True,
                "sortable": True,
                "no_index": False,
            },
        )
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        # Verify canonical order: INDEXMISSING, SORTABLE
        assert suffix == ["INDEXMISSING", "SORTABLE"]


class TestModifierOrderingConsistency:
    """Test that all field types follow the same ordering rules."""

    def test_all_fields_index_missing_before_sortable(self):
        """Test that all field types put INDEXMISSING before SORTABLE."""
        fields = [
            TextField(name="text", attrs={"sortable": True, "index_missing": True}),
            NumericField(
                name="numeric", attrs={"sortable": True, "index_missing": True}
            ),
            TagField(name="tag", attrs={"sortable": True, "index_missing": True}),
            GeoField(name="geo", attrs={"sortable": True, "index_missing": True}),
        ]

        for field in fields:
            redis_field = field.as_redis_field()
            suffix = redis_field.args_suffix

            assert (
                "INDEXMISSING" in suffix
            ), f"{field.__class__.__name__} missing INDEXMISSING"
            assert "SORTABLE" in suffix, f"{field.__class__.__name__} missing SORTABLE"
            assert suffix.index("INDEXMISSING") < suffix.index(
                "SORTABLE"
            ), f"{field.__class__.__name__} has wrong order"

    def test_noindex_comes_last(self):
        """Test that NOINDEX always comes last."""
        fields = [
            TextField(name="text", attrs={"sortable": True, "no_index": True}),
            NumericField(name="numeric", attrs={"sortable": True, "no_index": True}),
            TagField(name="tag", attrs={"sortable": True, "no_index": True}),
            GeoField(name="geo", attrs={"sortable": True, "no_index": True}),
        ]

        for field in fields:
            redis_field = field.as_redis_field()
            suffix = redis_field.args_suffix

            if "NOINDEX" in suffix:
                assert (
                    suffix[-1] == "NOINDEX"
                ), f"{field.__class__.__name__} NOINDEX not at end"


class TestNormalizeFieldModifiersHelper:
    """Test the _normalize_field_modifiers helper function."""

    def test_basic_reordering(self):
        """Test basic reordering of INDEXMISSING and SORTABLE."""
        field = RedisTextField("test")
        field.args_suffix = ["SORTABLE", "INDEXMISSING"]
        canonical_order = ["INDEXMISSING", "SORTABLE"]

        _normalize_field_modifiers(field, canonical_order)

        assert field.args_suffix == ["INDEXMISSING", "SORTABLE"]

    def test_unf_added_with_sortable(self):
        """Test that UNF is added when want_unf=True and SORTABLE is present."""
        field = RedisTextField("test")
        field.args_suffix = ["SORTABLE"]
        canonical_order = ["SORTABLE", "UNF"]

        _normalize_field_modifiers(field, canonical_order, want_unf=True)

        assert field.args_suffix == ["SORTABLE", "UNF"]

    def test_unf_not_duplicated(self):
        """Test that UNF is not duplicated if already present."""
        field = RedisTextField("test")
        field.args_suffix = ["SORTABLE", "UNF"]
        canonical_order = ["SORTABLE", "UNF"]

        _normalize_field_modifiers(field, canonical_order, want_unf=True)

        assert field.args_suffix == ["SORTABLE", "UNF"]

    def test_unf_not_added_without_sortable(self):
        """Test that UNF is not added if SORTABLE is not present."""
        field = RedisTextField("test")
        field.args_suffix = ["INDEXMISSING"]
        canonical_order = ["INDEXMISSING", "SORTABLE", "UNF"]

        _normalize_field_modifiers(field, canonical_order, want_unf=True)

        assert field.args_suffix == ["INDEXMISSING"]

    def test_all_modifiers_canonical_order(self):
        """Test canonical order with all modifiers."""
        field = RedisTextField("test")
        field.args_suffix = ["NOINDEX", "UNF", "SORTABLE", "INDEXMISSING", "INDEXEMPTY"]
        canonical_order = ["INDEXEMPTY", "INDEXMISSING", "SORTABLE", "UNF", "NOINDEX"]

        _normalize_field_modifiers(field, canonical_order, want_unf=True)

        assert field.args_suffix == [
            "INDEXEMPTY",
            "INDEXMISSING",
            "SORTABLE",
            "UNF",
            "NOINDEX",
        ]

    def test_empty_suffix(self):
        """Test with empty args_suffix."""
        field = RedisTextField("test")
        field.args_suffix = []
        canonical_order = ["INDEXMISSING", "SORTABLE"]

        _normalize_field_modifiers(field, canonical_order)

        assert field.args_suffix == []


class TestFieldModifierScenario:
    """Test field modifier ordering scenario."""

    def test_work_experience_summary_field(self):
        """Test TextField with INDEXMISSING SORTABLE UNF (field modifier scenario)."""
        field = TextField(
            name="work_experience_summary",
            attrs={"index_missing": True, "sortable": True, "unf": True},
        )
        redis_field = field.as_redis_field()
        suffix = redis_field.args_suffix

        # Verify exact order from field modifier requirements
        assert suffix == ["INDEXMISSING", "SORTABLE", "UNF"]

    def test_field_modifier_scenario_redis_args(self):
        """Test that redis_args() produces correct command for field modifier scenario."""
        field = TextField(
            name="work_experience_summary",
            attrs={"index_missing": True, "sortable": True, "unf": True},
        )
        redis_field = field.as_redis_field()
        args = redis_field.redis_args()

        # Verify the args contain the field name and modifiers in correct order
        assert "work_experience_summary" in args
        assert "TEXT" in args

        # Find the position of TEXT and verify modifiers come after it
        text_idx = args.index("TEXT")
        remaining_args = args[text_idx + 1 :]

        # Verify INDEXMISSING comes before SORTABLE
        if "INDEXMISSING" in remaining_args and "SORTABLE" in remaining_args:
            assert remaining_args.index("INDEXMISSING") < remaining_args.index(
                "SORTABLE"
            )

        # Verify SORTABLE comes before UNF
        if "SORTABLE" in remaining_args and "UNF" in remaining_args:
            assert remaining_args.index("SORTABLE") < remaining_args.index("UNF")
