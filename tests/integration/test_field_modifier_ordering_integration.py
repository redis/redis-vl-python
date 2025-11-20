"""
Integration tests for field modifier ordering against live Redis.

These tests verify that indices can be created successfully with various
field modifier combinations and that the modifiers are correctly set in Redis.
"""

import pytest
import redis

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema


@pytest.fixture
def redis_client():
    """Fixture to provide a Redis client."""
    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    yield client
    # Cleanup is handled by individual tests


class TestTextFieldModifierOrderingIntegration:
    """Integration tests for TextField modifier ordering."""

    def test_textfield_sortable_and_index_missing(self, redis_client):
        """Test TextField with sortable and index_missing creates successfully."""
        schema_dict = {
            "index": {
                "name": "test_text_sortable_missing",
                "prefix": "text_sm",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "title",
                    "type": "text",
                    "attrs": {"sortable": True, "index_missing": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url="redis://localhost:6379")

        try:
            index.create(overwrite=True)

            # Verify index was created
            info = redis_client.execute_command("FT.INFO", "test_text_sortable_missing")
            assert info is not None

            # Verify field attributes contain both SORTABLE and INDEXMISSING
            attrs = info[7][0]  # Get field attributes
            assert "SORTABLE" in attrs
            assert "INDEXMISSING" in attrs
        finally:
            index.delete(drop=True)

    def test_textfield_all_modifiers(self, redis_client):
        """Test TextField with all modifiers."""
        schema_dict = {
            "index": {
                "name": "test_text_all_mods",
                "prefix": "text_all",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "content",
                    "type": "text",
                    "attrs": {
                        "index_empty": True,
                        "index_missing": True,
                        "sortable": True,
                        "unf": True,
                    },
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url="redis://localhost:6379")

        try:
            index.create(overwrite=True)

            # Verify index was created
            info = redis_client.execute_command("FT.INFO", "test_text_all_mods")
            assert info is not None

            # Verify all modifiers are present
            attrs = info[7][0]
            assert "SORTABLE" in attrs
            assert "INDEXMISSING" in attrs
            assert "INDEXEMPTY" in attrs
            assert "UNF" in attrs
        finally:
            index.delete(drop=True)


class TestTagFieldModifierOrderingIntegration:
    """Integration tests for TagField modifier ordering."""

    def test_tagfield_sortable_and_index_missing(self, redis_client):
        """Test TagField with sortable and index_missing creates successfully."""
        schema_dict = {
            "index": {
                "name": "test_tag_sortable_missing",
                "prefix": "tag_sm",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "tags",
                    "type": "tag",
                    "attrs": {"sortable": True, "index_missing": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url="redis://localhost:6379")

        try:
            index.create(overwrite=True)

            # Verify index was created
            info = redis_client.execute_command("FT.INFO", "test_tag_sortable_missing")
            assert info is not None

            # Verify field attributes contain both SORTABLE and INDEXMISSING
            attrs = info[7][0]
            assert "SORTABLE" in attrs
            assert "INDEXMISSING" in attrs
        finally:
            index.delete(drop=True)

    def test_tagfield_all_modifiers(self, redis_client):
        """Test TagField with all modifiers."""
        schema_dict = {
            "index": {
                "name": "test_tag_all_mods",
                "prefix": "tag_all",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "categories",
                    "type": "tag",
                    "attrs": {
                        "index_empty": True,
                        "index_missing": True,
                        "sortable": True,
                    },
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url="redis://localhost:6379")

        try:
            index.create(overwrite=True)

            # Verify index was created
            info = redis_client.execute_command("FT.INFO", "test_tag_all_mods")
            assert info is not None

            # Verify all modifiers are present
            attrs = info[7][0]
            assert "SORTABLE" in attrs
            assert "INDEXMISSING" in attrs
            assert "INDEXEMPTY" in attrs
        finally:
            index.delete(drop=True)


class TestGeoFieldModifierOrderingIntegration:
    """Integration tests for GeoField modifier ordering."""

    def test_geofield_sortable_and_index_missing(self, redis_client):
        """Test GeoField with sortable and index_missing creates successfully."""
        schema_dict = {
            "index": {
                "name": "test_geo_sortable_missing",
                "prefix": "geo_sm",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "location",
                    "type": "geo",
                    "attrs": {"sortable": True, "index_missing": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url="redis://localhost:6379")

        try:
            index.create(overwrite=True)

            # Verify index was created
            info = redis_client.execute_command("FT.INFO", "test_geo_sortable_missing")
            assert info is not None

            # Verify field attributes contain both SORTABLE and INDEXMISSING
            attrs = info[7][0]
            assert "SORTABLE" in attrs
            assert "INDEXMISSING" in attrs
        finally:
            index.delete(drop=True)


class TestNumericFieldModifierOrderingIntegration:
    """Integration tests for NumericField modifier ordering."""

    def test_numericfield_sortable_and_index_missing(self, redis_client):
        """Test NumericField with sortable and index_missing creates successfully."""
        schema_dict = {
            "index": {
                "name": "test_numeric_sortable_missing",
                "prefix": "num_sm",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "price",
                    "type": "numeric",
                    "attrs": {"sortable": True, "index_missing": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url="redis://localhost:6379")

        try:
            index.create(overwrite=True)

            # Verify index was created
            info = redis_client.execute_command(
                "FT.INFO", "test_numeric_sortable_missing"
            )
            assert info is not None

            # Verify field attributes contain both SORTABLE and INDEXMISSING
            attrs = info[7][0]
            assert "SORTABLE" in attrs
            assert "INDEXMISSING" in attrs
        finally:
            index.delete(drop=True)


class TestMultiFieldModifierOrderingIntegration:
    """Integration tests with multiple fields and modifiers."""

    def test_mixed_field_types_with_modifiers(self, redis_client):
        """Test index with multiple field types all using modifiers."""
        schema_dict = {
            "index": {
                "name": "test_mixed_fields",
                "prefix": "mixed",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "title",
                    "type": "text",
                    "attrs": {"sortable": True, "index_missing": True},
                },
                {
                    "name": "tags",
                    "type": "tag",
                    "attrs": {"sortable": True, "index_missing": True},
                },
                {
                    "name": "price",
                    "type": "numeric",
                    "attrs": {"sortable": True, "index_missing": True},
                },
                {
                    "name": "location",
                    "type": "geo",
                    "attrs": {"sortable": True, "index_missing": True},
                },
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url="redis://localhost:6379")

        try:
            index.create(overwrite=True)

            # Verify index was created
            info = redis_client.execute_command("FT.INFO", "test_mixed_fields")
            assert info is not None

            # Verify all fields were created
            attrs_list = info[7]
            assert len(attrs_list) == 4

            # Verify each field has the correct modifiers
            for attrs in attrs_list:
                assert "SORTABLE" in attrs
                assert "INDEXMISSING" in attrs
        finally:
            index.delete(drop=True)


class TestMLPCommandsScenarioIntegration:
    """Integration tests for the exact scenario from mlp_commands.txt."""

    def test_mlp_commands_index_creation(self, redis_client):
        """Test creating index matching mlp_commands.txt scenario.

        This test verifies that an index with INDEXMISSING SORTABLE UNF
        modifiers can be created successfully with correct field ordering.
        """
        schema_dict = {
            "index": {
                "name": "testidx",
                "prefix": "test:",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "work_experience_summary",
                    "type": "text",
                    "attrs": {"index_missing": True, "sortable": True, "unf": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url="redis://localhost:6379")

        try:
            # This should succeed with correct modifier ordering
            index.create(overwrite=True)

            # Verify index was created
            info = redis_client.execute_command("FT.INFO", "testidx")
            assert info is not None

            # Verify field attributes
            attrs = info[7][0]
            assert "INDEXMISSING" in attrs
            assert "SORTABLE" in attrs
            assert "UNF" in attrs
        finally:
            index.delete(drop=True)

    def test_indexmissing_enables_ismissing_query(self, redis_client):
        """Test that INDEXMISSING enables ismissing() query function."""
        schema_dict = {
            "index": {
                "name": "test_ismissing",
                "prefix": "ismiss:",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "optional_field",
                    "type": "text",
                    "attrs": {"index_missing": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url="redis://localhost:6379")

        try:
            index.create(overwrite=True)

            # Create documents with and without the field
            redis_client.hset("ismiss:1", "optional_field", "has value")
            redis_client.hset("ismiss:2", "other_field", "no optional_field")
            redis_client.hset("ismiss:3", "optional_field", "also has value")

            # Query for missing fields
            result = redis_client.execute_command(
                "FT.SEARCH", "test_ismissing", "ismissing(@optional_field)", "DIALECT", "2"
            )

            # Should return 1 result (ismiss:2)
            assert result[0] == 1
            assert "ismiss:2" in str(result)

        finally:
            redis_client.delete("ismiss:1", "ismiss:2", "ismiss:3")
            index.delete(drop=True)
