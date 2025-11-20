"""
Integration tests for field modifier ordering against live Redis.

These tests verify that indices can be created successfully with various
field modifier combinations. The key test is that index.create() succeeds
without error - if the modifiers are in the wrong order, Redis will reject
the FT.CREATE command.
"""

import pytest

from redisvl.index import SearchIndex
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.schema import IndexSchema

MIN_SEARCH_VERSION_FOR_INDEXMISSING = 21000  # RediSearch 2.10.0+


def skip_if_search_version_below_for_indexmissing(client) -> None:
    """Skip tests that require INDEXMISSING/INDEXEMPTY if RediSearch is too old."""
    modules = RedisConnectionFactory.get_modules(client)
    search_ver = modules.get("search", 0)
    searchlight_ver = modules.get("searchlight", 0)
    current_ver = max(search_ver, searchlight_ver)
    if current_ver < MIN_SEARCH_VERSION_FOR_INDEXMISSING:
        pytest.skip(
            "INDEXMISSING/INDEXEMPTY require RediSearch 2.10+ "
            f"(found module version {current_ver})"
        )


class TestTextFieldModifierOrderingIntegration:
    """Integration tests for TextField modifier ordering."""

    def test_textfield_sortable_and_index_missing(self, client, redis_url, worker_id):
        """Test TextField with sortable and index_missing creates successfully."""
        skip_if_search_version_below_for_indexmissing(client)
        schema_dict = {
            "index": {
                "name": f"test_text_sortable_missing_{worker_id}",
                "prefix": f"text_sm_{worker_id}",
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
        index = SearchIndex(schema=schema, redis_url=redis_url)

        try:
            # This should succeed - if modifiers are in wrong order, it will fail
            index.create(overwrite=True)

            # Verify index was created
            info = client.execute_command(
                "FT.INFO", f"test_text_sortable_missing_{worker_id}"
            )
            assert info is not None
        finally:
            try:
                index.delete(drop=True)
            except Exception:
                pass  # Index may not exist if test was skipped or failed early

    def test_textfield_all_modifiers(self, client, redis_url, worker_id):
        """Test TextField with all modifiers."""
        skip_if_search_version_below_for_indexmissing(client)
        schema_dict = {
            "index": {
                "name": f"test_text_all_mods_{worker_id}",
                "prefix": f"text_all_{worker_id}",
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
        index = SearchIndex(schema=schema, redis_url=redis_url)

        try:
            # This should succeed - if modifiers are in wrong order, it will fail
            index.create(overwrite=True)

            # Verify index was created
            info = client.execute_command("FT.INFO", f"test_text_all_mods_{worker_id}")
            assert info is not None
        finally:
            try:
                index.delete(drop=True)
            except Exception:
                pass  # Index may not exist if test was skipped or failed early


class TestTagFieldModifierOrderingIntegration:
    """Integration tests for TagField modifier ordering."""

    def test_tagfield_sortable_and_index_missing(self, client, redis_url, worker_id):
        """Test TagField with sortable and index_missing creates successfully."""
        skip_if_search_version_below_for_indexmissing(client)
        schema_dict = {
            "index": {
                "name": f"test_tag_sortable_missing_{worker_id}",
                "prefix": f"tag_sm_{worker_id}",
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
        index = SearchIndex(schema=schema, redis_url=redis_url)

        try:
            # This should succeed - if modifiers are in wrong order, it will fail
            index.create(overwrite=True)

            # Verify index was created
            info = client.execute_command(
                "FT.INFO", f"test_tag_sortable_missing_{worker_id}"
            )
            assert info is not None
        finally:
            try:
                index.delete(drop=True)
            except Exception:
                pass  # Index may not exist if test was skipped or failed early

    def test_tagfield_all_modifiers(self, client, redis_url, worker_id):
        """Test TagField with all modifiers."""
        skip_if_search_version_below_for_indexmissing(client)
        schema_dict = {
            "index": {
                "name": f"test_tag_all_mods_{worker_id}",
                "prefix": f"tag_all_{worker_id}",
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
        index = SearchIndex(schema=schema, redis_url=redis_url)

        try:
            # This should succeed - if modifiers are in wrong order, it will fail
            index.create(overwrite=True)

            # Verify index was created
            info = client.execute_command("FT.INFO", f"test_tag_all_mods_{worker_id}")
            assert info is not None
        finally:
            try:
                index.delete(drop=True)
            except Exception:
                pass  # Index may not exist if test was skipped or failed early


class TestGeoFieldModifierOrderingIntegration:
    """Integration tests for GeoField modifier ordering."""

    def test_geofield_sortable_and_index_missing(self, client, redis_url, worker_id):
        """Test GeoField with sortable and index_missing creates successfully."""
        skip_if_search_version_below_for_indexmissing(client)
        schema_dict = {
            "index": {
                "name": f"test_geo_sortable_missing_{worker_id}",
                "prefix": f"geo_sm_{worker_id}",
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
        index = SearchIndex(schema=schema, redis_url=redis_url)

        try:
            # This should succeed - if modifiers are in wrong order, it will fail
            index.create(overwrite=True)

            # Verify index was created
            info = client.execute_command(
                "FT.INFO", f"test_geo_sortable_missing_{worker_id}"
            )
            assert info is not None
        finally:
            try:
                index.delete(drop=True)
            except Exception:
                pass  # Index may not exist if test was skipped or failed early


class TestNumericFieldModifierOrderingIntegration:
    """Integration tests for NumericField modifier ordering."""

    def test_numericfield_sortable_and_index_missing(
        self, client, redis_url, worker_id
    ):
        """Test NumericField with sortable and index_missing creates successfully."""
        skip_if_search_version_below_for_indexmissing(client)
        schema_dict = {
            "index": {
                "name": f"test_numeric_sortable_missing_{worker_id}",
                "prefix": f"num_sm_{worker_id}",
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
        index = SearchIndex(schema=schema, redis_url=redis_url)

        try:
            # This should succeed - if modifiers are in wrong order, it will fail
            index.create(overwrite=True)

            # Verify index was created
            info = client.execute_command(
                "FT.INFO", f"test_numeric_sortable_missing_{worker_id}"
            )
            assert info is not None
        finally:
            try:
                index.delete(drop=True)
            except Exception:
                pass  # Index may not exist if test was skipped or failed early


class TestMultiFieldModifierOrderingIntegration:
    """Integration tests for multiple field types with modifiers."""

    def test_mixed_field_types_with_modifiers(self, client, redis_url, worker_id):
        """Test index with multiple field types all using modifiers."""
        skip_if_search_version_below_for_indexmissing(client)
        schema_dict = {
            "index": {
                "name": f"test_mixed_fields_{worker_id}",
                "prefix": f"mixed_{worker_id}",
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
        index = SearchIndex(schema=schema, redis_url=redis_url)

        try:
            # This should succeed - if modifiers are in wrong order, it will fail
            index.create(overwrite=True)

            # Verify index was created
            info = client.execute_command("FT.INFO", f"test_mixed_fields_{worker_id}")
            assert info is not None

            # Verify all fields were created
            attrs_list = info[7]
            assert len(attrs_list) == 4
        finally:
            try:
                index.delete(drop=True)
            except Exception:
                pass  # Index may not exist if test was skipped or failed early


class TestMultipleCommandsScenarioIntegration:
    """Integration tests for mlp_commands.txt scenario."""

    def test_mlp_commands_index_creation(self, client, redis_url, worker_id):
        """Test creating index with INDEXMISSING SORTABLE UNF modifiers.

        This test verifies that an index with all three modifiers
        (INDEXMISSING, SORTABLE, UNF) can be created successfully with
        correct field ordering.
        """
        skip_if_search_version_below_for_indexmissing(client)
        schema_dict = {
            "index": {
                "name": f"testidx_{worker_id}",
                "prefix": f"test_{worker_id}:",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "description",
                    "type": "text",
                    "attrs": {"index_missing": True, "sortable": True, "unf": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url=redis_url)

        try:
            # This should succeed with correct modifier ordering
            index.create(overwrite=True)

            # Verify index was created
            info = client.execute_command("FT.INFO", f"testidx_{worker_id}")
            assert info is not None
        finally:
            try:
                index.delete(drop=True)
            except Exception:
                pass  # Index may not exist if test was skipped or failed early

    def test_indexmissing_enables_ismissing_query(self, client, redis_url, worker_id):
        """Test that INDEXMISSING enables ismissing() query function."""
        skip_if_search_version_below_for_indexmissing(client)
        schema_dict = {
            "index": {
                "name": f"test_ismissing_{worker_id}",
                "prefix": f"ismiss_{worker_id}:",
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
        index = SearchIndex(schema=schema, redis_url=redis_url)

        try:
            index.create(overwrite=True)

            # Create documents with and without the field
            client.hset(f"ismiss_{worker_id}:1", "optional_field", "has value")
            client.hset(f"ismiss_{worker_id}:2", "other_field", "no optional_field")
            client.hset(f"ismiss_{worker_id}:3", "optional_field", "also has value")

            # Query for missing fields
            result = client.execute_command(
                "FT.SEARCH",
                f"test_ismissing_{worker_id}",
                "ismissing(@optional_field)",
                "DIALECT",
                "2",
            )

            # Should return 1 result (ismiss_{worker_id}:2)
            assert result[0] == 1
            assert f"ismiss_{worker_id}:2" in str(result)

        finally:
            try:
                client.delete(
                    f"ismiss_{worker_id}:1",
                    f"ismiss_{worker_id}:2",
                    f"ismiss_{worker_id}:3",
                )
                index.delete(drop=True)
            except Exception:
                pass  # Index may not exist if test was skipped or failed early


class TestFieldTypeModifierSupport:
    """Test that field types only support their documented modifiers."""

    def test_numeric_field_does_not_support_index_empty(
        self, client, redis_url, worker_id
    ):
        """Verify that NumericField does not have index_empty attribute.

        INDEXEMPTY is only supported for TEXT and TAG fields according to
        RediSearch documentation. NumericFieldAttributes should not have
        an index_empty attribute.
        """
        import inspect

        from redisvl.schema.fields import NumericFieldAttributes

        # Verify NumericFieldAttributes doesn't have index_empty
        attrs = inspect.signature(NumericFieldAttributes).parameters
        assert (
            "index_empty" not in attrs
        ), "NumericFieldAttributes should not have index_empty parameter"

        # Verify the attribute doesn't exist on the class
        field_attrs = NumericFieldAttributes()
        assert not hasattr(
            field_attrs, "index_empty"
        ), "NumericFieldAttributes should not have index_empty attribute"

    def test_geo_field_does_not_support_index_empty(self, client, redis_url, worker_id):
        """Verify that GeoField does not have index_empty attribute.

        INDEXEMPTY is only supported for TEXT and TAG fields according to
        RediSearch documentation. GeoFieldAttributes should not have
        an index_empty attribute.
        """
        import inspect

        from redisvl.schema.fields import GeoFieldAttributes

        # Verify GeoFieldAttributes doesn't have index_empty
        attrs = inspect.signature(GeoFieldAttributes).parameters
        assert (
            "index_empty" not in attrs
        ), "GeoFieldAttributes should not have index_empty parameter"

        # Verify the attribute doesn't exist on the class
        field_attrs = GeoFieldAttributes()
        assert not hasattr(
            field_attrs, "index_empty"
        ), "GeoFieldAttributes should not have index_empty attribute"

    def test_text_field_supports_index_empty(self, client, redis_url, worker_id):
        """Verify that TextField supports index_empty attribute.

        INDEXEMPTY is supported for TEXT fields according to RediSearch documentation.
        """
        from redisvl.schema.fields import TextFieldAttributes

        # Verify TextFieldAttributes has index_empty
        field_attrs = TextFieldAttributes(index_empty=True)
        assert hasattr(
            field_attrs, "index_empty"
        ), "TextFieldAttributes should have index_empty attribute"
        assert field_attrs.index_empty is True

    def test_tag_field_supports_index_empty(self, client, redis_url, worker_id):
        """Verify that TagField supports index_empty attribute.

        INDEXEMPTY is supported for TAG fields according to RediSearch documentation.
        """
        from redisvl.schema.fields import TagFieldAttributes

        # Verify TagFieldAttributes has index_empty
        field_attrs = TagFieldAttributes(index_empty=True)
        assert hasattr(
            field_attrs, "index_empty"
        ), "TagFieldAttributes should have index_empty attribute"
        assert field_attrs.index_empty is True
