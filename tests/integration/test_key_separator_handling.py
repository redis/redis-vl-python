"""
Test proper handling of key separators and prefixes.

These tests verify that key separators are handled correctly when:
1. Prefix ends with the separator
2. Custom separators are used
3. Keys are constructed in different components
"""

import pytest
from redis import Redis
from redis.commands.search.index_definition import IndexType

from redisvl.extensions.router import Route, SemanticRouter
from redisvl.index import SearchIndex
from redisvl.index.storage import HashStorage, JsonStorage
from redisvl.schema import IndexSchema


class TestKeySeparatorHandling:
    """Tests for proper key separator handling across the codebase."""

    def test_prefix_ending_with_separator_no_double_separator(self):
        """Test that prefix ending with separator doesn't create double separators."""
        # Create schema with prefix ending in separator
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "user:",  # Prefix ends with separator
                "key_separator": ":",
                "storage_type": "hash",
            },
            "fields": [{"name": "content", "type": "text"}],
        }
        schema = IndexSchema.from_dict(schema_dict)
        storage = HashStorage(index_schema=schema)

        # Create a key
        key = storage._key("123", schema.index.prefix, schema.index.key_separator)

        # Should not have double separator
        assert key == "user:123", f"Expected 'user:123' but got '{key}'"
        assert "::" not in key, f"Key has double separator: {key}"

    def test_custom_separator_used_consistently(self):
        """Test that custom key_separator is used throughout."""
        # Create schema with custom separator
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "user",
                "key_separator": "-",  # Custom separator
                "storage_type": "json",
            },
            "fields": [{"name": "content", "type": "text"}],
        }
        schema = IndexSchema.from_dict(schema_dict)
        storage = JsonStorage(index_schema=schema)

        # Create a key with custom separator
        key = storage._key("456", schema.index.prefix, schema.index.key_separator)

        # Should use custom separator
        assert key == "user-456", f"Expected 'user-456' but got '{key}'"
        assert ":" not in key, f"Key uses default separator instead of custom: {key}"

    def test_empty_prefix_handled_correctly(self):
        """Test that empty prefix is handled correctly."""
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "",  # Empty prefix
                "key_separator": ":",
                "storage_type": "hash",
            },
            "fields": [{"name": "content", "type": "text"}],
        }
        schema = IndexSchema.from_dict(schema_dict)
        storage = HashStorage(index_schema=schema)

        # Create a key with empty prefix
        key = storage._key("789", schema.index.prefix, schema.index.key_separator)

        # Should return just the ID without prefix or separator
        assert key == "789", f"Expected '789' but got '{key}'"

    def test_semantic_router_uses_index_separator(self, redis_url):
        """Test that SemanticRouter uses the index's key_separator."""
        # Create a route
        route = Route(
            name="test_route", references=["hello", "hi"], distance_threshold=0.5
        )

        # Create router with routes
        router = SemanticRouter(
            name="test_router_sep",
            routes=[route],
            redis_url=redis_url,
            overwrite=True,
        )

        # Modify the index schema to use custom separator
        router._index.schema.index.key_separator = "|"
        router._index.schema.index.prefix = "router"

        # Check that route reference keys use the custom separator
        route_key = router._route_ref_key(router._index, "test_route", "ref123")

        # Should use custom separator
        assert "|" in route_key, f"Route key doesn't use custom separator: {route_key}"
        assert (
            route_key.count(":") == 0
        ), f"Route key uses default separator: {route_key}"
        assert (
            route_key == "router|test_route|ref123"
        ), f"Unexpected route key: {route_key}"

    def test_prefix_with_separator_and_custom_separator(self):
        """Test handling when prefix contains old separator and we use a new one."""
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "app:user",  # Prefix contains ':'
                "key_separator": "-",  # But we use '-' as separator
                "storage_type": "hash",
            },
            "fields": [{"name": "content", "type": "text"}],
        }
        schema = IndexSchema.from_dict(schema_dict)
        storage = HashStorage(index_schema=schema)

        # Create a key
        key = storage._key("999", schema.index.prefix, schema.index.key_separator)

        # Should use the key_separator, not the : in prefix
        assert key == "app:user-999", f"Expected 'app:user-999' but got '{key}'"

    def test_special_characters_in_separator(self):
        """Test that special characters work as separators."""
        special_separators = ["_", "::", "->", ".", "/"]

        for sep in special_separators:
            schema_dict = {
                "index": {
                    "name": "test_index",
                    "prefix": "data",
                    "key_separator": sep,
                    "storage_type": "json",
                },
                "fields": [{"name": "content", "type": "text"}],
            }
            schema = IndexSchema.from_dict(schema_dict)
            storage = JsonStorage(index_schema=schema)

            key = storage._key("id", schema.index.prefix, schema.index.key_separator)
            expected = f"data{sep}id"
            assert (
                key == expected
            ), f"For separator '{sep}': expected '{expected}' but got '{key}'"

    def test_trailing_separator_normalization(self):
        """Test that trailing separators in prefix are normalized."""
        test_cases = [
            ("user:", ":", "123", "user:123"),  # Prefix ends with separator
            ("user::", ":", "456", "user:456"),  # Prefix ends with double separator
            ("user", ":", "789", "user:789"),  # Normal case
            ("user-", "-", "abc", "user-abc"),  # Custom separator
        ]

        for prefix, separator, id_val, expected in test_cases:
            schema_dict = {
                "index": {
                    "name": "test_index",
                    "prefix": prefix,
                    "key_separator": separator,
                    "storage_type": "hash",
                },
                "fields": [{"name": "content", "type": "text"}],
            }
            schema = IndexSchema.from_dict(schema_dict)
            storage = HashStorage(index_schema=schema)

            key = storage._key(id_val, schema.index.prefix, schema.index.key_separator)

            # Check for expected normalization
            assert (
                key == expected
            ), f"For prefix='{prefix}', sep='{separator}', id='{id_val}': expected '{expected}' but got '{key}'"


class TestSemanticRouterKeyConstruction:
    """Test SemanticRouter's key construction with separators."""

    def test_router_respects_modified_key_separator(self, redis_url):
        """Test that SemanticRouter respects modified key separators."""
        route = Route(
            name="test_route", references=["hello", "hi"], distance_threshold=0.5
        )

        router = SemanticRouter(
            name="router_sep_test",
            routes=[route],
            redis_url=redis_url,
            overwrite=True,
        )

        # Test with different separators
        for separator in [":", "-", "_", "|"]:
            router._index.schema.index.key_separator = separator
            router._index.schema.index.prefix = "routes"

            # Test internal key generation
            route_key = router._route_ref_key(router._index, "route1", "ref1")

            # Should use the configured separator
            expected = f"routes{separator}route1{separator}ref1"
            assert (
                route_key == expected
            ), f"For sep '{separator}': Expected '{expected}' but got '{route_key}'"

    def test_router_with_prefix_ending_in_separator(self, redis_url):
        """Test SemanticRouter when prefix ends with separator."""
        route = Route(
            name="test_route", references=["hello", "hi"], distance_threshold=0.5
        )

        router = SemanticRouter(
            name="router_trailing_test",
            routes=[route],
            redis_url=redis_url,
            overwrite=True,
        )

        # Modify to have prefix ending with separator
        router._index.schema.index.prefix = "routes:"
        router._index.schema.index.key_separator = ":"

        # Generate a route key
        route_key = router._route_ref_key(router._index, "route1", "ref1")

        # Should not have double separator
        assert "::" not in route_key, f"Route key has double separator: {route_key}"
        assert route_key == "routes:route1:ref1", f"Unexpected route key: {route_key}"


class TestSearchIndexKeyConstruction:
    """Test SearchIndex's key construction with separators."""

    def test_search_index_key_construction(self, redis_url):
        """Test that SearchIndex properly handles key construction."""
        schema_dict = {
            "index": {
                "name": "search_test",
                "prefix": "doc:",  # Ends with separator
                "key_separator": ":",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "text", "type": "text"},
                {"name": "tag", "type": "tag"},
            ],
        }

        index = SearchIndex(
            IndexSchema.from_dict(schema_dict),
            redis_url=redis_url,
        )
        index.create(overwrite=True)

        # Add a document
        data = [{"id": "123", "text": "test content", "tag": "test"}]
        keys = index.load(data, id_field="id")

        # Check the generated key
        assert len(keys) == 1
        key = keys[0]

        # Should not have double separator
        assert "::" not in key, f"Key has double separator: {key}"
        assert key == "doc:123", f"Expected 'doc:123' but got '{key}'"

        # Clean up
        index.delete(drop=True)
