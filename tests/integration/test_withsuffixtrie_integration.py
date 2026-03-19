"""
Integration tests for withsuffixtrie attribute on Text and Tag fields.

Tests verify that the WITHSUFFIXTRIE modifier is correctly passed to Redis
when creating indexes, enabling optimized suffix and contains queries.
"""

import pytest

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema


class TestTextFieldWithSuffixTrie:
    """Integration tests for TextField withsuffixtrie attribute."""

    def test_textfield_withsuffixtrie_creates_successfully(
        self, client, redis_url, worker_id
    ):
        """Test TextField with withsuffixtrie creates successfully."""
        schema_dict = {
            "index": {
                "name": f"test_text_suffix_{worker_id}",
                "prefix": f"text_suffix_{worker_id}:",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "email",
                    "type": "text",
                    "attrs": {"withsuffixtrie": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url=redis_url)
        index.create(overwrite=True)

        # Verify index was created and has WITHSUFFIXTRIE
        info = client.execute_command("FT.INFO", f"test_text_suffix_{worker_id}")

        # Find the field attributes in the info response
        # FT.INFO returns a flat list, we need to find the attributes section
        info_dict = _parse_ft_info(info)
        field_attrs = _get_field_attributes(info_dict, "email")

        assert (
            "WITHSUFFIXTRIE" in field_attrs
        ), f"WITHSUFFIXTRIE not found in field attributes: {field_attrs}"

        # Cleanup
        index.delete(drop=True)

    def test_textfield_withsuffixtrie_and_sortable(self, client, redis_url, worker_id):
        """Test TextField with withsuffixtrie and sortable combined."""
        schema_dict = {
            "index": {
                "name": f"test_text_suffix_sort_{worker_id}",
                "prefix": f"text_suffix_sort_{worker_id}:",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "title",
                    "type": "text",
                    "attrs": {"withsuffixtrie": True, "sortable": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url=redis_url)
        index.create(overwrite=True)

        info = client.execute_command("FT.INFO", f"test_text_suffix_sort_{worker_id}")
        info_dict = _parse_ft_info(info)
        field_attrs = _get_field_attributes(info_dict, "title")

        assert "WITHSUFFIXTRIE" in field_attrs
        assert "SORTABLE" in field_attrs

        index.delete(drop=True)


class TestTagFieldWithSuffixTrie:
    """Integration tests for TagField withsuffixtrie attribute."""

    def test_tagfield_withsuffixtrie_creates_successfully(
        self, client, redis_url, worker_id
    ):
        """Test TagField with withsuffixtrie creates successfully."""
        schema_dict = {
            "index": {
                "name": f"test_tag_suffix_{worker_id}",
                "prefix": f"tag_suffix_{worker_id}:",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "domain",
                    "type": "tag",
                    "attrs": {"withsuffixtrie": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url=redis_url)
        index.create(overwrite=True)

        info = client.execute_command("FT.INFO", f"test_tag_suffix_{worker_id}")
        info_dict = _parse_ft_info(info)
        field_attrs = _get_field_attributes(info_dict, "domain")

        assert (
            "WITHSUFFIXTRIE" in field_attrs
        ), f"WITHSUFFIXTRIE not found in field attributes: {field_attrs}"

        index.delete(drop=True)

    def test_tagfield_withsuffixtrie_and_case_sensitive(
        self, client, redis_url, worker_id
    ):
        """Test TagField with withsuffixtrie and case_sensitive combined."""
        schema_dict = {
            "index": {
                "name": f"test_tag_suffix_cs_{worker_id}",
                "prefix": f"tag_suffix_cs_{worker_id}:",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "sku",
                    "type": "tag",
                    "attrs": {"withsuffixtrie": True, "case_sensitive": True},
                }
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        index = SearchIndex(schema=schema, redis_url=redis_url)
        index.create(overwrite=True)

        info = client.execute_command("FT.INFO", f"test_tag_suffix_cs_{worker_id}")
        info_dict = _parse_ft_info(info)
        field_attrs = _get_field_attributes(info_dict, "sku")

        assert "WITHSUFFIXTRIE" in field_attrs
        assert "CASESENSITIVE" in field_attrs

        index.delete(drop=True)


# Helper functions to parse FT.INFO response


def _parse_ft_info(info) -> dict:
    """Parse FT.INFO response into a dictionary."""
    result = {}
    if isinstance(info, list):
        i = 0
        while i < len(info) - 1:
            key = info[i]
            value = info[i + 1]
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            result[key] = value
            i += 2
    return result


def _get_field_attributes(info_dict: dict, field_name: str) -> list:
    """Extract field attributes from parsed FT.INFO for a specific field."""
    attributes = info_dict.get("attributes", [])
    if isinstance(attributes, list):
        for field_info in attributes:
            if isinstance(field_info, list):
                # Field info is a list like [b'identifier', b'email', b'type', b'TEXT', ...]
                # Convert bytes to strings for comparison
                field_info_str = [
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in field_info
                ]
                # Check if this is the field we're looking for
                for i, item in enumerate(field_info_str):
                    if item == "identifier" and i + 1 < len(field_info_str):
                        if field_info_str[i + 1] == field_name:
                            return field_info_str
    return []
