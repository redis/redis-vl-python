"""
Test for IndexSchema type annotation issue.
TDD test file for issue #361.

This test verifies that the IndexSchema fields type annotation
matches what the validator actually expects.
"""

from typing import Dict, List

import pytest

from redisvl.schema import IndexSchema
from redisvl.schema.fields import BaseField, TagField, TextField


class TestIndexSchemaTypeIssue:
    """Test IndexSchema type annotation and validation consistency."""

    def test_fields_as_list_works(self):
        """Test that fields as a list (what validator expects) works."""
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "test",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "text_field", "type": "text"},
                {"name": "tag_field", "type": "tag"},
            ],
        }

        # This should work since validator expects list
        schema = IndexSchema.from_dict(schema_dict)
        assert schema is not None
        assert isinstance(schema.fields, dict)  # After processing, it becomes a dict
        assert "text_field" in schema.fields
        assert "tag_field" in schema.fields

    def test_fields_as_dict_should_work_per_type_annotation(self):
        """Test that fields as dict (per type annotation) now works after fix."""
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "test",
                "storage_type": "hash",
            },
            "fields": {
                "text_field": {"name": "text_field", "type": "text"},
                "tag_field": {"name": "tag_field", "type": "tag"},
            },
        }

        # According to type annotation `fields: Dict[str, BaseField] = {}`
        # this should work, and now it does after fixing the validator
        schema = IndexSchema.from_dict(schema_dict)
        assert schema is not None
        assert isinstance(schema.fields, dict)
        assert "text_field" in schema.fields
        assert "tag_field" in schema.fields

    def test_type_annotation_matches_runtime_behavior(self):
        """Test that the type annotation should match what the code actually accepts."""
        # Get the type annotation
        from typing import get_type_hints

        hints = get_type_hints(IndexSchema)
        fields_type = hints.get("fields")

        # The annotation says Dict[str, BaseField]
        assert fields_type == Dict[str, BaseField]

        # But the validator expects List and converts to Dict
        # This is the inconsistency reported in issue #361

    def test_fields_empty_list_works(self):
        """Test that empty fields list works."""
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "test",
                "storage_type": "hash",
            },
            "fields": [],  # Empty list should work
        }

        schema = IndexSchema.from_dict(schema_dict)
        assert schema is not None
        assert schema.fields == {}

    def test_fields_default_value(self):
        """Test default value for fields."""
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "test",
                "storage_type": "hash",
            },
            # No fields provided
        }

        schema = IndexSchema.from_dict(schema_dict)
        assert schema is not None
        assert schema.fields == {}  # Default should be empty dict

    def test_direct_instantiation_with_dict_fields(self):
        """Test direct instantiation with dict fields (should work after fix)."""
        # After processing, fields are stored as Dict[str, BaseField]
        # So direct instantiation with proper dict should work

        # Create fields
        text_field = TextField(name="text_field")
        tag_field = TagField(name="tag_field")

        # Try to create schema with fields as dict
        # This is what the type annotation suggests should work
        schema = IndexSchema(
            index={
                "name": "test_index",
                "prefix": "test",
                "storage_type": "hash",
            },
            fields={"text_field": text_field, "tag_field": tag_field},
        )
        assert schema is not None
        assert isinstance(schema.fields, dict)
        assert "text_field" in schema.fields
        assert "tag_field" in schema.fields

    def test_yaml_format_fields_as_list(self):
        """Test that YAML format uses fields as list."""
        yaml_content = """
        index:
            name: test_index
            prefix: test
            storage_type: hash
        fields:
            - name: text_field
              type: text
            - name: tag_field
              type: tag
        """

        # YAML format uses list, which is what validator expects
        # This test documents the expected YAML format
        import yaml

        schema_dict = yaml.safe_load(yaml_content)
        assert isinstance(schema_dict["fields"], list)

        schema = IndexSchema.from_dict(schema_dict)
        assert schema is not None
        assert isinstance(schema.fields, dict)

    def test_dict_fields_with_name_mismatch_fails(self):
        """Test that dict fields with mismatched names fail properly."""
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "test",
                "storage_type": "hash",
            },
            "fields": {
                "wrong_key": {"name": "correct_name", "type": "text"},  # Key != name
            },
        }

        with pytest.raises(ValueError, match="Field name mismatch"):
            IndexSchema.from_dict(schema_dict)

    def test_dict_fields_with_invalid_field_type_fails(self):
        """Test that dict fields with invalid field types fail properly."""
        schema_dict = {
            "index": {
                "name": "test_index",
                "prefix": "test",
                "storage_type": "hash",
            },
            "fields": {
                "text_field": "invalid_field_type",  # Should be dict or BaseField
            },
        }

        with pytest.raises(ValueError, match="Invalid field type"):
            IndexSchema.from_dict(schema_dict)
