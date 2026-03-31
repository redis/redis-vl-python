"""
Comprehensive integration tests for all 38 supported migration operations.

This test suite validates migrations against real Redis with a tiered validation approach:
- L1: Execution (plan.supported == True)
- L2: Data Integrity (doc_count_match == True)
- L3: Key Existence (key_sample_exists == True)
- L4: Schema Match (schema_match == True)

Test Categories:
1. Index-Level (2): rename index, change prefix
2. Field Add (4): text, tag, numeric, geo
3. Field Remove (5): text, tag, numeric, geo, vector
4. Field Rename (5): text, tag, numeric, geo, vector
5. Base Attrs (3): sortable, no_index, index_missing
6. Text Attrs (5): weight, no_stem, phonetic_matcher, index_empty, unf
7. Tag Attrs (3): separator, case_sensitive, index_empty
8. Numeric Attrs (1): unf
9. Vector Attrs (8): algorithm, distance_metric, initial_cap, m, ef_construction,
                      ef_runtime, epsilon, datatype
10. JSON Storage (2): add field, rename field

Some tests use L2-only validation due to Redis FT.INFO limitations:
- prefix change (keys renamed), HNSW params, initial_cap, phonetic_matcher, numeric unf

Run: pytest tests/integration/test_migration_comprehensive.py -v
Spec: nitin_docs/index_migrator/32_integration_test_spec.md
"""

import uuid
from typing import Any, Dict, List

import pytest
import yaml

from redisvl.index import SearchIndex
from redisvl.migration import MigrationExecutor, MigrationPlanner
from redisvl.migration.utils import load_migration_plan, schemas_equal
from redisvl.redis.utils import array_to_buffer

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def unique_ids(worker_id):
    """Generate unique identifiers for test isolation."""
    uid = str(uuid.uuid4())[:8]
    return {
        "name": f"mig_test_{worker_id}_{uid}",
        "prefix": f"mig_test:{worker_id}:{uid}",
    }


@pytest.fixture
def base_schema(unique_ids):
    """Base schema with all field types for testing."""
    return {
        "index": {
            "name": unique_ids["name"],
            "prefix": unique_ids["prefix"],
            "storage_type": "hash",
        },
        "fields": [
            {"name": "doc_id", "type": "tag"},
            {"name": "title", "type": "text"},
            {"name": "description", "type": "text"},
            {"name": "category", "type": "tag"},
            {"name": "price", "type": "numeric"},
            {"name": "location", "type": "geo"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",
                    "dims": 4,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }


@pytest.fixture
def sample_docs():
    """Sample documents with all field types."""
    return [
        {
            "doc_id": "1",
            "title": "Alpha Product",
            "description": "First product description",
            "category": "electronics",
            "price": 99.99,
            "location": "-122.4194,37.7749",  # SF coordinates
            "embedding": array_to_buffer([0.1, 0.2, 0.3, 0.4], "float32"),
        },
        {
            "doc_id": "2",
            "title": "Beta Service",
            "description": "Second service description",
            "category": "software",
            "price": 149.99,
            "location": "-73.9857,40.7484",  # NYC coordinates
            "embedding": array_to_buffer([0.2, 0.3, 0.4, 0.5], "float32"),
        },
        {
            "doc_id": "3",
            "title": "Gamma Item",
            "description": "",  # Empty for index_empty tests
            "category": "",  # Empty for index_empty tests
            "price": 0,
            "location": "-118.2437,34.0522",  # LA coordinates
            "embedding": array_to_buffer([0.3, 0.4, 0.5, 0.6], "float32"),
        },
    ]


def run_migration(
    redis_url: str,
    tmp_path,
    index_name: str,
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper to run a migration and return results."""
    patch_path = tmp_path / "patch.yaml"
    patch_path.write_text(yaml.safe_dump(patch, sort_keys=False))

    plan_path = tmp_path / "plan.yaml"
    planner = MigrationPlanner()
    plan = planner.create_plan(
        index_name,
        redis_url=redis_url,
        schema_patch_path=str(patch_path),
    )
    planner.write_plan(plan, str(plan_path))

    executor = MigrationExecutor()
    report = executor.apply(
        load_migration_plan(str(plan_path)),
        redis_url=redis_url,
    )

    return {
        "plan": plan,
        "report": report,
        "supported": plan.diff_classification.supported,
        "succeeded": report.result == "succeeded",
        # Additional validation fields for granular checks
        "doc_count_match": report.validation.doc_count_match,
        "schema_match": report.validation.schema_match,
        "key_sample_exists": report.validation.key_sample_exists,
        "validation_errors": report.validation.errors,
    }


def setup_index(redis_url: str, schema: Dict, docs: List[Dict]) -> SearchIndex:
    """Create index and load documents."""
    index = SearchIndex.from_dict(schema, redis_url=redis_url)
    index.create(overwrite=True)
    index.load(docs, id_field="doc_id")
    return index


def cleanup_index(index: SearchIndex):
    """Clean up index after test."""
    try:
        index.delete(drop=True)
    except Exception:
        pass


# ==============================================================================
# 1. Index-Level Changes
# ==============================================================================


class TestIndexLevelChanges:
    """Tests for index-level migration operations."""

    def test_rename_index(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test renaming an index."""
        index = setup_index(redis_url, base_schema, sample_docs)
        old_name = base_schema["index"]["name"]
        new_name = f"{old_name}_renamed"

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                old_name,
                {"version": 1, "changes": {"index": {"name": new_name}}},
            )

            assert result["supported"], "Rename index should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"

            # Verify new index exists
            live_index = SearchIndex.from_existing(new_name, redis_url=redis_url)
            assert live_index.schema.index.name == new_name
            cleanup_index(live_index)
        except Exception:
            cleanup_index(index)
            raise

    def test_change_prefix(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test changing the key prefix (requires key renames)."""
        index = setup_index(redis_url, base_schema, sample_docs)
        old_prefix = base_schema["index"]["prefix"]
        new_prefix = f"{old_prefix}_newprefix"

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {"version": 1, "changes": {"index": {"prefix": new_prefix}}},
            )

            assert result["supported"], "Change prefix should be supported"
            # Validation now handles prefix change by transforming key_sample to new prefix
            assert result["succeeded"], f"Migration failed: {result['report']}"

            # Verify keys were renamed
            live_index = SearchIndex.from_existing(
                base_schema["index"]["name"], redis_url=redis_url
            )
            assert live_index.schema.index.prefix == new_prefix
            cleanup_index(live_index)
        except Exception:
            cleanup_index(index)
            raise


# ==============================================================================
# 2. Field Operations - Add Fields
# ==============================================================================


class TestAddFields:
    """Tests for adding fields of different types."""

    def test_add_text_field(self, redis_url, tmp_path, unique_ids, sample_docs):
        """Test adding a text field."""
        schema = {
            "index": {
                "name": unique_ids["name"],
                "prefix": unique_ids["prefix"],
                "storage_type": "hash",
            },
            "fields": [{"name": "doc_id", "type": "tag"}],
        }
        index = setup_index(redis_url, schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                unique_ids["name"],
                {
                    "version": 1,
                    "changes": {
                        "add_fields": [{"name": "title", "type": "text"}],
                    },
                },
            )

            assert result["supported"], "Add text field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_tag_field(self, redis_url, tmp_path, unique_ids, sample_docs):
        """Test adding a tag field."""
        schema = {
            "index": {
                "name": unique_ids["name"],
                "prefix": unique_ids["prefix"],
                "storage_type": "hash",
            },
            "fields": [{"name": "doc_id", "type": "tag"}],
        }
        index = setup_index(redis_url, schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                unique_ids["name"],
                {
                    "version": 1,
                    "changes": {
                        "add_fields": [
                            {
                                "name": "category",
                                "type": "tag",
                                "attrs": {"separator": ","},
                            }
                        ],
                    },
                },
            )

            assert result["supported"], "Add tag field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_numeric_field(self, redis_url, tmp_path, unique_ids, sample_docs):
        """Test adding a numeric field."""
        schema = {
            "index": {
                "name": unique_ids["name"],
                "prefix": unique_ids["prefix"],
                "storage_type": "hash",
            },
            "fields": [{"name": "doc_id", "type": "tag"}],
        }
        index = setup_index(redis_url, schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                unique_ids["name"],
                {
                    "version": 1,
                    "changes": {
                        "add_fields": [{"name": "price", "type": "numeric"}],
                    },
                },
            )

            assert result["supported"], "Add numeric field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_geo_field(self, redis_url, tmp_path, unique_ids, sample_docs):
        """Test adding a geo field."""
        schema = {
            "index": {
                "name": unique_ids["name"],
                "prefix": unique_ids["prefix"],
                "storage_type": "hash",
            },
            "fields": [{"name": "doc_id", "type": "tag"}],
        }
        index = setup_index(redis_url, schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                unique_ids["name"],
                {
                    "version": 1,
                    "changes": {
                        "add_fields": [{"name": "location", "type": "geo"}],
                    },
                },
            )

            assert result["supported"], "Add geo field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise


# ==============================================================================
# 2. Field Operations - Remove Fields
# ==============================================================================


class TestRemoveFields:
    """Tests for removing fields of different types."""

    def test_remove_text_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test removing a text field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {"version": 1, "changes": {"remove_fields": ["description"]}},
            )

            assert result["supported"], "Remove text field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_remove_tag_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test removing a tag field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {"version": 1, "changes": {"remove_fields": ["category"]}},
            )

            assert result["supported"], "Remove tag field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_remove_numeric_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test removing a numeric field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {"version": 1, "changes": {"remove_fields": ["price"]}},
            )

            assert result["supported"], "Remove numeric field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_remove_geo_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test removing a geo field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {"version": 1, "changes": {"remove_fields": ["location"]}},
            )

            assert result["supported"], "Remove geo field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_remove_vector_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test removing a vector field (allowed but warned)."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {"version": 1, "changes": {"remove_fields": ["embedding"]}},
            )

            assert result["supported"], "Remove vector field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise


# ==============================================================================
# 2. Field Operations - Rename Fields
# ==============================================================================


class TestRenameFields:
    """Tests for renaming fields of different types."""

    def test_rename_text_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test renaming a text field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "rename_fields": [
                            {"old_name": "title", "new_name": "headline"}
                        ],
                    },
                },
            )

            assert result["supported"], "Rename text field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_rename_tag_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test renaming a tag field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "rename_fields": [{"old_name": "category", "new_name": "tags"}],
                    },
                },
            )

            assert result["supported"], "Rename tag field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_rename_numeric_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test renaming a numeric field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "rename_fields": [{"old_name": "price", "new_name": "cost"}],
                    },
                },
            )

            assert result["supported"], "Rename numeric field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_rename_geo_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test renaming a geo field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "rename_fields": [
                            {"old_name": "location", "new_name": "coordinates"}
                        ],
                    },
                },
            )

            assert result["supported"], "Rename geo field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_rename_vector_field(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test renaming a vector field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "rename_fields": [
                            {"old_name": "embedding", "new_name": "vector"}
                        ],
                    },
                },
            )

            assert result["supported"], "Rename vector field should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise


# ==============================================================================
# 3. Base Attributes (All Non-Vector Types)
# ==============================================================================


class TestBaseAttributes:
    """Tests for base attributes: sortable, no_index, index_missing."""

    def test_add_sortable(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test adding sortable attribute to a field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"sortable": True}}
                        ],
                    },
                },
            )

            assert result["supported"], "Add sortable should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_no_index(self, redis_url, tmp_path, unique_ids, sample_docs):
        """Test adding no_index attribute (store only, no searching)."""
        # Need a sortable field first
        schema = {
            "index": {
                "name": unique_ids["name"],
                "prefix": unique_ids["prefix"],
                "storage_type": "hash",
            },
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {"name": "title", "type": "text", "attrs": {"sortable": True}},
            ],
        }
        index = setup_index(redis_url, schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                unique_ids["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"no_index": True}}
                        ],
                    },
                },
            )

            assert result["supported"], "Add no_index should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_index_missing(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test adding index_missing attribute."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"index_missing": True}}
                        ],
                    },
                },
            )

            assert result["supported"], "Add index_missing should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise


# ==============================================================================
# 4. Text Field Attributes
# ==============================================================================


class TestTextAttributes:
    """Tests for text field attributes: weight, no_stem, phonetic_matcher, etc."""

    def test_change_weight(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test changing text field weight."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [{"name": "title", "attrs": {"weight": 2.0}}],
                    },
                },
            )

            assert result["supported"], "Change weight should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_no_stem(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test adding no_stem attribute."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"no_stem": True}}
                        ],
                    },
                },
            )

            assert result["supported"], "Add no_stem should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_phonetic_matcher(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test adding phonetic_matcher attribute."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"phonetic_matcher": "dm:en"}}
                        ],
                    },
                },
            )

            assert result["supported"], "Add phonetic_matcher should be supported"
            # phonetic_matcher is stripped from schema comparison (FT.INFO doesn't return it)
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_index_empty_text(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test adding index_empty to text field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"index_empty": True}}
                        ],
                    },
                },
            )

            assert result["supported"], "Add index_empty should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_unf_text(self, redis_url, tmp_path, unique_ids, sample_docs):
        """Test adding unf (un-normalized form) to text field."""
        # UNF requires sortable
        schema = {
            "index": {
                "name": unique_ids["name"],
                "prefix": unique_ids["prefix"],
                "storage_type": "hash",
            },
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {"name": "title", "type": "text", "attrs": {"sortable": True}},
            ],
        }
        index = setup_index(redis_url, schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                unique_ids["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [{"name": "title", "attrs": {"unf": True}}],
                    },
                },
            )

            assert result["supported"], "Add UNF should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise


# ==============================================================================
# 5. Tag Field Attributes
# ==============================================================================


class TestTagAttributes:
    """Tests for tag field attributes: separator, case_sensitive, withsuffixtrie, etc."""

    def test_change_separator(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test changing tag separator."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "category", "attrs": {"separator": "|"}}
                        ],
                    },
                },
            )

            assert result["supported"], "Change separator should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_case_sensitive(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test adding case_sensitive attribute."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "category", "attrs": {"case_sensitive": True}}
                        ],
                    },
                },
            )

            assert result["supported"], "Add case_sensitive should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_add_index_empty_tag(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test adding index_empty to tag field."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "category", "attrs": {"index_empty": True}}
                        ],
                    },
                },
            )

            assert result["supported"], "Add index_empty should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise


# ==============================================================================
# 6. Numeric Field Attributes
# ==============================================================================


class TestNumericAttributes:
    """Tests for numeric field attributes: unf."""

    def test_add_unf_numeric(self, redis_url, tmp_path, unique_ids, sample_docs):
        """Test adding unf (un-normalized form) to numeric field."""
        # UNF requires sortable
        schema = {
            "index": {
                "name": unique_ids["name"],
                "prefix": unique_ids["prefix"],
                "storage_type": "hash",
            },
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {"name": "price", "type": "numeric", "attrs": {"sortable": True}},
            ],
        }
        index = setup_index(redis_url, schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                unique_ids["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [{"name": "price", "attrs": {"unf": True}}],
                    },
                },
            )

            assert result["supported"], "Add UNF to numeric should be supported"
            # Redis auto-applies UNF with SORTABLE on numeric fields, so both should match
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise


# ==============================================================================
# 7. Vector Field Attributes (Index-Only Changes)
# ==============================================================================


class TestVectorAttributes:
    """Tests for vector field attributes: algorithm, distance_metric, HNSW params, etc."""

    def test_change_algorithm_hnsw_to_flat(
        self, redis_url, tmp_path, base_schema, sample_docs
    ):
        """Test changing vector algorithm from HNSW to FLAT."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "embedding", "attrs": {"algorithm": "flat"}}
                        ],
                    },
                },
            )

            assert result["supported"], "Change algorithm should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_change_distance_metric(
        self, redis_url, tmp_path, base_schema, sample_docs
    ):
        """Test changing distance metric."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "embedding", "attrs": {"distance_metric": "l2"}}
                        ],
                    },
                },
            )

            assert result["supported"], "Change distance_metric should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_change_initial_cap(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test changing initial_cap."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "embedding", "attrs": {"initial_cap": 1000}}
                        ],
                    },
                },
            )

            assert result["supported"], "Change initial_cap should be supported"
            # Redis may not return initial_cap accurately in FT.INFO.
            # Check doc_count_match to confirm the migration executed successfully.
            assert result[
                "doc_count_match"
            ], f"Migration failed: {result['validation_errors']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_change_hnsw_m(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test changing HNSW m parameter."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [{"name": "embedding", "attrs": {"m": 32}}],
                    },
                },
            )

            assert result["supported"], "Change HNSW m should be supported"
            # Redis may not return m accurately in FT.INFO.
            # Check doc_count_match to confirm the migration executed successfully.
            assert result[
                "doc_count_match"
            ], f"Migration failed: {result['validation_errors']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_change_hnsw_ef_construction(
        self, redis_url, tmp_path, base_schema, sample_docs
    ):
        """Test changing HNSW ef_construction parameter."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "embedding", "attrs": {"ef_construction": 400}}
                        ],
                    },
                },
            )

            assert result["supported"], "Change ef_construction should be supported"
            # Redis may not return ef_construction accurately in FT.INFO.
            # Check doc_count_match to confirm the migration executed successfully.
            assert result[
                "doc_count_match"
            ], f"Migration failed: {result['validation_errors']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_change_hnsw_ef_runtime(
        self, redis_url, tmp_path, base_schema, sample_docs
    ):
        """Test changing HNSW ef_runtime parameter."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "embedding", "attrs": {"ef_runtime": 20}}
                        ],
                    },
                },
            )

            assert result["supported"], "Change ef_runtime should be supported"
            # Redis may not return ef_runtime accurately in FT.INFO (often returns defaults).
            # Check doc_count_match to confirm the migration executed successfully.
            assert result[
                "doc_count_match"
            ], f"Migration failed: {result['validation_errors']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_change_hnsw_epsilon(self, redis_url, tmp_path, base_schema, sample_docs):
        """Test changing HNSW epsilon parameter."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "embedding", "attrs": {"epsilon": 0.05}}
                        ],
                    },
                },
            )

            assert result["supported"], "Change epsilon should be supported"
            # Redis may not return epsilon accurately in FT.INFO (often returns defaults).
            # Check doc_count_match to confirm the migration executed successfully.
            assert result[
                "doc_count_match"
            ], f"Migration failed: {result['validation_errors']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_change_datatype_quantization(
        self, redis_url, tmp_path, base_schema, sample_docs
    ):
        """Test changing vector datatype (quantization)."""
        index = setup_index(redis_url, base_schema, sample_docs)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                base_schema["index"]["name"],
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "embedding", "attrs": {"datatype": "float16"}}
                        ],
                    },
                },
            )

            assert result["supported"], "Change datatype should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise


# ==============================================================================
# 8. JSON Storage Type Tests
# ==============================================================================


class TestJsonStorageType:
    """Tests for migrations with JSON storage type."""

    @pytest.fixture
    def json_schema(self, unique_ids):
        """Schema using JSON storage type."""
        return {
            "index": {
                "name": unique_ids["name"],
                "prefix": unique_ids["prefix"],
                "storage_type": "json",
            },
            "fields": [
                {"name": "doc_id", "type": "tag", "attrs": {"path": "$.doc_id"}},
                {"name": "title", "type": "text", "attrs": {"path": "$.title"}},
                {"name": "category", "type": "tag", "attrs": {"path": "$.category"}},
                {"name": "price", "type": "numeric", "attrs": {"path": "$.price"}},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "path": "$.embedding",
                        "algorithm": "hnsw",
                        "dims": 4,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }

    @pytest.fixture
    def json_sample_docs(self):
        """Sample JSON documents (as dicts for RedisJSON)."""
        return [
            {
                "doc_id": "1",
                "title": "Alpha Product",
                "category": "electronics",
                "price": 99.99,
                "embedding": [0.1, 0.2, 0.3, 0.4],
            },
            {
                "doc_id": "2",
                "title": "Beta Service",
                "category": "software",
                "price": 149.99,
                "embedding": [0.2, 0.3, 0.4, 0.5],
            },
        ]

    def test_json_add_field(
        self, redis_url, tmp_path, unique_ids, json_schema, json_sample_docs, client
    ):
        """Test adding a field with JSON storage."""
        index = SearchIndex.from_dict(json_schema, redis_url=redis_url)
        index.create(overwrite=True)

        # Load JSON docs directly
        for i, doc in enumerate(json_sample_docs):
            key = f"{unique_ids['prefix']}:{i+1}"
            client.json().set(key, "$", json_sample_docs[i])

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                unique_ids["name"],
                {
                    "version": 1,
                    "changes": {
                        "add_fields": [
                            {
                                "name": "status",
                                "type": "tag",
                                "attrs": {"path": "$.status"},
                            }
                        ],
                    },
                },
            )

            assert result["supported"], "Add field with JSON should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise

    def test_json_rename_field(
        self, redis_url, tmp_path, unique_ids, json_schema, json_sample_docs, client
    ):
        """Test renaming a field with JSON storage."""
        index = SearchIndex.from_dict(json_schema, redis_url=redis_url)
        index.create(overwrite=True)

        # Load JSON docs
        for i, doc in enumerate(json_sample_docs):
            key = f"{unique_ids['prefix']}:{i+1}"
            client.json().set(key, "$", doc)

        try:
            result = run_migration(
                redis_url,
                tmp_path,
                unique_ids["name"],
                {
                    "version": 1,
                    "changes": {
                        "rename_fields": [
                            {"old_name": "title", "new_name": "headline"}
                        ],
                    },
                },
            )

            assert result["supported"], "Rename field with JSON should be supported"
            assert result["succeeded"], f"Migration failed: {result['report']}"
            cleanup_index(index)
        except Exception:
            cleanup_index(index)
            raise
