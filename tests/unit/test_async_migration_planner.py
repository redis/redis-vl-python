"""Unit tests for AsyncMigrationPlanner.

These tests mirror the sync MigrationPlanner tests but use async/await patterns.
"""

from fnmatch import fnmatch

import pytest
import yaml

from redisvl.migration import AsyncMigrationPlanner, MigrationPlanner
from redisvl.schema.schema import IndexSchema


class AsyncDummyClient:
    """Async mock Redis client for testing."""

    def __init__(self, keys):
        self.keys = keys

    async def scan(self, cursor=0, match=None, count=None):
        matched = []
        for key in self.keys:
            decoded_key = key.decode() if isinstance(key, bytes) else str(key)
            if match is None or fnmatch(decoded_key, match):
                matched.append(key)
        return 0, matched


class AsyncDummyIndex:
    """Async mock SearchIndex for testing."""

    def __init__(self, schema, stats, keys):
        self.schema = schema
        self._stats = stats
        self._client = AsyncDummyClient(keys)

    @property
    def client(self):
        return self._client

    async def info(self):
        return self._stats


def _make_source_schema():
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "docs",
                "prefix": "docs",
                "key_separator": ":",
                "storage_type": "json",
            },
            "fields": [
                {
                    "name": "title",
                    "type": "text",
                    "path": "$.title",
                    "attrs": {"sortable": False},
                },
                {
                    "name": "price",
                    "type": "numeric",
                    "path": "$.price",
                    "attrs": {"sortable": True},
                },
                {
                    "name": "embedding",
                    "type": "vector",
                    "path": "$.embedding",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


@pytest.mark.asyncio
async def test_async_create_plan_from_schema_patch(monkeypatch, tmp_path):
    """Test async planner creates valid plan from schema patch."""
    source_schema = _make_source_schema()
    dummy_index = AsyncDummyIndex(
        source_schema,
        {"num_docs": 2, "indexing": False},
        [b"docs:1", b"docs:2", b"docs:3"],
    )

    async def mock_from_existing(*args, **kwargs):
        return dummy_index

    monkeypatch.setattr(
        "redisvl.migration.async_planner.AsyncSearchIndex.from_existing",
        mock_from_existing,
    )

    patch_path = tmp_path / "schema_patch.yaml"
    patch_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "changes": {
                    "add_fields": [
                        {
                            "name": "category",
                            "type": "tag",
                            "path": "$.category",
                            "attrs": {"separator": ","},
                        }
                    ],
                    "remove_fields": ["price"],
                    "update_fields": [
                        {
                            "name": "title",
                            "options": {"sortable": True},
                        }
                    ],
                },
            },
            sort_keys=False,
        )
    )

    planner = AsyncMigrationPlanner(key_sample_limit=2)
    plan = await planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        schema_patch_path=str(patch_path),
    )

    assert plan.diff_classification.supported is True
    assert plan.source.index_name == "docs"
    assert plan.source.keyspace.storage_type == "json"
    assert plan.source.keyspace.prefixes == ["docs"]
    assert plan.source.keyspace.key_separator == ":"
    assert plan.source.keyspace.key_sample == ["docs:1", "docs:2"]
    assert plan.warnings == ["Index downtime is required"]

    merged_fields = {
        field["name"]: field for field in plan.merged_target_schema["fields"]
    }
    assert plan.merged_target_schema["index"]["prefix"] == "docs"
    assert merged_fields["title"]["attrs"]["sortable"] is True
    assert "price" not in merged_fields
    assert merged_fields["category"]["type"] == "tag"

    # Test write_plan works (delegates to sync)
    plan_path = tmp_path / "migration_plan.yaml"
    planner.write_plan(plan, str(plan_path))
    written_plan = yaml.safe_load(plan_path.read_text())
    assert written_plan["mode"] == "drop_recreate"
    assert written_plan["diff_classification"]["supported"] is True


@pytest.mark.asyncio
async def test_async_planner_datatype_change_allowed(monkeypatch, tmp_path):
    """Changing vector datatype (quantization) is allowed - executor will re-encode."""
    source_schema = _make_source_schema()
    dummy_index = AsyncDummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])

    async def mock_from_existing(*args, **kwargs):
        return dummy_index

    monkeypatch.setattr(
        "redisvl.migration.async_planner.AsyncSearchIndex.from_existing",
        mock_from_existing,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {"name": "title", "type": "text", "path": "$.title"},
                    {"name": "price", "type": "numeric", "path": "$.price"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float16",  # Changed from float32
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = AsyncMigrationPlanner()
    plan = await planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is True
    assert len(plan.diff_classification.blocked_reasons) == 0

    # Verify datatype changes are detected
    datatype_changes = MigrationPlanner.get_vector_datatype_changes(
        plan.source.schema_snapshot, plan.merged_target_schema
    )
    assert "embedding" in datatype_changes
    assert datatype_changes["embedding"]["source"] == "float32"
    assert datatype_changes["embedding"]["target"] == "float16"


@pytest.mark.asyncio
async def test_async_planner_algorithm_change_allowed(monkeypatch, tmp_path):
    """Changing vector algorithm is allowed (index-only change)."""
    source_schema = _make_source_schema()
    dummy_index = AsyncDummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])

    async def mock_from_existing(*args, **kwargs):
        return dummy_index

    monkeypatch.setattr(
        "redisvl.migration.async_planner.AsyncSearchIndex.from_existing",
        mock_from_existing,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {"name": "title", "type": "text", "path": "$.title"},
                    {"name": "price", "type": "numeric", "path": "$.price"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "hnsw",  # Changed from flat
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = AsyncMigrationPlanner()
    plan = await planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is True
    assert len(plan.diff_classification.blocked_reasons) == 0


@pytest.mark.asyncio
async def test_async_planner_prefix_change_is_supported(monkeypatch, tmp_path):
    """Prefix change is supported: executor will rename keys."""
    source_schema = _make_source_schema()
    dummy_index = AsyncDummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])

    async def mock_from_existing(*args, **kwargs):
        return dummy_index

    monkeypatch.setattr(
        "redisvl.migration.async_planner.AsyncSearchIndex.from_existing",
        mock_from_existing,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs_v2",  # Changed prefix
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": source_schema.to_dict()["fields"],
            },
            sort_keys=False,
        )
    )

    planner = AsyncMigrationPlanner()
    plan = await planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    # Prefix change is now supported
    assert plan.diff_classification.supported is True
    assert plan.rename_operations.change_prefix == "docs_v2"
    # Should have a warning about key renaming
    assert any("prefix" in w.lower() for w in plan.warnings)
