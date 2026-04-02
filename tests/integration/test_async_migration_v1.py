"""Integration tests for async migration (Phase 1.5).

These tests verify the async migration components work correctly with a real
Redis instance, mirroring the sync tests in test_migration_v1.py.
"""

import uuid

import pytest
import yaml

from redisvl.index import AsyncSearchIndex
from redisvl.migration import (
    AsyncMigrationExecutor,
    AsyncMigrationPlanner,
    AsyncMigrationValidator,
)
from redisvl.migration.utils import load_migration_plan, schemas_equal
from redisvl.redis.utils import array_to_buffer


@pytest.mark.asyncio
async def test_async_drop_recreate_plan_apply_validate_flow(
    redis_url, worker_id, tmp_path
):
    """Test full async migration flow: plan -> apply -> validate."""
    unique_id = str(uuid.uuid4())[:8]
    index_name = f"async_migration_v1_{worker_id}_{unique_id}"
    prefix = f"async_migration_v1:{worker_id}:{unique_id}"

    source_index = AsyncSearchIndex.from_dict(
        {
            "index": {
                "name": index_name,
                "prefix": prefix,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {"name": "title", "type": "text"},
                {"name": "price", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        },
        redis_url=redis_url,
    )

    docs = [
        {
            "doc_id": "1",
            "title": "alpha",
            "price": 1,
            "category": "news",
            "embedding": array_to_buffer([0.1, 0.2, 0.3], "float32"),
        },
        {
            "doc_id": "2",
            "title": "beta",
            "price": 2,
            "category": "sports",
            "embedding": array_to_buffer([0.2, 0.1, 0.4], "float32"),
        },
    ]

    await source_index.create(overwrite=True)
    await source_index.load(docs, id_field="doc_id")

    # Create schema patch
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
                            "attrs": {"separator": ","},
                        }
                    ],
                    "remove_fields": ["price"],
                    "update_fields": [{"name": "title", "attrs": {"sortable": True}}],
                },
            },
            sort_keys=False,
        )
    )

    # Create plan using async planner
    plan_path = tmp_path / "migration_plan.yaml"
    planner = AsyncMigrationPlanner()
    plan = await planner.create_plan(
        index_name,
        redis_url=redis_url,
        schema_patch_path=str(patch_path),
    )
    assert plan.diff_classification.supported is True
    planner.write_plan(plan, str(plan_path))

    # Create query checks
    query_check_path = tmp_path / "query_checks.yaml"
    query_check_path.write_text(
        yaml.safe_dump({"fetch_ids": ["1", "2"]}, sort_keys=False)
    )

    # Apply migration using async executor
    executor = AsyncMigrationExecutor()
    report = await executor.apply(
        load_migration_plan(str(plan_path)),
        redis_url=redis_url,
        query_check_file=str(query_check_path),
    )

    # Verify migration succeeded
    assert report.result == "succeeded"
    assert report.validation.schema_match is True
    assert report.validation.doc_count_match is True
    assert report.validation.key_sample_exists is True
    assert report.validation.indexing_failures_delta == 0
    assert not report.validation.errors
    assert report.benchmark_summary.documents_indexed_per_second is not None

    # Verify schema matches target
    live_index = await AsyncSearchIndex.from_existing(index_name, redis_url=redis_url)
    assert schemas_equal(live_index.schema.to_dict(), plan.merged_target_schema)

    # Test standalone async validator
    validator = AsyncMigrationValidator()
    validation, _target_info, _duration = await validator.validate(
        load_migration_plan(str(plan_path)),
        redis_url=redis_url,
        query_check_file=str(query_check_path),
    )
    assert validation.schema_match is True
    assert validation.doc_count_match is True
    assert validation.key_sample_exists is True
    assert not validation.errors

    # Cleanup
    await live_index.delete(drop=True)
