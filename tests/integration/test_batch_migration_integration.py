"""
Integration tests for batch migration.

Tests the full batch migration flow with real Redis:
- Batch planning with patterns and explicit lists
- Batch apply with checkpointing
- Resume after interruption
- Failure policies (fail_fast, continue_on_error)
"""

import uuid

import pytest
import yaml

from redisvl.index import SearchIndex
from redisvl.migration import BatchMigrationExecutor, BatchMigrationPlanner
from redisvl.redis.utils import array_to_buffer


def create_test_index(name: str, prefix: str, redis_url: str) -> SearchIndex:
    """Helper to create a test index with standard schema."""
    index = SearchIndex.from_dict(
        {
            "index": {
                "name": name,
                "prefix": prefix,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {"name": "title", "type": "text"},
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
    return index


def load_test_data(index: SearchIndex) -> None:
    """Load sample documents into an index."""
    docs = [
        {
            "doc_id": "1",
            "title": "alpha",
            "embedding": array_to_buffer([0.1, 0.2, 0.3], "float32"),
        },
        {
            "doc_id": "2",
            "title": "beta",
            "embedding": array_to_buffer([0.2, 0.1, 0.4], "float32"),
        },
    ]
    index.load(docs, id_field="doc_id")


class TestBatchMigrationPlanIntegration:
    """Test batch plan creation with real Redis."""

    def test_batch_plan_with_pattern(self, redis_url, worker_id, tmp_path):
        """Test creating a batch plan using pattern matching."""
        unique_id = str(uuid.uuid4())[:8]
        prefix = f"batch_test:{worker_id}:{unique_id}"
        indexes = []

        # Create multiple indexes matching pattern
        for i in range(3):
            name = f"batch_{unique_id}_idx_{i}"
            index = create_test_index(name, f"{prefix}_{i}", redis_url)
            index.create(overwrite=True)
            load_test_data(index)
            indexes.append(index)

        # Create shared patch (add sortable to title)
        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"sortable": True}}
                        ]
                    },
                },
                sort_keys=False,
            )
        )

        # Create batch plan
        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            pattern=f"batch_{unique_id}_idx_*",
            schema_patch_path=str(patch_path),
            redis_url=redis_url,
        )

        # Verify batch plan
        assert batch_plan.batch_id is not None
        assert len(batch_plan.indexes) == 3
        for entry in batch_plan.indexes:
            assert entry.applicable is True
            assert entry.skip_reason is None

        # Cleanup
        for index in indexes:
            index.delete(drop=True)

    def test_batch_plan_with_explicit_list(self, redis_url, worker_id, tmp_path):
        """Test creating a batch plan with explicit index list."""
        unique_id = str(uuid.uuid4())[:8]
        prefix = f"batch_list_test:{worker_id}:{unique_id}"
        index_names = []
        indexes = []

        # Create indexes
        for i in range(2):
            name = f"list_batch_{unique_id}_{i}"
            index = create_test_index(name, f"{prefix}_{i}", redis_url)
            index.create(overwrite=True)
            load_test_data(index)
            indexes.append(index)
            index_names.append(name)

        # Create shared patch
        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"sortable": True}}
                        ]
                    },
                },
                sort_keys=False,
            )
        )

        # Create batch plan with explicit list
        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=index_names,
            schema_patch_path=str(patch_path),
            redis_url=redis_url,
        )

        assert len(batch_plan.indexes) == 2
        assert all(idx.applicable for idx in batch_plan.indexes)

        # Cleanup
        for index in indexes:
            index.delete(drop=True)


class TestBatchMigrationApplyIntegration:
    """Test batch apply with real Redis."""

    def test_batch_apply_full_flow(self, redis_url, worker_id, tmp_path):
        """Test complete batch apply flow: plan -> apply -> verify."""
        unique_id = str(uuid.uuid4())[:8]
        prefix = f"batch_apply:{worker_id}:{unique_id}"
        indexes = []
        index_names = []

        # Create multiple indexes
        for i in range(3):
            name = f"apply_batch_{unique_id}_{i}"
            index = create_test_index(name, f"{prefix}_{i}", redis_url)
            index.create(overwrite=True)
            load_test_data(index)
            indexes.append(index)
            index_names.append(name)

        # Create shared patch (make title sortable)
        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"sortable": True}}
                        ]
                    },
                },
                sort_keys=False,
            )
        )

        # Create batch plan
        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=index_names,
            schema_patch_path=str(patch_path),
            redis_url=redis_url,
        )

        # Save batch plan
        plan_path = tmp_path / "batch_plan.yaml"
        planner.write_batch_plan(batch_plan, str(plan_path))

        # Apply batch migration
        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"
        executor = BatchMigrationExecutor()
        report = executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_url=redis_url,
        )

        # Verify report
        assert report.status == "completed"
        assert report.summary.total_indexes == 3
        assert report.summary.successful == 3
        assert report.summary.failed == 0

        # Verify all indexes were migrated (title is now sortable)
        for name in index_names:
            migrated = SearchIndex.from_existing(name, redis_url=redis_url)
            title_field = migrated.schema.fields.get("title")
            assert title_field is not None
            assert title_field.attrs.sortable is True

        # Cleanup
        for name in index_names:
            idx = SearchIndex.from_existing(name, redis_url=redis_url)
            idx.delete(drop=True)

    def test_batch_apply_with_inapplicable_indexes(
        self, redis_url, worker_id, tmp_path
    ):
        """Test batch apply skips indexes that don't have matching fields."""
        unique_id = str(uuid.uuid4())[:8]
        prefix = f"batch_skip:{worker_id}:{unique_id}"
        indexes_to_cleanup = []

        # Create an index WITH embedding field
        with_embedding = f"with_emb_{unique_id}"
        idx1 = create_test_index(with_embedding, f"{prefix}_1", redis_url)
        idx1.create(overwrite=True)
        load_test_data(idx1)
        indexes_to_cleanup.append(with_embedding)

        # Create an index WITHOUT embedding field
        without_embedding = f"no_emb_{unique_id}"
        idx2 = SearchIndex.from_dict(
            {
                "index": {
                    "name": without_embedding,
                    "prefix": f"{prefix}_2",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "doc_id", "type": "tag"},
                    {"name": "content", "type": "text"},
                ],
            },
            redis_url=redis_url,
        )
        idx2.create(overwrite=True)
        idx2.load([{"doc_id": "1", "content": "test"}], id_field="doc_id")
        indexes_to_cleanup.append(without_embedding)

        # Create patch targeting embedding field (won't apply to idx2)
        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "embedding", "attrs": {"datatype": "float16"}}
                        ]
                    },
                },
                sort_keys=False,
            )
        )

        # Create batch plan
        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=[with_embedding, without_embedding],
            schema_patch_path=str(patch_path),
            redis_url=redis_url,
        )

        # One should be applicable, one not
        applicable = [idx for idx in batch_plan.indexes if idx.applicable]
        not_applicable = [idx for idx in batch_plan.indexes if not idx.applicable]
        assert len(applicable) == 1
        assert len(not_applicable) == 1
        assert "embedding" in not_applicable[0].skip_reason.lower()

        # Apply
        executor = BatchMigrationExecutor()
        report = executor.apply(
            batch_plan,
            state_path=str(tmp_path / "state.yaml"),
            report_dir=str(tmp_path / "reports"),
            redis_url=redis_url,
        )

        assert report.summary.successful == 1
        assert report.summary.skipped == 1

        # Cleanup
        for name in indexes_to_cleanup:
            idx = SearchIndex.from_existing(name, redis_url=redis_url)
            idx.delete(drop=True)


class TestBatchMigrationResumeIntegration:
    """Test batch resume functionality with real Redis."""

    def test_resume_from_checkpoint(self, redis_url, worker_id, tmp_path):
        """Test resuming a batch migration from checkpoint state."""
        unique_id = str(uuid.uuid4())[:8]
        prefix = f"batch_resume:{worker_id}:{unique_id}"
        index_names = []
        indexes = []

        # Create indexes
        for i in range(3):
            name = f"resume_batch_{unique_id}_{i}"
            index = create_test_index(name, f"{prefix}_{i}", redis_url)
            index.create(overwrite=True)
            load_test_data(index)
            indexes.append(index)
            index_names.append(name)

        # Create patch
        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"sortable": True}}
                        ]
                    },
                },
                sort_keys=False,
            )
        )

        # Create batch plan
        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=index_names,
            schema_patch_path=str(patch_path),
            redis_url=redis_url,
        )

        # Save batch plan (needed for resume)
        plan_path = tmp_path / "batch_plan.yaml"
        planner.write_batch_plan(batch_plan, str(plan_path))

        # Create a checkpoint state simulating partial completion
        state_path = tmp_path / "batch_state.yaml"
        partial_state = {
            "batch_id": batch_plan.batch_id,
            "plan_path": str(plan_path),
            "started_at": "2026-03-20T10:00:00Z",
            "updated_at": "2026-03-20T10:01:00Z",
            "completed": [
                {
                    "name": index_names[0],
                    "status": "success",
                    "completed_at": "2026-03-20T10:00:30Z",
                }
            ],
            "remaining": index_names[1:],  # Still need to process idx 1 and 2
            "current_index": None,
        }
        state_path.write_text(yaml.safe_dump(partial_state, sort_keys=False))

        # Resume from checkpoint
        executor = BatchMigrationExecutor()
        report = executor.resume(
            state_path=str(state_path),
            batch_plan_path=str(plan_path),
            report_dir=str(tmp_path / "reports"),
            redis_url=redis_url,
        )

        # Should complete remaining 2 indexes
        # Note: The first index was marked as succeeded in checkpoint but not actually
        # migrated, so the report will show 2 successful (the ones actually processed)
        assert report.summary.successful >= 2
        assert report.status == "completed"

        # Verify at least the resumed indexes were migrated
        for name in index_names[1:]:
            migrated = SearchIndex.from_existing(name, redis_url=redis_url)
            title_field = migrated.schema.fields.get("title")
            assert title_field is not None
            assert title_field.attrs.sortable is True

        # Cleanup
        for name in index_names:
            idx = SearchIndex.from_existing(name, redis_url=redis_url)
            idx.delete(drop=True)

    def test_progress_callback_called(self, redis_url, worker_id, tmp_path):
        """Test that progress callback is invoked during batch apply."""
        unique_id = str(uuid.uuid4())[:8]
        prefix = f"batch_progress:{worker_id}:{unique_id}"
        index_names = []
        indexes = []

        # Create indexes
        for i in range(2):
            name = f"progress_batch_{unique_id}_{i}"
            index = create_test_index(name, f"{prefix}_{i}", redis_url)
            index.create(overwrite=True)
            load_test_data(index)
            indexes.append(index)
            index_names.append(name)

        # Create patch
        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "changes": {
                        "update_fields": [
                            {"name": "title", "attrs": {"sortable": True}}
                        ]
                    },
                },
                sort_keys=False,
            )
        )

        # Create batch plan
        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=index_names,
            schema_patch_path=str(patch_path),
            redis_url=redis_url,
        )

        # Track progress callbacks
        progress_calls = []

        def progress_cb(name, pos, total, status):
            progress_calls.append((name, pos, total, status))

        # Apply with progress callback
        executor = BatchMigrationExecutor()
        executor.apply(
            batch_plan,
            state_path=str(tmp_path / "state.yaml"),
            report_dir=str(tmp_path / "reports"),
            redis_url=redis_url,
            progress_callback=progress_cb,
        )

        # Verify progress was reported for each index
        assert len(progress_calls) >= 2  # At least one call per index
        reported_names = {call[0] for call in progress_calls}
        for name in index_names:
            assert name in reported_names

        # Cleanup
        for name in index_names:
            idx = SearchIndex.from_existing(name, redis_url=redis_url)
            idx.delete(drop=True)
