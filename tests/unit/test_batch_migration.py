"""
Unit tests for BatchMigrationPlanner and BatchMigrationExecutor.

Tests use mocked Redis clients to verify:
- Pattern matching and index selection
- Applicability checking
- Checkpoint persistence and resume
- Failure policies
- Progress callbacks
"""

from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from redisvl.migration import (
    BatchMigrationExecutor,
    BatchMigrationPlanner,
    BatchPlan,
    BatchState,
    SchemaPatch,
)
from redisvl.migration.models import BatchIndexEntry, BatchIndexState
from redisvl.schema.schema import IndexSchema

# =============================================================================
# Test Fixtures and Mock Helpers
# =============================================================================


class MockRedisClient:
    """Mock Redis client for batch migration tests."""

    def __init__(self, indexes: List[str] = None, keys: Dict[str, List[str]] = None):
        self.indexes = indexes or []
        self.keys = keys or {}
        self._data: Dict[str, Dict[str, bytes]] = {}

    def execute_command(self, *args, **kwargs):
        if args[0] == "FT._LIST":
            return [idx.encode() for idx in self.indexes]
        raise NotImplementedError(f"Command not mocked: {args}")

    def scan(self, cursor=0, match=None, count=None):
        matched = []
        all_keys = []
        for prefix_keys in self.keys.values():
            all_keys.extend(prefix_keys)

        for key in all_keys:
            decoded_key = key.decode() if isinstance(key, bytes) else str(key)
            if match is None or fnmatch(decoded_key, match):
                matched.append(key if isinstance(key, bytes) else key.encode())
        return 0, matched

    def hget(self, key, field):
        return self._data.get(key, {}).get(field)

    def hset(self, key, field, value):
        if key not in self._data:
            self._data[key] = {}
        self._data[key][field] = value

    def pipeline(self):
        return MockPipeline(self)


class MockPipeline:
    """Mock Redis pipeline."""

    def __init__(self, client: MockRedisClient):
        self._client = client
        self._commands: List[tuple] = []

    def hset(self, key, field, value):
        self._commands.append(("hset", key, field, value))
        return self

    def execute(self):
        results = []
        for cmd in self._commands:
            if cmd[0] == "hset":
                self._client.hset(cmd[1], cmd[2], cmd[3])
                results.append(1)
        self._commands = []
        return results


def make_dummy_index(name: str, schema_dict: Dict[str, Any], stats: Dict[str, Any]):
    """Create a mock SearchIndex for testing."""
    mock_index = Mock()
    mock_index.name = name
    mock_index.schema = IndexSchema.from_dict(schema_dict)
    mock_index._redis_client = MockRedisClient()
    mock_index.client = mock_index._redis_client
    mock_index.info = Mock(return_value=stats)
    mock_index.delete = Mock()
    mock_index.create = Mock()
    mock_index.exists = Mock(return_value=True)
    return mock_index


def make_test_schema(name: str, prefix: str = None, dims: int = 3) -> Dict[str, Any]:
    """Create a test schema dictionary."""
    return {
        "index": {
            "name": name,
            "prefix": prefix or name,
            "key_separator": ":",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "title", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": dims,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }


def make_shared_patch(
    update_fields: List[Dict] = None,
    add_fields: List[Dict] = None,
    remove_fields: List[str] = None,
) -> Dict[str, Any]:
    """Create a test schema patch dictionary."""
    return {
        "version": 1,
        "changes": {
            "update_fields": update_fields or [],
            "add_fields": add_fields or [],
            "remove_fields": remove_fields or [],
            "index": {},
        },
    }


def make_batch_plan(
    batch_id: str,
    indexes: List[BatchIndexEntry],
    failure_policy: str = "fail_fast",
    requires_quantization: bool = False,
) -> BatchPlan:
    """Create a BatchPlan with default values for testing."""
    return BatchPlan(
        batch_id=batch_id,
        shared_patch=SchemaPatch(
            version=1,
            changes={"update_fields": [], "add_fields": [], "remove_fields": []},
        ),
        indexes=indexes,
        requires_quantization=requires_quantization,
        failure_policy=failure_policy,
        created_at="2026-03-20T10:00:00Z",
    )


# =============================================================================
# BatchMigrationPlanner Tests
# =============================================================================


class TestBatchMigrationPlannerPatternMatching:
    """Test pattern matching for index discovery."""

    def test_pattern_matches_multiple_indexes(self, monkeypatch, tmp_path):
        """Pattern should match multiple indexes."""
        mock_client = MockRedisClient(
            indexes=["products_idx", "users_idx", "orders_idx", "logs_idx"]
        )

        def mock_list_indexes(**kwargs):
            return ["products_idx", "users_idx", "orders_idx", "logs_idx"]

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.list_indexes", mock_list_indexes
        )

        # Mock from_existing for each index
        def mock_from_existing(name, **kwargs):
            return make_dummy_index(
                name, make_test_schema(name), {"num_docs": 10, "indexing": False}
            )

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                make_shared_patch(
                    update_fields=[
                        {"name": "embedding", "attrs": {"algorithm": "hnsw"}}
                    ]
                )
            )
        )

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            pattern="*_idx",
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        assert len(batch_plan.indexes) == 4
        assert all(idx.name.endswith("_idx") for idx in batch_plan.indexes)

    def test_pattern_no_matches_raises_error(self, monkeypatch, tmp_path):
        """Empty pattern results should raise ValueError."""
        mock_client = MockRedisClient(indexes=["products", "users"])

        def mock_list_indexes(**kwargs):
            return ["products", "users"]

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.list_indexes", mock_list_indexes
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(yaml.safe_dump(make_shared_patch()))

        planner = BatchMigrationPlanner()
        with pytest.raises(ValueError, match="No indexes found"):
            planner.create_batch_plan(
                pattern="*_idx",  # Won't match anything
                schema_patch_path=str(patch_path),
                redis_client=mock_client,
            )

    def test_pattern_with_special_characters(self, monkeypatch, tmp_path):
        """Pattern matching with special characters in index names."""
        mock_client = MockRedisClient(
            indexes=["app:prod:idx", "app:dev:idx", "app:staging:idx"]
        )

        def mock_list_indexes(**kwargs):
            return ["app:prod:idx", "app:dev:idx", "app:staging:idx"]

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.list_indexes", mock_list_indexes
        )

        def mock_from_existing(name, **kwargs):
            return make_dummy_index(
                name, make_test_schema(name), {"num_docs": 5, "indexing": False}
            )

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(yaml.safe_dump(make_shared_patch()))

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            pattern="app:*:idx",
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        assert len(batch_plan.indexes) == 3


class TestBatchMigrationPlannerIndexSelection:
    """Test explicit index list selection."""

    def test_explicit_index_list(self, monkeypatch, tmp_path):
        """Explicit index list should be used directly."""
        mock_client = MockRedisClient(indexes=["idx1", "idx2", "idx3", "idx4", "idx5"])

        def mock_from_existing(name, **kwargs):
            return make_dummy_index(
                name, make_test_schema(name), {"num_docs": 10, "indexing": False}
            )

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(yaml.safe_dump(make_shared_patch()))

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=["idx1", "idx3", "idx5"],
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        assert len(batch_plan.indexes) == 3
        assert [idx.name for idx in batch_plan.indexes] == ["idx1", "idx3", "idx5"]

    def test_duplicate_index_names(self, monkeypatch, tmp_path):
        """Duplicate index names in list should be preserved (user intent)."""
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        def mock_from_existing(name, **kwargs):
            return make_dummy_index(
                name, make_test_schema(name), {"num_docs": 10, "indexing": False}
            )

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(yaml.safe_dump(make_shared_patch()))

        planner = BatchMigrationPlanner()
        # Duplicates are preserved - user explicitly listed them twice
        batch_plan = planner.create_batch_plan(
            indexes=["idx1", "idx1", "idx2"],
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        assert len(batch_plan.indexes) == 3

    def test_non_existent_index(self, monkeypatch, tmp_path):
        """Non-existent index should be marked as not applicable."""
        mock_client = MockRedisClient(indexes=["idx1"])

        def mock_from_existing(name, **kwargs):
            if name == "idx1":
                return make_dummy_index(
                    name, make_test_schema(name), {"num_docs": 10, "indexing": False}
                )
            raise Exception(f"Index '{name}' not found")

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(yaml.safe_dump(make_shared_patch()))

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=["idx1", "nonexistent"],
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        assert len(batch_plan.indexes) == 2
        assert batch_plan.indexes[0].applicable is True
        assert batch_plan.indexes[1].applicable is False
        assert "not found" in batch_plan.indexes[1].skip_reason.lower()

    def test_indexes_from_file(self, monkeypatch, tmp_path):
        """Load index names from file."""
        mock_client = MockRedisClient(indexes=["idx1", "idx2", "idx3"])

        def mock_from_existing(name, **kwargs):
            return make_dummy_index(
                name, make_test_schema(name), {"num_docs": 10, "indexing": False}
            )

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        # Create indexes file
        indexes_file = tmp_path / "indexes.txt"
        indexes_file.write_text("idx1\n# comment\nidx2\n\nidx3\n")

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(yaml.safe_dump(make_shared_patch()))

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes_file=str(indexes_file),
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        assert len(batch_plan.indexes) == 3
        assert [idx.name for idx in batch_plan.indexes] == ["idx1", "idx2", "idx3"]


class TestBatchMigrationPlannerApplicability:
    """Test applicability checking for shared patches."""

    def test_missing_field_marks_not_applicable(self, monkeypatch, tmp_path):
        """Index missing field in update_fields should be marked not applicable."""
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        def mock_from_existing(name, **kwargs):
            if name == "idx1":
                # Has embedding field
                return make_dummy_index(
                    name, make_test_schema(name), {"num_docs": 10, "indexing": False}
                )
            # idx2 - no embedding field
            schema = {
                "index": {"name": name, "prefix": name, "storage_type": "hash"},
                "fields": [{"name": "title", "type": "text"}],
            }
            return make_dummy_index(name, schema, {"num_docs": 5, "indexing": False})

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                make_shared_patch(
                    update_fields=[
                        {"name": "embedding", "attrs": {"algorithm": "hnsw"}}
                    ]
                )
            )
        )

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=["idx1", "idx2"],
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        idx1_entry = next(e for e in batch_plan.indexes if e.name == "idx1")
        idx2_entry = next(e for e in batch_plan.indexes if e.name == "idx2")

        assert idx1_entry.applicable is True
        assert idx2_entry.applicable is False
        assert "embedding" in idx2_entry.skip_reason.lower()

    def test_field_already_exists_marks_not_applicable(self, monkeypatch, tmp_path):
        """Adding field that already exists should mark not applicable."""
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        def mock_from_existing(name, **kwargs):
            schema = make_test_schema(name)
            # Add 'category' field to idx2
            if name == "idx2":
                schema["fields"].append({"name": "category", "type": "tag"})
            return make_dummy_index(name, schema, {"num_docs": 10, "indexing": False})

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                make_shared_patch(add_fields=[{"name": "category", "type": "tag"}])
            )
        )

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=["idx1", "idx2"],
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        idx1_entry = next(e for e in batch_plan.indexes if e.name == "idx1")
        idx2_entry = next(e for e in batch_plan.indexes if e.name == "idx2")

        assert idx1_entry.applicable is True
        assert idx2_entry.applicable is False
        assert "category" in idx2_entry.skip_reason.lower()

    def test_blocked_change_marks_not_applicable(self, monkeypatch, tmp_path):
        """Blocked changes (e.g., dims change) should mark not applicable."""
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        def mock_from_existing(name, **kwargs):
            dims = 3 if name == "idx1" else 768
            return make_dummy_index(
                name,
                make_test_schema(name, dims=dims),
                {"num_docs": 10, "indexing": False},
            )

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                make_shared_patch(
                    update_fields=[
                        {"name": "embedding", "attrs": {"dims": 1536}}  # Change dims
                    ]
                )
            )
        )

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=["idx1", "idx2"],
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        # Both should be not applicable because dims change is blocked
        for entry in batch_plan.indexes:
            assert entry.applicable is False
            assert "dims" in entry.skip_reason.lower()


class TestBatchMigrationPlannerQuantization:
    """Test quantization detection in batch plans."""

    def test_detects_quantization_required(self, monkeypatch, tmp_path):
        """Batch plan should detect when quantization is required."""
        mock_client = MockRedisClient(indexes=["idx1"])

        def mock_from_existing(name, **kwargs):
            return make_dummy_index(
                name, make_test_schema(name), {"num_docs": 10, "indexing": False}
            )

        monkeypatch.setattr(
            "redisvl.migration.batch_planner.SearchIndex.from_existing",
            mock_from_existing,
        )
        monkeypatch.setattr(
            "redisvl.migration.planner.SearchIndex.from_existing", mock_from_existing
        )

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(
            yaml.safe_dump(
                make_shared_patch(
                    update_fields=[
                        {"name": "embedding", "attrs": {"datatype": "float16"}}
                    ]
                )
            )
        )

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=["idx1"],
            schema_patch_path=str(patch_path),
            redis_client=mock_client,
        )

        assert batch_plan.requires_quantization is True


class TestBatchMigrationPlannerEdgeCases:
    """Test edge cases and error handling."""

    def test_multiple_source_specification_error(self, tmp_path):
        """Should error when multiple source types are specified."""
        mock_client = MockRedisClient(indexes=["idx1"])

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(yaml.safe_dump(make_shared_patch()))

        planner = BatchMigrationPlanner()
        with pytest.raises(ValueError, match="only one of"):
            planner.create_batch_plan(
                indexes=["idx1"],
                pattern="*",  # Can't specify both
                schema_patch_path=str(patch_path),
                redis_client=mock_client,
            )

    def test_no_source_specification_error(self, tmp_path):
        """Should error when no source is specified."""
        mock_client = MockRedisClient(indexes=["idx1"])

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(yaml.safe_dump(make_shared_patch()))

        planner = BatchMigrationPlanner()
        with pytest.raises(ValueError, match="Must provide one of"):
            planner.create_batch_plan(
                schema_patch_path=str(patch_path),
                redis_client=mock_client,
            )

    def test_missing_patch_file_error(self):
        """Should error when patch file doesn't exist."""
        mock_client = MockRedisClient(indexes=["idx1"])

        planner = BatchMigrationPlanner()
        with pytest.raises(FileNotFoundError):
            planner.create_batch_plan(
                indexes=["idx1"],
                schema_patch_path="/nonexistent/patch.yaml",
                redis_client=mock_client,
            )

    def test_missing_indexes_file_error(self, tmp_path):
        """Should error when indexes file doesn't exist."""
        mock_client = MockRedisClient(indexes=["idx1"])

        patch_path = tmp_path / "patch.yaml"
        patch_path.write_text(yaml.safe_dump(make_shared_patch()))

        planner = BatchMigrationPlanner()
        with pytest.raises(FileNotFoundError):
            planner.create_batch_plan(
                indexes_file="/nonexistent/indexes.txt",
                schema_patch_path=str(patch_path),
                redis_client=mock_client,
            )


# =============================================================================
# BatchMigrationExecutor Tests
# =============================================================================


class MockMigrationPlan:
    """Mock migration plan for testing."""

    def __init__(self, index_name: str):
        self.source = Mock()
        self.source.schema_snapshot = make_test_schema(index_name)
        self.merged_target_schema = make_test_schema(index_name)


class MockMigrationReport:
    """Mock migration report for testing."""

    def __init__(self, result: str = "succeeded", errors: List[str] = None):
        self.result = result
        self.validation = Mock(errors=errors or [])

    def model_dump(self, **kwargs):
        return {"result": self.result}


def create_mock_executor(
    succeed_on: List[str] = None,
    fail_on: List[str] = None,
    track_calls: List[str] = None,
):
    """Create a properly configured BatchMigrationExecutor with mocks.

    Args:
        succeed_on: Index names that should succeed.
        fail_on: Index names that should fail.
        track_calls: List to append index names as they're migrated.

    Returns:
        A BatchMigrationExecutor with mocked planner and executor.
    """
    succeed_on = succeed_on or []
    fail_on = fail_on or []
    if track_calls is None:
        track_calls = []

    # Create mock planner
    mock_planner = Mock()

    def create_plan_from_patch(index_name, **kwargs):
        track_calls.append(index_name)
        return MockMigrationPlan(index_name)

    mock_planner.create_plan_from_patch = create_plan_from_patch

    # Create mock executor
    mock_single_executor = Mock()

    def apply(plan, **kwargs):
        # Determine if this should succeed or fail based on tracked calls
        if track_calls:
            last_index = track_calls[-1]
            if last_index in fail_on:
                return MockMigrationReport(
                    result="failed", errors=["Simulated failure"]
                )
        return MockMigrationReport(result="succeeded")

    mock_single_executor.apply = apply

    # Create the batch executor with injected mocks
    batch_executor = BatchMigrationExecutor(executor=mock_single_executor)
    batch_executor._planner = mock_planner

    return batch_executor, track_calls


class TestBatchMigrationExecutorCheckpointing:
    """Test checkpoint persistence and state management."""

    def test_checkpoint_created_at_start(self, tmp_path):
        """Checkpoint state file should be created when migration starts."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-001",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),
            ],
            failure_policy="fail_fast",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        executor, _ = create_mock_executor(succeed_on=["idx1", "idx2"])
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # Verify checkpoint file was created
        assert state_path.exists()
        state_data = yaml.safe_load(state_path.read_text())
        assert state_data["batch_id"] == "test-batch-001"

    def test_checkpoint_updated_after_each_index(self, monkeypatch, tmp_path):
        """Checkpoint should be updated after each index is processed."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-002",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),
                BatchIndexEntry(name="idx3", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"
        checkpoint_snapshots = []

        # Capture checkpoints as they're written
        original_write = BatchMigrationExecutor._write_state

        def capture_checkpoint(self, state, path):
            checkpoint_snapshots.append(
                {"remaining": list(state.remaining), "completed": len(state.completed)}
            )
            return original_write(self, state, path)

        monkeypatch.setattr(BatchMigrationExecutor, "_write_state", capture_checkpoint)

        executor, _ = create_mock_executor(succeed_on=["idx1", "idx2", "idx3"])
        mock_client = MockRedisClient(indexes=["idx1", "idx2", "idx3"])

        executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # Verify checkpoints were written progressively
        # Each index should trigger 2 writes: start and end
        assert len(checkpoint_snapshots) >= 6  # At least 2 per index

    def test_resume_from_checkpoint(self, tmp_path):
        """Resume should continue from where migration left off."""
        # Create a checkpoint state simulating interrupted migration
        batch_plan = make_batch_plan(
            batch_id="test-batch-003",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),
                BatchIndexEntry(name="idx3", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        # Write the batch plan
        plan_path = tmp_path / "batch_plan.yaml"
        with open(plan_path, "w") as f:
            yaml.safe_dump(batch_plan.model_dump(exclude_none=True), f, sort_keys=False)

        # Write a checkpoint state (idx1 completed, idx2 and idx3 remaining)
        state_path = tmp_path / "batch_state.yaml"
        checkpoint_state = BatchState(
            batch_id="test-batch-003",
            plan_path=str(plan_path),
            started_at="2026-03-20T10:00:00Z",
            updated_at="2026-03-20T10:05:00Z",
            remaining=["idx2", "idx3"],
            completed=[
                BatchIndexState(
                    name="idx1",
                    status="succeeded",
                    completed_at="2026-03-20T10:05:00Z",
                )
            ],
            current_index=None,
        )
        with open(state_path, "w") as f:
            yaml.safe_dump(
                checkpoint_state.model_dump(exclude_none=True), f, sort_keys=False
            )

        report_dir = tmp_path / "reports"
        migrated_indexes: List[str] = []

        executor, migrated_indexes = create_mock_executor(
            succeed_on=["idx2", "idx3"],
        )
        mock_client = MockRedisClient(indexes=["idx1", "idx2", "idx3"])

        # Resume from checkpoint
        report = executor.resume(
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # idx1 should NOT be migrated again (already completed)
        assert "idx1" not in migrated_indexes
        # Only idx2 and idx3 should be migrated
        assert migrated_indexes == ["idx2", "idx3"]
        # Report should show all 3 as succeeded
        assert report.summary.successful == 3


class TestBatchMigrationExecutorFailurePolicies:
    """Test failure policy behavior (fail_fast vs continue_on_error)."""

    def test_fail_fast_stops_on_first_error(self, tmp_path):
        """fail_fast policy should stop processing after first failure."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-fail-fast",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),  # This will fail
                BatchIndexEntry(name="idx3", applicable=True),
            ],
            failure_policy="fail_fast",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        executor, migrated_indexes = create_mock_executor(
            succeed_on=["idx1", "idx3"],
            fail_on=["idx2"],
        )
        mock_client = MockRedisClient(indexes=["idx1", "idx2", "idx3"])

        report = executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # idx3 should NOT have been attempted due to fail_fast
        assert "idx3" not in migrated_indexes
        assert migrated_indexes == ["idx1", "idx2"]

        # Report should show partial results
        assert report.summary.successful == 1
        assert report.summary.failed == 1
        assert report.summary.skipped == 1  # idx3 was skipped

    def test_continue_on_error_processes_all(self, tmp_path):
        """continue_on_error policy should process all indexes."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-continue",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),  # This will fail
                BatchIndexEntry(name="idx3", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        executor, migrated_indexes = create_mock_executor(
            succeed_on=["idx1", "idx3"],
            fail_on=["idx2"],
        )
        mock_client = MockRedisClient(indexes=["idx1", "idx2", "idx3"])

        report = executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # ALL indexes should have been attempted
        assert migrated_indexes == ["idx1", "idx2", "idx3"]

        # Report should show mixed results
        assert report.summary.successful == 2  # idx1 and idx3
        assert report.summary.failed == 1  # idx2
        assert report.summary.skipped == 0
        assert report.status == "partial_failure"

    def test_retry_failed_on_resume(self, tmp_path):
        """retry_failed=True should retry previously failed indexes."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-retry",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        plan_path = tmp_path / "batch_plan.yaml"
        with open(plan_path, "w") as f:
            yaml.safe_dump(batch_plan.model_dump(exclude_none=True), f, sort_keys=False)

        # Create checkpoint with idx1 failed
        state_path = tmp_path / "batch_state.yaml"
        checkpoint_state = BatchState(
            batch_id="test-batch-retry",
            plan_path=str(plan_path),
            started_at="2026-03-20T10:00:00Z",
            updated_at="2026-03-20T10:05:00Z",
            remaining=[],  # All "done" but idx1 failed
            completed=[
                BatchIndexState(
                    name="idx1", status="failed", completed_at="2026-03-20T10:03:00Z"
                ),
                BatchIndexState(
                    name="idx2", status="succeeded", completed_at="2026-03-20T10:05:00Z"
                ),
            ],
            current_index=None,
        )
        with open(state_path, "w") as f:
            yaml.safe_dump(
                checkpoint_state.model_dump(exclude_none=True), f, sort_keys=False
            )

        report_dir = tmp_path / "reports"

        executor, migrated_indexes = create_mock_executor(succeed_on=["idx1", "idx2"])
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        report = executor.resume(
            state_path=str(state_path),
            retry_failed=True,
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # idx1 should be retried, idx2 should not (already succeeded)
        assert "idx1" in migrated_indexes
        assert "idx2" not in migrated_indexes
        assert report.summary.successful == 2


class TestBatchMigrationExecutorProgressCallback:
    """Test progress callback functionality."""

    def test_progress_callback_called_for_each_index(self, tmp_path):
        """Progress callback should be invoked for each index."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-progress",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),
                BatchIndexEntry(name="idx3", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"
        progress_events = []

        def progress_callback(index_name, position, total, status):
            progress_events.append(
                {"index": index_name, "pos": position, "total": total, "status": status}
            )

        executor, _ = create_mock_executor(succeed_on=["idx1", "idx2", "idx3"])
        mock_client = MockRedisClient(indexes=["idx1", "idx2", "idx3"])

        executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
            progress_callback=progress_callback,
        )

        # Should have 2 events per index (starting + final status)
        assert len(progress_events) == 6
        # Check first index events
        assert progress_events[0] == {
            "index": "idx1",
            "pos": 1,
            "total": 3,
            "status": "starting",
        }
        assert progress_events[1] == {
            "index": "idx1",
            "pos": 1,
            "total": 3,
            "status": "succeeded",
        }


class TestBatchMigrationExecutorEdgeCases:
    """Test edge cases and error scenarios."""

    def test_exception_during_migration_captured(self, tmp_path):
        """Exception during migration should be captured in state."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-exception",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        # Track calls and raise exception for idx1
        call_count = [0]

        # Create mock planner that raises on idx1
        mock_planner = Mock()

        def create_plan_from_patch(index_name, **kwargs):
            call_count[0] += 1
            if index_name == "idx1":
                raise RuntimeError("Connection lost to Redis")
            return MockMigrationPlan(index_name)

        mock_planner.create_plan_from_patch = create_plan_from_patch

        # Create mock executor
        mock_single_executor = Mock()
        mock_single_executor.apply = Mock(
            return_value=MockMigrationReport(result="succeeded")
        )

        # Create batch executor with mocks
        executor = BatchMigrationExecutor(executor=mock_single_executor)
        executor._planner = mock_planner
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        report = executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # Both should have been attempted
        assert call_count[0] == 2
        # idx1 failed with exception, idx2 succeeded
        assert report.summary.failed == 1
        assert report.summary.successful == 1

        # Check error message is captured
        idx1_report = next(r for r in report.indexes if r.name == "idx1")
        assert "Connection lost" in idx1_report.error

    def test_non_applicable_indexes_skipped(self, tmp_path):
        """Non-applicable indexes should be skipped and reported."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-skip",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(
                    name="idx2",
                    applicable=False,
                    skip_reason="Missing field: embedding",
                ),
                BatchIndexEntry(name="idx3", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        executor, migrated_indexes = create_mock_executor(succeed_on=["idx1", "idx3"])
        mock_client = MockRedisClient(indexes=["idx1", "idx2", "idx3"])

        report = executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # idx2 should NOT be migrated
        assert "idx2" not in migrated_indexes
        assert migrated_indexes == ["idx1", "idx3"]

        # Report should show idx2 as skipped
        assert report.summary.successful == 2
        assert report.summary.skipped == 1

        idx2_report = next(r for r in report.indexes if r.name == "idx2")
        assert idx2_report.status == "skipped"
        assert "Missing field" in idx2_report.error

    def test_empty_batch_plan(self, monkeypatch, tmp_path):
        """Empty batch plan should complete immediately."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-empty",
            indexes=[],  # No indexes
            failure_policy="fail_fast",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        executor = BatchMigrationExecutor()
        mock_client = MockRedisClient(indexes=[])

        report = executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        assert report.status == "completed"
        assert report.summary.total_indexes == 0
        assert report.summary.successful == 0

    def test_missing_redis_connection_error(self, tmp_path):
        """Should error when no Redis connection is provided."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-no-redis",
            indexes=[BatchIndexEntry(name="idx1", applicable=True)],
            failure_policy="fail_fast",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        executor = BatchMigrationExecutor()

        with pytest.raises(ValueError, match="redis"):
            executor.apply(
                batch_plan,
                state_path=str(state_path),
                report_dir=str(report_dir),
                # No redis_url or redis_client provided
            )

    def test_resume_missing_state_file_error(self, tmp_path):
        """Resume should error when state file doesn't exist."""
        executor = BatchMigrationExecutor()
        mock_client = MockRedisClient(indexes=[])

        with pytest.raises(FileNotFoundError, match="State file"):
            executor.resume(
                state_path=str(tmp_path / "nonexistent_state.yaml"),
                report_dir=str(tmp_path / "reports"),
                redis_client=mock_client,
            )

    def test_resume_missing_plan_file_error(self, tmp_path):
        """Resume should error when plan file doesn't exist."""
        # Create state file pointing to nonexistent plan
        state_path = tmp_path / "batch_state.yaml"
        state = BatchState(
            batch_id="test-batch",
            plan_path="/nonexistent/plan.yaml",
            started_at="2026-03-20T10:00:00Z",
            updated_at="2026-03-20T10:05:00Z",
            remaining=["idx1"],
            completed=[],
            current_index=None,
        )
        with open(state_path, "w") as f:
            yaml.safe_dump(state.model_dump(exclude_none=True), f)

        executor = BatchMigrationExecutor()
        mock_client = MockRedisClient(indexes=["idx1"])

        with pytest.raises(FileNotFoundError, match="Batch plan"):
            executor.resume(
                state_path=str(state_path),
                report_dir=str(tmp_path / "reports"),
                redis_client=mock_client,
            )


class TestBatchMigrationExecutorReportGeneration:
    """Test batch report generation."""

    def test_report_contains_all_indexes(self, tmp_path):
        """Final report should contain entries for all indexes."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-report",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(
                    name="idx2", applicable=False, skip_reason="Missing field"
                ),
                BatchIndexEntry(name="idx3", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        executor, _ = create_mock_executor(succeed_on=["idx1", "idx3"])
        mock_client = MockRedisClient(indexes=["idx1", "idx2", "idx3"])

        report = executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # All indexes should be in report
        index_names = {r.name for r in report.indexes}
        assert index_names == {"idx1", "idx2", "idx3"}

        # Verify totals
        assert report.summary.total_indexes == 3
        assert report.summary.successful == 2
        assert report.summary.skipped == 1

    def test_per_index_reports_written(self, tmp_path):
        """Individual reports should be written for each migrated index."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-files",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        executor, _ = create_mock_executor(succeed_on=["idx1", "idx2"])
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        # Report files should exist
        assert (report_dir / "idx1_report.yaml").exists()
        assert (report_dir / "idx2_report.yaml").exists()

    def test_completed_status_when_all_succeed(self, tmp_path):
        """Status should be 'completed' when all indexes succeed."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-complete",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        executor, _ = create_mock_executor(succeed_on=["idx1", "idx2"])
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        report = executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        assert report.status == "completed"

    def test_failed_status_when_all_fail(self, tmp_path):
        """Status should be 'failed' when all indexes fail."""
        batch_plan = make_batch_plan(
            batch_id="test-batch-all-fail",
            indexes=[
                BatchIndexEntry(name="idx1", applicable=True),
                BatchIndexEntry(name="idx2", applicable=True),
            ],
            failure_policy="continue_on_error",
        )

        state_path = tmp_path / "batch_state.yaml"
        report_dir = tmp_path / "reports"

        # Create a mock that raises exceptions for all indexes
        mock_planner = Mock()
        mock_planner.create_plan_from_patch = Mock(
            side_effect=RuntimeError("All migrations fail")
        )

        mock_single_executor = Mock()
        executor = BatchMigrationExecutor(executor=mock_single_executor)
        executor._planner = mock_planner
        mock_client = MockRedisClient(indexes=["idx1", "idx2"])

        report = executor.apply(
            batch_plan,
            state_path=str(state_path),
            report_dir=str(report_dir),
            redis_client=mock_client,
        )

        assert report.status == "failed"
        assert report.summary.failed == 2
        assert report.summary.successful == 0
