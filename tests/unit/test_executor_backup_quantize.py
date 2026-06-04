"""Tests for the new two-phase quantize flow in MigrationExecutor.

Verifies:
  - dump_vectors: pipeline-reads originals, writes to backup file
  - quantize_from_backup: reads backup file, converts, pipeline-writes
  - Resume: reloads backup file, skips completed batches
  - BGSAVE is NOT called
"""

import struct
from unittest.mock import MagicMock, patch

import pytest

from redisvl.migration.models import (
    DiffClassification,
    KeyspaceSnapshot,
    MigrationPlan,
    MigrationValidation,
    RenameOperations,
    SourceSnapshot,
)


def _make_float32_vector(dims: int = 4, seed: float = 0.0) -> bytes:
    return struct.pack(f"<{dims}f", *[seed + i for i in range(dims)])


def _make_migration_plan(
    *,
    storage_type: str = "hash",
    source_dtype: str = "float32",
    target_dtype: str = "float16",
    change_prefix: str | None = None,
) -> MigrationPlan:
    source_prefix = "doc:"
    target_prefix = change_prefix or source_prefix
    source_schema = {
        "index": {
            "name": "idx",
            "prefix": source_prefix,
            "storage_type": storage_type,
        },
        "fields": [
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 4,
                    "distance_metric": "cosine",
                    "datatype": source_dtype,
                },
            }
        ],
    }
    target_schema = {
        "index": {
            "name": "idx",
            "prefix": target_prefix,
            "storage_type": storage_type,
        },
        "fields": [
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 4,
                    "distance_metric": "cosine",
                    "datatype": target_dtype,
                },
            }
        ],
    }
    return MigrationPlan(
        source=SourceSnapshot(
            index_name="idx",
            schema_snapshot=source_schema,
            stats_snapshot={"num_docs": 2},
            keyspace=KeyspaceSnapshot(
                storage_type=storage_type,
                prefixes=[source_prefix],
                key_separator=":",
                key_sample=["doc:1"],
            ),
        ),
        requested_changes={},
        merged_target_schema=target_schema,
        diff_classification=DiffClassification(supported=True),
        rename_operations=RenameOperations(change_prefix=change_prefix),
    )


def _successful_validation():
    return (
        MigrationValidation(
            schema_match=True,
            doc_count_match=True,
            key_sample_exists=True,
        ),
        {"num_docs": 2, "vector_index_sz_mb": 1},
        0.01,
    )


class TestDumpVectors:
    """Test Phase 1: dumping original vectors to backup file."""

    def test_dump_creates_backup_and_reads_via_pipeline(self, tmp_path):
        from redisvl.migration.executor import MigrationExecutor

        executor = MigrationExecutor()
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        dims = 4
        keys = [f"doc:{i}" for i in range(6)]
        vec = _make_float32_vector(dims)
        # 6 keys × 1 field = 6 results per execute
        mock_pipe.execute.return_value = [vec] * 6

        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": dims}
        }

        backup_path = str(tmp_path / "test_backup")
        backup = executor._dump_vectors(
            client=mock_client,
            index_name="myindex",
            keys=keys,
            datatype_changes=datatype_changes,
            backup_path=backup_path,
            batch_size=3,
        )

        # Should use pipeline reads, not individual hget
        mock_client.hget.assert_not_called()
        # 2 batches of 3 keys = 2 pipeline.execute() calls
        assert mock_pipe.execute.call_count == 2
        # Backup file created and dump complete
        assert backup.header.phase == "ready"
        assert backup.header.dump_completed_batches == 2
        # All data readable
        batches = list(backup.iter_batches())
        assert len(batches) == 2
        assert len(batches[0][0]) == 3  # first batch has 3 keys
        assert len(batches[1][0]) == 3  # second batch has 3 keys


class TestQuantizeFromBackup:
    """Test Phase 2: reading from backup, converting, writing to Redis."""

    def _create_dumped_backup(self, tmp_path, num_keys=4, dims=4, batch_size=2):
        from redisvl.migration.backup import VectorBackup

        backup_path = str(tmp_path / "test_backup")
        backup = VectorBackup.create(
            path=backup_path,
            index_name="myindex",
            fields={
                "embedding": {"source": "float32", "target": "float16", "dims": dims}
            },
            batch_size=batch_size,
        )
        for batch_idx in range(num_keys // batch_size):
            start = batch_idx * batch_size
            keys = [f"doc:{j}" for j in range(start, start + batch_size)]
            vec = _make_float32_vector(dims)
            originals = {k: {"embedding": vec} for k in keys}
            backup.write_batch(batch_idx, keys, originals)
        backup.mark_dump_complete()
        return backup

    def test_quantize_writes_converted_via_pipeline(self, tmp_path):
        from redisvl.migration.executor import MigrationExecutor

        executor = MigrationExecutor()
        backup = self._create_dumped_backup(tmp_path)

        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": 4}
        }

        docs = executor._quantize_from_backup(
            client=mock_client,
            backup=backup,
            datatype_changes=datatype_changes,
        )

        # Should write via pipeline, not individual hset
        mock_client.hset.assert_not_called()
        # 2 batches = 2 pipeline.execute() calls
        assert mock_pipe.execute.call_count == 2
        # Each batch has 2 keys × 1 field = 2 hset calls per batch
        assert mock_pipe.hset.call_count == 4
        # 4 docs quantized
        assert docs == 4
        # Backup should be marked complete
        assert backup.header.phase == "completed"

    def test_quantize_writes_correct_float16_data(self, tmp_path):
        from redisvl.migration.executor import MigrationExecutor

        executor = MigrationExecutor()
        backup = self._create_dumped_backup(tmp_path, num_keys=2, batch_size=2)

        written_data = {}
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        def capture_hset(key, field, value):
            written_data[key] = {field: value}

        mock_pipe.hset.side_effect = capture_hset

        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": 4}
        }

        executor._quantize_from_backup(
            client=mock_client,
            backup=backup,
            datatype_changes=datatype_changes,
        )

        # Verify written data is float16 (2 bytes per dim = 8 bytes total)
        for key, fields in written_data.items():
            assert len(fields["embedding"]) == 4 * 2  # dims * sizeof(float16)

    def test_quantize_maps_backup_keys_to_live_prefix(self, tmp_path):
        from redisvl.migration.executor import MigrationExecutor

        executor = MigrationExecutor()
        backup = self._create_dumped_backup(tmp_path, num_keys=2, batch_size=2)

        written_keys = []
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        def capture_hset(key, field, value):
            written_keys.append(key)

        mock_pipe.hset.side_effect = capture_hset

        executor._quantize_from_backup(
            client=mock_client,
            backup=backup,
            datatype_changes={
                "embedding": {"source": "float32", "target": "float16", "dims": 4}
            },
            key_transform=lambda key: key.replace("doc:", "new:", 1),
        )

        assert written_keys == ["new:0", "new:1"]


class TestQuantizeResume:
    """Test resume after crash during quantize phase."""

    def test_resume_skips_completed_batches(self, tmp_path):
        from redisvl.migration.backup import VectorBackup
        from redisvl.migration.executor import MigrationExecutor

        # Create backup with 4 batches, mark 2 as quantized
        backup_path = str(tmp_path / "test_backup")
        backup = VectorBackup.create(
            path=backup_path,
            index_name="myindex",
            fields={"embedding": {"source": "float32", "target": "float16", "dims": 4}},
            batch_size=2,
        )
        vec = _make_float32_vector(4)
        for batch_idx in range(4):
            keys = [f"doc:{batch_idx * 2}", f"doc:{batch_idx * 2 + 1}"]
            backup.write_batch(batch_idx, keys, {k: {"embedding": vec} for k in keys})
        backup.mark_dump_complete()
        backup.start_quantize()
        backup.mark_batch_quantized(0)
        backup.mark_batch_quantized(1)
        # Simulate crash — save and reload
        del backup
        backup = VectorBackup.load(backup_path)

        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        executor = MigrationExecutor()
        docs = executor._quantize_from_backup(
            client=mock_client,
            backup=backup,
            datatype_changes={
                "embedding": {"source": "float32", "target": "float16", "dims": 4}
            },
        )

        # Only 2 remaining batches × 2 keys = 4 docs, but should only process 2 batches
        assert mock_pipe.execute.call_count == 2
        assert mock_pipe.hset.call_count == 4  # 2 batches × 2 keys
        assert docs == 4


class TestBackupKeyMapping:
    def test_key_prefix_mapping_persists_in_header(self, tmp_path):
        from redisvl.migration.backup import VectorBackup

        backup_path = str(tmp_path / "test_backup")
        VectorBackup.create(
            path=backup_path,
            index_name="idx",
            fields={"embedding": {"source": "float32", "target": "float16"}},
            batch_size=2,
            key_prefix={"source": "old:", "target": "new:"},
        )

        loaded = VectorBackup.load(backup_path)

        assert loaded is not None
        assert loaded.map_key("old:1") == "new:1"
        assert loaded.map_key("other:1") == "other:1"


class TestMandatoryBackupEnforcement:
    """Test that every migration apply requires a backup directory."""

    def test_none_backup_dir_raises(self):
        """Passing backup_dir=None must raise before migration starts."""
        from redisvl.migration.executor import _require_backup_dir

        with pytest.raises(ValueError, match="backup directory is required"):
            _require_backup_dir(None)

    def test_empty_string_backup_dir_raises(self):
        """Passing backup_dir='' must raise before migration starts."""
        from redisvl.migration.executor import _require_backup_dir

        with pytest.raises(ValueError, match="backup directory is required"):
            _require_backup_dir("")

    def test_valid_backup_dir_is_created(self, tmp_path):
        """A valid missing backup directory is created up front."""
        from redisvl.migration.executor import _require_backup_dir

        backup_dir = tmp_path / "nested" / "backups"
        assert not backup_dir.exists()

        resolved = _require_backup_dir(str(backup_dir))

        assert resolved == str(backup_dir)
        assert backup_dir.is_dir()

    def test_unwritable_existing_backup_dir_raises(self, tmp_path):
        """An existing directory that cannot be written fails the preflight."""
        from redisvl.migration.executor import _require_backup_dir

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        with patch(
            "redisvl.migration.executor.tempfile.mkstemp",
            side_effect=PermissionError("permission denied"),
        ):
            with pytest.raises(ValueError, match="backup directory"):
                _require_backup_dir(str(backup_dir))


class TestApplyCrashResume:
    def _mock_source_and_target(self):
        mock_client = MagicMock()
        mock_client.info.return_value = {}
        mock_client.config_get.return_value = {}
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        source_index = MagicMock()
        source_index._redis_client = mock_client
        source_index.delete = MagicMock()

        target_index = MagicMock()
        target_index.create = MagicMock()
        return mock_client, source_index, target_index

    def _create_ready_backup(self, backup_path, plan=None):
        from redisvl.migration.backup import VectorBackup
        from redisvl.migration.executor import _checkpoint_identity

        plan = plan or _make_migration_plan()
        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": 4}
        }
        backup = VectorBackup.create(
            path=backup_path,
            index_name="idx",
            fields=datatype_changes,
            batch_size=1,
            **_checkpoint_identity(plan, datatype_changes),
        )
        vec = _make_float32_vector(4)
        backup.write_batch(0, ["doc:1"], {"doc:1": {"embedding": vec}})
        backup.mark_dump_complete()
        return backup

    def test_ready_backup_with_live_source_drops_before_resume(self, tmp_path):
        from redisvl.migration.backup import VectorBackup
        from redisvl.migration.executor import MigrationExecutor, _resolve_backup_path

        plan = _make_migration_plan()
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        backup_path = _resolve_backup_path(str(backup_dir), "idx")
        self._create_ready_backup(backup_path, plan)

        executor = MigrationExecutor()
        executor.validator.validate = MagicMock(return_value=_successful_validation())
        _, source_index, target_index = self._mock_source_and_target()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot",
                side_effect=[True, False],
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_existing",
                return_value=source_index,
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_dict",
                return_value=target_index,
            ),
            patch(
                "redisvl.migration.executor.wait_for_index_ready",
                return_value=({"num_docs": 2, "vector_index_sz_mb": 1}, 0.01),
            ),
        ):
            report = executor.apply(
                plan,
                redis_client=MagicMock(),
                backup_dir=str(backup_dir),
            )

        source_index.delete.assert_called_once_with(drop=False)
        target_index.create.assert_called_once()
        assert report.result == "succeeded"
        reloaded = VectorBackup.load(backup_path)
        assert reloaded is not None
        assert reloaded.header.phase == "validated"

    def test_completed_backup_without_target_creates_target(self, tmp_path):
        from redisvl.migration.backup import VectorBackup
        from redisvl.migration.executor import MigrationExecutor, _resolve_backup_path

        plan = _make_migration_plan()
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        backup_path = _resolve_backup_path(str(backup_dir), "idx")
        backup = self._create_ready_backup(backup_path, plan)
        backup.start_quantize()
        backup.mark_batch_quantized(0)
        backup.mark_complete()

        executor = MigrationExecutor()
        executor.validator.validate = MagicMock(return_value=_successful_validation())
        _, source_index, target_index = self._mock_source_and_target()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot",
                side_effect=[False, False],
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_dict",
                side_effect=[source_index, target_index],
            ),
            patch(
                "redisvl.migration.executor.wait_for_index_ready",
                return_value=({"num_docs": 2, "vector_index_sz_mb": 1}, 0.01),
            ),
        ):
            report = executor.apply(
                plan,
                redis_client=MagicMock(),
                backup_dir=str(backup_dir),
            )

        source_index.delete.assert_not_called()
        target_index.create.assert_called_once()
        assert report.result == "succeeded"
        reloaded = VectorBackup.load(backup_path)
        assert reloaded is not None
        assert reloaded.header.phase == "validated"

    def test_completed_backup_with_live_target_skips_create(self, tmp_path):
        from redisvl.migration.backup import VectorBackup
        from redisvl.migration.executor import MigrationExecutor, _resolve_backup_path

        plan = _make_migration_plan()
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        backup_path = _resolve_backup_path(str(backup_dir), "idx")
        backup = self._create_ready_backup(backup_path, plan)
        backup.start_quantize()
        backup.mark_batch_quantized(0)
        backup.mark_complete()

        executor = MigrationExecutor()
        executor.validator.validate = MagicMock(return_value=_successful_validation())
        _, source_index, target_index = self._mock_source_and_target()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot",
                side_effect=[False, True],
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_dict",
                side_effect=[source_index, target_index],
            ),
        ):
            report = executor.apply(
                plan,
                redis_client=MagicMock(),
                backup_dir=str(backup_dir),
            )

        target_index.create.assert_not_called()
        assert report.result == "succeeded"
        reloaded = VectorBackup.load(backup_path)
        assert reloaded is not None
        assert reloaded.header.phase == "validated"

    def test_multi_worker_requires_redis_url_before_loading_index(self, tmp_path):
        from redisvl.migration.executor import MigrationExecutor

        executor = MigrationExecutor()
        plan = _make_migration_plan()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot"
            ) as matches_mock,
            patch("redisvl.migration.executor.SearchIndex.from_existing") as from_mock,
        ):
            report = executor.apply(
                plan,
                redis_client=MagicMock(),
                backup_dir=str(tmp_path / "backups"),
                num_workers=2,
            )

        matches_mock.assert_not_called()
        from_mock.assert_not_called()
        assert report.result == "failed"
        assert "redis_url is required" in report.validation.errors[0]

    def test_multi_worker_manifest_resumes_after_source_drop(self, tmp_path):
        from redisvl.migration.backup import MultiWorkerBackupManifest
        from redisvl.migration.executor import (
            MigrationExecutor,
            _checkpoint_identity,
            _resolve_backup_path,
        )
        from redisvl.migration.quantize import MultiWorkerResult

        plan = _make_migration_plan()
        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": 4}
        }
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        backup_path = _resolve_backup_path(str(backup_dir), "idx")
        worker_paths = [
            str(backup_dir / "migration_backup_idx_worker0"),
            str(backup_dir / "migration_backup_idx_worker1"),
        ]
        manifest = MultiWorkerBackupManifest.create(
            backup_path,
            index_name="idx",
            batch_size=1,
            requested_workers=2,
            key_slices=[["doc:1"], ["doc:2"]],
            worker_backup_paths=worker_paths,
            **_checkpoint_identity(plan, datatype_changes),
        )
        manifest.mark_index_dropped()

        executor = MigrationExecutor()
        executor.validator.validate = MagicMock(return_value=_successful_validation())
        _, source_index, target_index = self._mock_source_and_target()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot",
                side_effect=[False, False],
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_dict",
                side_effect=[source_index, target_index],
            ),
            patch(
                "redisvl.migration.quantize.multi_worker_quantize",
                return_value=MultiWorkerResult(
                    total_docs_quantized=2,
                    num_workers=2,
                    backup_paths=worker_paths,
                ),
            ) as quantize_mock,
            patch(
                "redisvl.migration.executor.wait_for_index_ready",
                return_value=({"num_docs": 2, "vector_index_sz_mb": 1}, 0.01),
            ),
        ):
            report = executor.apply(
                plan,
                redis_url="redis://localhost:6379",
                backup_dir=str(backup_dir),
                num_workers=2,
            )

        source_index.delete.assert_not_called()
        quantize_mock.assert_called_once()
        target_index.create.assert_called_once()
        assert report.result == "succeeded"
        reloaded = MultiWorkerBackupManifest.load(backup_path)
        assert reloaded is not None
        assert reloaded.phase == "validated"

    def test_multi_worker_manifest_resumes_without_num_workers_arg(self, tmp_path):
        from redisvl.migration.backup import MultiWorkerBackupManifest
        from redisvl.migration.executor import (
            MigrationExecutor,
            _checkpoint_identity,
            _resolve_backup_path,
        )
        from redisvl.migration.quantize import MultiWorkerResult

        plan = _make_migration_plan()
        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": 4}
        }
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        backup_path = _resolve_backup_path(str(backup_dir), "idx")
        worker_paths = [
            str(backup_dir / "migration_backup_idx_worker0"),
            str(backup_dir / "migration_backup_idx_worker1"),
        ]
        manifest = MultiWorkerBackupManifest.create(
            backup_path,
            index_name="idx",
            batch_size=7,
            requested_workers=2,
            key_slices=[["doc:1"], ["doc:2"]],
            worker_backup_paths=worker_paths,
            **_checkpoint_identity(plan, datatype_changes),
        )
        manifest.mark_index_dropped()

        executor = MigrationExecutor()
        executor.validator.validate = MagicMock(return_value=_successful_validation())
        _, source_index, target_index = self._mock_source_and_target()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot",
                side_effect=[False, False],
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_dict",
                side_effect=[source_index, target_index],
            ),
            patch(
                "redisvl.migration.quantize.multi_worker_quantize",
                return_value=MultiWorkerResult(
                    total_docs_quantized=2,
                    num_workers=2,
                    backup_paths=worker_paths,
                ),
            ) as quantize_mock,
            patch(
                "redisvl.migration.executor.wait_for_index_ready",
                return_value=({"num_docs": 2, "vector_index_sz_mb": 1}, 0.01),
            ),
        ):
            report = executor.apply(
                plan,
                redis_url="redis://localhost:6379",
                backup_dir=str(backup_dir),
            )

        assert report.result == "succeeded"
        quantize_mock.assert_called_once()
        assert quantize_mock.call_args.kwargs["num_workers"] == 2
        assert quantize_mock.call_args.kwargs["batch_size"] == 7

    def test_checkpoint_plan_mismatch_with_missing_source_fails_before_create(
        self, tmp_path
    ):
        from redisvl.migration.executor import MigrationExecutor, _resolve_backup_path

        original_plan = _make_migration_plan(target_dtype="float16")
        retry_plan = _make_migration_plan(target_dtype="int8")
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        backup_path = _resolve_backup_path(str(backup_dir), "idx")
        backup = self._create_ready_backup(backup_path, original_plan)
        backup.start_quantize()
        backup.mark_batch_quantized(0)
        backup.mark_complete()

        executor = MigrationExecutor()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot",
                side_effect=[False, False],
            ),
            patch("redisvl.migration.executor.SearchIndex.from_dict") as from_dict,
        ):
            report = executor.apply(
                retry_plan,
                redis_client=MagicMock(),
                backup_dir=str(backup_dir),
            )

        from_dict.assert_not_called()
        assert report.result == "failed"
        assert "does not match this migration plan" in report.validation.errors[0]

    def test_empty_quantization_reports_no_backup_path(self, tmp_path):
        from redisvl.migration.executor import MigrationExecutor

        plan = _make_migration_plan()
        plan.source.stats_snapshot["num_docs"] = 0
        executor = MigrationExecutor()
        executor.validator.validate = MagicMock(return_value=_successful_validation())
        _, source_index, target_index = self._mock_source_and_target()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot",
                side_effect=[True, False],
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_existing",
                return_value=source_index,
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_dict",
                return_value=target_index,
            ),
            patch.object(executor, "_enumerate_indexed_keys", return_value=iter(())),
            patch(
                "redisvl.migration.executor.wait_for_index_ready",
                return_value=({"num_docs": 0, "vector_index_sz_mb": 0}, 0.01),
            ),
        ):
            report = executor.apply(
                plan,
                redis_client=MagicMock(),
                backup_dir=str(tmp_path / "backups"),
            )

        assert report.result == "succeeded"
        assert report.backup is not None
        assert report.backup.backup_paths == []


class TestSameWidthGuard:
    def test_hash_same_width_returns_before_drop(self, tmp_path):
        from redisvl.migration.executor import MigrationExecutor

        plan = _make_migration_plan(source_dtype="float16", target_dtype="bfloat16")
        executor = MigrationExecutor()
        _, source_index, target_index = TestApplyCrashResume()._mock_source_and_target()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot",
                side_effect=[True, False],
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_existing",
                return_value=source_index,
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_dict",
                return_value=target_index,
            ),
        ):
            report = executor.apply(
                plan,
                redis_client=MagicMock(),
                backup_dir=str(tmp_path / "backups"),
            )

        source_index.delete.assert_not_called()
        target_index.create.assert_not_called()
        assert "same-width datatype" in report.validation.errors[0]

    def test_json_same_width_is_not_blocked_by_hash_byte_guard(self, tmp_path):
        from redisvl.migration.executor import MigrationExecutor

        plan = _make_migration_plan(
            storage_type="json", source_dtype="float16", target_dtype="bfloat16"
        )
        executor = MigrationExecutor()
        executor.validator.validate = MagicMock(return_value=_successful_validation())
        _, source_index, target_index = TestApplyCrashResume()._mock_source_and_target()

        with (
            patch(
                "redisvl.migration.executor.current_source_matches_snapshot",
                side_effect=[True, False],
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_existing",
                return_value=source_index,
            ),
            patch(
                "redisvl.migration.executor.SearchIndex.from_dict",
                return_value=target_index,
            ),
            patch(
                "redisvl.migration.executor.wait_for_index_ready",
                return_value=({"num_docs": 2, "vector_index_sz_mb": 1}, 0.01),
            ),
        ):
            report = executor.apply(
                plan,
                redis_client=MagicMock(),
                backup_dir=str(tmp_path / "backups"),
            )

        assert report.result == "succeeded"
        target_index.create.assert_called_once()


class TestEnumerateScanFallback:
    """SCAN-fallback conditions in MigrationExecutor._enumerate_indexed_keys."""

    def _build_executor_with_info(self, info_dict):
        """Construct an executor and a mock client whose ft().info() returns info_dict."""
        from redisvl.migration.executor import MigrationExecutor

        executor = MigrationExecutor()
        mock_client = MagicMock()
        mock_ft = MagicMock()
        mock_ft.info.return_value = info_dict
        mock_client.ft.return_value = mock_ft
        return executor, mock_client

    def test_falls_back_to_scan_when_percent_indexed_below_one(self):
        """percent_indexed < 1.0 must trigger SCAN fallback to avoid silent loss."""
        executor, mock_client = self._build_executor_with_info(
            {"hash_indexing_failures": 0, "percent_indexed": "0.5"}
        )

        with (
            patch.object(
                executor,
                "_enumerate_with_scan",
                return_value=iter(["doc:1", "doc:2"]),
            ) as scan_mock,
            patch.object(
                executor,
                "_enumerate_with_aggregate",
                return_value=iter(["should-not-be-used"]),
            ) as aggregate_mock,
        ):
            keys = list(executor._enumerate_indexed_keys(mock_client, "idx"))

        scan_mock.assert_called_once()
        aggregate_mock.assert_not_called()
        assert keys == ["doc:1", "doc:2"]

    def test_uses_aggregate_when_fully_indexed(self):
        """percent_indexed == 1.0 with no failures should use FT.AGGREGATE."""
        executor, mock_client = self._build_executor_with_info(
            {"hash_indexing_failures": 0, "percent_indexed": "1"}
        )

        with (
            patch.object(
                executor,
                "_enumerate_with_scan",
                return_value=iter(["should-not-be-used"]),
            ) as scan_mock,
            patch.object(
                executor,
                "_enumerate_with_aggregate",
                return_value=iter(["doc:1", "doc:2"]),
            ) as aggregate_mock,
        ):
            keys = list(executor._enumerate_indexed_keys(mock_client, "idx"))

        scan_mock.assert_not_called()
        aggregate_mock.assert_called_once()
        assert keys == ["doc:1", "doc:2"]

    def test_failures_take_precedence_over_percent_indexed(self):
        """hash_indexing_failures > 0 always triggers SCAN, regardless of percent_indexed."""
        executor, mock_client = self._build_executor_with_info(
            {"hash_indexing_failures": 7, "percent_indexed": "1"}
        )

        with patch.object(
            executor,
            "_enumerate_with_scan",
            return_value=iter(["doc:1"]),
        ) as scan_mock:
            keys = list(executor._enumerate_indexed_keys(mock_client, "idx"))

        scan_mock.assert_called_once()
        assert keys == ["doc:1"]

    def test_treats_missing_percent_indexed_as_complete(self):
        """Missing percent_indexed key should default to 1.0 (use FT.AGGREGATE)."""
        executor, mock_client = self._build_executor_with_info(
            {"hash_indexing_failures": 0}
        )

        with (
            patch.object(
                executor,
                "_enumerate_with_scan",
                return_value=iter(["should-not-be-used"]),
            ) as scan_mock,
            patch.object(
                executor,
                "_enumerate_with_aggregate",
                return_value=iter(["doc:1"]),
            ) as aggregate_mock,
        ):
            keys = list(executor._enumerate_indexed_keys(mock_client, "idx"))

        scan_mock.assert_not_called()
        aggregate_mock.assert_called_once()
        assert keys == ["doc:1"]
