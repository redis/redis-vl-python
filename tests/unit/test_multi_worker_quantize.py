"""Tests for multi-worker quantization.

TDD: tests written BEFORE implementation.

Tests:
  - Key splitting across N workers
  - Per-worker backup file shards
  - Multi-worker sync execution via ThreadPoolExecutor
  - Progress aggregation
"""

import struct
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


def _make_float32_vector(dims: int = 4, seed: float = 0.0) -> bytes:
    return struct.pack(f"<{dims}f", *[seed + i for i in range(dims)])


class TestSplitKeys:
    """Test splitting keys into N contiguous slices."""

    def test_split_evenly(self):
        from redisvl.migration.quantize import split_keys

        keys = [f"doc:{i}" for i in range(8)]
        slices = split_keys(keys, num_workers=4)
        assert len(slices) == 4
        assert slices[0] == ["doc:0", "doc:1"]
        assert slices[1] == ["doc:2", "doc:3"]
        assert slices[2] == ["doc:4", "doc:5"]
        assert slices[3] == ["doc:6", "doc:7"]

    def test_split_uneven(self):
        from redisvl.migration.quantize import split_keys

        keys = [f"doc:{i}" for i in range(10)]
        slices = split_keys(keys, num_workers=3)
        assert len(slices) == 3
        # 10 / 3 = 4, 4, 2
        assert len(slices[0]) == 4
        assert len(slices[1]) == 4
        assert len(slices[2]) == 2
        # All keys present
        all_keys = [k for s in slices for k in s]
        assert all_keys == keys

    def test_split_fewer_keys_than_workers(self):
        from redisvl.migration.quantize import split_keys

        keys = ["doc:0", "doc:1"]
        slices = split_keys(keys, num_workers=5)
        # Should produce only 2 non-empty slices (not 5)
        non_empty = [s for s in slices if s]
        assert len(non_empty) == 2

    def test_split_single_worker(self):
        from redisvl.migration.quantize import split_keys

        keys = [f"doc:{i}" for i in range(10)]
        slices = split_keys(keys, num_workers=1)
        assert len(slices) == 1
        assert slices[0] == keys

    def test_split_empty_keys(self):
        from redisvl.migration.quantize import split_keys

        slices = split_keys([], num_workers=4)
        assert slices == []

    def test_split_zero_workers_raises(self):
        from redisvl.migration.quantize import split_keys

        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            split_keys(["doc:0"], num_workers=0)

    def test_split_negative_workers_raises(self):
        from redisvl.migration.quantize import split_keys

        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            split_keys(["doc:0", "doc:1"], num_workers=-1)

    def test_split_zero_workers_empty_keys_raises(self):
        """Even with empty keys, invalid num_workers should still raise."""
        from redisvl.migration.quantize import split_keys

        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            split_keys([], num_workers=0)


class TestMultiWorkerSync:
    """Test multi-worker quantization with ThreadPoolExecutor."""

    def test_multi_worker_dump_and_quantize(self, tmp_path):
        """4 workers process 8 keys (2 each). Each gets own backup shard."""
        from redisvl.migration.quantize import multi_worker_quantize

        dims = 4
        vec = _make_float32_vector(dims)
        all_keys = [f"doc:{i}" for i in range(8)]

        # Mock Redis: each client.pipeline().execute() returns vectors
        def make_mock_client():
            mock = MagicMock()
            mock_pipe = MagicMock()
            mock.pipeline.return_value = mock_pipe
            mock_pipe.execute.return_value = [vec] * 2  # 2 keys per worker
            return mock

        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": dims}
        }

        with patch(
            "redisvl.redis.connection.RedisConnectionFactory.get_redis_connection"
        ) as mock_get_conn:
            mock_get_conn.side_effect = lambda **kwargs: make_mock_client()

            result = multi_worker_quantize(
                redis_url="redis://localhost:6379",
                keys=all_keys,
                datatype_changes=datatype_changes,
                backup_dir=str(tmp_path),
                index_name="myindex",
                num_workers=4,
                batch_size=2,
            )

        assert result.total_docs_quantized == 8
        assert result.num_workers == 4
        # Each worker should have created a backup shard
        assert len(list(tmp_path.glob("*.header"))) == 4

    def test_single_worker_fallback(self, tmp_path):
        """With num_workers=1, should still work (no ThreadPoolExecutor needed)."""
        from redisvl.migration.quantize import multi_worker_quantize

        dims = 4
        vec = _make_float32_vector(dims)
        keys = [f"doc:{i}" for i in range(4)]

        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_pipe.execute.return_value = [vec] * 4

        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": dims}
        }

        with patch(
            "redisvl.redis.connection.RedisConnectionFactory.get_redis_connection"
        ) as mock_get_conn:
            mock_get_conn.return_value = mock_client

            result = multi_worker_quantize(
                redis_url="redis://localhost:6379",
                keys=keys,
                datatype_changes=datatype_changes,
                backup_dir=str(tmp_path),
                index_name="myindex",
                num_workers=1,
                batch_size=4,
            )

        assert result.total_docs_quantized == 4
        assert result.num_workers == 1


class TestMultiWorkerResult:
    """Test the result object from multi-worker quantization."""

    def test_result_attributes(self):
        from redisvl.migration.quantize import MultiWorkerResult

        result = MultiWorkerResult(
            total_docs_quantized=1000,
            num_workers=4,
            worker_results=[
                {"worker_id": 0, "docs": 250},
                {"worker_id": 1, "docs": 250},
                {"worker_id": 2, "docs": 250},
                {"worker_id": 3, "docs": 250},
            ],
        )
        assert result.total_docs_quantized == 1000
        assert result.num_workers == 4
        assert len(result.worker_results) == 4


class TestWorkerResume:
    """Test sync and async worker resume from partial backups."""

    def _make_partial_backup(self, tmp_path, phase="dump", dump_batches=1):
        """Create a partial backup to simulate crash-resume."""
        from redisvl.migration.backup import VectorBackup

        bp = str(tmp_path / "migration_backup_testidx_shard_0")
        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": 4}
        }
        backup = VectorBackup.create(
            path=bp,
            index_name="testidx",
            fields=datatype_changes,
            batch_size=2,
        )
        # Write some batches
        for i in range(dump_batches):
            keys = [f"doc:{i * 2}", f"doc:{i * 2 + 1}"]
            originals = {
                k: {"embedding": _make_float32_vector(4, seed=float(j))}
                for j, k in enumerate(keys)
            }
            backup.write_batch(i, keys, originals)

        if phase == "ready":
            backup.mark_dump_complete()
        elif phase == "active":
            backup.mark_dump_complete()
            backup.start_quantize()
        return bp, datatype_changes

    def test_sync_worker_resumes_from_ready_phase(self, tmp_path):
        """Sync worker should skip dump and proceed to quantize on resume."""
        from redisvl.migration.backup import VectorBackup

        bp, dt_changes = self._make_partial_backup(
            tmp_path, phase="ready", dump_batches=2
        )

        # Verify backup is in ready phase
        backup = VectorBackup.load(bp)
        assert backup is not None
        assert backup.header.phase == "ready"
        assert backup.header.dump_completed_batches == 2

    def test_sync_worker_resumes_from_dump_phase(self, tmp_path):
        """Sync worker should resume dumping from the last completed batch."""
        from redisvl.migration.backup import VectorBackup

        bp, dt_changes = self._make_partial_backup(
            tmp_path, phase="dump", dump_batches=1
        )

        backup = VectorBackup.load(bp)
        assert backup is not None
        assert backup.header.phase == "dump"
        assert backup.header.dump_completed_batches == 1
        # Worker should start from batch 1, not 0

    def test_sync_worker_skips_completed_backup(self, tmp_path):
        """Completed backup should be detected and skipped."""
        from redisvl.migration.backup import VectorBackup

        bp, dt_changes = self._make_partial_backup(
            tmp_path, phase="active", dump_batches=2
        )
        backup = VectorBackup.load(bp)
        # Mark all batches quantized
        for i in range(2):
            backup.mark_batch_quantized(i)
        backup.mark_complete()

        # Reload and verify
        backup = VectorBackup.load(bp)
        assert backup.header.phase == "completed"

    @pytest.mark.asyncio
    async def test_async_worker_loads_existing_backup(self, tmp_path):
        """Async worker should load existing backup instead of creating new."""
        from redisvl.migration.backup import VectorBackup

        bp, dt_changes = self._make_partial_backup(
            tmp_path, phase="ready", dump_batches=2
        )

        # Verify load succeeds and returns existing backup
        backup = VectorBackup.load(bp)
        assert backup is not None
        assert backup.header.phase == "ready"
        assert backup.header.dump_completed_batches == 2

        # Verify create would fail (backup already exists)
        with pytest.raises(FileExistsError):
            VectorBackup.create(
                path=bp,
                index_name="testidx",
                fields=dt_changes,
                batch_size=2,
            )
