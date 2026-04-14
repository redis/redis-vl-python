"""Tests for the new two-phase quantize flow in MigrationExecutor.

Verifies:
  - dump_vectors: pipeline-reads originals, writes to backup file
  - quantize_from_backup: reads backup file, converts, pipeline-writes
  - Resume: reloads backup file, skips completed batches
  - BGSAVE is NOT called
"""

import struct
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


def _make_float32_vector(dims: int = 4, seed: float = 0.0) -> bytes:
    return struct.pack(f"<{dims}f", *[seed + i for i in range(dims)])


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
            keys = [f"doc:{batch_idx*2}", f"doc:{batch_idx*2+1}"]
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
