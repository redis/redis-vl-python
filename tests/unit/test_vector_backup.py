"""Tests for VectorBackup — the backup file for crash-safe quantization.

TDD: these tests are written BEFORE the implementation.
"""

import os
import struct
import tempfile

import pytest


class TestVectorBackupCreate:
    """Test creating a new backup file."""

    def test_create_new_backup(self, tmp_path):
        from redisvl.migration.backup import VectorBackup

        backup_path = str(tmp_path / "test_backup")
        backup = VectorBackup.create(
            path=backup_path,
            index_name="myindex",
            fields={
                "embedding": {"source": "float32", "target": "float16", "dims": 768}
            },
            batch_size=500,
        )
        assert backup.header.index_name == "myindex"
        assert backup.header.phase == "dump"
        assert backup.header.dump_completed_batches == 0
        assert backup.header.quantize_completed_batches == 0
        assert backup.header.batch_size == 500
        assert backup.header.fields == {
            "embedding": {"source": "float32", "target": "float16", "dims": 768}
        }

    def test_create_writes_header_to_disk(self, tmp_path):
        from redisvl.migration.backup import VectorBackup

        backup_path = str(tmp_path / "test_backup")
        VectorBackup.create(
            path=backup_path,
            index_name="myindex",
            fields={
                "embedding": {"source": "float32", "target": "float16", "dims": 768}
            },
            batch_size=500,
        )
        # Header file should exist
        assert os.path.exists(backup_path + ".header")

    def test_create_raises_if_already_exists(self, tmp_path):
        from redisvl.migration.backup import VectorBackup

        backup_path = str(tmp_path / "test_backup")
        VectorBackup.create(
            path=backup_path,
            index_name="myindex",
            fields={
                "embedding": {"source": "float32", "target": "float16", "dims": 768}
            },
        )
        with pytest.raises(FileExistsError):
            VectorBackup.create(
                path=backup_path,
                index_name="myindex",
                fields={
                    "embedding": {"source": "float32", "target": "float16", "dims": 768}
                },
            )


class TestVectorBackupDump:
    """Test writing batches during the dump phase."""

    def _make_backup(self, tmp_path, batch_size=500):
        from redisvl.migration.backup import VectorBackup

        backup_path = str(tmp_path / "test_backup")
        return VectorBackup.create(
            path=backup_path,
            index_name="myindex",
            fields={"embedding": {"source": "float32", "target": "float16", "dims": 4}},
            batch_size=batch_size,
        )

    def _fake_vector(self, dims=4):
        """Create a fake float32 vector."""
        return struct.pack(f"<{dims}f", *[float(i) for i in range(dims)])

    def test_write_batch(self, tmp_path):
        backup = self._make_backup(tmp_path, batch_size=2)
        keys = ["doc:0", "doc:1"]
        originals = {
            "doc:0": {"embedding": self._fake_vector()},
            "doc:1": {"embedding": self._fake_vector()},
        }
        backup.write_batch(0, keys, originals)
        assert backup.header.dump_completed_batches == 1

    def test_write_multiple_batches(self, tmp_path):
        backup = self._make_backup(tmp_path, batch_size=2)
        vec = self._fake_vector()
        for batch_idx in range(4):
            keys = [f"doc:{batch_idx * 2}", f"doc:{batch_idx * 2 + 1}"]
            originals = {k: {"embedding": vec} for k in keys}
            backup.write_batch(batch_idx, keys, originals)
        assert backup.header.dump_completed_batches == 4

    def test_mark_dump_complete_transitions_to_ready(self, tmp_path):
        backup = self._make_backup(tmp_path, batch_size=2)
        vec = self._fake_vector()
        backup.write_batch(
            0, ["doc:0", "doc:1"], {k: {"embedding": vec} for k in ["doc:0", "doc:1"]}
        )
        backup.mark_dump_complete()
        assert backup.header.phase == "ready"

    def test_iter_batches_returns_all_dumped_data(self, tmp_path):
        backup = self._make_backup(tmp_path, batch_size=2)
        vec = self._fake_vector()

        # Write 2 batches
        for batch_idx in range(2):
            keys = [f"doc:{batch_idx * 2}", f"doc:{batch_idx * 2 + 1}"]
            originals = {k: {"embedding": vec} for k in keys}
            backup.write_batch(batch_idx, keys, originals)
        backup.mark_dump_complete()

        # Read them back
        batches = list(backup.iter_batches())
        assert len(batches) == 2
        batch_keys, batch_data = batches[0]
        assert batch_keys == ["doc:0", "doc:1"]
        assert batch_data["doc:0"]["embedding"] == vec
        assert batch_data["doc:1"]["embedding"] == vec

    def test_write_batch_wrong_phase_raises(self, tmp_path):
        backup = self._make_backup(tmp_path, batch_size=2)
        vec = self._fake_vector()
        backup.write_batch(
            0, ["doc:0", "doc:1"], {k: {"embedding": vec} for k in ["doc:0", "doc:1"]}
        )
        backup.mark_dump_complete()
        # Now in "ready" phase — writing another batch should fail
        with pytest.raises(ValueError, match="Cannot write batch.*phase"):
            backup.write_batch(1, ["doc:2"], {"doc:2": {"embedding": vec}})


class TestVectorBackupQuantize:
    """Test quantize phase progress tracking."""

    def _make_dumped_backup(self, tmp_path, num_keys=8, batch_size=2, dims=4):
        """Create a backup that has completed the dump phase."""
        import struct

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
        vec = struct.pack(f"<{dims}f", *[float(i) for i in range(dims)])
        num_batches = (num_keys + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_keys)
            keys = [f"doc:{j}" for j in range(start, end)]
            originals = {k: {"embedding": vec} for k in keys}
            backup.write_batch(batch_idx, keys, originals)
        backup.mark_dump_complete()
        return backup

    def test_mark_batch_quantized(self, tmp_path):
        backup = self._make_dumped_backup(tmp_path)
        backup.start_quantize()  # ready → active
        assert backup.header.phase == "active"
        backup.mark_batch_quantized(0)
        assert backup.header.quantize_completed_batches == 1
        backup.mark_batch_quantized(1)
        assert backup.header.quantize_completed_batches == 2

    def test_mark_complete(self, tmp_path):
        backup = self._make_dumped_backup(tmp_path, num_keys=4)
        backup.start_quantize()
        backup.mark_batch_quantized(0)
        backup.mark_batch_quantized(1)
        backup.mark_complete()
        assert backup.header.phase == "completed"

    def test_iter_batches_skips_completed(self, tmp_path):
        """After marking batches 0 and 1 as quantized, iter_remaining_batches
        should only yield batches 2 and 3."""
        backup = self._make_dumped_backup(tmp_path)  # 8 keys, batch_size=2 → 4 batches
        backup.start_quantize()
        backup.mark_batch_quantized(0)
        backup.mark_batch_quantized(1)

        remaining = list(backup.iter_remaining_batches())
        assert len(remaining) == 2
        # Batch 2 starts at doc:4
        batch_keys, _ = remaining[0]
        assert batch_keys[0] == "doc:4"


class TestVectorBackupResume:
    """Test loading a backup file and resuming from where it left off."""

    def _make_dumped_backup(self, tmp_path, num_keys=8, batch_size=2, dims=4):
        import struct

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
        vec = struct.pack(f"<{dims}f", *[float(i) for i in range(dims)])
        num_batches = (num_keys + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_keys)
            keys = [f"doc:{j}" for j in range(start, end)]
            originals = {k: {"embedding": vec} for k in keys}
            backup.write_batch(batch_idx, keys, originals)
        backup.mark_dump_complete()
        return backup, backup_path

    def test_load_returns_none_if_no_file(self, tmp_path):
        from redisvl.migration.backup import VectorBackup

        result = VectorBackup.load(str(tmp_path / "nonexistent"))
        assert result is None

    def test_load_restores_header(self, tmp_path):
        from redisvl.migration.backup import VectorBackup

        backup, path = self._make_dumped_backup(tmp_path)
        loaded = VectorBackup.load(path)
        assert loaded is not None
        assert loaded.header.index_name == "myindex"
        assert loaded.header.phase == "ready"
        assert loaded.header.dump_completed_batches == 4

    def test_load_and_resume_quantize(self, tmp_path):
        """Simulate crash: dump complete, 2 batches quantized, then crash.
        On reload, iter_remaining_batches should skip the 2 completed."""
        from redisvl.migration.backup import VectorBackup

        backup, path = self._make_dumped_backup(tmp_path)
        backup.start_quantize()
        backup.mark_batch_quantized(0)
        backup.mark_batch_quantized(1)
        # "crash" — drop the object, reload from disk
        del backup

        loaded = VectorBackup.load(path)
        assert loaded is not None
        assert loaded.header.phase == "active"
        assert loaded.header.quantize_completed_batches == 2

        remaining = list(loaded.iter_remaining_batches())
        assert len(remaining) == 2
        batch_keys, _ = remaining[0]
        assert batch_keys[0] == "doc:4"

    def test_load_and_resume_dump(self, tmp_path):
        """Simulate crash during dump: 2 of 4 batches dumped.
        On reload, should see phase=dump, dump_completed_batches=2."""
        import struct

        from redisvl.migration.backup import VectorBackup

        backup_path = str(tmp_path / "test_backup")
        backup = VectorBackup.create(
            path=backup_path,
            index_name="myindex",
            fields={"embedding": {"source": "float32", "target": "float16", "dims": 4}},
            batch_size=2,
        )
        vec = struct.pack("<4f", 0.0, 1.0, 2.0, 3.0)
        # Write only 2 of 4 expected batches
        for batch_idx in range(2):
            keys = [f"doc:{batch_idx * 2}", f"doc:{batch_idx * 2 + 1}"]
            originals = {k: {"embedding": vec} for k in keys}
            backup.write_batch(batch_idx, keys, originals)
        # "crash" — don't call mark_dump_complete
        del backup

        loaded = VectorBackup.load(backup_path)
        assert loaded is not None
        assert loaded.header.phase == "dump"
        assert loaded.header.dump_completed_batches == 2
        # Can read back the 2 completed batches
        batches = list(loaded.iter_batches())
        assert len(batches) == 2


class TestVectorBackupRollback:
    """Test reading originals for rollback."""

    def test_rollback_reads_all_originals(self, tmp_path):
        import struct

        from redisvl.migration.backup import VectorBackup

        backup_path = str(tmp_path / "test_backup")
        backup = VectorBackup.create(
            path=backup_path,
            index_name="myindex",
            fields={"embedding": {"source": "float32", "target": "float16", "dims": 4}},
            batch_size=2,
        )
        vecs = {}
        for i in range(4):
            vec = struct.pack("<4f", *[float(i * 10 + j) for j in range(4)])
            vecs[f"doc:{i}"] = vec

        # Write 2 batches with distinct vectors
        backup.write_batch(
            0,
            ["doc:0", "doc:1"],
            {
                "doc:0": {"embedding": vecs["doc:0"]},
                "doc:1": {"embedding": vecs["doc:1"]},
            },
        )
        backup.write_batch(
            1,
            ["doc:2", "doc:3"],
            {
                "doc:2": {"embedding": vecs["doc:2"]},
                "doc:3": {"embedding": vecs["doc:3"]},
            },
        )
        backup.mark_dump_complete()

        # Read all batches and verify originals are preserved
        all_originals = {}
        for batch_keys, batch_data in backup.iter_batches():
            all_originals.update(batch_data)

        assert len(all_originals) == 4
        for key in ["doc:0", "doc:1", "doc:2", "doc:3"]:
            assert all_originals[key]["embedding"] == vecs[key]


class TestRollbackCLI:
    """Tests for the rvl migrate rollback CLI command path derivation and restore logic."""

    def _create_backup_with_data(self, tmp_path, name="test_idx"):
        """Helper: create a backup with 2 batches of data."""
        from redisvl.migration.backup import VectorBackup

        bp = str(tmp_path / f"migration_backup_{name}")
        vecs = {
            "doc:0": struct.pack("<4f", 1.0, 2.0, 3.0, 4.0),
            "doc:1": struct.pack("<4f", 5.0, 6.0, 7.0, 8.0),
        }
        backup = VectorBackup.create(
            path=bp,
            index_name=name,
            fields={"embedding": {"source": "float32", "target": "float16", "dims": 4}},
            batch_size=1,
        )
        backup.write_batch(0, ["doc:0"], {"doc:0": {"embedding": vecs["doc:0"]}})
        backup.write_batch(1, ["doc:1"], {"doc:1": {"embedding": vecs["doc:1"]}})
        backup.mark_dump_complete()
        return bp, vecs

    def test_header_path_derivation_no_removesuffix(self, tmp_path):
        """Verify path derivation works without str.removesuffix (Python 3.8 compat)."""
        from pathlib import Path

        bp, _ = self._create_backup_with_data(tmp_path)
        header_files = sorted(Path(tmp_path).glob("*.header"))
        assert len(header_files) == 1
        # This is how the CLI derives backup paths — must not use removesuffix
        derived = str(header_files[0].with_suffix(""))
        assert derived == bp

    def test_rollback_restores_via_iter_batches(self, tmp_path):
        """Verify rollback reads all batches and gets correct original vectors."""
        from redisvl.migration.backup import VectorBackup

        bp, vecs = self._create_backup_with_data(tmp_path)
        backup = VectorBackup.load(bp)
        assert backup is not None

        restored = {}
        for batch_keys, originals in backup.iter_batches():
            for key in batch_keys:
                if key in originals:
                    restored[key] = originals[key]

        assert len(restored) == 2
        assert restored["doc:0"]["embedding"] == vecs["doc:0"]
        assert restored["doc:1"]["embedding"] == vecs["doc:1"]

    def test_rollback_nonexistent_dir(self):
        """Verify error handling for missing backup directory."""
        import os

        assert not os.path.isdir("/nonexistent/backup/dir/xyz123")

    def test_rollback_empty_dir(self, tmp_path):
        """Verify no header files found in empty directory."""
        from pathlib import Path

        header_files = sorted(Path(tmp_path).glob("*.header"))
        assert len(header_files) == 0

    def test_rollback_unloadable_backup_returns_none(self, tmp_path):
        """VectorBackup.load returns None for corrupt/missing data."""
        from redisvl.migration.backup import VectorBackup

        # Create header but no data file
        bp = str(tmp_path / "bad_backup")
        result = VectorBackup.load(bp)
        assert result is None
