"""Vector backup file for crash-safe quantization.

Stores original vector bytes on disk so that:
- Quantization can resume from where it left off after a crash
- Original vectors can be restored (rollback) at any time
- No BGSAVE or Redis-side checkpointing is needed

File layout:
  <path>.header    — JSON file with phase, progress counters, metadata
  <path>.data      — Binary file with length-prefixed pickle blobs per batch
  <path>.manifest  — Optional JSON manifest for multi-worker resume
"""

import json
import os
import pickle
import struct
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple


@dataclass
class BackupHeader:
    """Metadata and progress tracking for a vector backup."""

    index_name: str
    fields: Dict[str, Dict[str, Any]]
    batch_size: int
    phase: str = "dump"  # dump → ready → index_dropped → active → completed
    dump_completed_batches: int = 0
    quantize_completed_batches: int = 0
    key_prefix: Optional[Dict[str, str]] = None
    source_schema_hash: Optional[str] = None
    target_schema_hash: Optional[str] = None
    datatype_changes_hash: Optional[str] = None
    plan_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "index_name": self.index_name,
            "fields": self.fields,
            "batch_size": self.batch_size,
            "phase": self.phase,
            "dump_completed_batches": self.dump_completed_batches,
            "quantize_completed_batches": self.quantize_completed_batches,
            "key_prefix": self.key_prefix,
            "source_schema_hash": self.source_schema_hash,
            "target_schema_hash": self.target_schema_hash,
            "datatype_changes_hash": self.datatype_changes_hash,
            "plan_hash": self.plan_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BackupHeader":
        return cls(
            index_name=d["index_name"],
            fields=d["fields"],
            batch_size=d.get("batch_size", 500),
            phase=d.get("phase", "dump"),
            dump_completed_batches=d.get("dump_completed_batches", 0),
            quantize_completed_batches=d.get("quantize_completed_batches", 0),
            key_prefix=d.get("key_prefix"),
            source_schema_hash=d.get("source_schema_hash"),
            target_schema_hash=d.get("target_schema_hash"),
            datatype_changes_hash=d.get("datatype_changes_hash"),
            plan_hash=d.get("plan_hash"),
        )


class VectorBackup:
    """Manages a vector backup file for crash-safe quantization.

    Two files on disk:
      <path>.header  — small JSON, atomically updated after each batch
      <path>.data    — append-only binary, one length-prefixed pickle blob per batch
    """

    def __init__(self, path: str, header: BackupHeader) -> None:
        self._path = path
        self._header_path = path + ".header"
        self._data_path = path + ".data"
        self.header = header

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str,
        index_name: str,
        fields: Dict[str, Dict[str, Any]],
        batch_size: int = 500,
        key_prefix: Optional[Dict[str, str]] = None,
        source_schema_hash: Optional[str] = None,
        target_schema_hash: Optional[str] = None,
        datatype_changes_hash: Optional[str] = None,
        plan_hash: Optional[str] = None,
    ) -> "VectorBackup":
        """Create a new backup file. Raises FileExistsError if one already exists."""
        header_path = path + ".header"
        if os.path.exists(header_path):
            raise FileExistsError(f"Backup already exists at {header_path}")

        header = BackupHeader(
            index_name=index_name,
            fields=fields,
            batch_size=batch_size,
            key_prefix=key_prefix,
            source_schema_hash=source_schema_hash,
            target_schema_hash=target_schema_hash,
            datatype_changes_hash=datatype_changes_hash,
            plan_hash=plan_hash,
        )
        backup = cls(path, header)
        backup._save_header()
        return backup

    @classmethod
    def load(cls, path: str) -> Optional["VectorBackup"]:
        """Load an existing backup from disk. Returns None if not found."""
        header_path = path + ".header"
        if not os.path.exists(header_path):
            return None
        with open(header_path, "r") as f:
            header = BackupHeader.from_dict(json.load(f))
        return cls(path, header)

    # ------------------------------------------------------------------
    # Header persistence (atomic write via temp + rename)
    # ------------------------------------------------------------------

    def _save_header(self) -> None:
        """Atomically write header to disk."""
        dir_path = os.path.dirname(self._header_path) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.header.to_dict(), f)
            os.replace(tmp, self._header_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Dump phase: write batches of original vectors
    # ------------------------------------------------------------------

    def write_batch(
        self,
        batch_idx: int,
        keys: List[str],
        originals: Dict[str, Dict[str, bytes]],
    ) -> None:
        """Append a batch of original vectors to the data file.

        Args:
            batch_idx: Sequential batch index (0, 1, 2, ...)
            keys: Ordered list of Redis keys in this batch
            originals: {key: {field_name: original_bytes}}
        """
        if self.header.phase != "dump":
            raise ValueError(
                f"Cannot write batch in phase '{self.header.phase}'. "
                "Only allowed during 'dump' phase."
            )
        blob = pickle.dumps({"keys": keys, "vectors": originals})
        # Length-prefixed: 4 bytes big-endian length + blob
        length_prefix = struct.pack(">I", len(blob))
        with open(self._data_path, "ab") as f:
            f.write(length_prefix)
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())

        self.header.dump_completed_batches = batch_idx + 1
        self._save_header()

    def mark_dump_complete(self) -> None:
        """Transition from dump → ready."""
        if self.header.phase != "dump":
            raise ValueError(
                f"Cannot mark dump complete in phase '{self.header.phase}'"
            )
        self.header.phase = "ready"
        self._save_header()

    def mark_index_dropped(self) -> None:
        """Record that the source index definition has been dropped."""
        if self.header.phase not in ("ready", "index_dropped"):
            raise ValueError(
                f"Cannot mark index dropped in phase '{self.header.phase}'"
            )
        self.header.phase = "index_dropped"
        self._save_header()

    # ------------------------------------------------------------------
    # Quantize phase: track which batches have been written to Redis
    # ------------------------------------------------------------------

    def start_quantize(self) -> None:
        """Transition from ready/index_dropped → active."""
        if self.header.phase not in ("ready", "index_dropped", "active"):
            raise ValueError(f"Cannot start quantize in phase '{self.header.phase}'")
        self.header.phase = "active"
        self._save_header()

    def mark_batch_quantized(self, batch_idx: int) -> None:
        """Record that a batch has been successfully written to Redis.

        Called ONLY after pipeline_write succeeds.
        """
        self.header.quantize_completed_batches = batch_idx + 1
        self._save_header()

    def mark_complete(self) -> None:
        """Transition from active → completed."""
        self.header.phase = "completed"
        self._save_header()

    def mark_target_created(self) -> None:
        """Record that the target index has been created after quantization."""
        if self.header.phase not in ("completed", "target_created", "validated"):
            raise ValueError(
                f"Cannot mark target created in phase '{self.header.phase}'"
            )
        self.header.phase = "target_created"
        self._save_header()

    def mark_validated(self) -> None:
        """Record that post-migration validation has passed."""
        if self.header.phase not in ("completed", "target_created", "validated"):
            raise ValueError(f"Cannot mark validated in phase '{self.header.phase}'")
        self.header.phase = "validated"
        self._save_header()

    def map_key(self, key: str) -> str:
        """Map a backed-up key to its current live key, if a prefix changed."""
        key_prefix = self.header.key_prefix
        if not key_prefix:
            return key
        old_prefix = key_prefix.get("source")
        new_prefix = key_prefix.get("target")
        if old_prefix is None or new_prefix is None:
            return key
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix) :]
        return key

    # ------------------------------------------------------------------
    # Reading batches back
    # ------------------------------------------------------------------

    def iter_batches(
        self,
    ) -> Generator[Tuple[List[str], Dict[str, Dict[str, bytes]]], None, None]:
        """Iterate ALL batches in the data file.

        Yields (keys, originals) for each batch.
        """
        if not os.path.exists(self._data_path):
            return
        with open(self._data_path, "rb") as f:
            for _ in range(self.header.dump_completed_batches):
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    return
                length = struct.unpack(">I", length_bytes)[0]
                blob = f.read(length)
                if len(blob) < length:
                    return
                batch = pickle.loads(blob)
                yield batch["keys"], batch["vectors"]

    def iter_remaining_batches(
        self,
    ) -> Generator[Tuple[List[str], Dict[str, Dict[str, bytes]]], None, None]:
        """Iterate batches that have NOT been quantized yet.

        Skips the first `quantize_completed_batches` batches.
        """
        skip = self.header.quantize_completed_batches
        for idx, (keys, vectors) in enumerate(self.iter_batches()):
            if idx < skip:
                continue
            yield keys, vectors


@dataclass
class MultiWorkerBackupManifest:
    """Checkpoint manifest for executor-level multi-worker resume."""

    path: str
    index_name: str
    batch_size: int
    requested_workers: int
    actual_workers: int
    worker_backup_paths: List[str]
    key_slices: List[List[str]]
    phase: str = "prepared"
    key_prefix: Optional[Dict[str, str]] = None
    source_schema_hash: Optional[str] = None
    target_schema_hash: Optional[str] = None
    datatype_changes_hash: Optional[str] = None
    plan_hash: Optional[str] = None

    @property
    def _manifest_path(self) -> str:
        return self.path + ".manifest"

    @property
    def keys(self) -> List[str]:
        return [key for key_slice in self.key_slices for key in key_slice]

    def to_dict(self) -> dict:
        return {
            "index_name": self.index_name,
            "batch_size": self.batch_size,
            "requested_workers": self.requested_workers,
            "actual_workers": self.actual_workers,
            "worker_backup_paths": self.worker_backup_paths,
            "key_slices": self.key_slices,
            "phase": self.phase,
            "key_prefix": self.key_prefix,
            "source_schema_hash": self.source_schema_hash,
            "target_schema_hash": self.target_schema_hash,
            "datatype_changes_hash": self.datatype_changes_hash,
            "plan_hash": self.plan_hash,
        }

    @classmethod
    def create(
        cls,
        path: str,
        *,
        index_name: str,
        batch_size: int,
        requested_workers: int,
        key_slices: List[List[str]],
        worker_backup_paths: List[str],
        key_prefix: Optional[Dict[str, str]] = None,
        source_schema_hash: Optional[str] = None,
        target_schema_hash: Optional[str] = None,
        datatype_changes_hash: Optional[str] = None,
        plan_hash: Optional[str] = None,
    ) -> "MultiWorkerBackupManifest":
        manifest_path = path + ".manifest"
        if os.path.exists(manifest_path):
            raise FileExistsError(f"Backup manifest already exists at {manifest_path}")
        manifest = cls(
            path=path,
            index_name=index_name,
            batch_size=batch_size,
            requested_workers=requested_workers,
            actual_workers=len(key_slices),
            worker_backup_paths=worker_backup_paths,
            key_slices=key_slices,
            key_prefix=key_prefix,
            source_schema_hash=source_schema_hash,
            target_schema_hash=target_schema_hash,
            datatype_changes_hash=datatype_changes_hash,
            plan_hash=plan_hash,
        )
        manifest._save()
        return manifest

    @classmethod
    def load(cls, path: str) -> Optional["MultiWorkerBackupManifest"]:
        manifest_path = path + ".manifest"
        if not os.path.exists(manifest_path):
            return None
        with open(manifest_path, "r") as f:
            data = json.load(f)
        return cls(
            path=path,
            index_name=data["index_name"],
            batch_size=data.get("batch_size", 500),
            requested_workers=data.get("requested_workers", data.get("num_workers", 1)),
            actual_workers=data.get("actual_workers", 0),
            worker_backup_paths=data.get("worker_backup_paths", []),
            key_slices=data.get("key_slices", []),
            phase=data.get("phase", "prepared"),
            key_prefix=data.get("key_prefix"),
            source_schema_hash=data.get("source_schema_hash"),
            target_schema_hash=data.get("target_schema_hash"),
            datatype_changes_hash=data.get("datatype_changes_hash"),
            plan_hash=data.get("plan_hash"),
        )

    def _save(self) -> None:
        dir_path = os.path.dirname(self._manifest_path) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, self._manifest_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def mark_index_dropped(self) -> None:
        if self.phase not in ("prepared", "index_dropped"):
            raise ValueError(f"Cannot mark index dropped in phase '{self.phase}'")
        self.phase = "index_dropped"
        self._save()

    def update_key_slices(self, key_slices: List[List[str]]) -> None:
        self.key_slices = key_slices
        self.actual_workers = len(key_slices)
        self._save()

    def mark_keys_renamed(self) -> None:
        if self.phase not in ("index_dropped", "keys_renamed"):
            raise ValueError(f"Cannot mark keys renamed in phase '{self.phase}'")
        self.phase = "keys_renamed"
        self._save()

    def mark_quantizing(self) -> None:
        if self.phase not in (
            "prepared",
            "index_dropped",
            "keys_renamed",
            "quantizing",
        ):
            raise ValueError(f"Cannot mark quantizing in phase '{self.phase}'")
        self.phase = "quantizing"
        self._save()

    def mark_quantized(self) -> None:
        if self.phase not in ("quantizing", "quantized"):
            raise ValueError(f"Cannot mark quantized in phase '{self.phase}'")
        self.phase = "quantized"
        self._save()

    def mark_target_created(self) -> None:
        if self.phase not in ("quantized", "target_created", "validated"):
            raise ValueError(f"Cannot mark target created in phase '{self.phase}'")
        self.phase = "target_created"
        self._save()

    def mark_validated(self) -> None:
        if self.phase not in ("quantized", "target_created", "validated"):
            raise ValueError(f"Cannot mark validated in phase '{self.phase}'")
        self.phase = "validated"
        self._save()

    def map_key(self, key: str) -> str:
        key_prefix = self.key_prefix
        if not key_prefix:
            return key
        old_prefix = key_prefix.get("source")
        new_prefix = key_prefix.get("target")
        if old_prefix is None or new_prefix is None:
            return key
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix) :]
        return key
