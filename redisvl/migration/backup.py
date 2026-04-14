"""Vector backup file for crash-safe quantization.

Stores original vector bytes on disk so that:
- Quantization can resume from where it left off after a crash
- Original vectors can be restored (rollback) at any time
- No BGSAVE or Redis-side checkpointing is needed

File layout:
  <path>.header  — JSON file with phase, progress counters, metadata
  <path>.data    — Binary file with length-prefixed pickle blobs per batch
"""

import json
import os
import pickle
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


@dataclass
class BackupHeader:
    """Metadata and progress tracking for a vector backup."""

    index_name: str
    fields: Dict[str, Dict[str, Any]]
    batch_size: int
    phase: str = "dump"  # dump → ready → active → completed
    dump_completed_batches: int = 0
    quantize_completed_batches: int = 0

    def to_dict(self) -> dict:
        return {
            "index_name": self.index_name,
            "fields": self.fields,
            "batch_size": self.batch_size,
            "phase": self.phase,
            "dump_completed_batches": self.dump_completed_batches,
            "quantize_completed_batches": self.quantize_completed_batches,
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
    ) -> "VectorBackup":
        """Create a new backup file. Raises FileExistsError if one already exists."""
        header_path = path + ".header"
        if os.path.exists(header_path):
            raise FileExistsError(f"Backup already exists at {header_path}")

        header = BackupHeader(
            index_name=index_name,
            fields=fields,
            batch_size=batch_size,
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

    # ------------------------------------------------------------------
    # Quantize phase: track which batches have been written to Redis
    # ------------------------------------------------------------------

    def start_quantize(self) -> None:
        """Transition from ready → active."""
        if self.header.phase not in ("ready", "active"):
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
