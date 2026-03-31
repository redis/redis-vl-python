"""Crash-safe quantization utilities for index migration.

Provides idempotent dtype detection, checkpointing, BGSAVE safety,
and bounded undo buffering for reliable vector re-encoding.
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

from redisvl.migration.models import DTYPE_BYTES

logger = logging.getLogger(__name__)

# Dtypes that share byte widths and are functionally interchangeable
# for idempotent detection purposes (same byte length per element).
_DTYPE_FAMILY: Dict[str, str] = {
    "float64": "8byte",
    "float32": "4byte",
    "float16": "2byte",
    "bfloat16": "2byte",
    "int8": "1byte",
    "uint8": "1byte",
}


# ---------------------------------------------------------------------------
# Idempotent Dtype Detection
# ---------------------------------------------------------------------------


def detect_vector_dtype(data: bytes, expected_dims: int) -> Optional[str]:
    """Inspect raw vector bytes and infer the storage dtype.

    Uses byte length and expected dimensions to determine which dtype
    the vector is currently stored as. Returns the canonical representative
    for each byte-width family (float16 for 2-byte, int8 for 1-byte),
    since dtypes within a family cannot be distinguished by length alone.

    Args:
        data: Raw vector bytes from Redis.
        expected_dims: Number of dimensions expected for this vector field.

    Returns:
        Detected dtype string (e.g. "float32", "float16", "int8") or None
        if the size does not match any known dtype.
    """
    if not data or expected_dims <= 0:
        return None

    nbytes = len(data)

    # Check each dtype in decreasing element size to avoid ambiguity.
    # Only canonical representatives are checked (float16 covers bfloat16,
    # int8 covers uint8) since they share byte widths.
    for dtype in ("float64", "float32", "float16", "int8"):
        if nbytes == expected_dims * DTYPE_BYTES[dtype]:
            return dtype

    return None


def is_already_quantized(
    data: bytes,
    expected_dims: int,
    source_dtype: str,
    target_dtype: str,
) -> bool:
    """Check whether a vector has already been converted to the target dtype.

    Uses byte-width families to handle ambiguous dtypes. For example,
    if source is float32 and target is float16, a vector detected as
    2-bytes-per-element is considered already quantized (the byte width
    shrank from 4 to 2, so conversion already happened).

    However, same-width conversions (e.g. float16 -> bfloat16 or
    int8 -> uint8) are NOT skipped because the encoding semantics
    differ even though the byte length is identical. We cannot
    distinguish these by length, so we must always re-encode.

    Args:
        data: Raw vector bytes.
        expected_dims: Number of dimensions.
        source_dtype: The dtype the vector was originally stored as.
        target_dtype: The dtype we want to convert to.

    Returns:
        True if the vector already matches the target dtype (skip conversion).
    """
    detected = detect_vector_dtype(data, expected_dims)
    if detected is None:
        return False

    detected_family = _DTYPE_FAMILY.get(detected)
    target_family = _DTYPE_FAMILY.get(target_dtype)
    source_family = _DTYPE_FAMILY.get(source_dtype)

    # If detected byte-width matches target family, the vector looks converted.
    # But if source and target share the same byte-width family (e.g.
    # float16 -> bfloat16), we cannot tell whether conversion happened,
    # so we must NOT skip -- always re-encode for same-width migrations.
    if source_family == target_family:
        return False

    return detected_family == target_family


# ---------------------------------------------------------------------------
# Quantization Checkpoint
# ---------------------------------------------------------------------------


class QuantizationCheckpoint(BaseModel):
    """Tracks migration progress for crash-safe resume."""

    index_name: str
    total_keys: int
    completed_keys: int = 0
    completed_batches: int = 0
    last_batch_keys: List[str] = Field(default_factory=list)
    processed_keys: List[str] = Field(default_factory=list)
    status: str = "in_progress"
    checkpoint_path: str = ""

    def record_batch(self, keys: List[str]) -> None:
        """Record a successfully processed batch.

        Does not auto-save to disk. Call save() after record_batch()
        to persist the checkpoint for crash recovery.
        """
        self.completed_keys += len(keys)
        self.completed_batches += 1
        self.last_batch_keys = list(keys)
        self.processed_keys.extend(keys)

    def mark_complete(self) -> None:
        """Mark the migration as completed."""
        self.status = "completed"

    def save(self) -> None:
        """Persist checkpoint to disk atomically.

        Writes to a temporary file first, then renames. This ensures a
        crash mid-write does not corrupt the checkpoint file.
        """
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=".checkpoint_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(self.model_dump(), f, sort_keys=False)
            os.replace(tmp_path, str(path))
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> Optional["QuantizationCheckpoint"]:
        """Load a checkpoint from disk. Returns None if file does not exist.

        Always sets checkpoint_path to the path used to load, not the
        value stored in the file. This ensures subsequent save() calls
        write to the correct location even if the file was moved.
        """
        p = Path(path)
        if not p.exists():
            return None
        with open(p, "r") as f:
            data = yaml.safe_load(f)
        if not data:
            return None
        checkpoint = cls.model_validate(data)
        checkpoint.checkpoint_path = str(p)
        return checkpoint

    def get_remaining_keys(self, all_keys: List[str]) -> List[str]:
        """Return keys that have not yet been processed."""
        done = set(self.processed_keys)
        return [k for k in all_keys if k not in done]


# ---------------------------------------------------------------------------
# BGSAVE Safety Net
# ---------------------------------------------------------------------------


def trigger_bgsave_and_wait(
    client: Any,
    *,
    timeout_seconds: int = 300,
    poll_interval: float = 1.0,
) -> bool:
    """Trigger a Redis BGSAVE and wait for it to complete.

    If a BGSAVE is already in progress, waits for it instead.

    Args:
        client: Sync Redis client.
        timeout_seconds: Max seconds to wait for BGSAVE to finish.
        poll_interval: Seconds between status polls.

    Returns:
        True if BGSAVE completed successfully.
    """
    try:
        client.bgsave()
    except Exception as exc:
        if "already in progress" not in str(exc).lower():
            raise
        logger.info("BGSAVE already in progress, waiting for it to finish.")

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        info = client.info()
        if isinstance(info, dict) and not info.get("rdb_bgsave_in_progress", 0):
            return True
        time.sleep(poll_interval)

    raise TimeoutError(f"BGSAVE did not complete within {timeout_seconds}s")


async def async_trigger_bgsave_and_wait(
    client: Any,
    *,
    timeout_seconds: int = 300,
    poll_interval: float = 1.0,
) -> bool:
    """Async version of trigger_bgsave_and_wait."""
    try:
        await client.bgsave()
    except Exception as exc:
        if "already in progress" not in str(exc).lower():
            raise
        logger.info("BGSAVE already in progress, waiting for it to finish.")

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        info = await client.info()
        if isinstance(info, dict) and not info.get("rdb_bgsave_in_progress", 0):
            return True
        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"BGSAVE did not complete within {timeout_seconds}s")


# ---------------------------------------------------------------------------
# Bounded Undo Buffer
# ---------------------------------------------------------------------------


class BatchUndoBuffer:
    """Stores original vector values for the current batch to allow rollback.

    Memory-bounded: only holds data for one batch at a time. Call clear()
    after each successful batch commit.
    """

    def __init__(self) -> None:
        self._entries: List[Tuple[str, str, bytes]] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def store(self, key: str, field: str, original_value: bytes) -> None:
        """Record the original value of a field before mutation."""
        self._entries.append((key, field, original_value))

    def rollback(self, pipe: Any) -> None:
        """Restore all stored originals via the given pipeline (sync)."""
        if not self._entries:
            return
        for key, field, value in self._entries:
            pipe.hset(key, field, value)
        pipe.execute()

    async def async_rollback(self, pipe: Any) -> None:
        """Restore all stored originals via the given pipeline (async)."""
        if not self._entries:
            return
        for key, field, value in self._entries:
            pipe.hset(key, field, value)
        await pipe.execute()

    def clear(self) -> None:
        """Discard all stored entries."""
        self._entries.clear()
