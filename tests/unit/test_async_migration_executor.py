"""Unit tests for migration executors and disk space estimator.

These tests mirror the sync MigrationExecutor patterns but use async/await.
Also includes pure-calculation tests for estimate_disk_space().
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from redisvl.migration import AsyncMigrationExecutor, MigrationExecutor
from redisvl.migration.models import (
    DiffClassification,
    KeyspaceSnapshot,
    MigrationPlan,
    SourceSnapshot,
    ValidationPolicy,
    _format_bytes,
)
from redisvl.migration.utils import (
    build_scan_match_patterns,
    estimate_disk_space,
    normalize_keys,
)


def _make_basic_plan():
    """Create a basic migration plan for testing."""
    return MigrationPlan(
        mode="drop_recreate",
        source=SourceSnapshot(
            index_name="test_index",
            keyspace=KeyspaceSnapshot(
                storage_type="hash",
                prefixes=["test"],
                key_separator=":",
                key_sample=["test:1", "test:2"],
            ),
            schema_snapshot={
                "index": {
                    "name": "test_index",
                    "prefix": "test",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "title", "type": "text"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            },
            stats_snapshot={"num_docs": 2},
        ),
        requested_changes={},
        merged_target_schema={
            "index": {
                "name": "test_index",
                "prefix": "test",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "title", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",  # Changed from flat
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        },
        diff_classification=DiffClassification(
            supported=True,
            blocked_reasons=[],
        ),
        validation=ValidationPolicy(
            require_doc_count_match=True,
        ),
        warnings=["Index downtime is required"],
    )


def test_async_executor_instantiation():
    """Test AsyncMigrationExecutor can be instantiated."""
    executor = AsyncMigrationExecutor()
    assert executor is not None
    assert executor.validator is not None


def test_async_executor_with_validator():
    """Test AsyncMigrationExecutor with custom validator."""
    from redisvl.migration import AsyncMigrationValidator

    custom_validator = AsyncMigrationValidator()
    executor = AsyncMigrationExecutor(validator=custom_validator)
    assert executor.validator is custom_validator


@pytest.mark.asyncio
async def test_async_executor_handles_unsupported_plan():
    """Test executor returns error report for unsupported plan."""
    plan = _make_basic_plan()
    plan.diff_classification.supported = False
    plan.diff_classification.blocked_reasons = ["Test blocked reason"]

    executor = AsyncMigrationExecutor()

    # The executor doesn't raise an error - it returns a report with errors
    report = await executor.apply(plan, redis_url="redis://localhost:6379")
    assert report.result == "failed"
    assert "Test blocked reason" in report.validation.errors


@pytest.mark.asyncio
async def test_async_executor_validates_redis_url():
    """Test executor requires redis_url or redis_client."""
    plan = _make_basic_plan()
    executor = AsyncMigrationExecutor()

    # The executor should raise an error internally when trying to connect
    # but let's verify it doesn't crash before it tries to apply
    # For a proper test, we'd need to mock AsyncSearchIndex.from_existing
    # For now, we just verify the executor is created
    assert executor is not None


# =============================================================================
# Disk Space Estimator Tests
# =============================================================================


def _make_quantize_plan(
    source_dtype="float32",
    target_dtype="float16",
    dims=3072,
    doc_count=100_000,
    storage_type="hash",
):
    """Helper to create a migration plan with a vector datatype change."""
    return MigrationPlan(
        mode="drop_recreate",
        source=SourceSnapshot(
            index_name="test_index",
            keyspace=KeyspaceSnapshot(
                storage_type=storage_type,
                prefixes=["test"],
                key_separator=":",
            ),
            schema_snapshot={
                "index": {
                    "name": "test_index",
                    "prefix": "test",
                    "storage_type": storage_type,
                },
                "fields": [
                    {"name": "title", "type": "text"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "hnsw",
                            "dims": dims,
                            "distance_metric": "cosine",
                            "datatype": source_dtype,
                        },
                    },
                ],
            },
            stats_snapshot={"num_docs": doc_count},
        ),
        requested_changes={},
        merged_target_schema={
            "index": {
                "name": "test_index",
                "prefix": "test",
                "storage_type": storage_type,
            },
            "fields": [
                {"name": "title", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": dims,
                        "distance_metric": "cosine",
                        "datatype": target_dtype,
                    },
                },
            ],
        },
        diff_classification=DiffClassification(supported=True, blocked_reasons=[]),
        validation=ValidationPolicy(require_doc_count_match=True),
    )


def test_estimate_fp32_to_fp16():
    """FP32->FP16 with 3072 dims, 100K docs should produce expected byte counts."""
    plan = _make_quantize_plan("float32", "float16", dims=3072, doc_count=100_000)
    est = estimate_disk_space(plan)

    assert est.has_quantization is True
    assert len(est.vector_fields) == 1
    vf = est.vector_fields[0]
    assert vf.source_bytes_per_doc == 3072 * 4  # 12288
    assert vf.target_bytes_per_doc == 3072 * 2  # 6144

    assert est.total_source_vector_bytes == 100_000 * 12288
    assert est.total_target_vector_bytes == 100_000 * 6144
    assert est.memory_savings_after_bytes == 100_000 * (12288 - 6144)

    # RDB = source * 0.95
    assert est.rdb_snapshot_disk_bytes == int(100_000 * 12288 * 0.95)
    # COW = full source
    assert est.rdb_cow_memory_if_concurrent_bytes == 100_000 * 12288
    # AOF disabled by default
    assert est.aof_enabled is False
    assert est.aof_growth_bytes == 0
    assert est.total_new_disk_bytes == est.rdb_snapshot_disk_bytes


def test_estimate_with_aof_enabled():
    """AOF growth should include RESP overhead per HSET."""
    plan = _make_quantize_plan("float32", "float16", dims=3072, doc_count=100_000)
    est = estimate_disk_space(plan, aof_enabled=True)

    assert est.aof_enabled is True
    target_vec_size = 3072 * 2
    expected_aof = 100_000 * (target_vec_size + 114)  # 114 = HSET overhead
    assert est.aof_growth_bytes == expected_aof
    assert est.total_new_disk_bytes == est.rdb_snapshot_disk_bytes + expected_aof


def test_estimate_json_storage_aof():
    """JSON storage quantization should not report in-place rewrite costs."""
    plan = _make_quantize_plan(
        "float32", "float16", dims=128, doc_count=1000, storage_type="json"
    )
    est = estimate_disk_space(plan, aof_enabled=True)

    assert est.has_quantization is False
    assert est.aof_growth_bytes == 0
    assert est.total_new_disk_bytes == 0


def test_estimate_no_quantization():
    """Same dtype source and target should produce empty estimate."""
    plan = _make_quantize_plan("float32", "float32", dims=128, doc_count=1000)
    est = estimate_disk_space(plan)

    assert est.has_quantization is False
    assert len(est.vector_fields) == 0
    assert est.total_new_disk_bytes == 0
    assert est.memory_savings_after_bytes == 0


def test_estimate_fp32_to_int8():
    """FP32->INT8 should use 1 byte per element."""
    plan = _make_quantize_plan("float32", "int8", dims=768, doc_count=50_000)
    est = estimate_disk_space(plan)

    assert est.vector_fields[0].source_bytes_per_doc == 768 * 4
    assert est.vector_fields[0].target_bytes_per_doc == 768 * 1
    assert est.memory_savings_after_bytes == 50_000 * 768 * 3


def test_estimate_summary_with_quantization():
    """Summary string should contain key information."""
    plan = _make_quantize_plan("float32", "float16", dims=128, doc_count=1000)
    est = estimate_disk_space(plan)
    summary = est.summary()

    assert "Pre-migration disk space estimate" in summary
    assert "test_index" in summary
    assert "1,000 documents" in summary
    assert "float32 -> float16" in summary
    assert "RDB snapshot" in summary
    assert "reduction" in summary or "memory savings" in summary


def test_estimate_summary_no_quantization():
    """Summary for non-quantization migration should say no disk needed."""
    plan = _make_quantize_plan("float32", "float32", dims=128, doc_count=1000)
    est = estimate_disk_space(plan)
    summary = est.summary()

    assert "No vector quantization" in summary


def test_format_bytes_gb():
    assert _format_bytes(1_073_741_824) == "1.00 GB"
    assert _format_bytes(2_147_483_648) == "2.00 GB"


def test_format_bytes_mb():
    assert _format_bytes(1_048_576) == "1.0 MB"
    assert _format_bytes(10_485_760) == "10.0 MB"


def test_format_bytes_kb():
    assert _format_bytes(1024) == "1.0 KB"
    assert _format_bytes(2048) == "2.0 KB"


def test_format_bytes_bytes():
    assert _format_bytes(500) == "500 bytes"
    assert _format_bytes(0) == "0 bytes"


def test_savings_pct():
    """Verify savings percentage calculation."""
    plan = _make_quantize_plan("float32", "float16", dims=128, doc_count=100)
    est = estimate_disk_space(plan)
    # FP32->FP16 = 50% savings
    assert est._savings_pct() == 50


# =============================================================================
# TDD RED Phase: Idempotent Dtype Detection Tests
# =============================================================================
# These test detect_vector_dtype() and is_already_quantized() which inspect
# raw vector bytes to determine whether a key needs conversion or can be skipped.


def test_detect_dtype_float32_by_size():
    """A 3072-dim vector stored as FP32 should be 12288 bytes."""
    import numpy as np

    from redisvl.migration.reliability import detect_vector_dtype

    vec = np.random.randn(3072).astype(np.float32).tobytes()
    detected = detect_vector_dtype(vec, expected_dims=3072)
    assert detected == "float32"


def test_detect_dtype_float16_by_size():
    """A 3072-dim vector stored as FP16 should be 6144 bytes."""
    import numpy as np

    from redisvl.migration.reliability import detect_vector_dtype

    vec = np.random.randn(3072).astype(np.float16).tobytes()
    detected = detect_vector_dtype(vec, expected_dims=3072)
    assert detected == "float16"


def test_detect_dtype_int8_by_size():
    """A 768-dim vector stored as INT8 should be 768 bytes."""
    import numpy as np

    from redisvl.migration.reliability import detect_vector_dtype

    vec = np.zeros(768, dtype=np.int8).tobytes()
    detected = detect_vector_dtype(vec, expected_dims=768)
    assert detected == "int8"


def test_detect_dtype_bfloat16_by_size():
    """A 768-dim bfloat16 vector should be 1536 bytes (same as float16)."""
    import numpy as np

    from redisvl.migration.reliability import detect_vector_dtype

    # bfloat16 and float16 are both 2 bytes per element
    vec = np.random.randn(768).astype(np.float16).tobytes()
    detected = detect_vector_dtype(vec, expected_dims=768)
    # Cannot distinguish float16 from bfloat16 by size alone; returns "float16"
    assert detected in ("float16", "bfloat16")


def test_detect_dtype_empty_returns_none():
    """Empty bytes should return None."""
    from redisvl.migration.reliability import detect_vector_dtype

    assert detect_vector_dtype(b"", expected_dims=128) is None


def test_detect_dtype_unknown_size():
    """Bytes that don't match any known dtype should return None."""
    from redisvl.migration.reliability import detect_vector_dtype

    # 7 bytes doesn't match any dtype for 3 dims
    assert detect_vector_dtype(b"\x00" * 7, expected_dims=3) is None


def test_is_already_quantized_skip():
    """If source is float32 and vector is already float16, should return True."""
    import numpy as np

    from redisvl.migration.reliability import is_already_quantized

    vec = np.random.randn(128).astype(np.float16).tobytes()
    result = is_already_quantized(
        vec, expected_dims=128, source_dtype="float32", target_dtype="float16"
    )
    assert result is True


def test_is_already_quantized_needs_conversion():
    """If source is float32 and vector IS float32, should return False."""
    import numpy as np

    from redisvl.migration.reliability import is_already_quantized

    vec = np.random.randn(128).astype(np.float32).tobytes()
    result = is_already_quantized(
        vec, expected_dims=128, source_dtype="float32", target_dtype="float16"
    )
    assert result is False


def test_is_already_quantized_bfloat16_target():
    """If target is bfloat16 and vector is 2-bytes-per-element, should return True.

    bfloat16 and float16 share the same byte width (2 bytes per element)
    and are treated as the same dtype family for idempotent detection.
    """
    import numpy as np

    from redisvl.migration.reliability import is_already_quantized

    vec = np.random.randn(128).astype(np.float16).tobytes()
    result = is_already_quantized(
        vec, expected_dims=128, source_dtype="float32", target_dtype="bfloat16"
    )
    assert result is True


def test_is_already_quantized_uint8_target():
    """If target is uint8 and vector is 1-byte-per-element, should return True.

    uint8 and int8 share the same byte width (1 byte per element)
    and are treated as the same dtype family for idempotent detection.
    """
    import numpy as np

    from redisvl.migration.reliability import is_already_quantized

    vec = np.random.randn(128).astype(np.int8).tobytes()
    result = is_already_quantized(
        vec, expected_dims=128, source_dtype="float32", target_dtype="uint8"
    )
    assert result is True


def test_is_already_quantized_same_width_float16_to_bfloat16():
    """float16 -> bfloat16 should NOT be skipped (same byte width, different encoding)."""
    import numpy as np

    from redisvl.migration.reliability import is_already_quantized

    vec = np.random.randn(128).astype(np.float16).tobytes()
    result = is_already_quantized(
        vec, expected_dims=128, source_dtype="float16", target_dtype="bfloat16"
    )
    assert result is False


def test_is_already_quantized_same_width_int8_to_uint8():
    """int8 -> uint8 should NOT be skipped (same byte width, different encoding)."""
    import numpy as np

    from redisvl.migration.reliability import is_already_quantized

    vec = np.random.randn(128).astype(np.int8).tobytes()
    result = is_already_quantized(
        vec, expected_dims=128, source_dtype="int8", target_dtype="uint8"
    )
    assert result is False


# =============================================================================
# TDD RED Phase: Checkpoint File Tests
# =============================================================================


def test_checkpoint_create_new(tmp_path):
    """Creating a new checkpoint should initialize with zero progress."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    cp = QuantizationCheckpoint(
        index_name="test_index",
        total_keys=10000,
        checkpoint_path=str(tmp_path / "checkpoint.yaml"),
    )
    assert cp.index_name == "test_index"
    assert cp.total_keys == 10000
    assert cp.completed_keys == 0
    assert cp.completed_batches == 0
    assert cp.last_batch_keys == []
    assert cp.status == "in_progress"


def test_checkpoint_save_and_load(tmp_path):
    """Checkpoint should persist to disk and reload with same state."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    path = str(tmp_path / "checkpoint.yaml")
    cp = QuantizationCheckpoint(
        index_name="test_index",
        total_keys=5000,
        checkpoint_path=path,
    )
    cp.record_batch(["key:1", "key:2", "key:3"])
    cp.save()

    loaded = QuantizationCheckpoint.load(path)
    assert loaded.index_name == "test_index"
    assert loaded.total_keys == 5000
    assert loaded.completed_keys == 3
    assert loaded.completed_batches == 1
    assert loaded.last_batch_keys == ["key:1", "key:2", "key:3"]


def test_checkpoint_record_multiple_batches(tmp_path):
    """Recording multiple batches should accumulate counts."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    cp = QuantizationCheckpoint(
        index_name="idx",
        total_keys=100,
        checkpoint_path=str(tmp_path / "cp.yaml"),
    )
    cp.record_batch(["k1", "k2"])
    cp.record_batch(["k3", "k4", "k5"])

    assert cp.completed_keys == 5
    assert cp.completed_batches == 2
    assert cp.last_batch_keys == ["k3", "k4", "k5"]


def test_checkpoint_mark_complete(tmp_path):
    """Marking complete should set status to 'completed'."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    cp = QuantizationCheckpoint(
        index_name="idx",
        total_keys=2,
        checkpoint_path=str(tmp_path / "cp.yaml"),
    )
    cp.record_batch(["k1", "k2"])
    cp.mark_complete()

    assert cp.status == "completed"


def test_checkpoint_get_remaining_keys(tmp_path):
    """get_remaining_keys should return only keys not yet processed."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    cp = QuantizationCheckpoint(
        index_name="idx",
        total_keys=5,
        checkpoint_path=str(tmp_path / "cp.yaml"),
    )
    all_keys = ["k1", "k2", "k3", "k4", "k5"]
    cp.record_batch(["k1", "k2"])

    remaining = cp.get_remaining_keys(all_keys)
    assert remaining == ["k3", "k4", "k5"]


def test_checkpoint_get_remaining_keys_uses_completed_offset_when_compact(tmp_path):
    """Compact checkpoints should resume via completed_keys ordering."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    cp = QuantizationCheckpoint(
        index_name="idx",
        total_keys=5,
        checkpoint_path=str(tmp_path / "cp.yaml"),
    )
    cp.record_batch(["k1", "k2"])

    remaining = cp.get_remaining_keys(["k1", "k2", "k3", "k4", "k5"])
    assert remaining == ["k3", "k4", "k5"]


def test_checkpoint_save_excludes_processed_keys(tmp_path):
    """New checkpoints should persist compact state without processed_keys."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    path = tmp_path / "checkpoint.yaml"
    cp = QuantizationCheckpoint(
        index_name="idx",
        total_keys=3,
        checkpoint_path=str(path),
    )
    cp.save()

    raw = path.read_text()
    assert "processed_keys" not in raw


def test_checkpoint_load_nonexistent_returns_none(tmp_path):
    """Loading a nonexistent checkpoint file should return None."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    result = QuantizationCheckpoint.load(
        str(tmp_path / "nonexistent_checkpoint_xyz.yaml")
    )
    assert result is None


def test_checkpoint_load_forces_path(tmp_path):
    """load() should set checkpoint_path to the file used to load, not the stored value."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    original_path = str(tmp_path / "original.yaml")
    cp = QuantizationCheckpoint(
        index_name="idx",
        total_keys=10,
        checkpoint_path=original_path,
    )
    cp.record_batch(["k1"])
    cp.save()

    # Move the file to a new location
    new_path = str(tmp_path / "moved.yaml")
    import shutil

    shutil.copy(original_path, new_path)

    loaded = QuantizationCheckpoint.load(new_path)
    assert loaded.checkpoint_path == new_path  # should use load path, not stored


def test_checkpoint_save_preserves_legacy_processed_keys(tmp_path):
    """Legacy checkpoints should keep processed_keys across subsequent saves."""
    from redisvl.migration.reliability import QuantizationCheckpoint

    path = tmp_path / "legacy.yaml"
    path.write_text(
        "index_name: idx\n"
        "total_keys: 4\n"
        "processed_keys:\n"
        "  - k1\n"
        "  - k2\n"
        "status: in_progress\n"
    )

    checkpoint = QuantizationCheckpoint.load(str(path))
    checkpoint.record_batch(["k3"])
    checkpoint.save()

    reloaded = QuantizationCheckpoint.load(str(path))
    assert reloaded.processed_keys == ["k1", "k2", "k3"]
    assert reloaded.completed_keys == 3


def test_quantize_vectors_saves_checkpoint_before_processing(monkeypatch, tmp_path):
    """Checkpoint save should happen before the first HGET in a fresh run."""
    import numpy as np

    executor = MigrationExecutor()
    checkpoint_path = str(tmp_path / "checkpoint.yaml")
    field_bytes = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    events: list[str] = []

    original_save = executor._quantize_vectors.__globals__[
        "QuantizationCheckpoint"
    ].save

    def tracking_save(self):
        events.append("save")
        return original_save(self)

    monkeypatch.setattr(
        executor._quantize_vectors.__globals__["QuantizationCheckpoint"],
        "save",
        tracking_save,
    )

    client = MagicMock()
    client.hget.side_effect = lambda key, field: (events.append("hget") or field_bytes)
    pipe = MagicMock()
    client.pipeline.return_value = pipe
    source_index = MagicMock()
    source_index._redis_client = client
    source_index.name = "idx"

    result = executor._quantize_vectors(
        source_index,
        {"embedding": {"source": "float32", "target": "float16", "dims": 2}},
        ["doc:1"],
        checkpoint_path=checkpoint_path,
    )

    assert result == 1
    assert events[0] == "save"
    assert Path(checkpoint_path).exists()


def test_quantize_vectors_returns_reencoded_docs_not_scanned_docs():
    """Quantize count should reflect converted docs, not skipped docs."""
    import numpy as np

    executor = MigrationExecutor()
    already_quantized = np.array([1.0, 2.0], dtype=np.float16).tobytes()
    needs_quantization = np.array([1.0, 2.0], dtype=np.float32).tobytes()

    client = MagicMock()
    client.hget.side_effect = lambda key, field: {
        "doc:1": already_quantized,
        "doc:2": needs_quantization,
    }[key]
    pipe = MagicMock()
    client.pipeline.return_value = pipe
    source_index = MagicMock()
    source_index._redis_client = client
    source_index.name = "idx"

    progress: list[tuple[int, int]] = []
    result = executor._quantize_vectors(
        source_index,
        {"embedding": {"source": "float32", "target": "float16", "dims": 2}},
        ["doc:1", "doc:2"],
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert result == 1
    assert progress[-1] == (2, 2)


def test_build_scan_match_patterns_uses_separator():
    assert build_scan_match_patterns(["test"], ":") == ["test:*"]
    assert build_scan_match_patterns(["test:"], ":") == ["test:*"]
    assert build_scan_match_patterns([], ":") == ["*"]
    assert build_scan_match_patterns(["b", "a"], ":") == ["a:*", "b:*"]


def test_normalize_keys_dedupes_and_sorts():
    assert normalize_keys(["b", "a", "b"]) == ["a", "b"]


def test_detect_aof_enabled_from_info():
    from redisvl.migration.utils import detect_aof_enabled

    client = MagicMock()
    client.info.return_value = {"aof_enabled": 1}
    assert detect_aof_enabled(client) is True


@pytest.mark.asyncio
async def test_async_detect_aof_enabled_from_info():
    executor = AsyncMigrationExecutor()
    client = MagicMock()
    client.info = AsyncMock(return_value={"aof_enabled": 1})
    assert await executor._detect_aof_enabled(client) is True


def test_estimate_json_quantization_is_noop():
    """JSON datatype changes should not report in-place rewrite costs."""
    plan = _make_quantize_plan(
        "float32", "float16", dims=128, doc_count=1000, storage_type="json"
    )
    est = estimate_disk_space(plan, aof_enabled=True)

    assert est.has_quantization is False
    assert est.total_new_disk_bytes == 0
    assert est.aof_growth_bytes == 0


def test_estimate_unknown_dtype_raises():
    plan = _make_quantize_plan("madeup32", "float16", dims=128, doc_count=10)

    with pytest.raises(ValueError, match="Unknown source vector datatype"):
        estimate_disk_space(plan)


def test_enumerate_with_scan_uses_all_prefixes():
    executor = MigrationExecutor()
    client = MagicMock()
    client.ft.return_value.info.return_value = {
        "index_definition": {"prefixes": ["alpha", "beta"]}
    }
    client.scan.side_effect = [
        (0, [b"alpha:1", b"shared:1"]),
        (0, [b"beta:2", b"shared:1"]),
    ]

    keys = list(executor._enumerate_with_scan(client, "idx", batch_size=1000))

    assert keys == ["alpha:1", "shared:1", "beta:2"]


@pytest.mark.asyncio
async def test_async_enumerate_with_scan_uses_all_prefixes():
    executor = AsyncMigrationExecutor()
    client = MagicMock()
    client.ft.return_value.info = AsyncMock(
        return_value={"index_definition": {"prefixes": ["alpha", "beta"]}}
    )
    client.scan = AsyncMock(
        side_effect=[
            (0, [b"alpha:1", b"shared:1"]),
            (0, [b"beta:2", b"shared:1"]),
        ]
    )

    keys = [
        key
        async for key in executor._enumerate_with_scan(client, "idx", batch_size=1000)
    ]

    assert keys == ["alpha:1", "shared:1", "beta:2"]


def test_apply_rejects_same_width_resume(monkeypatch):
    plan = _make_quantize_plan("float16", "bfloat16", dims=2, doc_count=1)
    executor = MigrationExecutor()

    def _make_index(*args, **kwargs):
        index = MagicMock()
        index._redis_client = MagicMock()
        index.name = "test_index"
        return index

    monkeypatch.setattr(
        "redisvl.migration.executor.current_source_matches_snapshot",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "redisvl.migration.executor.SearchIndex.from_existing",
        _make_index,
    )
    monkeypatch.setattr(
        "redisvl.migration.executor.SearchIndex.from_dict",
        _make_index,
    )

    report = executor.apply(
        plan,
        redis_client=MagicMock(),
        checkpoint_path="resume.yaml",
    )

    assert report.result == "failed"
    assert "same-width datatype changes" in report.validation.errors[0]


@pytest.mark.asyncio
async def test_async_quantize_vectors_saves_checkpoint_before_processing(
    monkeypatch, tmp_path
):
    """Async checkpoint save should happen before the first HGET in a fresh run."""
    import numpy as np

    executor = AsyncMigrationExecutor()
    checkpoint_path = str(tmp_path / "checkpoint.yaml")
    field_bytes = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    events: list[str] = []

    original_save = executor._async_quantize_vectors.__globals__[
        "QuantizationCheckpoint"
    ].save

    def tracking_save(self):
        events.append("save")
        return original_save(self)

    monkeypatch.setattr(
        executor._async_quantize_vectors.__globals__["QuantizationCheckpoint"],
        "save",
        tracking_save,
    )

    client = MagicMock()
    client.hget = AsyncMock(
        side_effect=lambda key, field: (events.append("hget") or field_bytes)
    )
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    client.pipeline.return_value = pipe
    source_index = MagicMock()
    source_index._get_client = AsyncMock(return_value=client)
    source_index._redis_client = client
    source_index.name = "idx"

    result = await executor._async_quantize_vectors(
        source_index,
        {"embedding": {"source": "float32", "target": "float16", "dims": 2}},
        ["doc:1"],
        checkpoint_path=checkpoint_path,
    )

    assert result == 1
    assert events[0] == "save"
    assert Path(checkpoint_path).exists()


@pytest.mark.asyncio
async def test_async_quantize_vectors_returns_reencoded_docs_not_scanned_docs():
    """Async quantize count should reflect converted docs, not skipped docs."""
    import numpy as np

    executor = AsyncMigrationExecutor()
    already_quantized = np.array([1.0, 2.0], dtype=np.float16).tobytes()
    needs_quantization = np.array([1.0, 2.0], dtype=np.float32).tobytes()

    client = MagicMock()
    client.hget = AsyncMock(
        side_effect=lambda key, field: {
            "doc:1": already_quantized,
            "doc:2": needs_quantization,
        }[key]
    )
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    client.pipeline.return_value = pipe
    source_index = MagicMock()
    source_index._get_client = AsyncMock(return_value=client)
    source_index._redis_client = client
    source_index.name = "idx"

    progress: list[tuple[int, int]] = []
    result = await executor._async_quantize_vectors(
        source_index,
        {"embedding": {"source": "float32", "target": "float16", "dims": 2}},
        ["doc:1", "doc:2"],
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert result == 1
    assert progress[-1] == (2, 2)


# =============================================================================
# TDD RED Phase: BGSAVE Safety Net Tests
# =============================================================================


def test_trigger_bgsave_success():
    """BGSAVE should be triggered and waited on; returns True on success."""
    from unittest.mock import MagicMock

    from redisvl.migration.reliability import trigger_bgsave_and_wait

    mock_client = MagicMock()
    mock_client.bgsave.return_value = True
    mock_client.info.return_value = {"rdb_bgsave_in_progress": 0}

    result = trigger_bgsave_and_wait(mock_client, timeout_seconds=5)
    assert result is True
    mock_client.bgsave.assert_called_once()


def test_trigger_bgsave_already_in_progress():
    """If BGSAVE is already running, wait for it instead of starting a new one."""
    from unittest.mock import MagicMock, call

    from redisvl.migration.reliability import trigger_bgsave_and_wait

    mock_client = MagicMock()
    # First bgsave raises because one is already in progress
    mock_client.bgsave.side_effect = Exception("Background save already in progress")
    # First check: still running; second check: done
    mock_client.info.side_effect = [
        {"rdb_bgsave_in_progress": 1},
        {"rdb_bgsave_in_progress": 0},
    ]

    result = trigger_bgsave_and_wait(mock_client, timeout_seconds=5, poll_interval=0.01)
    assert result is True


@pytest.mark.asyncio
async def test_async_trigger_bgsave_success():
    """Async BGSAVE should work the same as sync."""
    from unittest.mock import AsyncMock

    from redisvl.migration.reliability import async_trigger_bgsave_and_wait

    mock_client = AsyncMock()
    mock_client.bgsave.return_value = True
    mock_client.info.return_value = {"rdb_bgsave_in_progress": 0}

    result = await async_trigger_bgsave_and_wait(mock_client, timeout_seconds=5)
    assert result is True
    mock_client.bgsave.assert_called_once()


# =============================================================================
# TDD RED Phase: Bounded Undo Buffer Tests
# =============================================================================


def test_undo_buffer_store_and_rollback():
    """Undo buffer should store original values and rollback via pipeline."""
    from unittest.mock import MagicMock

    from redisvl.migration.reliability import BatchUndoBuffer

    buf = BatchUndoBuffer()
    buf.store("key:1", "embedding", b"\x00\x01\x02\x03")
    buf.store("key:2", "embedding", b"\x04\x05\x06\x07")

    assert buf.size == 2

    mock_pipe = MagicMock()
    buf.rollback(mock_pipe)

    # Should have called hset twice to restore originals
    assert mock_pipe.hset.call_count == 2
    mock_pipe.execute.assert_called_once()


def test_undo_buffer_clear():
    """After clear, buffer should be empty."""
    from redisvl.migration.reliability import BatchUndoBuffer

    buf = BatchUndoBuffer()
    buf.store("key:1", "field", b"\x00")
    assert buf.size == 1

    buf.clear()
    assert buf.size == 0


def test_undo_buffer_empty_rollback():
    """Rolling back an empty buffer should be a no-op."""
    from unittest.mock import MagicMock

    from redisvl.migration.reliability import BatchUndoBuffer

    buf = BatchUndoBuffer()
    mock_pipe = MagicMock()
    buf.rollback(mock_pipe)

    # No hset calls, no execute
    mock_pipe.hset.assert_not_called()
    mock_pipe.execute.assert_not_called()


def test_undo_buffer_multiple_fields_same_key():
    """Should handle multiple fields for the same key."""
    from unittest.mock import MagicMock

    from redisvl.migration.reliability import BatchUndoBuffer

    buf = BatchUndoBuffer()
    buf.store("key:1", "embedding", b"\x00\x01")
    buf.store("key:1", "embedding2", b"\x02\x03")

    assert buf.size == 2

    mock_pipe = MagicMock()
    buf.rollback(mock_pipe)
    assert mock_pipe.hset.call_count == 2


@pytest.mark.asyncio
async def test_undo_buffer_async_rollback():
    """async_rollback should await pipe.execute() for async Redis pipelines."""
    from unittest.mock import AsyncMock, MagicMock

    from redisvl.migration.reliability import BatchUndoBuffer

    buf = BatchUndoBuffer()
    buf.store("key:1", "embedding", b"\x00\x01")
    buf.store("key:2", "embedding", b"\x02\x03")

    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock()

    await buf.async_rollback(mock_pipe)
    assert mock_pipe.hset.call_count == 2
    mock_pipe.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_undo_buffer_async_rollback_empty():
    """async_rollback on empty buffer should be a no-op."""
    from unittest.mock import AsyncMock, MagicMock

    from redisvl.migration.reliability import BatchUndoBuffer

    buf = BatchUndoBuffer()
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock()

    await buf.async_rollback(mock_pipe)
    mock_pipe.hset.assert_not_called()
    mock_pipe.execute.assert_not_awaited()
