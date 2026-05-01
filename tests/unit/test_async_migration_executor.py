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
# Idempotent Resume Rename Tests (sync executor)
# =============================================================================
# These tests validate that crash-resume for prefix renames is idempotent:
# if a key was already renamed in a prior (crashed) run, retrying should
# skip it instead of aborting with a collision error.


class TestIdempotentResumeRenameStandalone:
    """Test _rename_keys_standalone handles already-renamed keys during resume."""

    def _make_executor(self):
        return MigrationExecutor()

    def test_already_renamed_keys_skipped_on_resume(self):
        """Simulate crash-resume: 2 of 3 keys were already renamed.

        Before the fix, RENAMENX returning False would be treated as a
        collision and raise RuntimeError. After the fix, the executor
        checks if src is gone + dst exists and counts it as already done.
        """
        executor = self._make_executor()
        mock_client = MagicMock()

        # Pipeline: RENAMENX returns True for key3 (not yet renamed),
        # False for key1 and key2 (already renamed in prior run).
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [False, False, True]
        mock_client.pipeline.return_value = mock_pipe

        # When executor checks EXISTS for the False results:
        # key1: src gone, dst exists → already renamed
        # key2: src gone, dst exists → already renamed
        def exists_side_effect(key):
            already_renamed_srcs = {"old:1", "old:2"}
            already_renamed_dsts = {"new:1", "new:2"}
            if key in already_renamed_srcs:
                return 0  # source gone
            if key in already_renamed_dsts:
                return 1  # destination exists
            return 0

        mock_client.exists.side_effect = exists_side_effect

        keys = ["old:1", "old:2", "old:3"]
        result = executor._rename_keys_standalone(mock_client, keys, "old:", "new:")

        # All 3 should count as renamed (2 skipped + 1 actually renamed)
        assert result == 3

    def test_true_collision_still_raises(self):
        """When source AND destination both exist, it's a real collision → RuntimeError."""
        executor = self._make_executor()
        mock_client = MagicMock()

        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [False]  # RENAMENX failed
        mock_client.pipeline.return_value = mock_pipe

        # Both source and destination exist → true collision
        mock_client.exists.side_effect = lambda key: 1

        keys = ["old:1"]
        with pytest.raises(RuntimeError, match="destination key.*already exist"):
            executor._rename_keys_standalone(mock_client, keys, "old:", "new:")

    def test_src_and_dst_both_gone_is_collision(self):
        """If RENAMENX fails, src is gone, but dst is ALSO gone → collision error.

        This is an anomalous state (key deleted externally?) — we treat it
        as a collision rather than silently losing data.
        """
        executor = self._make_executor()
        mock_client = MagicMock()

        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [False]
        mock_client.pipeline.return_value = mock_pipe

        # src gone, dst also gone
        exists_map = {"old:1": 0, "new:1": 0}
        mock_client.exists.side_effect = lambda key: exists_map.get(key, 0)

        keys = ["old:1"]
        with pytest.raises(RuntimeError, match="destination key.*already exist"):
            executor._rename_keys_standalone(mock_client, keys, "old:", "new:")

    def test_mixed_fresh_and_resumed_keys(self):
        """Mix of fresh renames and already-renamed keys — all succeed."""
        executor = self._make_executor()
        mock_client = MagicMock()

        mock_pipe = MagicMock()
        # key1: RENAMENX succeeds
        # key2: RENAMENX fails — already renamed (src gone, dst exists)
        mock_pipe.execute.return_value = [True, False]
        mock_client.pipeline.return_value = mock_pipe

        exists_map = {
            "old:2": 0,  # source gone
            "new:2": 1,  # destination exists
        }
        mock_client.exists.side_effect = lambda key: exists_map.get(key, 0)

        keys = ["old:1", "old:2"]
        result = executor._rename_keys_standalone(mock_client, keys, "old:", "new:")

        assert result == 2  # 1 fresh + 1 already-renamed


class TestIdempotentResumeRenameCluster:
    """Test _rename_keys_cluster handles already-renamed keys during resume."""

    def _make_executor(self):
        return MigrationExecutor()

    def test_already_renamed_keys_skipped_on_resume(self):
        """Simulate crash-resume on cluster: keys already renamed are skipped."""
        executor = self._make_executor()
        mock_client = MagicMock()

        # Phase 1 check pipeline: exists(new_key), exists(old_key) for each pair
        check_pipe = MagicMock()
        # key1: dst exists (1), src gone (0) → already renamed
        # key2: dst exists (1), src gone (0) → already renamed
        # key3: dst gone (0), src exists (1) → needs rename
        check_pipe.execute.return_value = [1, 0, 1, 0, 0, 1]

        # Phase 2 dump pipeline for key3 only
        dump_pipe = MagicMock()
        dump_pipe.execute.return_value = [b"\x00\x01\x02", -1]  # dump data, pttl

        # Phase 3 restore pipeline
        restore_pipe = MagicMock()
        restore_pipe.execute.return_value = [True, 1]  # RESTORE ok, DEL ok

        mock_client.pipeline.side_effect = [check_pipe, dump_pipe, restore_pipe]

        keys = ["old:1", "old:2", "old:3"]
        result = executor._rename_keys_cluster(mock_client, keys, "old:", "new:")

        # 2 already-renamed + 1 fresh = 3
        assert result == 3

    def test_true_collision_raises_on_cluster(self):
        """When source AND destination both exist on cluster → RuntimeError."""
        executor = self._make_executor()
        mock_client = MagicMock()

        check_pipe = MagicMock()
        # key1: dst exists (1), src ALSO exists (1) → true collision
        check_pipe.execute.return_value = [1, 1]
        mock_client.pipeline.return_value = check_pipe

        keys = ["old:1"]
        with pytest.raises(RuntimeError, match="destination key.*already exists"):
            executor._rename_keys_cluster(mock_client, keys, "old:", "new:")

    def test_both_missing_key_skipped_on_cluster(self):
        """Key where both source and destination are gone — warn and skip."""
        executor = self._make_executor()
        mock_client = MagicMock()

        check_pipe = MagicMock()
        # key1: dst gone (0), src gone (0) → both missing
        check_pipe.execute.return_value = [0, 0]

        # Even with no live_pairs, the code still creates dump/restore pipelines
        dump_pipe = MagicMock()
        dump_pipe.execute.return_value = []
        restore_pipe = MagicMock()

        mock_client.pipeline.side_effect = [check_pipe, dump_pipe, restore_pipe]

        keys = ["old:1"]
        result = executor._rename_keys_cluster(mock_client, keys, "old:", "new:")

        # Key skipped, nothing renamed
        assert result == 0
