"""
Integration tests for migration routes.

Tests the full Apply+Validate flow for all supported migration operations.
Requires Redis 8.0+ for INT8/UINT8 datatype tests.
"""

import uuid

import pytest

from redisvl.index import SearchIndex
from redisvl.migration import MigrationExecutor, MigrationPlanner
from redisvl.migration.models import FieldUpdate, SchemaPatch


def create_source_index(redis_url, worker_id, source_attrs):
    """Helper to create a source index with specified vector attributes."""
    unique_id = str(uuid.uuid4())[:8]
    index_name = f"mig_route_{worker_id}_{unique_id}"
    prefix = f"mig_route:{worker_id}:{unique_id}"

    base_attrs = {
        "dims": 128,
        "datatype": "float32",
        "distance_metric": "cosine",
        "algorithm": "flat",
    }
    base_attrs.update(source_attrs)

    index = SearchIndex.from_dict(
        {
            "index": {"name": index_name, "prefix": prefix, "storage_type": "json"},
            "fields": [
                {"name": "title", "type": "text", "path": "$.title"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "path": "$.embedding",
                    "attrs": base_attrs,
                },
            ],
        },
        redis_url=redis_url,
    )
    index.create(overwrite=True)
    return index, index_name


def run_migration(redis_url, index_name, patch_attrs):
    """Helper to run a migration with the given patch attributes."""
    patch = SchemaPatch(
        version=1,
        changes={
            "add_fields": [],
            "remove_fields": [],
            "update_fields": [FieldUpdate(name="embedding", attrs=patch_attrs)],
            "rename_fields": [],
            "index": {},
        },
    )

    planner = MigrationPlanner()
    plan = planner.create_plan_from_patch(
        index_name, schema_patch=patch, redis_url=redis_url
    )

    executor = MigrationExecutor()
    report = executor.apply(plan, redis_url=redis_url)
    return report, plan


class TestAlgorithmChanges:
    """Test algorithm migration routes."""

    def test_hnsw_to_flat(self, redis_url, worker_id):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "hnsw"}
        )
        try:
            report, _ = run_migration(redis_url, index_name, {"algorithm": "flat"})
            assert report.result == "succeeded"
            assert report.validation.schema_match is True

            live = SearchIndex.from_existing(index_name, redis_url=redis_url)
            assert str(live.schema.fields["embedding"].attrs.algorithm).endswith("FLAT")
        finally:
            index.delete(drop=True)

    def test_flat_to_hnsw_with_params(self, redis_url, worker_id):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "flat"}
        )
        try:
            report, _ = run_migration(
                redis_url,
                index_name,
                {"algorithm": "hnsw", "m": 32, "ef_construction": 200},
            )
            assert report.result == "succeeded"
            assert report.validation.schema_match is True

            live = SearchIndex.from_existing(index_name, redis_url=redis_url)
            attrs = live.schema.fields["embedding"].attrs
            assert str(attrs.algorithm).endswith("HNSW")
            assert attrs.m == 32
            assert attrs.ef_construction == 200
        finally:
            index.delete(drop=True)


class TestDatatypeChanges:
    """Test datatype migration routes."""

    @pytest.mark.parametrize(
        "source_dtype,target_dtype",
        [
            ("float32", "float16"),
            ("float32", "bfloat16"),
            ("float16", "float32"),
        ],
    )
    def test_flat_datatype_change(
        self, redis_url, worker_id, source_dtype, target_dtype
    ):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "flat", "datatype": source_dtype}
        )
        try:
            report, _ = run_migration(redis_url, index_name, {"datatype": target_dtype})
            assert report.result == "succeeded"
            assert report.validation.schema_match is True
        finally:
            index.delete(drop=True)

    @pytest.mark.parametrize("target_dtype", ["int8", "uint8"])
    def test_flat_quantized_datatype(self, redis_url, worker_id, target_dtype):
        """Test INT8/UINT8 datatypes (requires Redis 8.0+)."""
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "flat"}
        )
        try:
            report, _ = run_migration(redis_url, index_name, {"datatype": target_dtype})
            assert report.result == "succeeded"
            assert report.validation.schema_match is True
        finally:
            index.delete(drop=True)

    @pytest.mark.parametrize(
        "source_dtype,target_dtype",
        [
            ("float32", "float16"),
            ("float32", "bfloat16"),
        ],
    )
    def test_hnsw_datatype_change(
        self, redis_url, worker_id, source_dtype, target_dtype
    ):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "hnsw", "datatype": source_dtype}
        )
        try:
            report, _ = run_migration(redis_url, index_name, {"datatype": target_dtype})
            assert report.result == "succeeded"
            assert report.validation.schema_match is True
        finally:
            index.delete(drop=True)

    @pytest.mark.parametrize("target_dtype", ["int8", "uint8"])
    def test_hnsw_quantized_datatype(self, redis_url, worker_id, target_dtype):
        """Test INT8/UINT8 datatypes with HNSW (requires Redis 8.0+)."""
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "hnsw"}
        )
        try:
            report, _ = run_migration(redis_url, index_name, {"datatype": target_dtype})
            assert report.result == "succeeded"
            assert report.validation.schema_match is True
        finally:
            index.delete(drop=True)


class TestDistanceMetricChanges:
    """Test distance metric migration routes."""

    @pytest.mark.parametrize(
        "source_metric,target_metric",
        [
            ("cosine", "l2"),
            ("cosine", "ip"),
            ("l2", "cosine"),
            ("ip", "l2"),
        ],
    )
    def test_distance_metric_change(
        self, redis_url, worker_id, source_metric, target_metric
    ):
        index, index_name = create_source_index(
            redis_url,
            worker_id,
            {"algorithm": "flat", "distance_metric": source_metric},
        )
        try:
            report, _ = run_migration(
                redis_url, index_name, {"distance_metric": target_metric}
            )
            assert report.result == "succeeded"
            assert report.validation.schema_match is True
        finally:
            index.delete(drop=True)


class TestHNSWTuningParameters:
    """Test HNSW parameter tuning routes."""

    def test_hnsw_m_parameter(self, redis_url, worker_id):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "hnsw"}
        )
        try:
            report, _ = run_migration(redis_url, index_name, {"m": 64})
            assert report.result == "succeeded"
            assert report.validation.schema_match is True

            live = SearchIndex.from_existing(index_name, redis_url=redis_url)
            assert live.schema.fields["embedding"].attrs.m == 64
        finally:
            index.delete(drop=True)

    def test_hnsw_ef_construction_parameter(self, redis_url, worker_id):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "hnsw"}
        )
        try:
            report, _ = run_migration(redis_url, index_name, {"ef_construction": 500})
            assert report.result == "succeeded"
            assert report.validation.schema_match is True

            live = SearchIndex.from_existing(index_name, redis_url=redis_url)
            assert live.schema.fields["embedding"].attrs.ef_construction == 500
        finally:
            index.delete(drop=True)

    def test_hnsw_ef_runtime_parameter(self, redis_url, worker_id):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "hnsw"}
        )
        try:
            report, _ = run_migration(redis_url, index_name, {"ef_runtime": 50})
            assert report.result == "succeeded"
            assert report.validation.schema_match is True
        finally:
            index.delete(drop=True)

    def test_hnsw_epsilon_parameter(self, redis_url, worker_id):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "hnsw"}
        )
        try:
            report, _ = run_migration(redis_url, index_name, {"epsilon": 0.1})
            assert report.result == "succeeded"
            assert report.validation.schema_match is True
        finally:
            index.delete(drop=True)

    def test_hnsw_all_params_combined(self, redis_url, worker_id):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "hnsw"}
        )
        try:
            report, _ = run_migration(
                redis_url,
                index_name,
                {"m": 48, "ef_construction": 300, "ef_runtime": 75, "epsilon": 0.05},
            )
            assert report.result == "succeeded"
            assert report.validation.schema_match is True

            live = SearchIndex.from_existing(index_name, redis_url=redis_url)
            attrs = live.schema.fields["embedding"].attrs
            assert attrs.m == 48
            assert attrs.ef_construction == 300
        finally:
            index.delete(drop=True)


class TestCombinedChanges:
    """Test combined migration routes (multiple changes at once)."""

    def test_flat_to_hnsw_with_datatype_and_metric(self, redis_url, worker_id):
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "flat"}
        )
        try:
            report, _ = run_migration(
                redis_url,
                index_name,
                {"algorithm": "hnsw", "datatype": "float16", "distance_metric": "l2"},
            )
            assert report.result == "succeeded"
            assert report.validation.schema_match is True

            live = SearchIndex.from_existing(index_name, redis_url=redis_url)
            attrs = live.schema.fields["embedding"].attrs
            assert str(attrs.algorithm).endswith("HNSW")
        finally:
            index.delete(drop=True)

    def test_flat_to_hnsw_with_int8(self, redis_url, worker_id):
        """Combined algorithm + quantized datatype (requires Redis 8.0+)."""
        index, index_name = create_source_index(
            redis_url, worker_id, {"algorithm": "flat"}
        )
        try:
            report, _ = run_migration(
                redis_url,
                index_name,
                {"algorithm": "hnsw", "datatype": "int8"},
            )
            assert report.result == "succeeded"
            assert report.validation.schema_match is True
        finally:
            index.delete(drop=True)
