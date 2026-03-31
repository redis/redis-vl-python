from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from redisvl.index import SearchIndex
from redisvl.migration.models import (
    AOF_HSET_OVERHEAD_BYTES,
    AOF_JSON_SET_OVERHEAD_BYTES,
    DTYPE_BYTES,
    RDB_COMPRESSION_RATIO,
    DiskSpaceEstimate,
    MigrationPlan,
    MigrationReport,
    VectorFieldEstimate,
)
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.schema.schema import IndexSchema


def list_indexes(
    *, redis_url: Optional[str] = None, redis_client: Optional[Any] = None
):
    if redis_client is None:
        if not redis_url:
            raise ValueError("Must provide either redis_url or redis_client")
        redis_client = RedisConnectionFactory.get_redis_connection(redis_url=redis_url)
    index = SearchIndex.from_dict(
        {"index": {"name": "__redisvl_migration_helper__"}, "fields": []},
        redis_client=redis_client,
    )
    return index.listall()


def load_yaml(path: str) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    with open(resolved, "r") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: Dict[str, Any], path: str) -> None:
    resolved = Path(path).resolve()
    with open(resolved, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_migration_plan(path: str) -> MigrationPlan:
    return MigrationPlan.model_validate(load_yaml(path))


def write_migration_report(report: MigrationReport, path: str) -> None:
    write_yaml(report.model_dump(exclude_none=True), path)


def write_benchmark_report(report: MigrationReport, path: str) -> None:
    benchmark_report = {
        "version": report.version,
        "mode": report.mode,
        "source_index": report.source_index,
        "target_index": report.target_index,
        "result": report.result,
        "timings": report.timings.model_dump(exclude_none=True),
        "benchmark_summary": report.benchmark_summary.model_dump(exclude_none=True),
        "validation": {
            "schema_match": report.validation.schema_match,
            "doc_count_match": report.validation.doc_count_match,
            "indexing_failures_delta": report.validation.indexing_failures_delta,
            "key_sample_exists": report.validation.key_sample_exists,
        },
    }
    write_yaml(benchmark_report, path)


def normalize_keys(keys: List[str]) -> List[str]:
    """Deduplicate and sort keys for deterministic resume behavior."""
    return sorted(set(keys))


def build_scan_match_patterns(prefixes: List[str], key_separator: str) -> List[str]:
    """Build SCAN patterns for all configured prefixes."""
    if not prefixes:
        return ["*"]

    patterns = set()
    for prefix in prefixes:
        if not prefix:
            return ["*"]
        if key_separator and not prefix.endswith(key_separator):
            patterns.add(f"{prefix}{key_separator}*")
        else:
            patterns.add(f"{prefix}*")
    return sorted(patterns)


def detect_aof_enabled(client: Any) -> bool:
    """Best-effort detection of whether AOF is enabled on the live Redis."""
    try:
        info = client.info("persistence")
        if isinstance(info, dict) and "aof_enabled" in info:
            return bool(int(info["aof_enabled"]))
    except Exception:
        pass

    try:
        config = client.config_get("appendonly")
        if isinstance(config, dict):
            value = config.get("appendonly")
            if value is not None:
                return str(value).lower() in {"yes", "1", "true", "on"}
    except Exception:
        pass

    return False


def get_schema_field_path(schema: Dict[str, Any], field_name: str) -> Optional[str]:
    """Return the JSON path configured for a field, if present."""
    for field in schema.get("fields", []):
        if field.get("name") != field_name:
            continue
        path = field.get("path")
        if path is None:
            path = field.get("attrs", {}).get("path")
        return str(path) if path is not None else None
    return None


# Attributes excluded from schema validation comparison.
# These are query-time or creation-hint parameters that FT.INFO does not return
# and are not relevant for index structure validation (confirmed by RediSearch team).
# - ef_runtime, epsilon: query-time tuning knobs, not index definition attributes
# - initial_cap: creation-time memory pre-allocation hint, diverges after indexing
EXCLUDED_VECTOR_ATTRS = {"ef_runtime", "epsilon", "initial_cap"}
# phonetic_matcher: the matcher string (e.g. "dm:en") is not stored server-side,
#   only a boolean flag is kept, so it cannot be read back.
# withsuffixtrie: returned as a flag in FT.INFO but not as a KV attribute,
#   so RedisVL's parser does not capture it yet.
EXCLUDED_TEXT_ATTRS = {"phonetic_matcher", "withsuffixtrie"}
EXCLUDED_TAG_ATTRS = {"withsuffixtrie"}


def _strip_excluded_attrs(field: Dict[str, Any]) -> Dict[str, Any]:
    """Remove attributes not relevant for index validation comparison.

    These are either query-time parameters, creation-time hints, or attributes
    whose server-side representation differs from the schema definition.

    Also normalizes attributes that have implicit behavior:
    - For NUMERIC + SORTABLE, Redis auto-applies UNF, so we normalize to unf=True
    """
    field = field.copy()
    attrs = field.get("attrs", {})
    if not attrs:
        return field

    attrs = attrs.copy()
    field_type = field.get("type", "").lower()

    if field_type == "vector":
        for attr in EXCLUDED_VECTOR_ATTRS:
            attrs.pop(attr, None)
    elif field_type == "text":
        for attr in EXCLUDED_TEXT_ATTRS:
            attrs.pop(attr, None)
        # Normalize weight to int for comparison (FT.INFO may return float)
        if "weight" in attrs and isinstance(attrs["weight"], float):
            if attrs["weight"] == int(attrs["weight"]):
                attrs["weight"] = int(attrs["weight"])
    elif field_type == "tag":
        for attr in EXCLUDED_TAG_ATTRS:
            attrs.pop(attr, None)
    elif field_type == "numeric":
        # Redis auto-applies UNF when SORTABLE is set on NUMERIC fields.
        # Normalize unf to True when sortable is True to avoid false mismatches.
        if attrs.get("sortable"):
            attrs["unf"] = True

    field["attrs"] = attrs
    return field


def canonicalize_schema(
    schema_dict: Dict[str, Any],
    *,
    strip_unreliable: bool = False,
    strip_excluded: bool = False,
) -> Dict[str, Any]:
    """Canonicalize schema for comparison.

    Args:
        schema_dict: The schema dictionary to canonicalize.
        strip_unreliable: Deprecated alias for strip_excluded. Kept for
            backward compatibility.
        strip_excluded: If True, remove query-time and creation-hint attributes
            that are not part of index structure validation.
    """
    schema = IndexSchema.from_dict(schema_dict).to_dict()

    should_strip = strip_excluded or strip_unreliable
    fields = schema.get("fields", [])
    if should_strip:
        fields = [_strip_excluded_attrs(f) for f in fields]

    schema["fields"] = sorted(fields, key=lambda field: field["name"])
    prefixes = schema["index"].get("prefix")
    if isinstance(prefixes, list):
        schema["index"]["prefix"] = sorted(prefixes)
    stopwords = schema["index"].get("stopwords")
    if isinstance(stopwords, list):
        schema["index"]["stopwords"] = list(stopwords)
    return schema


def schemas_equal(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    strip_unreliable: bool = False,
    strip_excluded: bool = False,
) -> bool:
    """Compare two schemas for equality.

    Args:
        left: First schema dictionary.
        right: Second schema dictionary.
        strip_unreliable: Deprecated alias for strip_excluded. Kept for
            backward compatibility.
        strip_excluded: If True, exclude query-time and creation-hint attributes
            (ef_runtime, epsilon, initial_cap, phonetic_matcher) from comparison.
    """
    should_strip = strip_excluded or strip_unreliable
    return json.dumps(
        canonicalize_schema(left, strip_excluded=should_strip), sort_keys=True
    ) == json.dumps(
        canonicalize_schema(right, strip_excluded=should_strip), sort_keys=True
    )


def wait_for_index_ready(
    index: SearchIndex,
    *,
    timeout_seconds: int = 1800,
    poll_interval_seconds: float = 0.5,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[Dict[str, Any], float]:
    """Wait for index to finish indexing all documents.

    Args:
        index: The SearchIndex to monitor.
        timeout_seconds: Maximum time to wait.
        poll_interval_seconds: How often to check status.
        progress_callback: Optional callback(indexed_docs, total_docs, percent).
    """
    start = time.perf_counter()
    deadline = start + timeout_seconds
    latest_info = index.info()

    stable_ready_checks = 0
    while time.perf_counter() < deadline:
        latest_info = index.info()
        indexing = latest_info.get("indexing")
        percent_indexed = latest_info.get("percent_indexed")

        if percent_indexed is not None or indexing is not None:
            ready = float(percent_indexed or 0) >= 1.0 and not bool(indexing)
            if progress_callback:
                total_docs = int(latest_info.get("num_docs", 0))
                pct = float(percent_indexed or 0)
                indexed_docs = int(total_docs * pct)
                progress_callback(indexed_docs, total_docs, pct * 100)
        else:
            current_docs = latest_info.get("num_docs")
            if current_docs is None:
                ready = True
            else:
                if stable_ready_checks == 0:
                    stable_ready_checks = int(current_docs)
                    time.sleep(poll_interval_seconds)
                    continue
                ready = int(current_docs) == stable_ready_checks

        if ready:
            return latest_info, round(time.perf_counter() - start, 3)

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Index {index.schema.index.name} did not become ready within {timeout_seconds} seconds"
    )


def current_source_matches_snapshot(
    index_name: str,
    expected_schema: Dict[str, Any],
    *,
    redis_url: Optional[str] = None,
    redis_client: Optional[Any] = None,
) -> bool:
    current_index = SearchIndex.from_existing(
        index_name,
        redis_url=redis_url,
        redis_client=redis_client,
    )
    return schemas_equal(current_index.schema.to_dict(), expected_schema)


def timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def estimate_disk_space(
    plan: MigrationPlan,
    *,
    aof_enabled: bool = False,
) -> DiskSpaceEstimate:
    """Estimate disk space required for a migration with quantization.

    This is a pure calculation based on the migration plan. No Redis
    operations are performed.

    Args:
        plan: The migration plan containing source/target schemas.
        aof_enabled: Whether AOF persistence is active on the Redis instance.

    Returns:
        DiskSpaceEstimate with projected costs.
    """
    doc_count = int(plan.source.stats_snapshot.get("num_docs", 0) or 0)
    storage_type = plan.source.keyspace.storage_type
    index_name = plan.source.index_name

    # Find vector fields with datatype changes
    source_fields = {
        f["name"]: f for f in plan.source.schema_snapshot.get("fields", [])
    }
    target_fields = {f["name"]: f for f in plan.merged_target_schema.get("fields", [])}

    vector_field_estimates: list[VectorFieldEstimate] = []
    total_source_bytes = 0
    total_target_bytes = 0
    total_aof_growth = 0

    aof_overhead = (
        AOF_JSON_SET_OVERHEAD_BYTES
        if storage_type == "json"
        else AOF_HSET_OVERHEAD_BYTES
    )

    for name, source_field in source_fields.items():
        if source_field.get("type") != "vector":
            continue
        target_field = target_fields.get(name)
        if not target_field or target_field.get("type") != "vector":
            continue

        source_attrs = source_field.get("attrs", {})
        target_attrs = target_field.get("attrs", {})
        source_dtype = source_attrs.get("datatype", "float32").lower()
        target_dtype = target_attrs.get("datatype", "float32").lower()

        if source_dtype == target_dtype:
            continue

        if source_dtype not in DTYPE_BYTES:
            raise ValueError(
                f"Unknown source vector datatype '{source_dtype}' for field '{name}'. "
                f"Supported datatypes: {', '.join(sorted(DTYPE_BYTES.keys()))}"
            )
        if target_dtype not in DTYPE_BYTES:
            raise ValueError(
                f"Unknown target vector datatype '{target_dtype}' for field '{name}'. "
                f"Supported datatypes: {', '.join(sorted(DTYPE_BYTES.keys()))}"
            )

        if storage_type == "json":
            # JSON-backed migrations do not rewrite per-document vector payloads
            # during apply(); they rely on recreate + re-index instead.
            continue

        dims = int(source_attrs.get("dims", 0))
        source_bpe = DTYPE_BYTES[source_dtype]
        target_bpe = DTYPE_BYTES[target_dtype]

        source_vec_size = dims * source_bpe
        target_vec_size = dims * target_bpe

        vector_field_estimates.append(
            VectorFieldEstimate(
                field_name=name,
                dims=dims,
                source_dtype=source_dtype,
                target_dtype=target_dtype,
                source_bytes_per_doc=source_vec_size,
                target_bytes_per_doc=target_vec_size,
            )
        )

        field_source_total = doc_count * source_vec_size
        field_target_total = doc_count * target_vec_size
        total_source_bytes += field_source_total
        total_target_bytes += field_target_total

        if aof_enabled:
            total_aof_growth += doc_count * (target_vec_size + aof_overhead)

    rdb_snapshot_disk = int(total_source_bytes * RDB_COMPRESSION_RATIO)
    rdb_cow_memory = total_source_bytes
    total_new_disk = rdb_snapshot_disk + total_aof_growth
    memory_savings = total_source_bytes - total_target_bytes

    return DiskSpaceEstimate(
        index_name=index_name,
        doc_count=doc_count,
        storage_type=storage_type,
        vector_fields=vector_field_estimates,
        total_source_vector_bytes=total_source_bytes,
        total_target_vector_bytes=total_target_bytes,
        rdb_snapshot_disk_bytes=rdb_snapshot_disk,
        rdb_cow_memory_if_concurrent_bytes=rdb_cow_memory,
        aof_enabled=aof_enabled,
        aof_growth_bytes=total_aof_growth,
        total_new_disk_bytes=total_new_disk,
        memory_savings_after_bytes=memory_savings,
    )
