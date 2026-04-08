from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import yaml

from redisvl.index import SearchIndex
from redisvl.migration.models import MigrationPlan, MigrationReport
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


# Attributes that FT.INFO does NOT return reliably.
# These are stripped from schema comparison to avoid false validation failures.
# The migration still works, but we cannot verify via FT.INFO read-back.
UNRELIABLE_VECTOR_ATTRS = {"ef_runtime", "epsilon", "initial_cap"}
UNRELIABLE_TEXT_ATTRS = {"phonetic_matcher", "withsuffixtrie"}
UNRELIABLE_TAG_ATTRS = {"withsuffixtrie"}


def _strip_unreliable_attrs(field: Dict[str, Any]) -> Dict[str, Any]:
    """Remove attributes that FT.INFO doesn't return reliably.

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
        for attr in UNRELIABLE_VECTOR_ATTRS:
            attrs.pop(attr, None)
    elif field_type == "text":
        for attr in UNRELIABLE_TEXT_ATTRS:
            attrs.pop(attr, None)
        # Normalize weight to int for comparison (FT.INFO may return float)
        if "weight" in attrs and isinstance(attrs["weight"], float):
            if attrs["weight"] == int(attrs["weight"]):
                attrs["weight"] = int(attrs["weight"])
    elif field_type == "tag":
        for attr in UNRELIABLE_TAG_ATTRS:
            attrs.pop(attr, None)
    elif field_type == "numeric":
        # Redis auto-applies UNF when SORTABLE is set on NUMERIC fields.
        # Normalize unf to True when sortable is True to avoid false mismatches.
        if attrs.get("sortable"):
            attrs["unf"] = True

    field["attrs"] = attrs
    return field


def canonicalize_schema(
    schema_dict: Dict[str, Any], *, strip_unreliable: bool = False
) -> Dict[str, Any]:
    """Canonicalize schema for comparison.

    Args:
        schema_dict: The schema dictionary to canonicalize.
        strip_unreliable: If True, remove attributes that FT.INFO doesn't
            return reliably. Use this when comparing expected vs live schema.
    """
    schema = IndexSchema.from_dict(schema_dict).to_dict()

    fields = schema.get("fields", [])
    if strip_unreliable:
        fields = [_strip_unreliable_attrs(f) for f in fields]

    schema["fields"] = sorted(fields, key=lambda field: field["name"])
    prefixes = schema["index"].get("prefix")
    if isinstance(prefixes, list):
        schema["index"]["prefix"] = sorted(prefixes)
    stopwords = schema["index"].get("stopwords")
    if isinstance(stopwords, list):
        schema["index"]["stopwords"] = list(stopwords)
    return schema


def schemas_equal(
    left: Dict[str, Any], right: Dict[str, Any], *, strip_unreliable: bool = False
) -> bool:
    """Compare two schemas for equality.

    Args:
        left: First schema dictionary.
        right: Second schema dictionary.
        strip_unreliable: If True, ignore attributes that FT.INFO doesn't
            return reliably (ef_runtime, epsilon, initial_cap, phonetic_matcher).
    """
    return json.dumps(
        canonicalize_schema(left, strip_unreliable=strip_unreliable), sort_keys=True
    ) == json.dumps(
        canonicalize_schema(right, strip_unreliable=strip_unreliable), sort_keys=True
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
