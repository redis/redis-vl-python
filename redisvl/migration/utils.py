from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def canonicalize_schema(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    schema = IndexSchema.from_dict(schema_dict).to_dict()
    schema["fields"] = sorted(schema.get("fields", []), key=lambda field: field["name"])
    prefixes = schema["index"].get("prefix")
    if isinstance(prefixes, list):
        schema["index"]["prefix"] = sorted(prefixes)
    stopwords = schema["index"].get("stopwords")
    if isinstance(stopwords, list):
        schema["index"]["stopwords"] = list(stopwords)
    return schema


def schemas_equal(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return json.dumps(canonicalize_schema(left), sort_keys=True) == json.dumps(
        canonicalize_schema(right), sort_keys=True
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
