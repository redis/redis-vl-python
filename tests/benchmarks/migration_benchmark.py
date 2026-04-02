"""Migration Benchmark: Measure end-to-end migration time at scale.

Populates a realistic 16-field index (matching the KM production schema)
at 1K, 10K, 100K, and 1M vectors, then migrates:
  - Sub-1M: HNSW FP32 -> FLAT FP16
  - 1M:     HNSW FP32 -> HNSW FP16

Collects full MigrationTimings from MigrationExecutor.apply().

Usage:
  python tests/benchmarks/migration_benchmark.py \\
    --redis-url redis://localhost:6379 \\
    --sizes 1000 10000 100000 \\
    --trials 3 \\
    --output tests/benchmarks/results_migration.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from redis import Redis

from redisvl.index import SearchIndex
from redisvl.migration import (
    AsyncMigrationExecutor,
    AsyncMigrationPlanner,
    MigrationExecutor,
    MigrationPlanner,
)
from redisvl.migration.models import FieldUpdate, SchemaPatch, SchemaPatchChanges
from redisvl.migration.utils import wait_for_index_ready
from redisvl.redis.utils import array_to_buffer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_DIMS = 3072
INDEX_PREFIX = "KM:benchmark:"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
BATCH_SIZE = 500

# Vocabularies for synthetic data
TAG_VOCABS = {
    "doc_base_id": [f"base_{i}" for i in range(50)],
    "file_id": [f"file_{i:06d}" for i in range(200)],
    "created_by": ["alice", "bob", "carol", "dave", "eve"],
    "CUSIP": [f"{random.randint(100000000, 999999999)}" for _ in range(100)],
}

TEXT_WORDS = [
    "financial",
    "report",
    "quarterly",
    "analysis",
    "revenue",
    "growth",
    "market",
    "portfolio",
    "investment",
    "dividend",
    "equity",
    "bond",
    "asset",
    "liability",
    "balance",
    "income",
    "statement",
    "forecast",
    "risk",
    "compliance",
]


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def make_source_schema(index_name: str) -> Dict[str, Any]:
    """Build the 16-field HNSW FP32 source schema dict."""
    return {
        "index": {
            "name": index_name,
            "prefix": INDEX_PREFIX,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "doc_base_id", "type": "tag", "attrs": {"separator": ","}},
            {"name": "file_id", "type": "tag", "attrs": {"separator": ","}},
            {"name": "page_text", "type": "text", "attrs": {"weight": 1}},
            {"name": "chunk_number", "type": "numeric"},
            {"name": "start_page", "type": "numeric"},
            {"name": "end_page", "type": "numeric"},
            {"name": "created_by", "type": "tag", "attrs": {"separator": ","}},
            {"name": "file_name", "type": "text", "attrs": {"weight": 1}},
            {"name": "created_time", "type": "numeric"},
            {"name": "last_updated_by", "type": "text", "attrs": {"weight": 1}},
            {"name": "last_updated_time", "type": "numeric"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",
                    "datatype": "float32",
                    "dims": VECTOR_DIMS,
                    "distance_metric": "COSINE",
                    "m": HNSW_M,
                    "ef_construction": HNSW_EF_CONSTRUCTION,
                },
            },
            {
                "name": "CUSIP",
                "type": "tag",
                "attrs": {"separator": ",", "index_missing": True},
            },
            {
                "name": "description",
                "type": "text",
                "attrs": {"weight": 1, "index_missing": True},
            },
            {
                "name": "name",
                "type": "text",
                "attrs": {"weight": 1, "index_missing": True},
            },
            {"name": "price", "type": "numeric", "attrs": {"index_missing": True}},
        ],
    }


def make_migration_patch(target_algo: str) -> SchemaPatch:
    """Build a SchemaPatch to change embedding from FP32 to FP16 (and optionally HNSW to FLAT)."""
    attrs = {"datatype": "float16"}
    if target_algo == "FLAT":
        attrs["algorithm"] = "flat"
    return SchemaPatch(
        version=1,
        changes=SchemaPatchChanges(
            update_fields=[
                FieldUpdate(name="embedding", attrs=attrs),
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_random_text(min_words: int = 10, max_words: int = 50) -> str:
    """Generate a random sentence from the vocabulary."""
    n = random.randint(min_words, max_words)
    return " ".join(random.choice(TEXT_WORDS) for _ in range(n))


def generate_document(doc_id: int, vector: np.ndarray) -> Dict[str, Any]:
    """Generate a single document with all 16 fields."""
    doc: Dict[str, Any] = {
        "doc_base_id": random.choice(TAG_VOCABS["doc_base_id"]),
        "file_id": random.choice(TAG_VOCABS["file_id"]),
        "page_text": generate_random_text(),
        "chunk_number": random.randint(0, 100),
        "start_page": random.randint(1, 500),
        "end_page": random.randint(1, 500),
        "created_by": random.choice(TAG_VOCABS["created_by"]),
        "file_name": f"document_{doc_id}.pdf",
        "created_time": int(time.time()) - random.randint(0, 86400 * 365),
        "last_updated_by": random.choice(TAG_VOCABS["created_by"]),
        "last_updated_time": int(time.time()) - random.randint(0, 86400 * 30),
        "embedding": array_to_buffer(vector, dtype="float32"),
    }
    # INDEXMISSING fields: populate ~80% of docs
    if random.random() < 0.8:
        doc["CUSIP"] = random.choice(TAG_VOCABS["CUSIP"])
    if random.random() < 0.8:
        doc["description"] = generate_random_text(5, 20)
    if random.random() < 0.8:
        doc["name"] = f"Entity {doc_id}"
    if random.random() < 0.8:
        doc["price"] = round(random.uniform(1.0, 10000.0), 2)
    return doc


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def populate_index(
    redis_url: str,
    index_name: str,
    num_docs: int,
) -> float:
    """Create the source index and populate it with synthetic data.

    Returns the time taken in seconds.
    """
    schema_dict = make_source_schema(index_name)
    index = SearchIndex.from_dict(schema_dict, redis_url=redis_url)

    # Drop existing index if any
    try:
        index.delete(drop=True)
    except Exception:
        pass

    # Clean up any leftover keys from previous runs
    client = Redis.from_url(redis_url)
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor, match=f"{INDEX_PREFIX}*", count=5000)
        if keys:
            client.delete(*keys)
        if cursor == 0:
            break
    client.close()

    index.create(overwrite=True)

    print(f"  Populating {num_docs:,} documents...")
    start = time.perf_counter()

    # Generate vectors in batches to manage memory
    rng = np.random.default_rng(seed=42)
    client = Redis.from_url(redis_url)

    for batch_start in range(0, num_docs, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_docs)
        batch_count = batch_end - batch_start

        # Generate batch of random unit-normalized vectors
        vectors = rng.standard_normal((batch_count, VECTOR_DIMS)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        pipe = client.pipeline(transaction=False)
        for i in range(batch_count):
            doc_id = batch_start + i
            key = f"{INDEX_PREFIX}{doc_id}"
            doc = generate_document(doc_id, vectors[i])
            pipe.hset(key, mapping=doc)

        pipe.execute()

        if (batch_end % 10000 == 0) or batch_end == num_docs:
            elapsed = time.perf_counter() - start
            rate = batch_end / elapsed if elapsed > 0 else 0
            print(f"    {batch_end:,}/{num_docs:,} docs ({rate:,.0f} docs/sec)")

    populate_duration = time.perf_counter() - start
    client.close()

    # Wait for indexing to complete
    print("  Waiting for index to be ready...")
    idx = SearchIndex.from_existing(index_name, redis_url=redis_url)
    _, indexing_wait = wait_for_index_ready(idx)
    print(
        f"  Index ready (waited {indexing_wait:.1f}s after {populate_duration:.1f}s populate)"
    )

    return populate_duration + indexing_wait


# ---------------------------------------------------------------------------
# Migration execution
# ---------------------------------------------------------------------------


def run_migration(
    redis_url: str,
    index_name: str,
    target_algo: str,
) -> Dict[str, Any]:
    """Run a single migration and return the full report as a dict.

    Returns a dict with 'report' (model_dump) and 'enumerate_method'
    indicating whether FT.AGGREGATE or SCAN was used for key discovery.
    """
    import logging

    patch = make_migration_patch(target_algo)
    planner = MigrationPlanner()
    plan = planner.create_plan_from_patch(
        index_name,
        schema_patch=patch,
        redis_url=redis_url,
    )

    if not plan.diff_classification.supported:
        raise RuntimeError(
            f"Migration not supported: {plan.diff_classification.blocked_reasons}"
        )

    executor = MigrationExecutor()

    # Capture enumerate method by intercepting executor logger warnings
    enumerate_method = "FT.AGGREGATE"  # default (happy path)
    _orig_logger = logging.getLogger("redisvl.migration.executor")
    _orig_level = _orig_logger.level

    class _EnumMethodHandler(logging.Handler):
        def emit(self, record):
            nonlocal enumerate_method
            msg = record.getMessage()
            if "Using SCAN" in msg or "Falling back to SCAN" in msg:
                enumerate_method = "SCAN"

    handler = _EnumMethodHandler()
    _orig_logger.addHandler(handler)
    _orig_logger.setLevel(logging.WARNING)

    def progress(step: str, detail: Optional[str] = None) -> None:
        if detail:
            print(f"    [{step}] {detail}")

    try:
        report = executor.apply(
            plan,
            redis_url=redis_url,
            progress_callback=progress,
        )
    finally:
        _orig_logger.removeHandler(handler)
        _orig_logger.setLevel(_orig_level)

    return {"report": report.model_dump(), "enumerate_method": enumerate_method}


async def async_run_migration(
    redis_url: str,
    index_name: str,
    target_algo: str,
) -> Dict[str, Any]:
    """Run a single migration using AsyncMigrationExecutor.

    Returns a dict with 'report' (model_dump) and 'enumerate_method'
    indicating whether FT.AGGREGATE or SCAN was used for key discovery.
    """
    import logging

    patch = make_migration_patch(target_algo)
    planner = AsyncMigrationPlanner()
    plan = await planner.create_plan_from_patch(
        index_name,
        schema_patch=patch,
        redis_url=redis_url,
    )

    if not plan.diff_classification.supported:
        raise RuntimeError(
            f"Migration not supported: {plan.diff_classification.blocked_reasons}"
        )

    executor = AsyncMigrationExecutor()

    # Capture enumerate method by intercepting executor logger warnings
    enumerate_method = "FT.AGGREGATE"  # default (happy path)
    _orig_logger = logging.getLogger("redisvl.migration.async_executor")
    _orig_level = _orig_logger.level

    class _EnumMethodHandler(logging.Handler):
        def emit(self, record):
            nonlocal enumerate_method
            msg = record.getMessage()
            if "Using SCAN" in msg or "Falling back to SCAN" in msg:
                enumerate_method = "SCAN"

    handler = _EnumMethodHandler()
    _orig_logger.addHandler(handler)
    _orig_logger.setLevel(logging.WARNING)

    def progress(step: str, detail: Optional[str] = None) -> None:
        if detail:
            print(f"    [{step}] {detail}")

    try:
        report = await executor.apply(
            plan,
            redis_url=redis_url,
            progress_callback=progress,
        )
    finally:
        _orig_logger.removeHandler(handler)
        _orig_logger.setLevel(_orig_level)

    return {"report": report.model_dump(), "enumerate_method": enumerate_method}


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def run_benchmark(
    redis_url: str,
    sizes: List[int],
    trials: int,
    output_path: Optional[str],
    use_async: bool = False,
) -> Dict[str, Any]:
    """Run the full migration benchmark across all sizes and trials."""
    executor_label = "async" if use_async else "sync"
    results: Dict[str, Any] = {
        "benchmark": "migration_timing",
        "executor": executor_label,
        "schema_field_count": 16,
        "vector_dims": VECTOR_DIMS,
        "trials_per_size": trials,
        "results": [],
    }

    for size in sizes:
        target_algo = "HNSW" if size >= 1_000_000 else "FLAT"
        index_name = f"bench_migration_{size}"
        print(f"\n{'='*60}")
        print(
            f"Size: {size:,} | Migration: HNSW FP32 -> {target_algo} FP16 | Executor: {executor_label}"
        )
        print(f"{'='*60}")

        size_result = {
            "size": size,
            "source_algo": "HNSW",
            "source_dtype": "FLOAT32",
            "target_algo": target_algo,
            "target_dtype": "FLOAT16",
            "trials": [],
        }

        for trial_num in range(1, trials + 1):
            print(f"\n  Trial {trial_num}/{trials}")

            # Step 1: Populate
            populate_time = populate_index(redis_url, index_name, size)

            # Capture source memory
            client = Redis.from_url(redis_url)
            try:
                info_raw = client.execute_command("FT.INFO", index_name)
                # Parse the flat list into a dict
                info_dict = {}
                for i in range(0, len(info_raw), 2):
                    key = info_raw[i]
                    if isinstance(key, bytes):
                        key = key.decode()
                    info_dict[key] = info_raw[i + 1]
                source_mem_mb = float(info_dict.get("vector_index_sz_mb", 0))
                source_total_mb = float(info_dict.get("total_index_memory_sz_mb", 0))
                source_num_docs = int(info_dict.get("num_docs", 0))
            except Exception as e:
                print(f"    Warning: could not read source FT.INFO: {e}")
                source_mem_mb = 0.0
                source_total_mb = 0.0
                source_num_docs = 0
            finally:
                client.close()

            print(
                f"  Source: {source_num_docs:,} docs, "
                f"vector_idx={source_mem_mb:.1f}MB, "
                f"total_idx={source_total_mb:.1f}MB"
            )

            # Step 2: Migrate
            print(f"  Running migration ({executor_label})...")
            if use_async:
                migration_result = asyncio.run(
                    async_run_migration(redis_url, index_name, target_algo)
                )
            else:
                migration_result = run_migration(redis_url, index_name, target_algo)
            report_dict = migration_result["report"]
            enumerate_method = migration_result["enumerate_method"]

            # Capture target memory
            target_index_name = report_dict.get("target_index", index_name)
            client = Redis.from_url(redis_url)
            try:
                info_raw = client.execute_command("FT.INFO", target_index_name)
                info_dict = {}
                for i in range(0, len(info_raw), 2):
                    key = info_raw[i]
                    if isinstance(key, bytes):
                        key = key.decode()
                    info_dict[key] = info_raw[i + 1]
                target_mem_mb = float(info_dict.get("vector_index_sz_mb", 0))
                target_total_mb = float(info_dict.get("total_index_memory_sz_mb", 0))
            except Exception as e:
                print(f"    Warning: could not read target FT.INFO: {e}")
                target_mem_mb = 0.0
                target_total_mb = 0.0
            finally:
                client.close()

            timings = report_dict.get("timings", {})
            migrate_s = timings.get("total_migration_duration_seconds", 0) or 0
            total_s = round(populate_time + migrate_s, 3)

            # Vector memory savings (the real savings from FP32 -> FP16)
            vec_savings_pct = (
                round((1 - target_mem_mb / source_mem_mb) * 100, 1)
                if source_mem_mb > 0
                else 0
            )

            trial_result = {
                "trial": trial_num,
                "load_time_seconds": round(populate_time, 3),
                "migrate_time_seconds": round(migrate_s, 3),
                "total_time_seconds": total_s,
                "enumerate_method": enumerate_method,
                "timings": timings,
                "benchmark_summary": report_dict.get("benchmark_summary", {}),
                "source_vector_index_mb": round(source_mem_mb, 3),
                "source_total_index_mb": round(source_total_mb, 3),
                "target_vector_index_mb": round(target_mem_mb, 3),
                "target_total_index_mb": round(target_total_mb, 3),
                "vector_memory_savings_pct": vec_savings_pct,
                "validation_passed": report_dict.get("result") == "succeeded",
                "num_docs": source_num_docs,
            }

            # Print isolated timings
            _enum_s = timings.get("drop_duration_seconds", 0) or 0  # noqa: F841
            quant_s = timings.get("quantize_duration_seconds") or 0
            index_s = timings.get("initial_indexing_duration_seconds") or 0
            down_s = timings.get("downtime_duration_seconds") or 0
            print(
                f"""  Results
    load       = {populate_time:.1f}s
    migrate    = {migrate_s:.1f}s (enumerate + drop + quantize + create + reindex + validate)
    total      = {total_s:.1f}s
    enumerate  = {enumerate_method}
    quantize   = {quant_s:.1f}s
    reindex    = {index_s:.1f}s
    downtime   = {down_s:.1f}s
    vec memory = {source_mem_mb:.1f}MB -> {target_mem_mb:.1f}MB ({vec_savings_pct:.1f}% saved)
    passed     = {trial_result['validation_passed']}"""
            )

            size_result["trials"].append(trial_result)

            # Clean up for next trial (drop index + keys)
            client = Redis.from_url(redis_url)
            try:
                try:
                    client.execute_command("FT.DROPINDEX", target_index_name)
                except Exception:
                    pass
                # Delete document keys
                cursor = 0
                while True:
                    cursor, keys = client.scan(
                        cursor, match=f"{INDEX_PREFIX}*", count=5000
                    )
                    if keys:
                        client.delete(*keys)
                    if cursor == 0:
                        break
            finally:
                client.close()

        results["results"].append(size_result)

    # Save results
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Migration timing benchmark")
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis connection URL"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 10000, 100000],
        help="Corpus sizes to benchmark",
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of trials per size"
    )
    parser.add_argument(
        "--output",
        default="tests/benchmarks/results_migration.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        default=False,
        help="Use AsyncMigrationExecutor instead of sync MigrationExecutor",
    )
    args = parser.parse_args()

    executor_label = "AsyncMigrationExecutor" if args.use_async else "MigrationExecutor"
    print(
        f"""Migration Benchmark
  Redis: {args.redis_url}
  Sizes: {args.sizes}
  Trials: {args.trials}
  Vector dims: {VECTOR_DIMS}
  Fields: 16
  Executor: {executor_label}"""
    )

    run_benchmark(
        redis_url=args.redis_url,
        sizes=args.sizes,
        trials=args.trials,
        output_path=args.output,
        use_async=args.use_async,
    )


if __name__ == "__main__":
    main()
