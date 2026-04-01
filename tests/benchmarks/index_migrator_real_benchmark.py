from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import yaml
from datasets import load_dataset
from redis import Redis
from sentence_transformers import SentenceTransformer

from redisvl.index import SearchIndex
from redisvl.migration import MigrationPlanner
from redisvl.query import VectorQuery
from redisvl.redis.utils import array_to_buffer

AG_NEWS_LABELS = {
    0: "world",
    1: "sports",
    2: "business",
    3: "sci_tech",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real local benchmark for migrating from HNSW/FP32 to FLAT/FP16 "
            "with a real internet dataset and sentence-transformers embeddings."
        )
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379",
        help="Redis URL for the local benchmark target.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 10000, 100000],
        help="Dataset sizes to benchmark.",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=25,
        help="Number of held-out query documents to benchmark search latency.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to fetch for overlap checks.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=256,
        help="Batch size for sentence-transformers encoding.",
    )
    parser.add_argument(
        "--load-batch-size",
        type=int,
        default=500,
        help="Batch size for SearchIndex.load calls.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--dataset-csv",
        default="",
        help=(
            "Optional path to a local AG News CSV file with label,title,description columns. "
            "If provided, the benchmark skips Hugging Face dataset downloads."
        ),
    )
    parser.add_argument(
        "--output",
        default="index_migrator_benchmark_results.json",
        help="Where to write the benchmark report.",
    )
    return parser.parse_args()


def build_schema(
    *,
    index_name: str,
    prefix: str,
    dims: int,
    algorithm: str,
    datatype: str,
) -> Dict[str, Any]:
    return {
        "index": {
            "name": index_name,
            "prefix": prefix,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "doc_id", "type": "tag"},
            {"name": "label", "type": "tag"},
            {"name": "text", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": dims,
                    "distance_metric": "cosine",
                    "algorithm": algorithm,
                    "datatype": datatype,
                },
            },
        ],
    }


def load_ag_news_records(num_docs: int, query_count: int) -> List[Dict[str, Any]]:
    dataset = load_dataset("ag_news", split=f"train[:{num_docs + query_count}]")
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        records.append(
            {
                "doc_id": f"ag-news-{idx}",
                "text": row["text"],
                "label": AG_NEWS_LABELS[int(row["label"])],
            }
        )
    return records


def load_ag_news_records_from_csv(
    csv_path: str,
    *,
    required_docs: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx >= required_docs:
                break
            label, title, description = row
            text = f"{title}. {description}".strip()
            records.append(
                {
                    "doc_id": f"ag-news-{idx}",
                    "text": text,
                    "label": AG_NEWS_LABELS[int(label) - 1],
                }
            )

    if len(records) < required_docs:
        raise ValueError(
            f"Expected at least {required_docs} records in {csv_path}, found {len(records)}"
        )
    return records


def encode_texts(
    model_name: str,
    texts: Sequence[str],
    batch_size: int,
) -> tuple[np.ndarray, float]:
    try:
        encoder = SentenceTransformer(model_name, local_files_only=True)
    except OSError:
        # Model not cached locally yet; download it
        print(f"Model '{model_name}' not found locally, downloading...")
        encoder = SentenceTransformer(model_name)
    start = time.perf_counter()
    embeddings = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    duration = time.perf_counter() - start
    return np.asarray(embeddings, dtype=np.float32), duration


def iter_documents(
    records: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    *,
    dtype: str,
) -> Iterable[Dict[str, Any]]:
    for record, embedding in zip(records, embeddings):
        yield {
            "doc_id": record["doc_id"],
            "label": record["label"],
            "text": record["text"],
            "embedding": array_to_buffer(embedding, dtype),
        }


def wait_for_index_ready(
    index: SearchIndex,
    *,
    timeout_seconds: int = 1800,
    poll_interval_seconds: float = 0.5,
) -> Dict[str, Any]:
    deadline = time.perf_counter() + timeout_seconds
    latest_info = index.info()
    while time.perf_counter() < deadline:
        latest_info = index.info()
        percent_indexed = float(latest_info.get("percent_indexed", 1) or 1)
        indexing = latest_info.get("indexing", 0)
        if percent_indexed >= 1.0 and not indexing:
            return latest_info
        time.sleep(poll_interval_seconds)
    raise TimeoutError(
        f"Index {index.schema.index.name} did not finish indexing within {timeout_seconds} seconds"
    )


def get_memory_snapshot(client: Redis) -> Dict[str, Any]:
    info = client.info("memory")
    used_memory_bytes = int(info.get("used_memory", 0))
    return {
        "used_memory_bytes": used_memory_bytes,
        "used_memory_mb": round(used_memory_bytes / (1024 * 1024), 3),
        "used_memory_human": info.get("used_memory_human"),
    }


def summarize_index_info(index_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_docs": int(index_info.get("num_docs", 0) or 0),
        "percent_indexed": float(index_info.get("percent_indexed", 0) or 0),
        "hash_indexing_failures": int(index_info.get("hash_indexing_failures", 0) or 0),
        "vector_index_sz_mb": float(index_info.get("vector_index_sz_mb", 0) or 0),
        "total_indexing_time": float(index_info.get("total_indexing_time", 0) or 0),
    }


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(np.asarray(values), pct)), 3)


def run_query_benchmark(
    index: SearchIndex,
    query_embeddings: np.ndarray,
    *,
    dtype: str,
    top_k: int,
) -> Dict[str, Any]:
    latencies_ms: List[float] = []
    result_sets: List[List[str]] = []

    for query_embedding in query_embeddings:
        query = VectorQuery(
            vector=query_embedding.tolist(),
            vector_field_name="embedding",
            return_fields=["doc_id", "label"],
            num_results=top_k,
            dtype=dtype,
        )
        start = time.perf_counter()
        results = index.query(query)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        result_sets.append(
            [result.get("doc_id") or result.get("id") for result in results if result]
        )

    return {
        "count": len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "mean_ms": round(statistics.mean(latencies_ms), 3),
        "result_sets": result_sets,
    }


def compute_overlap(
    source_result_sets: Sequence[Sequence[str]],
    target_result_sets: Sequence[Sequence[str]],
    *,
    top_k: int,
) -> Dict[str, Any]:
    overlap_ratios: List[float] = []
    for source_results, target_results in zip(source_result_sets, target_result_sets):
        source_set = set(source_results[:top_k])
        target_set = set(target_results[:top_k])
        overlap_ratios.append(len(source_set.intersection(target_set)) / max(top_k, 1))
    return {
        "mean_overlap_at_k": round(statistics.mean(overlap_ratios), 4),
        "min_overlap_at_k": round(min(overlap_ratios), 4),
        "max_overlap_at_k": round(max(overlap_ratios), 4),
    }


def run_quantization_migration(
    planner: MigrationPlanner,
    client: Redis,
    source_index_name: str,
    source_schema: Dict[str, Any],
    dims: int,
) -> Dict[str, Any]:
    """Run full HNSW/FP32 -> FLAT/FP16 migration with quantization."""
    from redisvl.migration import MigrationExecutor

    target_schema = build_schema(
        index_name=source_schema["index"]["name"],
        prefix=source_schema["index"]["prefix"],
        dims=dims,
        algorithm="flat",  # Change algorithm
        datatype="float16",  # Change datatype (quantization)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        target_schema_path = Path(tmpdir) / "target_schema.yaml"
        plan_path = Path(tmpdir) / "migration_plan.yaml"
        with open(target_schema_path, "w") as f:
            yaml.safe_dump(target_schema, f, sort_keys=False)

        plan_start = time.perf_counter()
        plan = planner.create_plan(
            source_index_name,
            redis_client=client,
            target_schema_path=str(target_schema_path),
        )
        planner.write_plan(plan, str(plan_path))
        plan_duration = time.perf_counter() - plan_start

        if not plan.diff_classification.supported:
            raise AssertionError(
                f"Expected planner to ALLOW quantization migration, "
                f"but it blocked with: {plan.diff_classification.blocked_reasons}"
            )

        # Check datatype changes detected
        datatype_changes = MigrationPlanner.get_vector_datatype_changes(
            plan.source.schema_snapshot, plan.merged_target_schema
        )

        # Execute migration
        executor = MigrationExecutor()
        migrate_start = time.perf_counter()
        report = executor.apply(plan, redis_client=client)
        migrate_duration = time.perf_counter() - migrate_start

        if report.result != "succeeded":
            raise AssertionError(f"Migration failed: {report.validation.errors}")

        return {
            "test": "quantization_migration",
            "plan_duration_seconds": round(plan_duration, 3),
            "migration_duration_seconds": round(migrate_duration, 3),
            "quantize_duration_seconds": report.timings.quantize_duration_seconds,
            "supported": plan.diff_classification.supported,
            "datatype_changes": datatype_changes,
            "result": report.result,
            "plan_path": str(plan_path),
        }


def assert_planner_allows_algorithm_change(
    planner: MigrationPlanner,
    client: Redis,
    source_index_name: str,
    source_schema: Dict[str, Any],
    dims: int,
) -> Dict[str, Any]:
    """Test that algorithm-only changes (HNSW -> FLAT) are allowed."""
    target_schema = build_schema(
        index_name=source_schema["index"]["name"],
        prefix=source_schema["index"]["prefix"],
        dims=dims,
        algorithm="flat",  # Different algorithm - should be allowed
        datatype="float32",  # Same datatype
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        target_schema_path = Path(tmpdir) / "target_schema.yaml"
        plan_path = Path(tmpdir) / "migration_plan.yaml"
        with open(target_schema_path, "w") as f:
            yaml.safe_dump(target_schema, f, sort_keys=False)

        start = time.perf_counter()
        plan = planner.create_plan(
            source_index_name,
            redis_client=client,
            target_schema_path=str(target_schema_path),
        )
        planner.write_plan(plan, str(plan_path))
        duration = time.perf_counter() - start

        if not plan.diff_classification.supported:
            raise AssertionError(
                f"Expected planner to ALLOW algorithm change (HNSW -> FLAT), "
                f"but it blocked with: {plan.diff_classification.blocked_reasons}"
            )

        return {
            "test": "algorithm_change_allowed",
            "plan_duration_seconds": round(duration, 3),
            "supported": plan.diff_classification.supported,
            "blocked_reasons": plan.diff_classification.blocked_reasons,
            "plan_path": str(plan_path),
        }


def benchmark_scale(
    *,
    client: Redis,
    all_records: Sequence[Dict[str, Any]],
    all_embeddings: np.ndarray,
    size: int,
    query_count: int,
    top_k: int,
    load_batch_size: int,
) -> Dict[str, Any]:
    records = list(all_records[:size])
    query_records = list(all_records[size : size + query_count])
    doc_embeddings = all_embeddings[:size]
    query_embeddings = all_embeddings[size : size + query_count]
    dims = int(all_embeddings.shape[1])

    client.flushall()

    baseline_memory = get_memory_snapshot(client)
    planner = MigrationPlanner(key_sample_limit=5)
    source_schema = build_schema(
        index_name=f"benchmark_source_{size}",
        prefix=f"benchmark:source:{size}",
        dims=dims,
        algorithm="hnsw",
        datatype="float32",
    )

    source_index = SearchIndex.from_dict(source_schema, redis_client=client)
    migrated_index = None  # Will be set after migration

    try:
        source_index.create(overwrite=True)
        source_load_start = time.perf_counter()
        source_index.load(
            iter_documents(records, doc_embeddings, dtype="float32"),
            id_field="doc_id",
            batch_size=load_batch_size,
        )
        source_info = wait_for_index_ready(source_index)
        source_setup_duration = time.perf_counter() - source_load_start
        source_memory = get_memory_snapshot(client)

        # Query source index before migration
        source_query_metrics = run_query_benchmark(
            source_index,
            query_embeddings,
            dtype="float32",
            top_k=top_k,
        )

        # Run full quantization migration: HNSW/FP32 -> FLAT/FP16
        quantization_result = run_quantization_migration(
            planner=planner,
            client=client,
            source_index_name=source_schema["index"]["name"],
            source_schema=source_schema,
            dims=dims,
        )

        # Get migrated index info and memory
        migrated_index = SearchIndex.from_existing(
            source_schema["index"]["name"], redis_client=client
        )
        target_info = wait_for_index_ready(migrated_index)
        overlap_memory = get_memory_snapshot(client)

        # Query migrated index
        target_query_metrics = run_query_benchmark(
            migrated_index,
            query_embeddings.astype(np.float16),
            dtype="float16",
            top_k=top_k,
        )

        overlap_metrics = compute_overlap(
            source_query_metrics["result_sets"],
            target_query_metrics["result_sets"],
            top_k=top_k,
        )

        post_cutover_memory = get_memory_snapshot(client)

        return {
            "size": size,
            "query_count": len(query_records),
            "vector_dims": dims,
            "source": {
                "algorithm": "hnsw",
                "datatype": "float32",
                "setup_duration_seconds": round(source_setup_duration, 3),
                "index_info": summarize_index_info(source_info),
                "query_metrics": {
                    k: v for k, v in source_query_metrics.items() if k != "result_sets"
                },
            },
            "migration": {
                "quantization": quantization_result,
            },
            "target": {
                "algorithm": "flat",
                "datatype": "float16",
                "migration_duration_seconds": quantization_result[
                    "migration_duration_seconds"
                ],
                "quantize_duration_seconds": quantization_result[
                    "quantize_duration_seconds"
                ],
                "index_info": summarize_index_info(target_info),
                "query_metrics": {
                    k: v for k, v in target_query_metrics.items() if k != "result_sets"
                },
            },
            "memory": {
                "baseline": baseline_memory,
                "after_source": source_memory,
                "during_overlap": overlap_memory,
                "after_cutover": post_cutover_memory,
                "overlap_increase_mb": round(
                    overlap_memory["used_memory_mb"] - source_memory["used_memory_mb"],
                    3,
                ),
                "net_change_after_cutover_mb": round(
                    post_cutover_memory["used_memory_mb"]
                    - source_memory["used_memory_mb"],
                    3,
                ),
            },
            "correctness": {
                "source_num_docs": int(source_info.get("num_docs", 0) or 0),
                "target_num_docs": int(target_info.get("num_docs", 0) or 0),
                "doc_count_match": int(source_info.get("num_docs", 0) or 0)
                == int(target_info.get("num_docs", 0) or 0),
                "migration_succeeded": quantization_result["result"] == "succeeded",
                **overlap_metrics,
            },
        }
    finally:
        try:
            migrated_index.delete(drop=True)
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    sizes = sorted(args.sizes)
    max_size = max(sizes)
    required_docs = max_size + args.query_count

    if args.dataset_csv:
        print(
            f"Loading AG News CSV from {args.dataset_csv} with {required_docs} records"
        )
        records = load_ag_news_records_from_csv(
            args.dataset_csv,
            required_docs=required_docs,
        )
    else:
        print(f"Loading AG News dataset with {required_docs} records")
        records = load_ag_news_records(
            required_docs - args.query_count,
            args.query_count,
        )
    print(f"Encoding {len(records)} texts with {args.model}")
    embeddings, embedding_duration = encode_texts(
        args.model,
        [record["text"] for record in records],
        args.embedding_batch_size,
    )

    client = Redis.from_url(args.redis_url, decode_responses=False)
    client.ping()

    report = {
        "dataset": "ag_news",
        "model": args.model,
        "sizes": sizes,
        "query_count": args.query_count,
        "top_k": args.top_k,
        "embedding_duration_seconds": round(embedding_duration, 3),
        "results": [],
    }

    for size in sizes:
        print(f"\nRunning benchmark for {size} documents")
        result = benchmark_scale(
            client=client,
            all_records=records,
            all_embeddings=embeddings,
            size=size,
            query_count=args.query_count,
            top_k=args.top_k,
            load_batch_size=args.load_batch_size,
        )
        report["results"].append(result)
        print(
            json.dumps(
                {
                    "size": size,
                    "source_setup_duration_seconds": result["source"][
                        "setup_duration_seconds"
                    ],
                    "migration_duration_seconds": result["target"][
                        "migration_duration_seconds"
                    ],
                    "quantize_duration_seconds": result["target"][
                        "quantize_duration_seconds"
                    ],
                    "migration_succeeded": result["correctness"]["migration_succeeded"],
                    "mean_overlap_at_k": result["correctness"]["mean_overlap_at_k"],
                    "memory_change_mb": result["memory"]["net_change_after_cutover_mb"],
                },
                indent=2,
            )
        )

    output_path = Path(args.output).resolve()
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nBenchmark report written to {output_path}")


if __name__ == "__main__":
    main()
