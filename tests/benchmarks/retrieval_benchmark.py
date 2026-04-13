"""Retrieval Benchmark: FP32 vs FP16 x HNSW vs FLAT

Replicates the methodology from the Redis SVS-VAMANA study using
pre-embedded datasets from HuggingFace (no embedding step required).

Comparison matrix (4 configurations):
  - HNSW / FLOAT32 (approximate, full precision)
  - HNSW / FLOAT16 (approximate, quantized)
  - FLAT / FLOAT32 (exact, full precision -- ground truth)
  - FLAT / FLOAT16 (exact, quantized)

Datasets:
  - dbpedia: 1536-dim OpenAI embeddings (KShivendu/dbpedia-entities-openai-1M)
  - cohere:  768-dim Cohere embeddings  (Cohere/wikipedia-22-12-en-embeddings)

Metrics:
  - Overlap@K (precision vs FLAT/FP32 ground truth)
  - Query latency: p50, p95, p99, mean
  - QPS (queries per second)
  - Memory footprint per configuration
  - Index build / load time

Usage:
  python tests/benchmarks/retrieval_benchmark.py \\
    --redis-url redis://localhost:6379 \\
    --dataset dbpedia \\
    --sizes 1000 10000 \\
    --top-k 10 \\
    --query-count 100 \\
    --output retrieval_benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from redis import Redis

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.redis.utils import array_to_buffer

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "dbpedia": {
        "hf_name": "KShivendu/dbpedia-entities-openai-1M",
        "embedding_column": "openai",
        "dims": 1536,
        "distance_metric": "cosine",
        "description": "DBpedia entities, OpenAI text-embedding-ada-002, 1536d",
    },
    "cohere": {
        "hf_name": "Cohere/wikipedia-22-12-en-embeddings",
        "embedding_column": "emb",
        "dims": 768,
        "distance_metric": "cosine",
        "description": "Wikipedia EN, Cohere multilingual encoder, 768d",
    },
    "random768": {
        "hf_name": None,
        "embedding_column": None,
        "dims": 768,
        "distance_metric": "cosine",
        "description": "Synthetic random unit vectors, 768d (Cohere-scale proxy)",
    },
}

# Index configurations to benchmark
INDEX_CONFIGS = [
    {"algorithm": "flat", "datatype": "float32", "label": "FLAT_FP32"},
    {"algorithm": "flat", "datatype": "float16", "label": "FLAT_FP16"},
    {"algorithm": "hnsw", "datatype": "float32", "label": "HNSW_FP32"},
    {"algorithm": "hnsw", "datatype": "float16", "label": "HNSW_FP16"},
]

# HNSW parameters matching SVS-VAMANA study
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_RUNTIME = 10

# Recall K values to compute recall curves
RECALL_K_VALUES = [1, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval benchmark: FP32 vs FP16 x HNSW vs FLAT."
    )
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="dbpedia",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 10000],
    )
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-runtime", type=int, default=10)
    parser.add_argument("--load-batch-size", type=int, default=500)
    parser.add_argument(
        "--recall-k-max",
        type=int,
        default=100,
        help="Max K for recall curve (queries will fetch this many results).",
    )
    parser.add_argument(
        "--output",
        default="retrieval_benchmark_results.json",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset_vectors(
    dataset_key: str,
    num_vectors: int,
) -> Tuple[np.ndarray, int]:
    """Load pre-embedded vectors from HuggingFace or generate synthetic."""
    ds_info = DATASETS[dataset_key]
    dims = ds_info["dims"]

    if ds_info["hf_name"] is None:
        # Synthetic random unit vectors
        print(f"Generating {num_vectors} random unit vectors ({dims}d) ...")
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((num_vectors, dims)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        print(f"  Generated shape: {vectors.shape}")
        return vectors, dims

    # Local import to avoid requiring datasets for synthetic mode
    from datasets import load_dataset

    hf_name = ds_info["hf_name"]
    emb_col = ds_info["embedding_column"]

    print(f"Loading {num_vectors} vectors from {hf_name} ...")
    ds = load_dataset(hf_name, split=f"train[:{num_vectors}]")
    vectors = np.array(ds[emb_col], dtype=np.float32)
    print(f"  Loaded shape: {vectors.shape}")
    return vectors, dims


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def build_schema(
    *,
    index_name: str,
    prefix: str,
    dims: int,
    algorithm: str,
    datatype: str,
    distance_metric: str,
    ef_runtime: int = HNSW_EF_RUNTIME,
) -> Dict[str, Any]:
    """Build an index schema dict for a given config."""
    vector_attrs: Dict[str, Any] = {
        "dims": dims,
        "distance_metric": distance_metric,
        "algorithm": algorithm,
        "datatype": datatype,
    }
    if algorithm == "hnsw":
        vector_attrs["m"] = HNSW_M
        vector_attrs["ef_construction"] = HNSW_EF_CONSTRUCTION
        vector_attrs["ef_runtime"] = ef_runtime

    return {
        "index": {
            "name": index_name,
            "prefix": prefix,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "doc_id", "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": vector_attrs,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Data loading into Redis
# ---------------------------------------------------------------------------


def iter_documents(
    vectors: np.ndarray,
    *,
    dtype: str,
) -> Iterable[Dict[str, Any]]:
    """Yield documents ready for SearchIndex.load()."""
    for i, vec in enumerate(vectors):
        yield {
            "doc_id": f"doc-{i}",
            "embedding": array_to_buffer(vec, dtype),
        }


def wait_for_index_ready(
    index: SearchIndex,
    *,
    timeout_seconds: int = 3600,
    poll_interval: float = 0.5,
) -> Dict[str, Any]:
    """Block until the index reports 100% indexed."""
    deadline = time.perf_counter() + timeout_seconds
    info = index.info()
    while time.perf_counter() < deadline:
        info = index.info()
        pct = float(info.get("percent_indexed", 0))
        indexing = info.get("indexing", 0)
        if pct >= 1.0 and not indexing:
            return info
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Index {index.schema.index.name} not ready within {timeout_seconds}s"
    )


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def get_memory_mb(client: Redis) -> float:
    info = client.info("memory")
    return round(int(info.get("used_memory", 0)) / (1024 * 1024), 3)


# ---------------------------------------------------------------------------
# Query execution & overlap
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(np.asarray(values), pct)), 6)


def run_queries(
    index: SearchIndex,
    query_vectors: np.ndarray,
    *,
    dtype: str,
    top_k: int,
) -> Dict[str, Any]:
    """Run query vectors; return latency stats and result doc-id lists."""
    latencies_ms: List[float] = []
    result_sets: List[List[str]] = []

    for qvec in query_vectors:
        q = VectorQuery(
            vector=qvec.tolist(),
            vector_field_name="embedding",
            return_fields=["doc_id"],
            num_results=top_k,
            dtype=dtype,
        )
        t0 = time.perf_counter()
        results = index.query(q)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        result_sets.append([r.get("doc_id") or r.get("id", "") for r in results if r])

    total_s = sum(latencies_ms) / 1000
    qps = len(latencies_ms) / total_s if total_s > 0 else 0

    return {
        "count": len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "mean_ms": round(statistics.mean(latencies_ms), 3),
        "qps": round(qps, 2),
        "result_sets": result_sets,
    }


def compute_overlap(
    ground_truth: List[List[str]],
    candidate: List[List[str]],
    *,
    top_k: int,
) -> Dict[str, Any]:
    """Compute Overlap@K (precision) of candidate vs ground truth."""
    ratios: List[float] = []
    for gt, cand in zip(ground_truth, candidate):
        gt_set = set(gt[:top_k])
        cand_set = set(cand[:top_k])
        ratios.append(len(gt_set & cand_set) / max(top_k, 1))
    return {
        "mean_overlap_at_k": round(statistics.mean(ratios), 4),
        "min_overlap_at_k": round(min(ratios), 4),
        "max_overlap_at_k": round(max(ratios), 4),
        "std_overlap_at_k": (
            round(statistics.stdev(ratios), 4) if len(ratios) > 1 else 0.0
        ),
    }


def compute_recall(
    ground_truth: List[List[str]],
    candidate: List[List[str]],
    *,
    k_values: Sequence[int],
    ground_truth_depth: int,
) -> Dict[str, Any]:
    """Compute Recall@K at multiple K values.

    For each K, recall is defined as:
        |candidate_top_K intersection ground_truth_top_GT_DEPTH| / GT_DEPTH

    The ground truth set is FIXED at ground_truth_depth (e.g., top-100 from
    FLAT FP32). As K increases from 1 to ground_truth_depth, recall should
    climb from low to 1.0 (for exact search) or near-1.0 (for approximate).

    This is the standard recall metric from ANN benchmarks -- it answers
    "what fraction of the true nearest neighbors did we find?"
    """
    recall_at_k: Dict[str, float] = {}
    recall_detail: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        ratios: List[float] = []
        for gt, cand in zip(ground_truth, candidate):
            gt_set = set(gt[:ground_truth_depth])
            cand_set = set(cand[:k])
            denom = min(ground_truth_depth, len(gt_set))
            if denom == 0:
                # Empty ground truth means nothing to recall; use 0.0
                ratios.append(0.0)
            else:
                ratios.append(len(gt_set & cand_set) / denom)
        mean_recall = round(statistics.mean(ratios), 4)
        recall_at_k[f"recall@{k}"] = mean_recall
        recall_detail[f"recall@{k}"] = {
            "mean": mean_recall,
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
            "std": round(statistics.stdev(ratios), 4) if len(ratios) > 1 else 0.0,
        }
    return {
        "recall_at_k": recall_at_k,
        "recall_detail": recall_detail,
        "ground_truth_depth": ground_truth_depth,
    }


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------


def benchmark_single_config(
    *,
    client: Redis,
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    config: Dict[str, str],
    dims: int,
    distance_metric: str,
    size: int,
    top_k: int,
    ef_runtime: int,
    load_batch_size: int,
) -> Dict[str, Any]:
    """Build one index config, load data, query, and return metrics."""
    label = config["label"]
    algo = config["algorithm"]
    dtype = config["datatype"]

    index_name = f"bench_{label}_{size}"
    prefix = f"bench:{label}:{size}"

    schema = build_schema(
        index_name=index_name,
        prefix=prefix,
        dims=dims,
        algorithm=algo,
        datatype=dtype,
        distance_metric=distance_metric,
        ef_runtime=ef_runtime,
    )

    idx = SearchIndex.from_dict(schema, redis_client=client)
    try:
        idx.create(overwrite=True)

        # Load data
        load_start = time.perf_counter()
        idx.load(
            iter_documents(doc_vectors, dtype=dtype),
            id_field="doc_id",
            batch_size=load_batch_size,
        )
        info = wait_for_index_ready(idx)
        load_duration = time.perf_counter() - load_start

        memory_mb = get_memory_mb(client)

        # Query
        query_metrics = run_queries(
            idx,
            query_vectors,
            dtype=dtype,
            top_k=top_k,
        )

        return {
            "label": label,
            "algorithm": algo,
            "datatype": dtype,
            "load_duration_seconds": round(load_duration, 3),
            "num_docs": int(info.get("num_docs", 0) or 0),
            "vector_index_sz_mb": float(info.get("vector_index_sz_mb", 0) or 0),
            "memory_mb": memory_mb,
            "latency": {
                "queried_top_k": top_k,
                **{k: v for k, v in query_metrics.items() if k != "result_sets"},
            },
            "result_sets": query_metrics["result_sets"],
        }
    finally:
        try:
            idx.delete(drop=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scale-level benchmark (runs all 4 configs for one size)
# ---------------------------------------------------------------------------


def benchmark_scale(
    *,
    client: Redis,
    all_vectors: np.ndarray,
    size: int,
    query_count: int,
    dims: int,
    distance_metric: str,
    top_k: int,
    ef_runtime: int,
    load_batch_size: int,
    recall_k_max: int = 100,
) -> Dict[str, Any]:
    """Run all 4 index configs for a given dataset size."""
    doc_vectors = all_vectors[:size]
    query_vectors = all_vectors[size : size + query_count].copy()

    # Use the larger of top_k and recall_k_max for querying
    # so we have enough results for recall curve computation
    effective_top_k = max(top_k, recall_k_max)

    baseline_memory = get_memory_mb(client)

    config_results: Dict[str, Any] = {}
    ground_truth_results: List[List[str]] = []

    # Run FLAT_FP32 first to establish ground truth
    gt_config = INDEX_CONFIGS[0]  # FLAT_FP32
    assert gt_config["label"] == "FLAT_FP32"

    for config in INDEX_CONFIGS:
        label = config["label"]
        print(f"    [{label}] Building and querying ...")

        result = benchmark_single_config(
            client=client,
            doc_vectors=doc_vectors,
            query_vectors=query_vectors,
            config=config,
            dims=dims,
            distance_metric=distance_metric,
            size=size,
            top_k=effective_top_k,
            ef_runtime=ef_runtime,
            load_batch_size=load_batch_size,
        )

        if label == "FLAT_FP32":
            ground_truth_results = result["result_sets"]

        config_results[label] = result

    # Compute overlap vs ground truth for every config (at original top_k)
    overlap_results: Dict[str, Any] = {}
    for label, result in config_results.items():
        overlap = compute_overlap(
            ground_truth_results,
            result["result_sets"],
            top_k=top_k,
        )
        overlap_results[label] = overlap

    # Compute recall at multiple K values.
    # Ground truth depth is fixed at top_k (e.g., 10). We measure what
    # fraction of those top_k true results appear in candidate top-K as
    # K varies from 1 up to effective_top_k.
    valid_k_values = [k for k in RECALL_K_VALUES if k <= effective_top_k]
    recall_results: Dict[str, Any] = {}
    for label, result in config_results.items():
        recall = compute_recall(
            ground_truth_results,
            result["result_sets"],
            k_values=valid_k_values,
            ground_truth_depth=top_k,
        )
        recall_results[label] = recall

    # Strip raw result_sets from output (too large for JSON)
    for label in config_results:
        del config_results[label]["result_sets"]

    return {
        "size": size,
        "query_count": query_count,
        "dims": dims,
        "distance_metric": distance_metric,
        "top_k": top_k,
        "recall_k_max": recall_k_max,
        "ef_runtime": ef_runtime,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "baseline_memory_mb": baseline_memory,
        "configs": config_results,
        "overlap_vs_ground_truth": overlap_results,
        "recall_vs_ground_truth": recall_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    sizes = sorted(args.sizes)
    max_needed = max(sizes) + args.query_count
    ds_info = DATASETS[args.dataset]

    print(
        f"""Retrieval Benchmark
  Dataset:      {args.dataset} ({ds_info['description']})
  Dims:         {ds_info['dims']}
  Sizes:        {sizes}
  Query count:  {args.query_count}
  Top-K:        {args.top_k}
  Recall K max: {args.recall_k_max}
  EF runtime:   {args.ef_runtime}
  HNSW M:       {HNSW_M}
  EF construct: {HNSW_EF_CONSTRUCTION}
  Redis URL:    {args.redis_url}
  Configs:      {[c['label'] for c in INDEX_CONFIGS]}"""
    )

    # Load vectors once
    all_vectors, dims = load_dataset_vectors(args.dataset, max_needed)
    if all_vectors.shape[0] < max_needed:
        raise ValueError(
            f"Dataset has {all_vectors.shape[0]} vectors but need {max_needed} "
            f"(max_size={max(sizes)} + query_count={args.query_count})"
        )

    client = Redis.from_url(args.redis_url, decode_responses=False)
    client.ping()
    print("Connected to Redis")

    report = {
        "benchmark": "retrieval_fp32_vs_fp16",
        "dataset": args.dataset,
        "dataset_description": ds_info["description"],
        "dims": dims,
        "distance_metric": ds_info["distance_metric"],
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "ef_runtime": args.ef_runtime,
        "top_k": args.top_k,
        "recall_k_max": args.recall_k_max,
        "recall_k_values": [
            k for k in RECALL_K_VALUES if k <= max(args.top_k, args.recall_k_max)
        ],
        "query_count": args.query_count,
        "configs": [c["label"] for c in INDEX_CONFIGS],
        "results": [],
    }

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"  Size: {size:,} documents")
        print(f"{'='*60}")

        client.flushdb()

        result = benchmark_scale(
            client=client,
            all_vectors=all_vectors,
            size=size,
            query_count=args.query_count,
            dims=dims,
            distance_metric=ds_info["distance_metric"],
            top_k=args.top_k,
            ef_runtime=args.ef_runtime,
            load_batch_size=args.load_batch_size,
            recall_k_max=args.recall_k_max,
        )
        report["results"].append(result)

        # Print summary table for this size
        print(
            f"\n  {'Config':<12} {'Load(s)':>8} {'Memory(MB)':>11} "
            f"{'p50(ms)':>8} {'p95(ms)':>8} {'QPS':>7} {'Overlap@K':>10}"
        )
        print(f"  {'-'*12} {'-'*8} {'-'*11} {'-'*8} {'-'*8} {'-'*7} {'-'*10}")
        for label, cfg in result["configs"].items():
            overlap = result["overlap_vs_ground_truth"][label]
            print(
                f"  {label:<12} "
                f"{cfg['load_duration_seconds']:>8.1f} "
                f"{cfg['memory_mb']:>11.1f} "
                f"{cfg['latency']['p50_ms']:>8.2f} "
                f"{cfg['latency']['p95_ms']:>8.2f} "
                f"{cfg['latency']['qps']:>7.1f} "
                f"{overlap['mean_overlap_at_k']:>10.4f}"
            )

        # Print recall curve summary
        recall_data = result.get("recall_vs_ground_truth", {})
        if recall_data:
            first_label = next(iter(recall_data))
            k_keys = sorted(
                recall_data[first_label].get("recall_at_k", {}).keys(),
                key=lambda x: int(x.split("@")[1]),
            )
            header = f"  {'Config':<12} " + " ".join(f"{k:>10}" for k in k_keys)
            print(f"\n  Recall Curve:")
            print(header)
            print(f"  {'-'*12} " + " ".join(f"{'-'*10}" for _ in k_keys))
            for label, rdata in recall_data.items():
                vals = " ".join(
                    f"{rdata['recall_at_k'].get(k, 0):>10.4f}" for k in k_keys
                )
                print(f"  {label:<12} {vals}")

    # Write report
    output_path = Path(args.output).resolve()
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
