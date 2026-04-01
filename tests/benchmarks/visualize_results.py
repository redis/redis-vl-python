#!/usr/bin/env python3
"""
Visualization script for retrieval benchmark results.

Generates charts replicating the style of the Redis SVS-VAMANA blog post:
  1. Memory footprint comparison (FP32 vs FP16, bar chart)
  2. Precision (Overlap@K) comparison (grouped bar chart)
  3. QPS comparison (grouped bar chart)
  4. Latency comparison (p50/p95, grouped bar chart)
  5. QPS vs Overlap@K curve (line chart)

Usage:
    python tests/benchmarks/visualize_results.py \
        --input tests/benchmarks/results_dbpedia.json \
        --output-dir tests/benchmarks/charts/
"""

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Redis-inspired color palette
COLORS = {
    "FLAT_FP32": "#1E3A5F",  # dark navy
    "FLAT_FP16": "#3B82F6",  # bright blue
    "HNSW_FP32": "#DC2626",  # Redis red
    "HNSW_FP16": "#F97316",  # orange
}

LABELS = {
    "FLAT_FP32": "FLAT FP32",
    "FLAT_FP16": "FLAT FP16",
    "HNSW_FP32": "HNSW FP32",
    "HNSW_FP16": "HNSW FP16",
}


def load_results(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def setup_style():
    """Apply a clean, modern chart style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8F9FA",
            "axes.edgecolor": "#DEE2E6",
            "axes.grid": True,
            "grid.color": "#E9ECEF",
            "grid.alpha": 0.7,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
        }
    )


def chart_memory(results: List[Dict], dataset: str, output_dir: str):
    """Chart 1: Memory footprint comparison per size (grouped bar chart)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ["FLAT_FP32", "FLAT_FP16", "HNSW_FP32", "HNSW_FP16"]
    sizes = [r["size"] for r in results]
    x = np.arange(len(sizes))
    width = 0.18

    for i, cfg in enumerate(configs):
        mem = [r["configs"][cfg]["memory_mb"] for r in results]
        bars = ax.bar(
            x + i * width,
            mem,
            width,
            label=LABELS[cfg],
            color=COLORS[cfg],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, mem):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Total Memory (MB)")
    ax.set_title(f"Memory Footprint: FP32 vs FP16 -- {dataset}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset}_memory.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {dataset}_memory.png")


def chart_overlap(results: List[Dict], dataset: str, output_dir: str):
    """Chart 2: Overlap@K (precision) comparison per size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ["FLAT_FP32", "FLAT_FP16", "HNSW_FP32", "HNSW_FP16"]
    sizes = [r["size"] for r in results]
    x = np.arange(len(sizes))
    width = 0.18

    for i, cfg in enumerate(configs):
        overlap = [
            r["overlap_vs_ground_truth"][cfg]["mean_overlap_at_k"] for r in results
        ]
        bars = ax.bar(
            x + i * width,
            overlap,
            width,
            label=LABELS[cfg],
            color=COLORS[cfg],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, overlap):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Overlap@K (Precision vs FLAT FP32)")
    ax.set_title(f"Search Precision: FP32 vs FP16 -- {dataset}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset}_overlap.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {dataset}_overlap.png")


def chart_qps(results: List[Dict], dataset: str, output_dir: str):
    """Chart 3: QPS comparison per size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ["FLAT_FP32", "FLAT_FP16", "HNSW_FP32", "HNSW_FP16"]
    sizes = [r["size"] for r in results]
    x = np.arange(len(sizes))
    width = 0.18

    for i, cfg in enumerate(configs):
        qps = [r["configs"][cfg]["latency"]["qps"] for r in results]
        bars = ax.bar(
            x + i * width,
            qps,
            width,
            label=LABELS[cfg],
            color=COLORS[cfg],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, qps):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Queries Per Second (QPS)")
    ax.set_title(f"Query Throughput: FP32 vs FP16 -- {dataset}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset}_qps.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {dataset}_qps.png")


def chart_latency(results: List[Dict], dataset: str, output_dir: str):
    """Chart 4: p50 and p95 latency comparison per size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    configs = ["FLAT_FP32", "FLAT_FP16", "HNSW_FP32", "HNSW_FP16"]
    sizes = [r["size"] for r in results]
    x = np.arange(len(sizes))
    width = 0.18

    for ax, metric, title in zip(
        axes, ["p50_ms", "p95_ms"], ["p50 Latency", "p95 Latency"]
    ):
        for i, cfg in enumerate(configs):
            vals = [r["configs"][cfg]["latency"][metric] for r in results]
            bars = ax.bar(
                x + i * width,
                vals,
                width,
                label=LABELS[cfg],
                color=COLORS[cfg],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xlabel("Corpus Size")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{title} -- {dataset}")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f"{s:,}" for s in sizes])
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset}_latency.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {dataset}_latency.png")


def chart_qps_vs_overlap(results: List[Dict], dataset: str, output_dir: str):
    """Chart 5: QPS vs Overlap@K curve (Redis blog Chart 2 style)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ["FLAT_FP32", "FLAT_FP16", "HNSW_FP32", "HNSW_FP16"]
    markers = {"FLAT_FP32": "s", "FLAT_FP16": "D", "HNSW_FP32": "o", "HNSW_FP16": "^"}

    for cfg in configs:
        overlaps = []
        qps_vals = []
        for r in results:
            overlaps.append(r["overlap_vs_ground_truth"][cfg]["mean_overlap_at_k"])
            qps_vals.append(r["configs"][cfg]["latency"]["qps"])

        ax.plot(
            overlaps,
            qps_vals,
            marker=markers[cfg],
            markersize=8,
            linewidth=2,
            label=LABELS[cfg],
            color=COLORS[cfg],
        )
        # Annotate points with size
        for ov, qps, r in zip(overlaps, qps_vals, results):
            ax.annotate(
                f'{r["size"]//1000}K',
                (ov, qps),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color=COLORS[cfg],
            )

    ax.set_xlabel("Overlap@K (Precision)")
    ax.set_ylabel("Queries Per Second (QPS)")
    ax.set_title(f"Precision vs Throughput -- {dataset}")
    ax.legend(loc="best")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset}_qps_vs_overlap.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {dataset}_qps_vs_overlap.png")


def chart_memory_savings(results: List[Dict], dataset: str, output_dir: str):
    """Chart 6: Memory savings percentage (Redis blog Chart 1 style)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = [r["size"] for r in results]

    # Calculate savings: FP16 vs FP32 for both FLAT and HNSW
    pairs = [
        ("FLAT", "FLAT_FP32", "FLAT_FP16", "#3B82F6"),
        ("HNSW", "HNSW_FP32", "HNSW_FP16", "#F97316"),
    ]

    x = np.arange(len(sizes))
    width = 0.3

    for i, (label, fp32, fp16, color) in enumerate(pairs):
        savings = []
        for r in results:
            m32 = r["configs"][fp32]["memory_mb"]
            m16 = r["configs"][fp16]["memory_mb"]
            pct = (1 - m16 / m32) * 100 if m32 > 0 else 0.0
            savings.append(pct)

        bars = ax.bar(
            x + i * width,
            savings,
            width,
            label=f"{label} FP16 savings",
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, savings):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Memory Savings (%)")
    ax.set_title(f"FP16 Memory Savings vs FP32 -- {dataset}")
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 60)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset}_memory_savings.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {dataset}_memory_savings.png")


def chart_build_time(results: List[Dict], dataset: str, output_dir: str):
    """Chart 7: Index build/load time comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ["FLAT_FP32", "FLAT_FP16", "HNSW_FP32", "HNSW_FP16"]
    sizes = [r["size"] for r in results]
    x = np.arange(len(sizes))
    width = 0.18

    for i, cfg in enumerate(configs):
        times = [r["configs"][cfg]["load_duration_seconds"] for r in results]
        bars = ax.bar(
            x + i * width,
            times,
            width,
            label=LABELS[cfg],
            color=COLORS[cfg],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, times):
            if val > 0.1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Build Time (seconds)")
    ax.set_title(f"Index Build Time -- {dataset}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset}_build_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {dataset}_build_time.png")


def chart_recall_curve(results: List[Dict], dataset: str, output_dir: str):
    """Chart 8: Recall@K curve -- recall at multiple K values for the largest size."""
    # Use the largest corpus size for the recall curve
    r = results[-1]
    recall_data = r.get("recall_vs_ground_truth")
    if not recall_data:
        print(f"  Skipping recall curve (no recall data in results)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ["FLAT_FP32", "FLAT_FP16", "HNSW_FP32", "HNSW_FP16"]
    markers = {"FLAT_FP32": "s", "FLAT_FP16": "D", "HNSW_FP32": "o", "HNSW_FP16": "^"}
    linestyles = {
        "FLAT_FP32": "-",
        "FLAT_FP16": "--",
        "HNSW_FP32": "-",
        "HNSW_FP16": "--",
    }

    for cfg in configs:
        if cfg not in recall_data:
            continue
        recall_at_k = recall_data[cfg].get("recall_at_k", {})
        if not recall_at_k:
            continue
        k_vals = sorted([int(k.split("@")[1]) for k in recall_at_k.keys()])
        recalls = [recall_at_k[f"recall@{k}"] for k in k_vals]

        ax.plot(
            k_vals,
            recalls,
            marker=markers[cfg],
            markersize=7,
            linewidth=2,
            linestyle=linestyles[cfg],
            label=LABELS[cfg],
            color=COLORS[cfg],
        )

    ax.set_xlabel("K (number of results)")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"Recall@K Curve at {r['size']:,} documents -- {dataset}")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset}_recall_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {dataset}_recall_curve.png")


def chart_recall_by_size(results: List[Dict], dataset: str, output_dir: str):
    """Chart 9: Recall@10 comparison across corpus sizes (grouped bar chart)."""
    # Check if recall data exists
    if not results[0].get("recall_vs_ground_truth"):
        print(f"  Skipping recall by size (no recall data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ["FLAT_FP32", "FLAT_FP16", "HNSW_FP32", "HNSW_FP16"]
    sizes = [r["size"] for r in results]
    x = np.arange(len(sizes))
    width = 0.18

    for i, cfg in enumerate(configs):
        recalls = []
        for r in results:
            recall_data = r.get("recall_vs_ground_truth", {}).get(cfg, {})
            recall_at_k = recall_data.get("recall_at_k", {})
            recalls.append(recall_at_k.get("recall@10", 0))
        bars = ax.bar(
            x + i * width,
            recalls,
            width,
            label=LABELS[cfg],
            color=COLORS[cfg],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, recalls):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Recall@10: FP32 vs FP16 -- {dataset}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset}_recall.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {dataset}_recall.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results.")
    parser.add_argument(
        "--input", nargs="+", required=True, help="One or more result JSON files."
    )
    parser.add_argument(
        "--output-dir",
        default="tests/benchmarks/charts/",
        help="Directory to save chart images.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_style()

    for path in args.input:
        data = load_results(path)
        dataset = data["dataset"]
        results = data["results"]
        print(f"\nGenerating charts for {dataset} ({len(results)} sizes) ...")

        chart_memory(results, dataset, args.output_dir)
        chart_overlap(results, dataset, args.output_dir)
        chart_qps(results, dataset, args.output_dir)
        chart_latency(results, dataset, args.output_dir)
        chart_qps_vs_overlap(results, dataset, args.output_dir)
        chart_memory_savings(results, dataset, args.output_dir)
        chart_build_time(results, dataset, args.output_dir)
        chart_recall_curve(results, dataset, args.output_dir)
        chart_recall_by_size(results, dataset, args.output_dir)

    print(f"\nAll charts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
