# Migration Benchmark Report

## Overview

This report presents the results of benchmarking the RedisVL index migration process at three scales (1K, 10K, 100K documents). The migration converts an HNSW FP32 index to a FLAT FP16 index. All runs use a realistic 16-field schema modeled after a production knowledge management index with 3072-dimensional COSINE vectors.

The benchmark isolates **load time** (populating Redis with synthetic data) from **migrate time** (the actual migration operation). It also confirms which enumeration method the migrator used to discover document keys.

## Environment

All trials ran on a single local machine against a Docker-hosted Redis Stack instance. Each scale was tested 3 times. Results shown below are averages across the 3 trials.

## Results Summary

| Scale | Load Time | Migrate Time | Quantize | Reindex | Downtime | Vec Memory Before | Vec Memory After | Saved |
|---|---|---|---|---|---|---|---|---|
| 1K | 0.4s | 0.8s | 0.3s | 0.5s | 0.8s | 12.3 MB | 6.0 MB | 51% |
| 10K | 26.2s | 3.4s | 2.4s | 1.0s | 3.4s | 123.5 MB | 60.5 MB | 51% |
| 100K | 454s | 30.9s | 24.1s | 6.6s | 30.7s | 1,211 MB | 593 MB | 51% |

All 9 trials used **FT.AGGREGATE** for key enumeration (never SCAN). All trials passed validation.

## Enumeration Method

The migrator discovers which keys belong to the index before starting the migration. It uses `FT.AGGREGATE ... WITHCURSOR` as the primary method, which returns only indexed keys without scanning the full keyspace. SCAN is a fallback reserved for cases where `hash_indexing_failures > 0` or `FT.AGGREGATE` errors out. In all 9 trials, enumeration used `FT.AGGREGATE` and completed in under 150ms even at 100K.

## How Drop-Recreate Migration Works

The migration executor follows this sequence:

**STEP 1: Enumerate keys** (before any modifications)
- Uses FT.AGGREGATE WITHCURSOR to discover all document keys in the source index
- Fallback to SCAN if the index has hash_indexing_failures > 0 or FT.AGGREGATE fails
- Keys are stored in memory for the quantization step

**STEP 2: Drop source index**
- Issues FT.DROPINDEX (without KEEPDOCS) to remove the index structure
- **The underlying documents remain in Redis** - only the index metadata is deleted
- At this point, the index is gone but all document hashes/JSON still exist with their FP32 vectors

**STEP 3: Quantize vectors** (rewrite document payloads IN-PLACE)
- For each document in the enumerated key list:
  - HGETALL to read the document (including FP32 vector)
  - Convert FP32 → FP16 in Python
  - HSET to write back the FP16 vector to the same document
- Processes documents in batches of 500 using Redis pipelines
- **Memory note**: The old index is already dropped, so there is no "double index" overhead. Only the document data exists in Redis during this phase.

**STEP 4: Key renames** (if needed)
- If the migration changes the key prefix, RENAME each key from old prefix to new prefix
- Skipped if no prefix change

**STEP 5: Create target index**
- Issues FT.CREATE with the new schema (FLAT, FP16, etc.)
- Redis begins background indexing of existing documents

**STEP 6: Wait for re-indexing**
- Polls FT.INFO until indexing completes (num_docs == expected count)
- The index is unavailable for queries until this completes

## Phase Breakdown

Here is the average duration of each phase at 100K documents (the most representative scale).

| Phase | Duration | Share of Migration |
|---|---|---|
| Enumerate | 0.12s | 0.4% |
| Drop | 0.00s | 0.0% |
| Quantize | 24.1s | 77.9% |
| Create | 0.002s | 0.0% |
| Reindex | 6.6s | 21.3% |
| Validate | 0.012s | 0.0% |

Quantization dominates at every scale. This is the client-side step that reads each document's FP32 vector from Redis, converts it to FP16 in Python, and writes it back. It is inherently I/O-bound and proportional to document count.

## Scaling Analysis

The central question is whether migration time grows linearly with document count, which determines whether we can predict costs at 1M and 10M.

### Per-Document Costs

| Scale | Per-Doc Quantize | Per-Doc Reindex |
|---|---|---|
| 1K | 277 us | 511 us |
| 10K | 237 us | 102 us |
| 100K | 241 us | 66 us |

**Quantize scales linearly.** The per-document cost stabilizes around 240 microseconds from 10K onward. This makes sense because each document requires one HGETALL and one HSET regardless of index size. There is no interaction between documents during quantization.

**Reindex scales sub-linearly.** The per-document cost decreases as scale increases. This is expected for FLAT indexes where Redis performs a simple sequential scan to build the brute-force index. Fixed overhead (index creation, initial polling delay) is amortized over more documents. At 100K the reindex throughput reaches ~15K docs/sec.

### Scaling Ratios (10x increments)

| Metric | 1K to 10K (10x data) | 10K to 100K (10x data) |
|---|---|---|
| Quantize time | 8.5x | 10.2x |
| Reindex time | 2.0x | 6.5x |
| Total migrate | 4.3x | 9.1x |

The 10K-to-100K ratio is the most reliable predictor since 1K has proportionally more fixed overhead. Quantize is essentially 10x for 10x data (linear). Reindex is growing faster than at small scale but still sub-linear.

## Predictions for 1M and 10M

Using the per-document rates observed at 100K (the most representative scale) and assuming linear scaling for quantize with a conservative linear assumption for reindex.

### Per-Document Rates Used

| Component | Rate |
|---|---|
| Quantize | 241 us/doc (from 100K average) |
| Reindex | 66 us/doc (from 100K, likely optimistic at larger scale) |

### Projected Migration Times

| Scale | Quantize | Reindex | Total Migrate | Downtime |
|---|---|---|---|---|
| **1M** | ~241s (~4 min) | ~66s (~1.1 min) | ~5.2 min | ~5.2 min |
| **10M** | ~2,410s (~40 min) | ~660s (~11 min) | ~51 min | ~51 min |

### Caveats on These Predictions

**1M is realistic.** The quantize step is pure per-document I/O with no cross-document dependencies, so linear extrapolation is well-founded. Reindex for a FLAT index at 1M should also remain close to linear. Memory requirement would be roughly 11.4 GB for FP32 vectors plus metadata, so a machine with 32 GB RAM should handle it.

**10M carries significant risk factors.**

1. **Memory requirement.** 10M documents at 3072 dimensions requires ~57 GB for FP32 vectors plus metadata overhead. During quantization, the source index has already been dropped, so there is no "double index" memory overhead. Each batch (500 docs) temporarily holds both FP32 and FP16 representations during HSET, but this is a small incremental cost. The main memory requirement is the baseline FP32 data (~57 GB), not 80+ GB. After quantization completes, memory drops to ~28.5 GB (FP16 vectors). A machine with 64-128 GB RAM should handle this comfortably.

2. **Reindex may slow down.** FLAT index construction at 10M with 3072 dimensions means Redis must build a brute-force index over the FP16 vector data. Background indexing throughput may degrade at this scale, especially if Redis is under memory pressure or serving concurrent traffic.

3. **Quantize could slow down.** At 10M, the pipeline batches (500 docs each) would execute 20,000 batch cycles. If Redis starts swapping or if network latency increases under load, per-batch cost could rise above the observed 241 us/doc average.

4. **FLAT may not be the right target at 10M.** A 10M-document FLAT index would make every query a brute-force scan over 10M vectors, which is impractical for production. HNSW FP16 would be the appropriate target, and HNSW index construction is O(n log n) rather than O(n), which would increase the reindex phase significantly (potentially 2-3x longer).

### Adjusted Predictions with Risk

| Scale | Optimistic | Expected | Pessimistic |
|---|---|---|---|
| **1M** | 4.5 min | 5.5 min | 8 min |
| **10M (FLAT target)** | 50 min | 60 min | 90 min |
| **10M (HNSW target)** | 70 min | 90 min | 150+ min |

The pessimistic 10M estimate accounts for HNSW rebuild cost (O(n log n) indexing) and potential per-batch slowdown at scale. A production 10M migration would require a machine with 64-128 GB RAM and should be empirically tested before deployment. The memory requirement is the baseline FP32 data size (~57 GB), not double that, because the source index is dropped before quantization begins.

## Async vs Sync Executor Comparison

A second set of 9 trials was run using `AsyncMigrationExecutor` instead of the sync `MigrationExecutor`. The async executor is what the CLI (`rvl migrate apply`) uses internally and was expected to show improved throughput through non-blocking I/O.

### Async Results Summary

| Scale | Migrate Time | Quantize | Reindex | Downtime |
|---|---|---|---|---|
| 1K | 0.8s | 0.30s | 0.51s | 0.81s |
| 10K | 4.3s | 3.21s | 1.01s | 4.23s |
| 100K | 35.8s | 29.6s | 6.06s | 35.7s |

### Side-by-Side Comparison

| Scale | Sync Migrate | Async Migrate | Sync Quantize | Async Quantize | Async Overhead |
|---|---|---|---|---|---|
| 1K | 0.8s | 0.8s | 0.3s | 0.3s | ~0% |
| 10K | 3.4s | 4.3s | 2.4s | 3.2s | +33% |
| 100K | 30.9s | 35.8s | 24.1s | 29.6s | +23% |

### Why Async is Slower

The async executor adds overhead without gaining parallelism for three reasons.

**Single-connection I/O.** Both executors talk to Redis over a single TCP connection. The async event loop adds coroutine scheduling and context-switch overhead on every `await`, but cannot overlap commands because Redis processes them sequentially on one connection.

**CPU-bound quantization.** The FP32 to FP16 conversion uses `struct.unpack` and `struct.pack` in Python. This is CPU-bound work that gets no benefit from `asyncio`. The event loop overhead adds roughly 50 microseconds per document (296 us/doc async vs 241 us/doc sync at 100K).

**Identical batching strategy.** Both executors use the same `pipeline.execute()` pattern with batches of 500 documents. The async version does not overlap I/O across batches because each batch must complete before the next begins.

### When Async Would Help

The async executor exists for integration with async application code (the CLI, web frameworks, or other coroutine-based systems). It does not improve raw migration throughput. To actually speed up the quantize phase, the optimization path would be multi-connection parallelism (splitting the key list across N workers, each with its own Redis connection), not async/await on a single connection.

### N-Worker Parallelism Considerations

Multi-connection parallelism has production risks that should be weighed before enabling it. Redis is single-threaded for command processing, so N connections do not give N times server-side throughput. The client-side overlap of network round-trips provides the speedup, but the server processes commands sequentially from one queue. In production deployments with replicas, concurrent HSET writes from N workers increase replication backlog pressure. If the buffer fills, Redis disconnects the replica and triggers a full resync, which is catastrophic during migration. AOF persistence adds similar risk since N concurrent writers accelerate AOF buffer growth and could trigger an AOF rewrite that forks the process and temporarily doubles memory. Sharded deployments require shard-aware key partitioning to avoid hotspots, and Redis Cloud proxy layers add per-connection overhead that does not appear in local benchmarks. The safe default should remain N=1 with opt-in parallelism. See `local_docs/index_migrator/03_benchmarking.md` for the full risk analysis.

## Key Takeaways

1. **Migration is fast relative to data loading.** At 100K, loading took 7.5 minutes while migration took only 31 seconds. The migration operation itself is not the bottleneck in any deployment workflow.

2. **Quantization dominates migration time at every scale** (~78% of total). Any optimization effort should focus on the quantize step, such as parallelizing the read-convert-write pipeline across multiple connections.

3. **Sync executor is faster than async** for raw migration throughput. The async version adds ~23% overhead at scale due to event loop costs on CPU-bound work.

4. **FT.AGGREGATE is the default enumeration path** and it works reliably. SCAN fallback exists but did not trigger in any trial.

5. **Vector memory savings are exactly 51% at every scale.** FP16 cuts the vector index footprint in half with no variation. Non-vector index metadata is unchanged.

6. **Linear extrapolation is valid up to 1M.** Beyond that, memory pressure and index algorithm choice (FLAT vs HNSW) introduce non-linear factors that require empirical validation.

## Raw Data

- Sync results: `tests/benchmarks/results_migration.json`
- Async results: `tests/benchmarks/results_migration_async.json`

