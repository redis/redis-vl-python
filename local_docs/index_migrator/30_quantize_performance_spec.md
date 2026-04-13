# Quantization Performance Overhaul

## Problem Statement

Migrating 1M keys currently takes ~50 minutes. Quantization (re-encoding vectors from one dtype to another) accounts for 78% of that time. The bottleneck is the quantize loop performing individual `HGET` calls per key per field — one network round trip each — while only pipelining the `HSET` writes.

### Current per-document cost breakdown (from benchmark at 100K)

| Component | Per-doc cost | % of migration |
|-----------|-------------|----------------|
| Quantize  | 241 µs      | 78%            |
| Reindex   | 66 µs       | ~20%           |
| Other     | —           | ~2%            |

### Why it's slow

The inner loop of `_quantize_vectors` does:

```python
for key in batch:                              # 500 keys per batch
    for field_name, change in datatype_changes.items():
        field_data = client.hget(key, field_name)  # ← 1 round trip per key per field
        # ... convert ...
        pipe.hset(key, field_name, new_bytes)       # pipelined, but writes only
    pipe.execute()                                  # 1 round trip for all writes
```

For 1M keys × 1 field: **1,000,000 read round trips + 2,000 write round trips** = ~1,002,000 total round trips.

### Additional problems

1. **BGSAVE is heavy and imprecise.** Before mutation, the executor triggers `BGSAVE` as a safety snapshot. This snapshots the *entire* database, not just the vectors being modified. For large DBs it takes minutes and provides no targeted rollback capability.

2. **QuantizationCheckpoint tracks progress but not data.** The current checkpoint records which keys have been processed. On crash you can resume, but the original vector bytes are gone — they've been overwritten in Redis. There is no rollback to original values after a crash (the `BatchUndoBuffer` only survives within a single batch, not across process crashes).

3. **Single-worker execution.** One process, one Redis connection, sequential batches. No parallelism.

## Design

Three changes, applied in order:

### Change 1: Pipeline reads

Batch `HGET` reads into a pipeline, same as writes. This is the single biggest win with zero risk.

**Before:** 1,002,000 round trips for 1M keys.
**After:** 4,000 round trips for 1M keys (2,000 read + 2,000 write).

```python
for i in range(0, total_keys, batch_size):
    batch = keys[i : i + batch_size]

    # Phase A: pipelined reads
    read_pipe = client.pipeline(transaction=False)
    read_meta = []
    for key in batch:
        for field_name, change in datatype_changes.items():
            read_pipe.hget(key, field_name)
            read_meta.append((key, field_name, change))
    read_results = read_pipe.execute()

    # Phase B: convert + pipelined writes
    write_pipe = client.pipeline(transaction=False)
    for (key, field_name, change), field_data in zip(read_meta, read_results):
        if not field_data:
            continue
        # ... idempotent check, convert, store original for backup ...
        write_pipe.hset(key, field_name, new_bytes)
    if writes_pending:
        write_pipe.execute()
```

**Estimated improvement:** 50 min → 3-5 min (10-15x speedup from eliminating read round trips).

Applies to both sync (`MigrationExecutor`) and async (`AsyncMigrationExecutor`).

### Change 2: Replace BGSAVE + QuantizationCheckpoint with a vector backup file

Remove BGSAVE entirely. Replace the current checkpoint (which only tracks progress) with a backup file that stores the **original vector bytes** for every key being quantized.

**Two-phase approach (Alternative A):**

```
Phase 1 — DUMP:
  For each batch of 500 keys:
    Pipeline-read all vector fields
    Append {key: {field: original_bytes}} to backup file
    Flush to disk

Phase 2 — QUANTIZE:
  For each batch of 500 keys:
    Read original vectors FROM the backup file (not Redis — already have them from phase 1)
    Convert dtype
    Pipeline-write new vectors to Redis
    Update progress counter in backup file header
```

#### Backup file format

Binary file using msgpack for compactness and speed. Structure:

```
[header]
  index_name: str
  total_keys: int
  fields: {field_name: {source_dtype, target_dtype, dims}}
  phase: "dump" | "quantize" | "completed"
  dump_completed_batches: int
  quantize_completed_batches: int
  batch_size: int

[batch 0]
  keys: [key1, key2, ..., key500]
  vectors: {key1: {field1: bytes, field2: bytes}, key2: {...}, ...}

[batch 1]
  ...
```

Each batch is written as a length-prefixed msgpack blob. The header is rewritten (atomically via temp+rename) after each batch to update progress counters.

#### Resume semantics

- **Crash during phase 1 (dump):** Restart dump from `dump_completed_batches`. No Redis state was modified yet.
- **Crash during phase 2 (quantize):** Resume from `quantize_completed_batches`. Original bytes are in the backup file for any batch. Idempotent dtype detection still applies as a safety net.
- **Rollback:** Read originals from backup file, HSET them back. Works for any batch at any time.

#### What this replaces

| Old component | New replacement |
|---------------|----------------|
| `trigger_bgsave_and_wait` | Removed entirely |
| `async_trigger_bgsave_and_wait` | Removed entirely |
| `QuantizationCheckpoint` model | `VectorBackup` (new) |
| `BatchUndoBuffer` | Backup file (originals always on disk) |
| `is_already_quantized` check | Kept as safety net, but less critical since backup has originals |
| BGSAVE CLI step | Removed from progress labels |


#### Disk space

Backup file size = N_keys × N_fields × bytes_per_vector.

| Scale | Dims | Source dtype | Backup size |
|-------|------|-------------|-------------|
| 100K  | 768  | float32     | ~292 MB     |
| 1M    | 768  | float32     | ~2.9 GB     |
| 1M    | 1536 | float32     | ~5.7 GB     |
| 10M   | 1536 | float32     | ~57 GB      |

Formula: `N × dims × bytes_per_element` (plus ~100 bytes/key overhead for key names and msgpack framing).

The CLI should estimate and display the required disk space before starting, and abort if insufficient.

### Change 3: Multi-worker parallelism (opt-in)

Split the key list into N slices. Each worker gets its own Redis connection, its own backup file shard, and processes its slice independently.

```
                    ┌─────────────────────┐
                    │   Coordinator       │
                    │ (main thread)       │
                    │                     │
                    │ 1. Enumerate keys   │
                    │ 2. Split into N     │
                    │ 3. Launch workers   │
                    │ 4. Wait for all     │
                    │ 5. Merge progress   │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼──────────┐ ┌─▼──────────────┐
     │  Worker 0      │ │  Worker 1    │ │  Worker N-1    │
     │  keys[0:250K]  │ │  keys[250K:] │ │  keys[750K:]   │
     │  connection 0  │ │  connection 1│ │  connection N-1│
     │  backup_0.bin  │ │  backup_1.bin│ │  backup_N.bin  │
     │                │ │              │ │                │
     │  dump → quant  │ │  dump → quant│ │  dump → quant  │
     └────────────────┘ └──────────────┘ └────────────────┘
```

#### Implementation

**Sync executor:** `concurrent.futures.ThreadPoolExecutor`. GIL is released during socket I/O (Redis calls) and numpy operations, so threads provide real parallelism for the I/O-bound workload.

**Async executor:** `asyncio.gather` with N concurrent coroutines, each using its own Redis connection from a pool.

#### Worker function (pseudocode)

```python
def _quantize_worker(
    worker_id: int,
    redis_url: str,
    keys: List[str],
    datatype_changes: Dict,
    backup_path: str,
    batch_size: int,
    progress_queue: Queue,
) -> WorkerResult:
    """Independent worker: dump + quantize a slice of keys."""
    client = Redis.from_url(redis_url)  # own connection
    backup = VectorBackup.create(backup_path, ...)

    # Phase 1: dump originals
    for batch in chunked(keys, batch_size):
        originals = pipeline_read(client, batch, datatype_changes)
        backup.write_batch(batch, originals)
        progress_queue.put(("dump", worker_id, len(batch)))

    backup.mark_dump_complete()

    # Phase 2: quantize from backup
    for batch_idx, (batch_keys, originals) in enumerate(backup.iter_batches()):
        converted = {k: convert(v, datatype_changes) for k, v in originals.items()}
        pipeline_write(client, converted)
        backup.mark_batch_quantized(batch_idx)
        progress_queue.put(("quantize", worker_id, len(batch_keys)))

    backup.mark_complete()
    client.close()
    return WorkerResult(worker_id, docs_quantized=len(keys))
```

#### Configuration

| Parameter | Default | CLI flag | Notes |
|-----------|---------|----------|-------|
| `workers` | 1 | `--workers N` | Number of parallel workers |
| `batch_size` | 500 | `--batch-size N` | Keys per pipeline batch |
| `backup_dir` | `.` | `--backup-dir PATH` | Directory for backup files |

#### Safety constraints (from benchmark report)

Per the benchmark notes on N-worker risks:

1. **Default N=1.** Opt-in only. The user must explicitly pass `--workers N` to enable parallelism.
2. **Replication backlog.** N concurrent HSET writers increase replication lag. For replicated deployments, recommend N ≤ 4 and monitor replication offset.
3. **AOF pressure.** N writers accelerate AOF buffer growth. If AOF is enabled, warn the user and suggest lower N or disabling AOF during migration.
4. **Redis is single-threaded.** N connections do not give Nx server throughput. The speedup comes from overlapping client-side I/O (network round trips). Diminishing returns above N=4-8 for a single Redis instance.
5. **Cluster mode.** Each worker's key slice must map to keys the worker's connection can reach. For non-clustered Redis this is trivial. For cluster, keys are already partitioned by slot — the coordinator should group keys by slot and assign slot-contiguous ranges to workers.

## Migration flow (new)

```
STEP 1: Enumerate keys (FT.AGGREGATE / SCAN fallback)
           │
STEP 2: Field renames (if any, pipelined)
           │
STEP 3: Drop index (FT.DROPINDEX)
           │
STEP 4: Key renames (if any, DUMP/RESTORE/DEL)
           │
STEP 5: Quantize (NEW — two-phase with backup file)
         ├─ Phase A: Dump originals to backup file(s)
         │   N workers, pipelined reads, backup file per worker
         ├─ Phase B: Convert + write back to Redis
         │   N workers, read from backup file, pipelined writes
         │
STEP 6: Create index (FT.CREATE with new schema)
           │
STEP 7: Wait for indexing (poll FT.INFO percent_indexed)
           │
STEP 8: Validate (schema match, doc count, key sample, query checks)
```

Note: BGSAVE step removed entirely.

## Files changed

### New files

| File | Purpose |
|------|---------|
| `redisvl/migration/backup.py` | `VectorBackup` class — read/write backup files |

### Modified files

| File | Changes |
|------|---------|
| `redisvl/migration/executor.py` | Pipeline reads, replace checkpoint with backup, multi-worker |
| `redisvl/migration/async_executor.py` | Same changes for async path |
| `redisvl/migration/reliability.py` | Remove `QuantizationCheckpoint`, `trigger_bgsave_and_wait`, `async_trigger_bgsave_and_wait`, `BatchUndoBuffer`. Keep `is_already_quantized`, `detect_vector_dtype`, `is_same_width_dtype_conversion`. |
| `redisvl/cli/migrate.py` | Add `--workers`, `--batch-size`, `--backup-dir` flags. Remove BGSAVE progress step. |
| `docs/concepts/index-migrations.md` | Update flow description, remove BGSAVE reference |
| `docs/user_guide/how_to_guides/migrate-indexes.md` | Update CLI flags, flow description |

### Removed functionality

| Component | Reason |
|-----------|--------|
| `trigger_bgsave_and_wait` | Replaced by backup file |
| `async_trigger_bgsave_and_wait` | Replaced by backup file |
| `QuantizationCheckpoint` | Replaced by `VectorBackup` |
| `BatchUndoBuffer` | Replaced by backup file (originals always on disk) |
| `--resume` / `checkpoint_path` parameter | Replaced by `--backup-dir` resume semantics |
| BGSAVE progress step in CLI | No longer needed |

## Expected performance

### Pipeline reads only (Change 1, N=1)

| Scale | Current | After pipelining | Speedup |
|-------|---------|-----------------|---------|
| 100K  | 31s     | ~3s             | ~10x    |
| 1M    | ~50 min | ~5 min          | ~10x    |
| 10M   | ~8 hrs  | ~50 min         | ~10x    |

### Pipeline reads + 4 workers (Changes 1+3, N=4)

| Scale | Current | After both | Speedup |
|-------|---------|-----------|---------|
| 100K  | 31s     | ~1s       | ~30x    |
| 1M    | ~50 min | ~1.5 min  | ~33x    |
| 10M   | ~8 hrs  | ~15 min   | ~32x    |

Note: Worker speedup is sub-linear (not 4x) because Redis is single-threaded. The gains come from overlapping client-side network I/O. Actual speedup depends on network latency, Redis load, and whether the deployment is standalone or clustered.

### Two-phase overhead (Change 2)

The dump phase adds one extra read pass over all keys. But since reads are now pipelined, this adds ~50% to read time (one extra pipeline execution per batch). The quantize phase reads from the local backup file instead of Redis, which is faster than Redis reads. Net effect: roughly neutral — the dump overhead is offset by faster quantize reads.

## Implementation order

1. **Pipeline reads** (sync + async). Run benchmarks to validate 10x improvement.
2. **VectorBackup file** (new module). Unit test read/write/resume.
3. **Replace BGSAVE + checkpoint** in executor with backup file. Integration test.
4. **Multi-worker** (sync first, then async). Benchmark at N=1,2,4.
5. **CLI flags** (`--workers`, `--batch-size`, `--backup-dir`).
6. **Update docs** and existing tests.

## Open questions

1. **msgpack vs pickle vs custom binary?** msgpack is compact and fast but adds a dependency. pickle is stdlib but not portable. Custom binary is zero-dep but more code. Recommendation: msgpack (already a common dep in data workflows, ~10x faster than YAML).
2. **Should backup files be cleaned up automatically on success?** Recommend yes with a `--keep-backup` flag to retain.
3. **Should the dump phase estimate and display ETA?** Yes — can calculate from batch throughput after the first few batches.
4. **Cluster slot-aware worker assignment?** Defer to a follow-up. For now, workers are assigned contiguous key ranges which works for standalone Redis. Cluster support needs slot grouping.

## References

- Benchmark report: `local_docs/index_migrator/05_migration_benchmark_report.md`
- Scaling notes: `local_docs/index_migrator/notes_scaling_and_reliability.md`
- PR review (pipelined reads): nkode run `ce95e0e4`, finding #2
- PR review (async pipelining): Copilot comment on `async_executor.py:639-654`
- PR review (sync pipelining): Copilot comment on `executor.py:674-695`