---
myst:
  html_meta:
    "description lang=en": |
      How to migrate a RedisVL index schema without losing data.
---

# Migrate an Index

This guide shows how to safely change your index schema using the RedisVL migrator.

## Quick Start

Add a field to your index in 4 commands:

```bash
# 1. See what indexes exist
rvl index listall --url redis://localhost:6379

# 2. Use the wizard to build a migration plan
rvl migrate wizard --index myindex --url redis://localhost:6379

# 3. Apply the migration
rvl migrate apply --plan migration_plan.yaml --url redis://localhost:6379

# 4. Verify the result
rvl migrate validate --plan migration_plan.yaml --url redis://localhost:6379
```

## Prerequisites

- Redis with the Search module (Redis Stack, Redis Cloud, or Redis Enterprise)
- An existing index to migrate
- `redisvl` installed (`pip install redisvl`)

```bash
# Local development with Redis 8.0+ (recommended for full feature support)
docker run -d --name redis -p 6379:6379 redis:8.0
```

**Note:** Redis 8.0+ is required for INT8/UINT8 vector datatypes. SVS-VAMANA algorithm requires Redis 8.2+ and Intel AVX-512 hardware.


## How It Works

Every migration follows the same three-phase flow: **describe what changed** (the patch),
**generate a plan** (diffing the patch against the live schema), and **execute the plan**.

### Single-Index Flow: wizard/plan then apply

```
wizard (interactive)                   plan (non-interactive)
        |                                    |
        v                                    v
  SchemaPatch YAML  <----or---->  SchemaPatch YAML
        |                                    |
        +------ planner.create_plan() -------+
                       |
                       v
              MigrationPlan YAML
                       |
                       v
              executor.apply()
                       |
                       v
              MigrationReport YAML
```

**Phase 1: Build a SchemaPatch.**
A patch is a small YAML file that declares *what you want to change*, not the full target schema.
You can build it interactively with `rvl migrate wizard`, or write it by hand. The patch has
five sections, each optional:

| Patch Section | What it does |
|---|---|
| `add_fields` | Adds new field definitions to the index |
| `remove_fields` | Removes fields from the index (document data is kept, just no longer indexed) |
| `rename_fields` | Renames fields in both the index schema and all documents (HGET old, HSET new, HDEL old) |
| `update_fields` | Modifies field attributes: algorithm, datatype, distance metric, sortable, separator, etc. |
| `index` | Changes the index name or key prefix |

**Phase 2: Generate a MigrationPlan.**
The planner connects to Redis, snapshots the live index schema and stats,
then merges the patch into the source schema to produce a `merged_target_schema`.
It classifies every change as supported or blocked and extracts rename operations.

The plan YAML contains:
- `source`: frozen snapshot of the live index at planning time (schema, stats, key sample, prefixes)
- `requested_changes`: the patch that was applied
- `merged_target_schema`: source + patch = what the index will look like after migration
- `diff_classification`: whether the migration is supported and any blocked reasons
- `rename_operations`: extracted index renames, prefix changes, and field renames
- `warnings`: any important notes (downtime required, lossy quantization, etc.)

The same patch produces different plans per index because each index has a different source schema.

**Phase 3: Apply.**
The executor reads the plan and runs the migration steps:

1. Enumerate keys (SCAN with source prefix)
2. Field renames (pipelined HGET/HSET/HDEL)
3. Dump original vectors to backup file (if quantizing and backup-dir provided)
4. Drop index (FT.DROPINDEX, documents are preserved)
5. Key prefix renames (RENAME or DUMP/RESTORE for cluster)
6. Quantize vectors from backup (pipelined read/convert/write)
7. Create index (FT.CREATE with merged target schema)
8. Wait for re-indexing to complete
9. Validate (doc count, schema match, key sample)

### Batch Flow: wizard/plan then batch-plan then batch-apply

For applying the same change across multiple indexes:

```
SchemaPatch YAML  (shared, written once)
        |
        v
batch_planner.create_batch_plan()
  for each index:
    snapshot live schema
    merge patch into source
    if applicable: write per-index MigrationPlan
    if not: mark skip_reason
        |
        v
BatchPlan YAML
  shared_patch: { ... }
  indexes:
    - name: idx_a, applicable: true, plan_path: plans/idx_a.yaml
    - name: idx_b, applicable: true, plan_path: plans/idx_b.yaml
    - name: idx_c, applicable: false, skip_reason: "field not found"
        |
        v
batch_executor.apply()
  for each applicable index (sequentially):
    executor.apply(per_index_plan)
```

The batch planner takes a **single shared patch** and tests it against every target index.
Indexes where the patch doesn't apply (e.g., it references a field that doesn't exist in that
index, or the change is blocked) are marked `applicable: false` with a `skip_reason` and skipped
during apply. Each applicable index gets its own full `MigrationPlan` written to disk.

This means you can review each per-index plan individually before running `batch-apply`.


## Step 1: Discover Available Indexes

```bash
rvl index listall --url redis://localhost:6379
```

**Example output:**
```
Indices:
  1. products_idx
  2. users_idx
  3. orders_idx
```

## Step 2: Build Your Schema Change

Choose one of these approaches:

### Option A: Use the Wizard (Recommended)

The wizard guides you through building a migration interactively. Run:

```bash
rvl migrate wizard --index myindex --url redis://localhost:6379
```

**Example wizard session (adding a field):**

```text
Building a migration plan for index 'myindex'
Current schema:
- Index name: myindex
- Storage type: hash
  - title (text)
  - embedding (vector)

Choose an action:
1. Add field        (text, tag, numeric, geo)
2. Update field     (sortable, weight, separator)
3. Remove field
4. Preview patch    (show pending changes as YAML)
5. Finish
Enter a number: 1

Field name: category
Field type options: text, tag, numeric, geo
Field type: tag
  Sortable: enables sorting and aggregation on this field
Sortable [y/n]: n
  Separator: character that splits multiple values (default: comma)
Separator [leave blank to keep existing/default]: |

Choose an action:
1. Add field        (text, tag, numeric, geo)
2. Update field     (sortable, weight, separator)
3. Remove field
4. Preview patch    (show pending changes as YAML)
5. Finish
Enter a number: 5

Migration plan written to /path/to/migration_plan.yaml
Mode: drop_recreate
Supported: True
Warnings:
- Index downtime is required
```

**Example wizard session (quantizing vectors):**

```text
Choose an action:
1. Add field        (text, tag, numeric, geo)
2. Update field     (sortable, weight, separator)
3. Remove field
4. Preview patch    (show pending changes as YAML)
5. Finish
Enter a number: 2

Updatable fields:
1. title (text)
2. embedding (vector)
Select a field to update by number or name: 2

Current vector config for 'embedding':
  algorithm: HNSW
  datatype: float32
  distance_metric: cosine
  dims: 384 (cannot be changed)
  m: 16
  ef_construction: 200

Leave blank to keep current value.
  Algorithm: vector search method (FLAT=brute force, HNSW=graph, SVS-VAMANA=compressed graph)
Algorithm [current: HNSW]:
  Datatype: float16, float32, bfloat16, float64, int8, uint8
            (float16 reduces memory ~50%, int8/uint8 reduce ~75%)
Datatype [current: float32]: float16
  Distance metric: how similarity is measured (cosine, l2, ip)
Distance metric [current: cosine]:
  M: number of connections per node (higher=better recall, more memory)
M [current: 16]:
  EF_CONSTRUCTION: build-time search depth (higher=better recall, slower build)
EF_CONSTRUCTION [current: 200]:

Choose an action:
...
5. Finish
Enter a number: 5

Migration plan written to /path/to/migration_plan.yaml
Mode: drop_recreate
Supported: True
```

### Option B: Write a Schema Patch (YAML)

Create `schema_patch.yaml` manually:

```yaml
version: 1
changes:
  add_fields:
    - name: category
      type: tag
      path: $.category
      attrs:
        separator: "|"
  remove_fields:
    - legacy_field
  update_fields:
    - name: title
      attrs:
        sortable: true
    - name: embedding
      attrs:
        datatype: float16        # quantize vectors
        algorithm: HNSW
        distance_metric: cosine
```

Then generate the plan:

```bash
rvl migrate plan \
  --index myindex \
  --schema-patch schema_patch.yaml \
  --url redis://localhost:6379 \
  --plan-out migration_plan.yaml
```

### Option C: Provide a Target Schema

If you have the complete target schema, use it directly:

```bash
rvl migrate plan \
  --index myindex \
  --target-schema target_schema.yaml \
  --url redis://localhost:6379 \
  --plan-out migration_plan.yaml
```

## Step 3: Review the Migration Plan

Before applying, review `migration_plan.yaml`:

```yaml
# migration_plan.yaml (example)
version: 1
mode: drop_recreate

source:
  schema_snapshot:
    index:
      name: myindex
      prefix: "doc:"
      storage_type: json
    fields:
      - name: title
        type: text
      - name: embedding
        type: vector
        attrs:
          dims: 384
          algorithm: hnsw
          datatype: float32
  stats_snapshot:
    num_docs: 10000
  keyspace:
    prefixes: ["doc:"]
    key_sample: ["doc:1", "doc:2", "doc:3"]

requested_changes:
  add_fields:
    - name: category
      type: tag

diff_classification:
  supported: true
  blocked_reasons: []

rename_operations:
  rename_index: null
  change_prefix: null
  rename_fields: []

merged_target_schema:
  index:
    name: myindex
    prefix: "doc:"
    storage_type: json
  fields:
    - name: title
      type: text
    - name: category
      type: tag
    - name: embedding
      type: vector
      attrs:
        dims: 384
        algorithm: hnsw
        datatype: float32

warnings:
  - "Index downtime is required"
```

**Key fields to check:**
- `diff_classification.supported` - Must be `true` to proceed
- `diff_classification.blocked_reasons` - Must be empty
- `warnings` - Top-level warnings about the migration
- `merged_target_schema` - The final schema after migration

## Understanding Downtime Requirements

**CRITICAL**: During a `drop_recreate` migration, your application must:

| Requirement | Description |
|-------------|-------------|
| **Pause reads** | Index is unavailable during migration |
| **Pause writes** | Writes during migration may be missed or cause conflicts |

### Why Both Reads AND Writes Must Be Paused

- **Reads**: The index definition is dropped and recreated. Any queries during this window will fail.
- **Writes**: Redis updates indexes synchronously on every write. If your app writes documents while the index is dropped, those writes are not indexed. Additionally, if you're quantizing vectors (float32 → float16), concurrent writes may conflict with the migration's re-encoding process.

### What "Downtime" Means

| Downtime Type | Reads | Writes | Safe? |
|---------------|-------|--------|-------|
| Full quiesce (recommended) | Stopped | Stopped | **YES** |
| Read-only pause | Stopped | Continuing | **NO** |
| Active | Active | Active | **NO** |

### Recovery from Interrupted Migration

| Interruption Point | Documents | Index | Recovery |
|--------------------|-----------|-------|----------|
| After drop, before quantize | Unchanged | **None** | Re-run apply (or pass `--backup-dir` to resume from backup) |
| During quantization | Partially quantized | **None** | Re-run with same `--backup-dir` to resume from last batch |
| After quantization, before create | Quantized | **None** | Re-run apply (will recreate index) |
| After create | Correct | Rebuilding | Wait for index ready |

The underlying documents are **never deleted** by `drop_recreate` mode. For large quantization jobs, use `--backup-dir` to enable crash-safe recovery. See [Crash-safe resume for quantization](#crash-safe-resume-for-quantization) below.

## Step 4: Apply the Migration

The `apply` command executes the migration. The index will be temporarily unavailable during the drop-recreate process.

```bash
rvl migrate apply \
  --plan migration_plan.yaml \
  --url redis://localhost:6379 \
  --report-out migration_report.yaml \
  --benchmark-out benchmark_report.yaml
```

### What `apply` does

The migration executor follows this sequence:

**STEP 1: Enumerate keys** (before any modifications)
- Discovers all document keys belonging to the source index
- Uses `FT.AGGREGATE WITHCURSOR` for efficient enumeration
- Falls back to `SCAN` if the index has indexing failures
- Keys are stored in memory for quantization or rename operations

**STEP 2: Drop source index**
- Issues `FT.DROPINDEX` to remove the index structure
- **The underlying documents remain in Redis** - only the index metadata is deleted
- After this point, the index is unavailable until step 6 completes

**STEP 3: Quantize vectors** (if changing vector datatype)
- For each document in the enumerated key list:
  - Reads the document (including the old vector)
  - Converts the vector to the new datatype (e.g., float32 → float16)
  - Writes back the converted vector to the same document
- Processes documents in batches of 500 using Redis pipelines
- Skipped for JSON storage (vectors are re-indexed automatically on recreate)
- **Backup support**: For large datasets, use `--backup-dir` to enable crash-safe recovery and rollback

**STEP 4: Key renames** (if changing key prefix)
- If the migration changes the key prefix, renames each key from old prefix to new prefix
- Skipped if no prefix change

**STEP 5: Create target index**
- Issues `FT.CREATE` with the merged target schema
- Redis begins background indexing of existing documents

**STEP 6: Wait for re-indexing**
- Polls `FT.INFO` until indexing completes
- The index becomes available for queries when this completes

**Summary**: The migration preserves all documents, drops only the index structure, performs any document-level transformations (quantization, renames), then recreates the index with the new schema.

### Async execution for large migrations

For large migrations (especially those involving vector quantization), use the `--async` flag:

```bash
rvl migrate apply \
  --plan migration_plan.yaml \
  --async \
  --url redis://localhost:6379
```

**What becomes async:**

- Document enumeration during quantization (uses `FT.AGGREGATE WITHCURSOR` for index-specific enumeration, falling back to SCAN only if indexing failures exist)
- Vector read/write operations (sequential async HGET, batched HSET via pipeline)
- Index readiness polling (uses `asyncio.sleep()` instead of blocking)
- Validation checks

**What stays sync:**

- CLI prompts and user interaction
- YAML file reading/writing
- Progress display

**When to use async:**

- Quantizing millions of vectors (float32 to float16)
- Integrating into an async application

For most migrations (index-only changes, small datasets), sync mode is sufficient and simpler.

See {doc}`/concepts/index-migrations` for detailed async vs sync guidance.

### Crash-safe resume for quantization

When migrating large datasets with vector quantization (e.g. float32 to float16), the re-encoding step can take minutes or hours. If the process is interrupted (crash, network drop, OOM kill), you don't want to start over. The `--backup-dir` flag enables crash-safe recovery.

#### How it works

When you pass `--backup-dir`, the migrator saves original vector bytes to disk before mutating them. Two files are created:

```
<backup-dir>/
  migration_backup_<index_name>.header   # JSON: phase, progress counters, field metadata
  migration_backup_<index_name>.data     # Binary: length-prefixed batches of original vectors
```

The **header file** is a small JSON file that tracks progress through a state machine:

```
dump → ready → active → completed
```

- **dump**: original vectors are being read from Redis and written to the data file, one batch at a time
- **ready**: all original vectors have been backed up; safe to proceed with quantization
- **active**: quantization is in progress; the header tracks which batches have been written back to Redis
- **completed**: all batches have been quantized and the migration finished successfully

The header is atomically updated (temp file + rename) after every batch, so a crash never corrupts it.

The **data file** is append-only binary. Each batch is stored as a 4-byte big-endian length prefix followed by a pickled blob containing the batch's keys and their original vector bytes.

On resume, the executor loads the header, sees how many batches were already quantized (`quantize_completed_batches`), and skips ahead in the data file to continue from the next unfinished batch.

**Disk usage:** approximately `num_docs × dims × bytes_per_element`. For example, 1M docs with 768-dim float32 vectors ≈ 2.9 GB.

#### Step-by-step: using crash-safe resume

**1. Estimate disk space (dry-run, no mutations):**

```bash
rvl migrate estimate --plan migration_plan.yaml
```

Example output:

```text
Pre-migration disk space estimate:
  Index: products_idx (1,000,000 documents)
  Vector field 'embedding': 768 dims, float32 -> float16

  RDB snapshot (BGSAVE):        ~2.87 GB
  AOF growth:                  not estimated (pass aof_enabled=True if AOF is on)
  Total new disk required:      ~2.87 GB

  Post-migration memory savings: ~1.43 GB (50% reduction)
```

If AOF is enabled:

```bash
rvl migrate estimate --plan migration_plan.yaml --aof-enabled
```

**2. Apply with backup enabled:**

```bash
rvl migrate apply \
  --plan migration_plan.yaml \
  --backup-dir /tmp/migration_backups \
  --url redis://localhost:6379 \
  --report-out migration_report.yaml
```

The `--backup-dir` flag takes a directory path. If no backup exists there, a new one is created. If one already exists (from a previous interrupted run), the migrator resumes from where it left off.

**3. If the process crashes or is interrupted:**

The header file will contain the progress:

```json
{
  "index_name": "products_idx",
  "fields": {"embedding": {"source": "float32", "target": "float16", "dims": 768}},
  "batch_size": 500,
  "phase": "active",
  "dump_completed_batches": 2000,
  "quantize_completed_batches": 900
}
```

This tells you: all 2000 batches of original vectors were backed up, and 900 of them have been quantized so far.

**4. Resume the migration:**

Re-run the exact same command:

```bash
rvl migrate apply \
  --plan migration_plan.yaml \
  --backup-dir /tmp/migration_backups \
  --url redis://localhost:6379 \
  --report-out migration_report.yaml
```

The migrator will:
- Detect the existing backup and skip already-quantized batches
- Continue quantizing from batch 901 onward
- Print progress like `Quantize vectors: 450,000/1,000,000 docs`

**5. On successful completion:**

The backup phase is set to `completed`. By default, backup files are **automatically deleted** after a successful migration. Pass `--keep-backup` to retain them for post-migration auditing or potential rollback.

#### Limitations

- **Same-width conversions** (float16 to bfloat16, or int8 to uint8) are **not supported** for resume. These conversions cannot be detected by byte-width inspection, so idempotent skip is impossible.
- **JSON storage** does not need vector re-encoding (Redis re-indexes JSON vectors on `FT.CREATE`). The backup is still created for consistency but no batched writes occur.
- The backup must match the migration plan. If you change the plan, delete the old backup directory and start fresh.

## Step 5: Validate the Result

Validation happens automatically during `apply`, but you can run it separately:

```bash
rvl migrate validate \
  --plan migration_plan.yaml \
  --url redis://localhost:6379 \
  --report-out migration_report.yaml
```

**Validation checks:**
- Live schema matches `merged_target_schema`
- Document count matches the source snapshot
- Sampled keys still exist
- No increase in indexing failures

## What's Supported

| Change | Supported | Notes |
|--------|-----------|-------|
| Add text/tag/numeric/geo field | ✅ | |
| Remove a field | ✅ | |
| Rename a field | ✅ | Renames field in all documents |
| Change key prefix | ✅ | Renames keys via RENAME command |
| Rename the index | ✅ | Index-only |
| Make a field sortable | ✅ | |
| Change field options (separator, stemming) | ✅ | |
| Change vector algorithm (FLAT ↔ HNSW ↔ SVS-VAMANA) | ✅ | Index-only |
| Change distance metric (COSINE ↔ L2 ↔ IP) | ✅ | Index-only |
| Tune HNSW parameters (M, EF_CONSTRUCTION) | ✅ | Index-only |
| Quantize vectors (float32 → float16/bfloat16/int8/uint8) | ✅ | Auto re-encode |

## What's Blocked

| Change | Why | Workaround |
|--------|-----|------------|
| Change vector dimensions | Requires re-embedding | Re-embed with new model, reload data |
| Change storage type (hash ↔ JSON) | Different data format | Export, transform, reload |
| Add a new vector field | Requires vectors for all docs | Add vectors first, then migrate |

## CLI Reference

### Single-Index Commands

| Command | Description |
|---------|-------------|
| `rvl migrate wizard` | Build a migration interactively |
| `rvl migrate plan` | Generate a migration plan |
| `rvl migrate apply` | Execute a migration |
| `rvl migrate estimate` | Estimate disk space for a migration (dry-run) |
| `rvl migrate validate` | Verify a migration result |

### Batch Commands

| Command | Description |
|---------|-------------|
| `rvl migrate batch-plan` | Create a batch migration plan |
| `rvl migrate batch-apply` | Execute a batch migration |
| `rvl migrate batch-resume` | Resume an interrupted batch |
| `rvl migrate batch-status` | Check batch progress |

**Common flags:**
- `--url` : Redis connection URL
- `--index` : Index name to migrate
- `--plan` / `--plan-out` : Path to migration plan
- `--async` : Use async executor for large migrations (apply only)
- `--report-out` : Path for validation report
- `--benchmark-out` : Path for performance metrics

**Apply flags (quantization & reliability):**
- `--backup-dir <dir>` : Directory for vector backup files. Enables crash-safe resume and manual rollback. Required when using `--workers` > 1.
- `--batch-size <N>` : Keys per pipeline batch (default 500). Values 200–1000 are typical.
- `--workers <N>` : Parallel quantization workers (default 1). Each worker opens its own Redis connection. See [Performance](#performance-tuning) for guidance.
- `--keep-backup` : Retain backup files after a successful migration (default: auto-cleanup).

**Batch-specific flags:**
- `--pattern` : Glob pattern to match index names (e.g., `*_idx`)
- `--indexes` : Explicit list of index names
- `--indexes-file` : File containing index names (one per line)
- `--schema-patch` : Path to shared schema patch YAML
- `--state` : Path to batch state file for resume
- `--failure-policy` : `fail_fast` or `continue_on_error`
- `--accept-data-loss` : Required for quantization (lossy changes)
- `--retry-failed` : Retry previously failed indexes on resume

## Troubleshooting

### Migration blocked: "unsupported change"

The planner detected a change that requires data transformation. Check `diff_classification.blocked_reasons` in the plan for details.

### Apply failed: "source schema mismatch"

The live index schema changed since the plan was generated. Re-run `rvl migrate plan` to create a fresh plan.

### Apply failed: "timeout waiting for index ready"

The index is taking longer to rebuild than expected. This can happen with large datasets. Check Redis logs and consider increasing the timeout or running during lower traffic periods.

### Validation failed: "document count mismatch"

Documents were added or removed between plan and apply. This is expected if your application is actively writing. Re-run `plan` and `apply` during a quieter period when the document count is stable, or verify the mismatch is due only to normal application traffic.

### How to recover from a failed migration

If `apply` fails mid-migration:

1. **Check if the index exists:** `rvl index info --index myindex`
2. **If the index exists but is wrong:** Re-run `apply` with the same plan
3. **If the index was dropped:** Recreate it from the plan's `merged_target_schema`

The underlying documents are never deleted by `drop_recreate`.

## Backup, Resume & Rollback

### How Backups Work

When you pass `--backup-dir` (or `backup_dir` in the Python API), the
migration executor saves **original vector bytes** to disk before mutating
them. This enables two key capabilities:

1. **Crash-safe resume** — if the process dies mid-migration, re-running the
   same command with the same `--backup-dir` automatically detects partial
   progress and resumes from the last completed batch.
2. **Manual rollback** — the backup files contain the original (pre-quantization)
   vector values, which can be restored to undo a migration.

Backup files are written to the specified directory with this layout:

```
<backup-dir>/
  migration_backup_<index_name>.header   # JSON: phase, progress counters, field metadata
  migration_backup_<index_name>.data     # Binary: length-prefixed batches of original vectors
```

**Disk usage:** approximately `num_docs × dims × bytes_per_element`.
For example, 1M docs with 768-dim float32 vectors ≈ 2.9 GB.

By default, backup files are **automatically deleted** after a successful
migration. Pass `--keep-backup` to retain them for post-migration auditing
or potential rollback.

### Crash-Safe Resume

If a migration is interrupted (crash, network error, Ctrl+C), simply re-run
the exact same command:

```bash
# Original command that was interrupted
rvl migrate apply --plan plan.yaml --url redis://localhost:6379 \
  --backup-dir /tmp/backups --workers 4

# Just re-run it — progress is resumed automatically
rvl migrate apply --plan plan.yaml --url redis://localhost:6379 \
  --backup-dir /tmp/backups --workers 4
```

The executor detects the existing backup header, reads how many batches were
completed, and resumes from the next unfinished batch. No data is duplicated
or lost.

```{note}
**Single-worker vs multi-worker resume:** In single-worker mode, the full
backup is written *before* the index is dropped, so a crash at any point
leaves a complete backup on disk. In multi-worker mode, dump and quantize
are fused (each worker reads, backs up, and converts its shard in one pass
*after* the index drop). A crash during this fused phase may leave partial
backup shards. Re-running detects and resumes from partial state.
```

### Rollback

If you need to undo a quantization migration and restore original vectors,
use the `rollback` command:

```bash
rvl migrate rollback --backup-dir /tmp/backups --url redis://localhost:6379
```

This reads every batch from the backup files and pipeline-HSETs the original
(pre-quantization) vector bytes back into Redis. After rollback completes:

- Your vector data is restored to its original datatype
- You will need to **manually recreate the original index schema** if the
  index was changed during migration (the rollback command restores data
  only, not the index definition)

```bash
# After rollback, recreate the original index if needed:
rvl index create --schema original_schema.yaml --url redis://localhost:6379
```

```{important}
Rollback requires that backup files were preserved. Either pass
`--keep-backup` during migration, or ensure the backup directory was not
cleaned up. Without backup files, rollback is not possible.
```

### Python API for Rollback

```python
from redisvl.migration.backup import VectorBackup
import redis

r = redis.from_url("redis://localhost:6379")
backup = VectorBackup.load("/tmp/backups/migration_backup_myindex")

for keys, originals in backup.iter_batches():
    pipe = r.pipeline(transaction=False)
    for key in keys:
        if key in originals:
            for field_name, original_bytes in originals[key].items():
                pipe.hset(key, field_name, original_bytes)
    pipe.execute()

print("Rollback complete")
```

## Python API

For programmatic migrations, use the migration classes directly:

### Sync API

```python
from redisvl.migration import MigrationPlanner, MigrationExecutor

planner = MigrationPlanner()
plan = planner.create_plan(
    "myindex",
    redis_url="redis://localhost:6379",
    schema_patch_path="schema_patch.yaml",
)

executor = MigrationExecutor()
report = executor.apply(plan, redis_url="redis://localhost:6379")
print(f"Migration result: {report.result}")
```

With backup and multi-worker quantization:

```python
report = executor.apply(
    plan,
    redis_url="redis://localhost:6379",
    backup_dir="/tmp/migration_backups",   # enables crash-safe resume
    batch_size=500,                        # keys per pipeline batch
    num_workers=4,                         # parallel quantization workers
    keep_backup=True,                      # retain backups for rollback
)
print(f"Quantized in {report.timings.quantize_duration_seconds}s")
```

### Async API

```python
import asyncio
from redisvl.migration import AsyncMigrationPlanner, AsyncMigrationExecutor

async def migrate():
    planner = AsyncMigrationPlanner()
    plan = await planner.create_plan(
        "myindex",
        redis_url="redis://localhost:6379",
        schema_patch_path="schema_patch.yaml",
    )

    executor = AsyncMigrationExecutor()
    report = await executor.apply(
        plan,
        redis_url="redis://localhost:6379",
        backup_dir="/tmp/migration_backups",
        num_workers=4,
    )
    print(f"Migration result: {report.result}")

asyncio.run(migrate())
```

## Batch Migration

When you need to apply the same schema change to multiple indexes, use batch migration. This is common for:

- Quantizing all indexes from float32 → float16
- Standardizing vector algorithms across indexes
- Coordinated migrations during maintenance windows

### Quick Start: Batch Migration

```bash
# 1. Create a shared patch (applies to any index with an 'embedding' field)
cat > quantize_patch.yaml << 'EOF'
version: 1
changes:
  update_fields:
    - name: embedding
      attrs:
        datatype: float16
EOF

# 2. Create a batch plan for all indexes matching a pattern
rvl migrate batch-plan \
  --pattern "*_idx" \
  --schema-patch quantize_patch.yaml \
  --plan-out batch_plan.yaml \
  --url redis://localhost:6379

# 3. Apply the batch plan
rvl migrate batch-apply \
  --plan batch_plan.yaml \
  --accept-data-loss \
  --url redis://localhost:6379

# 4. Check status
rvl migrate batch-status --state batch_state.yaml
```

### Batch Plan Options

**Select indexes by pattern:**
```bash
rvl migrate batch-plan \
  --pattern "*_idx" \
  --schema-patch quantize_patch.yaml \
  --plan-out batch_plan.yaml \
  --url redis://localhost:6379
```

**Select indexes by explicit list:**
```bash
rvl migrate batch-plan \
  --indexes "products_idx,users_idx,orders_idx" \
  --schema-patch quantize_patch.yaml \
  --plan-out batch_plan.yaml \
  --url redis://localhost:6379
```

**Select indexes from a file (for 100+ indexes):**
```bash
# Create index list file
echo -e "products_idx\nusers_idx\norders_idx" > indexes.txt

rvl migrate batch-plan \
  --indexes-file indexes.txt \
  --schema-patch quantize_patch.yaml \
  --plan-out batch_plan.yaml \
  --url redis://localhost:6379
```

### Batch Plan Review

The generated `batch_plan.yaml` shows which indexes will be migrated:

```yaml
version: 1
batch_id: "batch_20260320_100000"
mode: drop_recreate
failure_policy: fail_fast
requires_quantization: true

shared_patch:
  version: 1
  changes:
    update_fields:
      - name: embedding
        attrs:
          datatype: float16

indexes:
  - name: products_idx
    applicable: true
    skip_reason: null
  - name: users_idx
    applicable: true
    skip_reason: null
  - name: legacy_idx
    applicable: false
    skip_reason: "Field 'embedding' not found"

created_at: "2026-03-20T10:00:00Z"
```

**Key fields:**
- `applicable: true` means the patch applies to this index
- `skip_reason` explains why an index will be skipped

### Applying a Batch Plan

```bash
# Apply with fail-fast (default: stop on first error)
rvl migrate batch-apply \
  --plan batch_plan.yaml \
  --accept-data-loss \
  --url redis://localhost:6379

# Apply with continue-on-error (set at batch-plan time)
# Note: failure_policy is set during batch-plan, not batch-apply
rvl migrate batch-plan \
  --pattern "*_idx" \
  --schema-patch quantize_patch.yaml \
  --failure-policy continue_on_error \
  --plan-out batch_plan.yaml \
  --url redis://localhost:6379

rvl migrate batch-apply \
  --plan batch_plan.yaml \
  --accept-data-loss \
  --url redis://localhost:6379
```

**Flags for batch-apply:**
- `--accept-data-loss` : Required when quantizing vectors (float32 → float16 is lossy)
- `--state` : Path to batch state file (default: `batch_state.yaml`)
- `--report-dir` : Directory for per-index reports (default: `./reports/`)

**Note:** `--failure-policy` is set during `batch-plan`, not `batch-apply`. The policy is stored in the batch plan file.

### Resume After Failure

Batch migration automatically tracks progress in the state file. If interrupted:

```bash
# Resume from where it left off
rvl migrate batch-resume \
  --state batch_state.yaml \
  --accept-data-loss \
  --url redis://localhost:6379

# Retry previously failed indexes
rvl migrate batch-resume \
  --state batch_state.yaml \
  --retry-failed \
  --accept-data-loss \
  --url redis://localhost:6379
```

**Note:** If the batch plan involves quantization (e.g., `float32` → `float16`), you must pass `--accept-data-loss` to `batch-resume`, just as with `batch-apply`. Omit `--accept-data-loss` if the batch plan does not involve quantization.

### Checking Batch Status

```bash
rvl migrate batch-status --state batch_state.yaml
```

**Example output:**
```
Batch Migration Status
======================
Batch ID: batch_20260320_100000
Started: 2026-03-20T10:00:00Z
Updated: 2026-03-20T10:25:00Z

Completed: 2
  - products_idx: success (10:02:30)
  - users_idx: failed - Redis connection timeout (10:05:45)

In Progress: inventory_idx
Remaining: 1 (analytics_idx)
```

### Batch Report

After completion, a `batch_report.yaml` is generated:

```yaml
version: 1
batch_id: "batch_20260320_100000"
status: completed  # or partial_failure, failed
summary:
  total_indexes: 3
  successful: 3
  failed: 0
  skipped: 0
  total_duration_seconds: 127.5
indexes:
  - name: products_idx
    status: success
    report_path: ./reports/products_idx_report.yaml
  - name: users_idx
    status: success
    report_path: ./reports/users_idx_report.yaml
  - name: orders_idx
    status: success
    report_path: ./reports/orders_idx_report.yaml
completed_at: "2026-03-20T10:02:07Z"
```

### Python API for Batch Migration

```python
from redisvl.migration import BatchMigrationPlanner, BatchMigrationExecutor

# Create batch plan
planner = BatchMigrationPlanner()
batch_plan = planner.create_batch_plan(
    redis_url="redis://localhost:6379",
    pattern="*_idx",
    schema_patch_path="quantize_patch.yaml",
)

# Review applicability
for idx in batch_plan.indexes:
    if idx.applicable:
        print(f"Will migrate: {idx.name}")
    else:
        print(f"Skipping {idx.name}: {idx.skip_reason}")

# Execute batch
executor = BatchMigrationExecutor()
report = executor.apply(
    batch_plan,
    redis_url="redis://localhost:6379",
    state_path="batch_state.yaml",
    report_dir="./reports/",
    progress_callback=lambda name, pos, total, status: print(f"[{pos}/{total}] {name}: {status}"),
)

print(f"Batch status: {report.status}")
print(f"Successful: {report.summary.successful}/{report.summary.total_indexes}")
```

### Batch Migration Tips

1. **Test on a single index first**: Run a single-index migration to verify the patch works before applying to a batch.

2. **Use `continue_on_error` for large batches**: This ensures one failure doesn't block all remaining indexes.

3. **Schedule during low-traffic periods**: Each index has downtime during migration.

4. **Review skipped indexes**: The `skip_reason` often indicates schema differences that need attention.

5. **Keep state files**: The `batch_state.yaml` is essential for resume. Don't delete it until the batch completes successfully.

## Performance Tuning

### Quantization Throughput

Vector quantization (e.g. float32 → float16) is the most time-consuming
phase of a datatype migration. Observed throughput on a local Redis instance:

| Workers | Dims | Throughput | Notes |
|---------|------|------------|-------|
| 1       | 256  | ~70K docs/sec | Single worker is fastest for low dims |
| 4       | 256  | ~62K docs/sec | Worker overhead exceeds parallelism benefit |
| 1       | 1536 | ~15K docs/sec | Higher dims = more conversion work |
| 4       | 1536 | ~15K docs/sec | I/O-bound; Redis is the bottleneck |

**Guidance:**
- For **low-dimensional vectors** (≤ 256 dims), use `--workers 1` (the default). Per-vector conversion is so cheap that process-spawning and extra-connection overhead outweigh the parallelism benefit.
- For **high-dimensional vectors** (≥ 768 dims), `--workers 2-4` may help if the Redis server has available CPU headroom. Diminishing returns above 4–8 workers on a single Redis instance because Redis command processing is single-threaded.
- The main bottleneck for large migrations is typically **index rebuild time** (the `FT.CREATE` background indexing after vectors are written), not quantization itself.

### Batch Size

The `--batch-size` flag controls how many keys are read/written per Redis
pipeline round-trip. The default of 500 is a good balance. Larger batches
(1000+) reduce round-trips but increase per-batch memory and latency.

### Backup Disk Space

When `--backup-dir` is provided, original vectors are saved to disk before
mutation. Approximate size: `num_docs × dims × bytes_per_element`.

| Docs   | Dims | Source dtype | Backup size |
|--------|------|-------------|-------------|
| 100K   | 768  | float32     | ~292 MB     |
| 1M     | 768  | float32     | ~2.9 GB     |
| 1M     | 1536 | float32     | ~5.7 GB     |

### HNSW vs FLAT Index Capacity

```{note}
When migrating from **HNSW** to **FLAT**, the target index may report a
*higher* document count than the source. This is not a bug — it reflects
a fundamental difference in how the two algorithms store vectors.

HNSW maintains a navigable small-world graph with per-node neighbor lists.
This graph overhead limits how many vectors can fit in available memory.
FLAT stores vectors as a simple array with no graph overhead.

If the source HNSW index was operating near its memory capacity, some
documents may have been registered in Redis Search's document table but
not fully indexed into the HNSW graph. After migration to FLAT, those
same documents become fully searchable because FLAT requires less memory
per vector.

The migration validator compares the total key count
(`num_docs + hash_indexing_failures`) between source and target, so this
scenario is handled correctly in the general case.
```

## Learn more

- {doc}`/concepts/index-migrations`: How migrations work and which changes are supported
