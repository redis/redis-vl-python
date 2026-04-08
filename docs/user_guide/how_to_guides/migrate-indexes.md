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
rvl migrate list --url redis://localhost:6379

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

## Step 1: Discover Available Indexes

```bash
rvl migrate list --url redis://localhost:6379
```

**Example output:**
```
Available indexes:
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
  mode: drop_recreate
  warnings:
    - "Index will be unavailable during migration"
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

warnings: []
```

**Key fields to check:**
- `diff_classification.supported` - Must be `true` to proceed
- `diff_classification.blocked_reasons` - Must be empty
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
| After drop, before quantize | Unchanged | **None** | Re-run apply |
| After quantization, before create | Quantized | **None** | Manual FT.CREATE or re-run apply |
| After create | Correct | Rebuilding | Wait for index ready |

The underlying documents are **never deleted** by `drop_recreate` mode.

## Step 4: Apply the Migration

The `apply` command executes the migration. The index will be temporarily unavailable during the drop-recreate process.

```bash
rvl migrate apply \
  --plan migration_plan.yaml \
  --url redis://localhost:6379 \
  --report-out migration_report.yaml \
  --benchmark-out benchmark_report.yaml
```

What `apply` does:

1. checks that the live source schema still matches the saved source snapshot
2. drops only the index structure
3. preserves the existing documents
4. recreates the same index name with the merged target schema
5. waits for indexing readiness
6. validates the result
7. writes report artifacts

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
- Vector read/write operations (pipelined HGET/HSET)
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
| `rvl migrate list` | List all indexes |
| `rvl migrate wizard` | Build a migration interactively |
| `rvl migrate plan` | Generate a migration plan |
| `rvl migrate apply` | Execute a migration |
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

**Batch-specific flags:**
- `--pattern` : Glob pattern to match index names (e.g., `*_idx`)
- `--indexes` : Explicit list of index names
- `--indexes-file` : File containing index names (one per line)
- `--schema-patch` : Path to shared schema patch YAML
- `--state` : Path to checkpoint state file
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
    report = await executor.apply(plan, redis_url="redis://localhost:6379")
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
- `--state` : Path to checkpoint file (default: `batch_state.yaml`)
- `--report-dir` : Directory for per-index reports (default: `./reports/`)

**Note:** `--failure-policy` is set during `batch-plan`, not `batch-apply`. The policy is stored in the batch plan file.

### Resume After Failure

Batch migration automatically checkpoints progress. If interrupted:

```bash
# Resume from where it left off
rvl migrate batch-resume \
  --state batch_state.yaml \
  --url redis://localhost:6379

# Retry previously failed indexes
rvl migrate batch-resume \
  --state batch_state.yaml \
  --retry-failed \
  --url redis://localhost:6379
```

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
  - products_idx: succeeded (10:02:30)
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
    status: succeeded
    duration_seconds: 45.2
    docs_migrated: 15000
    report_path: ./reports/products_idx_report.yaml
  - name: users_idx
    status: succeeded
    duration_seconds: 38.1
    docs_migrated: 8500
  - name: orders_idx
    status: succeeded
    duration_seconds: 44.2
    docs_migrated: 22000
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

5. **Keep checkpoint files**: The `batch_state.yaml` is essential for resume. Don't delete it until the batch completes successfully.

## Learn more

- {doc}`/concepts/index-migrations`: How migrations work and which changes are supported
