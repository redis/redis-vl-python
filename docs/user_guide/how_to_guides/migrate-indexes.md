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
rvl migrate apply --plan migration_plan.yaml --allow-downtime --url redis://localhost:6379

# 4. Verify the result
rvl migrate validate --plan migration_plan.yaml --url redis://localhost:6379
```

## Prerequisites

- Redis with the Search module (Redis Stack, Redis Cloud, or Redis Enterprise)
- An existing index to migrate
- `redisvl` installed (`pip install redisvl`)

```bash
# Local development with Redis Stack
docker run -d --name redis -p 6379:6379 redis/redis-stack-server:latest
```

## Step 1: Discover Available Indexes

```bash
rvl migrate helper --url redis://localhost:6379
rvl migrate list --url redis://localhost:6379
```

**Example output:**
```
Index Migrator
==============
The migrator helps you safely change your index schema.

Supported changes:
  - Add, remove, or update text/tag/numeric/geo fields
  - Change vector algorithm (FLAT, HNSW, SVS-VAMANA)
  - Change distance metric (COSINE, L2, IP)
  - Quantize vectors (float32 → float16)

Commands:
  rvl migrate list      List all indexes
  rvl migrate wizard    Build a migration interactively
  rvl migrate plan      Generate a migration plan
  rvl migrate apply     Execute a migration
  rvl migrate validate  Verify a migration
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
index_name: myindex
migration_mode: drop_recreate

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
          algorithm: HNSW
          datatype: float32
  doc_count: 10000
  key_sample:
    - "doc:1"
    - "doc:2"
    - "doc:3"

diff_classification:
  supported: true
  mode: drop_recreate
  warnings:
    - "Index will be unavailable during migration"
  blocked_reasons: []

changes:
  add_fields:
    - name: category
      type: tag

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
        algorithm: HNSW
        datatype: float32
```

**Key fields to check:**
- `diff_classification.supported` - Must be `true` to proceed
- `diff_classification.blocked_reasons` - Must be empty
- `merged_target_schema` - The final schema after migration

## Step 4: Apply the Migration

The `apply` command requires `--allow-downtime` since the index will be temporarily unavailable.

```bash
rvl migrate apply \
  --plan migration_plan.yaml \
  --allow-downtime \
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
  --allow-downtime \
  --async \
  --url redis://localhost:6379
```

**What becomes async:**

- Keyspace SCAN during quantization (yields between batches of 500 keys)
- Vector read/write operations (pipelined HGET/HSET)
- Index readiness polling (uses `asyncio.sleep()` instead of blocking)
- Validation checks

**What stays sync:**

- CLI prompts and user interaction
- YAML file reading/writing
- Progress display

**When to use async:**

- Quantizing millions of vectors (float32 to float16)
- Redis instance has 40M+ keys
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
| Make a field sortable | ✅ | |
| Change field options (separator, stemming) | ✅ | |
| Change vector algorithm (FLAT ↔ HNSW ↔ SVS-VAMANA) | ✅ | Index-only |
| Change distance metric (COSINE ↔ L2 ↔ IP) | ✅ | Index-only |
| Tune HNSW parameters (M, EF_CONSTRUCTION) | ✅ | Index-only |
| Quantize vectors (float32 → float16) | ✅ | Auto re-encode |

## What's Blocked

| Change | Why | Workaround |
|--------|-----|------------|
| Change vector dimensions | Requires re-embedding | Re-embed with new model, reload data |
| Change prefix/keyspace | Documents at wrong keys | Create new index, reload data |
| Rename a field | Stored data uses old name | Create new index, reload data |
| Change storage type (hash ↔ JSON) | Different data format | Export, transform, reload |
| Add a new vector field | Requires vectors for all docs | Add vectors first, then migrate |

## CLI Reference

| Command | Description |
|---------|-------------|
| `rvl migrate helper` | Show supported changes and usage tips |
| `rvl migrate list` | List all indexes |
| `rvl migrate wizard` | Build a migration interactively |
| `rvl migrate plan` | Generate a migration plan |
| `rvl migrate apply` | Execute a migration |
| `rvl migrate validate` | Verify a migration result |

**Common flags:**
- `--url` : Redis connection URL
- `--index` : Index name to migrate
- `--plan` / `--plan-out` : Path to migration plan
- `--allow-downtime` : Acknowledge index unavailability (required for apply)
- `--async` : Use async executor for large migrations (apply only)
- `--report-out` : Path for validation report
- `--benchmark-out` : Path for performance metrics

## Troubleshooting

### Migration blocked: "unsupported change"

The planner detected a change that requires data transformation. Check `diff_classification.blocked_reasons` in the plan for details.

### Apply failed: "source schema mismatch"

The live index schema changed since the plan was generated. Re-run `rvl migrate plan` to create a fresh plan.

### Apply failed: "timeout waiting for index ready"

The index is taking longer to rebuild than expected. This can happen with large datasets. Check Redis logs and consider increasing the timeout or running during lower traffic periods.

### Validation failed: "document count mismatch"

Documents were added or removed between plan and apply. This is expected if your application is actively writing. Re-run validation with `--skip-count-check` if acceptable.

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

## Learn more

- {doc}`/concepts/index-migrations`: How migrations work and which changes are supported
