---
myst:
  html_meta:
    "description lang=en": |
      Learn how RedisVL index migrations work and which schema changes are supported.
---

# Index Migrations

Redis Search indexes are immutable. To change an index schema, you must drop the existing index and create a new one. RedisVL provides a migration workflow that automates this process while preserving your data.

This page explains how migrations work and which changes are supported. For step by step instructions, see the [migration guide](../user_guide/how_to_guides/migrate-indexes.md).

## Supported and blocked changes

The migrator classifies schema changes into two categories:

| Change | Status |
|--------|--------|
| Add or remove a field | Supported |
| Change field options (sortable, separator) | Supported |
| Change vector algorithm (FLAT, HNSW, SVS-VAMANA) | Supported |
| Change distance metric (COSINE, L2, IP) | Supported |
| Tune algorithm parameters (M, EF_CONSTRUCTION) | Supported |
| Quantize vectors (float32 to float16) | Supported |
| Change vector dimensions | Blocked |
| Change key prefix | Blocked |
| Rename a field | Blocked |
| Change storage type (hash to JSON) | Blocked |
| Add a new vector field | Blocked |

**Supported** changes can be applied automatically using `rvl migrate`. The migrator handles the index rebuild and any necessary data transformations.

**Blocked** changes require manual intervention because they involve incompatible data formats or missing data. The migrator will reject these changes and explain why.

## How the migrator works

The migrator uses a plan first workflow:

1. **Plan**: Capture the current schema, classify your changes, and generate a migration plan
2. **Review**: Inspect the plan before making any changes
3. **Apply**: Drop the index, transform data if needed, and recreate with the new schema
4. **Validate**: Verify the result matches expectations

This separation ensures you always know what will happen before any changes are made.

## Migration mode: drop_recreate

The `drop_recreate` mode rebuilds the index in place while preserving your documents.

The process:

1. Drop only the index structure (documents remain in Redis)
2. For datatype changes, re-encode vectors to the target precision
3. Recreate the index with the new schema
4. Wait for Redis to re-index the existing documents
5. Validate the result

**Tradeoff**: The index is unavailable during the rebuild. Review the migration plan carefully before applying.

## Index only vs document dependent changes

Schema changes fall into two categories based on whether they require modifying stored data.

**Index only changes** affect how Redis Search indexes data, not the data itself:

- Algorithm changes: The stored vector bytes are identical. Only the index structure differs.
- Distance metric changes: Same vectors, different similarity calculation.
- Adding or removing fields: The documents already contain the data. The index just starts or stops indexing it.

These changes complete quickly because they only require rebuilding the index.

**Document dependent changes** require modifying the stored data:

- Datatype changes (float32 to float16): Stored vector bytes must be re-encoded.
- Field renames: Stored field names must be updated in every document.
- Dimension changes: Vectors must be re-embedded with a different model.

The migrator handles datatype changes automatically. Other document dependent changes are blocked because they require application level logic or external services.

## Vector quantization

Changing vector precision from float32 to float16 reduces memory usage at the cost of slight precision loss. The migrator handles this automatically by:

1. Reading all vectors from Redis
2. Converting to the target precision
3. Writing updated vectors back
4. Recreating the index with the new schema

Typical reductions:

| Metric | Value |
|--------|-------|
| Index size reduction | ~50% |
| Memory reduction | ~35% |

Quantization time is proportional to document count. Plan for downtime accordingly.

## Why some changes are blocked

### Vector dimension changes

Vector dimensions are determined by your embedding model. A 384 dimensional vector from one model is mathematically incompatible with a 768 dimensional index expecting vectors from a different model. There is no way to resize an embedding.

**Resolution**: Re-embed your documents using the new model and load them into a new index.

### Prefix changes

Changing a prefix from `docs:` to `articles:` requires copying every document to a new key. This operation doubles storage temporarily and can leave orphaned keys if interrupted.

**Resolution**: Create a new index with the new prefix and reload your data.

### Field renames

Field names are stored in the documents themselves as hash field names or JSON keys. Renaming requires iterating through every document and updating the field name.

**Resolution**: Create a new index with the correct field name and reload your data.

### Storage type changes

Hash and JSON have different data layouts. Hash stores flat key value pairs. JSON stores nested structures. Converting between them requires understanding your schema and restructuring each document.

**Resolution**: Export your data, transform it to the new format, and reload into a new index.

### Adding a vector field

Adding a vector field means all existing documents need vectors for that field. The migrator cannot generate these vectors because it does not know which embedding model to use or what content to embed.

**Resolution**: Add vectors to your documents using your application, then run the migration.

## Downtime considerations

With `drop_recreate`, your index is unavailable between the drop and when re-indexing completes.

**CRITICAL**: Downtime requires both reads AND writes to be paused:

| Requirement | Reason |
|-------------|--------|
| **Pause reads** | Index is unavailable during migration |
| **Pause writes** | Redis updates indexes synchronously. Writes during migration may conflict with vector re-encoding or be missed |

Plan for:

- Search unavailability during the migration window
- Partial results while indexing is in progress
- Resource usage from the re-indexing process
- Quantization time if changing vector datatypes

The duration depends on document count, field count, and vector dimensions. For large indexes, consider running migrations during low traffic periods.

## Sync vs async execution

The migrator provides both synchronous and asynchronous execution modes.

### What becomes async and what stays sync

The migration workflow has distinct phases. Here is what each mode affects:

| Phase | Sync mode | Async mode | Notes |
|-------|-----------|------------|-------|
| **Plan generation** | `MigrationPlanner.create_plan()` | `AsyncMigrationPlanner.create_plan()` | Reads index metadata from Redis |
| **Schema snapshot** | Sync Redis calls | Async Redis calls | Single `FT.INFO` command |
| **Enumeration** | FT.AGGREGATE (or SCAN fallback) | FT.AGGREGATE (or SCAN fallback) | Before drop, only if quantization needed |
| **Drop index** | `index.delete()` | `await index.delete()` | Single `FT.DROPINDEX` command |
| **Quantization** | Sequential HGET + HSET | Pipelined HGET + batched HSET | Uses pre-enumerated keys |
| **Create index** | `index.create()` | `await index.create()` | Single `FT.CREATE` command |
| **Readiness polling** | `time.sleep()` loop | `asyncio.sleep()` loop | Polls `FT.INFO` until indexed |
| **Validation** | Sync Redis calls | Async Redis calls | Schema and doc count checks |
| **CLI interaction** | Always sync | Always sync | User prompts, file I/O |
| **YAML read/write** | Always sync | Always sync | Local filesystem only |

### When to use sync (default)

Sync execution is simpler and sufficient for most migrations:

- Small to medium indexes (under 100K documents)
- Index-only changes (algorithm, distance metric, field options)
- Interactive CLI usage where blocking is acceptable

For migrations without quantization, the Redis operations are fast single commands. Sync mode adds no meaningful overhead.

### When to use async

Async execution (`--async` flag) provides benefits in specific scenarios:

**Large quantization jobs (1M+ vectors)**

Converting float32 to float16 requires reading every vector, converting it, and writing it back. The async executor:

- Enumerates documents using `FT.AGGREGATE WITHCURSOR` for index-specific enumeration (falls back to `SCAN` only if indexing failures exist)
- Pipelines `HSET` operations in batches (100-1000 operations per pipeline is optimal for Redis)
- Yields to the event loop between batches so other tasks can proceed

**Large keyspaces (40M+ keys)**

When your Redis instance has many keys and the index has indexing failures (requiring SCAN fallback), async mode yields between batches.

**Async application integration**

If your application uses asyncio, you can integrate migration directly:

```python
import asyncio
from redisvl.migration import AsyncMigrationPlanner, AsyncMigrationExecutor

async def migrate():
    planner = AsyncMigrationPlanner()
    plan = await planner.create_plan("myindex", redis_url="redis://localhost:6379")

    executor = AsyncMigrationExecutor()
    report = await executor.apply(plan, redis_url="redis://localhost:6379")

asyncio.run(migrate())
```

### Why async helps with quantization

The migrator uses an optimized enumeration strategy:

1. **Index-based enumeration**: Uses `FT.AGGREGATE WITHCURSOR` to enumerate only indexed documents (not the entire keyspace)
2. **Fallback for safety**: If the index has indexing failures (`hash_indexing_failures > 0`), falls back to `SCAN` to ensure completeness
3. **Enumerate before drop**: Captures the document list while the index still exists, then drops and quantizes

This optimization provides 10-1000x speedup for sparse indexes (where only a small fraction of prefix-matching keys are indexed).

**Sync quantization:**
```
enumerate keys (FT.AGGREGATE or SCAN) -> store list
for each batch of 500 keys:
    for each key:
        HGET field (blocks)
        convert array
        pipeline.HSET(field, new_bytes)
    pipeline.execute() (blocks)
```

**Async quantization:**
```
enumerate keys (FT.AGGREGATE or SCAN) -> store list
for each batch of 500 keys:
    for each key:
        await HGET field (yields)
        convert array
        pipeline.HSET(field, new_bytes)
    await pipeline.execute() (yields)
```

Each `await` is a yield point where other coroutines can run. For millions of vectors, this prevents your application from freezing.

### What async does NOT improve

Async execution does not reduce:

- **Total migration time**: Same work, different scheduling
- **Redis server load**: Same commands execute on the server
- **Downtime window**: Index remains unavailable during rebuild
- **Network round trips**: Same number of Redis calls

The benefit is application responsiveness, not faster migration.

## Learn more

- [Migration guide](../user_guide/how_to_guides/migrate-indexes.md): Step by step instructions
- [Search and indexing](search-and-indexing.md): How Redis Search indexes work
