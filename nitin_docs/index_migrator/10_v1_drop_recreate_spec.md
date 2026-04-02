# Phase 1 Spec: `drop_recreate`

> **Status**: Implemented and shipped (PRs #567-#572). This spec has been updated with implementation notes where the shipped code diverged from the original design.

## Goal

Build a simple RedisVL migration workflow that:

- preserves existing documents
- captures the old index configuration before change
- applies only the user-requested schema changes
- generates a plan before any mutation
- supports both guided and scripted use
- explicitly surfaces downtime for the migrated index
- supports vector quantization (e.g., FP32 -> FP16) as an in-place rewrite *(added during implementation)*
- supports field renames, prefix changes, and index renames *(added during implementation)*
- supports async execution for large migrations *(added during implementation)*
- supports batch migration across multiple indexes *(added during implementation)*

## Supported Changes

The MVP supports schema changes that can be satisfied by rebuilding the index over the existing document set without rewriting or relocating stored documents.

Supported categories:

- add a new non-vector field that indexes data already present in stored documents
- remove an existing field from the index definition
- change index options on an existing non-vector field when the field name, field type, and storage path stay the same
- change index-level options that only affect index definition and do not relocate data

Supported field types for MVP changes:

- text
- tag
- numeric
- geo

The MVP always recreates the same logical index name unless the user is only generating a plan.

## Blocked Changes

The following changes are classified as unsupported and stop before `apply`:

- key separator changes
- storage type changes (hash <-> JSON)
- JSON path remodels
- vector dimension changes
- any change that requires a completely new stored payload shape

> **Implementation note**: Several items originally blocked in this spec were implemented during Phase 1:
> - ~~key prefix changes~~ - now supported via `index.prefix` in schema patch
> - ~~field renames~~ - now supported via `rename_fields` in schema patch
> - ~~vector datatype changes~~ - now supported as in-place quantization (e.g., FP32 -> FP16)
> - ~~new index name~~ - now supported via `index.name` in schema patch
>
> These were feasible because they could be done as in-place document rewrites without shadow indexes.

## Inputs

The workflow accepts:

- Redis connection parameters
- source index name
- one of:
  - `schema_patch.yaml`
  - `target_schema.yaml`
  - interactive wizard answers

Actual CLI surface (as shipped):

```text
rvl migrate helper
rvl migrate list
rvl migrate wizard --index <name> --plan-out <migration_plan.yaml>
rvl migrate plan --index <name> --schema-patch <patch.yaml>
rvl migrate plan --index <name> --target-schema <schema.yaml>
rvl migrate apply --plan <migration_plan.yaml> [--async] [--resume <checkpoint.yaml>]
rvl migrate estimate --plan <migration_plan.yaml>
rvl migrate validate --plan <migration_plan.yaml>
rvl migrate batch-plan --schema-patch <patch.yaml> (--pattern <glob> | --indexes <list>)
rvl migrate batch-apply --plan <batch_plan.yaml> [--accept-data-loss]
rvl migrate batch-resume --state <batch_state.yaml> [--retry-failed]
rvl migrate batch-status --state <batch_state.yaml>
```

> **Implementation note**: The `--allow-downtime` flag was removed. Downtime is implicit in `drop_recreate` mode. The `--accept-data-loss` flag is used only for quantization (lossy operation). The `helper` and `list` subcommands were added for discoverability. The `estimate` subcommand provides pre-migration disk space estimates.

Key optional flags:

- `--plan-out` / `--report-out` / `--benchmark-out`
- `--key-sample-limit`
- `--query-check-file`
- `--async` (for large migrations with quantization)
- `--resume` (crash-safe checkpoint resume)
- `--accept-data-loss` (for quantization acknowledgment)

### `schema_patch.yaml`

This is the authoritative input model for requested changes. Unspecified source configuration is preserved by default.

Example:

```yaml
version: 1
changes:
  add_fields:
    - name: category
      type: tag
      path: $.category
      separator: ","
  remove_fields:
    - legacy_score
  update_fields:
    - name: title
      options:
        sortable: true
```

### `target_schema.yaml`

This is a convenience input. The planner normalizes it into a schema patch by diffing it against the live source schema.

## Outputs

The workflow produces:

- `migration_plan.yaml`
- `migration_report.yaml`
- optional `benchmark_report.yaml`
- console summaries for plan, apply, and validate

### `migration_plan.yaml`

Required fields:

```yaml
version: 1
mode: drop_recreate
source:
  index_name: docs
  schema_snapshot: {}
  stats_snapshot: {}
  keyspace:
    storage_type: json
    prefixes: ["docs"]
    key_separator: ":"
    key_sample: ["docs:1", "docs:2"]
requested_changes: {}
merged_target_schema: {}
diff_classification:
  supported: true
  blocked_reasons: []
warnings:
  - index downtime is required
validation:
  require_doc_count_match: true
  require_schema_match: true
```

### `migration_report.yaml`

Required fields:

```yaml
version: 1
mode: drop_recreate
source_index: docs
result: succeeded
started_at: 2026-03-17T00:00:00Z
finished_at: 2026-03-17T00:05:00Z
timings:
  total_migration_duration_seconds: 300
  drop_duration_seconds: 3
  recreate_duration_seconds: 12
  initial_indexing_duration_seconds: 270
  validation_duration_seconds: 15
  downtime_duration_seconds: 285
validation:
  schema_match: true
  doc_count_match: true
  indexing_failures_delta: 0
  query_checks: []
benchmark_summary:
  documents_indexed_per_second: 3703.7
  source_index_size_mb: 2048
  target_index_size_mb: 1984
  index_size_delta_mb: -64
  baseline_query_p95_ms: 42
  during_migration_query_p95_ms: 90
  post_migration_query_p95_ms: 44
manual_actions: []
```

## CLI UX

### `plan`

- Capture the source snapshot from the live index.
- Normalize requested changes.
- Classify the diff as supported or blocked.
- Emit `migration_plan.yaml`.
- Print a short risk summary that includes downtime.

### `wizard`

- Read the live source schema first.
- Walk the user through supported change categories only.
- Reject unsupported requests during the wizard instead of silently converting them.
- Explain when a blocked request belongs to a future `iterative_shadow` migration.
- Emit the same `migration_plan.yaml` shape as `plan`.

### `apply`

- Accept only `migration_plan.yaml` as input.
- Refuse to run if the plan contains blocked reasons.
- Refuse to run if the current live schema no longer matches the saved source snapshot.
- Require `--accept-data-loss` when quantization is involved *(replaces original `--allow-downtime`)*.
- Support `--async` for large migrations with quantization.
- Support `--resume` for crash-safe checkpoint recovery.

### `validate`

- Re-run validation checks from the plan against the current live index.
- Emit `migration_report.yaml`.
- Emit `benchmark_report.yaml` when benchmark fields were collected.

## Execution Flow (as implemented)

1. Snapshot source state.
   - Load the live index schema using existing RedisVL introspection.
   - Capture live stats from index info.
   - Record storage type, prefixes, key separator, and a bounded key sample.
2. Normalize requested changes.
   - If the input is `target_schema.yaml`, diff it against the source schema and convert it to a patch.
   - If the input is wizard answers, convert them to the same patch model.
3. Merge and classify.
   - Apply only requested changes to the source schema.
   - Classify each diff as supported or blocked.
   - Detect rename operations (index name, prefix, field names).
   - Detect vector quantization operations.
   - Stop if any blocked diff exists.
4. Generate the plan.
   - Save source snapshot, requested changes, merged target schema, rename operations, validation policy, and warnings.
5. Apply the migration.
   - Confirm current live schema still matches the source snapshot.
   - **Quantize vectors** in-place if quantization is requested (crash-safe with checkpointing).
   - **Rename hash fields** if field renames are requested.
   - **Rename keys** if prefix change is requested.
   - Drop only the index structure.
   - Recreate the index (possibly with new name) using the merged target schema.
6. Wait for indexing completion.
   - Poll live index info until `indexing` is false and `percent_indexed` is complete.
   - Stop with timeout rather than waiting forever.
7. Validate.
   - Compare live schema to merged target schema.
   - Compare live doc count to source doc count.
   - Check indexing failure delta.
   - Check key sample existence.
   - Run optional query checks.
8. Emit the report with timings, validation, and benchmark summary.
9. Optionally emit separate benchmark report.

## Validation

Required validation checks:

- exact schema match against `merged_target_schema`
- live doc count equals source `num_docs`
- `hash_indexing_failures` does not increase
- key sample records still exist

Optional validation checks:

- query checks loaded from `--query-check-file`
- bounded sample fetch checks for representative document ids

Benchmark fields that should be collected during Phase 1:

- migration start and end timestamps
- index downtime duration
- readiness polling duration
- source and target document counts
- documents indexed per second
- source and target index footprint
- observed index footprint delta after recreate
- optional representative query latency before, during, and after migration

Validation is a hard failure for `apply`.

## Failure Handling

The implementation fails closed.

- Unsupported diff: stop at `plan`.
- Source snapshot mismatch at apply time: stop and ask the operator to regenerate the plan.
- Drop succeeds but recreate fails: documents remain; emit a failure report and a manual recovery command using the saved merged schema.
- Validation fails after recreate: leave the recreated index in place, emit a failure report, and stop.
- Interrupted quantization run: crash-safe checkpointing allows resume via `--resume <checkpoint.yaml>` *(added during implementation, replacing original "no checkpointing" stance)*.
- Pipeline errors during batch field renames or key renames: re-raised with context.

The implementation does not implement automatic rollback.

## Operational Guidance

This mode is downtime-accepting by design.

Engineers should assume:

- the index is unavailable between drop and recreated index readiness
- search quality can be degraded while initial indexing completes
- large indexes can place measurable pressure on shared Redis resources
- off-peak execution is preferred
- application-level maintenance handling is outside RedisVL
- blocked vector and payload-shape changes should be rerouted to Phase 2 planning instead of being forced into this path

Default key capture is intentionally small:

- keyspace definition is always recorded
- a bounded key sample is recorded
- a full key manifest is not part of the default MVP path

Benchmarking for Phase 1 should stay simple:

- capture timing and correctness metrics in structured reports
- support manual benchmark rehearsals using [03_benchmarking.md](./03_benchmarking.md)
- avoid introducing a dedicated benchmarking subsystem before the migrator exists
