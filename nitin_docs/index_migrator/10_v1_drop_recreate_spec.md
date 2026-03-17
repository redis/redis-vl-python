# Phase 1 Spec: `drop_recreate`

## Goal

Build a simple RedisVL migration workflow that:

- preserves existing documents
- captures the old index configuration before change
- applies only the user-requested schema changes
- generates a plan before any mutation
- supports both guided and scripted use
- explicitly accepts downtime for the migrated index

This phase is intentionally smaller than the full product goal. Vector datatype, precision, dimension, algorithm, and payload-shape-changing migrations are still in scope for the overall initiative, but they are deferred to `iterative_shadow`.

This is the only implementation target after the docs land.

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

The CLI must classify the following changes as unsupported in the MVP and stop before `apply`:

- key prefix changes
- key separator changes
- storage type changes
- JSON path remodels
- field renames
- vector dimension changes
- vector datatype changes
- vector precision changes
- any vector field algorithm change that depends on different stored payload shape
- any change that requires document rewrite or relocation
- any change that requires a new index name as part of the execution path

These changes should be reported as candidates for the Phase 2 `iterative_shadow` path rather than presented as unsupported forever.

## Inputs

The workflow accepts:

- Redis connection parameters
- source index name
- one of:
  - `schema_patch.yaml`
  - `target_schema.yaml`
  - interactive wizard answers

Recommended CLI surface:

```text
rvl migrate plan --index <name> --schema-patch <patch.yaml>
rvl migrate plan --index <name> --target-schema <schema.yaml>
rvl migrate wizard --index <name> --plan-out <migration_plan.yaml>
rvl migrate apply --plan <migration_plan.yaml> --allow-downtime
rvl migrate validate --plan <migration_plan.yaml>
```

Default optional flags:

- `--plan-out`
- `--report-out`
- `--key-sample-limit`
- `--query-check-file`
- `--non-interactive`

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
- Require `--allow-downtime`.
- Refuse to run if the plan contains blocked reasons.
- Refuse to run if the current live schema no longer matches the saved source snapshot.

### `validate`

- Re-run validation checks from the plan against the current live index.
- Emit `migration_report.yaml`.
- Emit `benchmark_report.yaml` when benchmark fields were collected.

## Execution Flow

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
   - Stop if any blocked diff exists.
4. Generate the plan.
   - Save source snapshot, requested changes, merged target schema, validation policy, and warnings.
5. Apply the migration.
   - Confirm current live schema still matches the source snapshot.
   - Drop only the index structure.
   - Recreate the same index name using the merged target schema.
6. Wait for indexing completion.
   - Poll live index info until `indexing` is false and `percent_indexed` is complete when those fields are available.
   - If those fields are unavailable, poll `num_docs` and readiness twice in a row before continuing.
   - Stop with timeout rather than waiting forever.
7. Validate.
   - Compare live schema to merged target schema.
   - Compare live doc count to source doc count.
   - Check indexing failure delta.
   - Run optional query checks.
8. Emit the report.
9. Emit benchmark artifacts when benchmark data was collected.

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

The MVP fails closed.

- Unsupported diff: stop at `plan`.
- Source snapshot mismatch at apply time: stop and ask the operator to regenerate the plan.
- Drop succeeds but recreate fails: documents remain; emit a failure report and a manual recovery command using the saved merged schema.
- Validation fails after recreate: leave the recreated index in place, emit a failure report, and stop.
- Interrupted run: no checkpointing in MVP. The operator reruns `plan` or reuses the existing plan after confirming the live source state is still compatible.

The MVP does not implement automatic rollback.

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
