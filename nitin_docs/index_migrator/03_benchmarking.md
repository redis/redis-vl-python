# Migration Benchmarking

## Goals

Migration benchmarking exists to answer practical operator questions:

- how long will the migration take
- how long will search be degraded or unavailable
- how much shared Redis capacity will the migration consume
- how much the target schema or vector shape will increase or reduce memory usage
- how much query performance changes during the migration window
- whether future migrations can be estimated from previous runs

The first benchmarking design should stay simple. It should collect structured measurements from real runs and manual rehearsals rather than introducing a separate performance framework before the migrator exists.

## Core Benchmark Questions

Every migration benchmark should answer:

1. How long did planning take?
2. How long did `apply` take end-to-end?
3. How long was the index unavailable or in degraded indexing state?
4. What document throughput did the migration achieve?
5. What query latency and error-rate changes occurred during the migration?
6. How much memory, flash, or disk footprint changed before, during, and after migration?
7. How accurate was the peak-overlap estimate?
8. Did the final migrated index match the expected schema and document count?

## Metrics

### Timing Metrics

- `plan_duration_seconds`
- `apply_duration_seconds`
- `validation_duration_seconds`
- `total_migration_duration_seconds`
- `drop_duration_seconds`
- `recreate_duration_seconds`
- `initial_indexing_duration_seconds`
- `downtime_duration_seconds` for `drop_recreate`
- `shadow_overlap_duration_seconds` for `iterative_shadow`
- `transform_duration_seconds` for payload rewrite work
- `backfill_duration_seconds` for target payload creation

### Throughput Metrics

- `source_num_docs`
- `target_num_docs`
- `documents_indexed_per_second`
- `documents_transformed_per_second`
- `bytes_rewritten_per_second`
- `progress_samples` captured during readiness polling

### Query Impact Metrics

- baseline query latency: `p50`, `p95`, `p99`
- during-migration query latency: `p50`, `p95`, `p99`
- post-migration query latency: `p50`, `p95`, `p99`
- query error rate during migration
- query result overlap or sample correctness checks

### Resource Impact Metrics

- source document footprint from live stats or sampling
- source index size from live stats
- target document footprint from live stats or sampling
- target index size from live stats
- total source footprint
- total target footprint
- footprint delta after migration
- estimated peak overlap footprint
- actual peak overlap footprint
- indexing failure delta
- memory headroom before migration
- memory headroom after migration
- peak memory headroom during overlap
- flash or disk footprint before and after when relevant
- source vector dimensions, datatype, precision, and algorithm
- target vector dimensions, datatype, precision, and algorithm
- source vector bytes per document
- target vector bytes per document

### Correctness Metrics

- schema match
- document count match
- indexing failure delta equals zero
- representative document fetch checks pass

## Benchmark Inputs

Each benchmark run should record the workload context, not just the raw timings.

Required context:

- migration mode
- dataset size
- storage type
- field mix
- whether vectors are present
- source and target vector configuration when vectors are present
- whether payload shape changes
- shard count
- replica count
- query load level during migration
- environment label such as `local`, `staging`, `redis_cloud`, or `redis_software`

Useful optional context:

- vector dimensions and datatype
- vector precision and algorithm
- auto-tiering enabled or disabled
- representative document size
- maintenance window target

## Benchmark Scenarios

Start with a small scenario matrix and expand only when needed.

Minimum Phase 1 benchmark scenarios:

- small index, low query load
- medium or large index, low query load
- medium or large index, representative read load

Minimum Phase 2 benchmark scenarios:

- one shadow migration on a sharded deployment with sufficient capacity
- one shadow migration that is blocked by the capacity gate
- one shadow migration under representative read load
- one algorithm migration such as `HNSW -> FLAT`
- one vector storage migration such as `FP32 -> FP16`
- one payload-shape-changing migration that requires transform or backfill

Scale variables should be changed one at a time where possible:

- document count
- shard count
- query concurrency
- storage type
- vector footprint

## Benchmark Artifacts

The benchmark workflow should use simple YAML artifacts.

### `benchmark_manifest.yaml`

```yaml
version: 1
label: staging-large-read-load
mode: drop_recreate
environment: staging
dataset:
  num_docs: 1000000
  storage_type: json
  vector_fields_present: true
platform:
  shard_count: 4
  replica_count: 1
workload:
  query_profile: representative-read
  query_check_file: queries.yaml
notes: ""
```

### `benchmark_report.yaml`

```yaml
version: 1
label: staging-large-read-load
mode: drop_recreate
timings:
  total_migration_duration_seconds: 540
  downtime_duration_seconds: 420
  validation_duration_seconds: 18
throughput:
  source_num_docs: 1000000
  documents_indexed_per_second: 2380.95
query_impact:
  baseline_p95_ms: 42
  during_migration_p95_ms: 95
  post_migration_p95_ms: 44
resource_impact:
  source_doc_footprint_mb: 6144
  source_index_size_mb: 8192
  target_doc_footprint_mb: 6144
  target_index_size_mb: 6144
  total_footprint_delta_mb: -2048
  estimated_peak_overlap_footprint_mb: 20480
  actual_peak_overlap_footprint_mb: 19840
  source_vector:
    dimensions: 1536
    datatype: float32
    algorithm: hnsw
  target_vector:
    dimensions: 1536
    datatype: float16
    algorithm: flat
correctness:
  schema_match: true
  doc_count_match: true
```

These artifacts are planning and validation aids. They should not become a separate system before the migrator itself is implemented.

## How Benchmarking Fits the Phases

### Phase 1: `drop_recreate`

Phase 1 should always record:

- start time
- end time
- index downtime duration
- readiness wait duration
- source and target document counts
- source and target index stats
- observed source-versus-target index footprint delta

Phase 1 should optionally record:

- representative query latency before, during, and after migration
- query correctness checks using the same file as validation queries

### Phase 2: `iterative_shadow`

Phase 2 should always record:

- source-to-shadow overlap duration
- planner estimate versus actual runtime
- capacity gate decision
- source and target document and index stats
- estimated versus actual peak overlap footprint
- observed memory savings or growth after the migration
- query impact during overlap

Phase 2 should use benchmark history as advisory input for ETA and risk reporting, not as a hard execution dependency.

## Exit Criteria

Benchmarking is good enough for the first implementation when:

- every migration report includes core timing and correctness metrics
- every shadow migration benchmark includes source-versus-target footprint deltas
- manual benchmark rehearsals can be run from a simple manifest
- the docs define what to collect before performance tuning begins
- benchmark requirements do not force a separate subsystem before the migrator ships
