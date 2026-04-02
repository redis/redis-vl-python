# Phase 2 Spec: `iterative_shadow`

## Goal

Add a conservative, capacity-aware shadow migration mode that works one index at a time and reduces disruption without attempting to automate cutover or platform scaling.

This phase exists to support the migration cases that Phase 1 intentionally does not handle safely:

- vector datatype changes such as `FP32 -> FP16`
- vector precision changes
- vector dimension changes
- vector algorithm changes such as `HNSW -> FLAT`
- payload-shape-changing model or algorithm migrations that require new stored fields or a new target keyspace

The first Phase 2 implementation should stay simple in operation even though it handles harder migration shapes:

- one index at a time
- capacity gate before each index
- operator-owned cutover
- no automatic scale-up or scale-down
- no multi-index concurrent shadowing
- explicit transform or backfill plan when the target payload shape changes

## Why It Is Not MVP

This mode is not the MVP because it introduces operational questions that Phase 1 does not need to solve:

- database-level capacity inventory
- target-footprint estimation for old and new document and index shapes
- overlap estimation for old and new payloads
- transform or backfill planning for target payload creation
- operator handoff between validation and cutover
- cleanup sequencing after cutover
- larger-scale manual testing on clustered deployments

Phase 1 should prove the core planning and reporting model first.

## Planner Inputs

The Phase 2 planner takes:

- source index name
- Redis connection settings
- supported schema patch or target schema input
- `platform_inventory.yaml`
- optional `transform_plan.yaml` when the migration requires new target payloads

### `platform_inventory.yaml`

```yaml
version: 1
platform: redis_cloud
database:
  name: customer-a-prod
  total_memory_mb: 131072
  available_memory_mb: 32768
  shard_count: 8
  replica_count: 1
  auto_tiering: false
  notes: ""
policy:
  reserve_percent: 15
```

Required inventory fields:

- platform
- total memory
- available memory
- shard count
- replica count
- reserve policy

Optional inventory fields:

- flash or disk notes
- environment labels
- operator comments
- benchmark history notes

### `transform_plan.yaml`

This file is required when the target schema cannot be built from the current stored payload.

Example:

```yaml
version: 1
target_keyspace:
  storage_type: json
  prefixes: ["docs_v2"]
  key_separator: ":"
transform:
  mode: rewrite
  vector_fields:
    - name: embedding
      source_path: $.embedding
      target_path: $.embedding_v2
      source_dimensions: 1536
      target_dimensions: 1536
      source_datatype: float32
      target_datatype: float16
      source_algorithm: hnsw
      target_algorithm: flat
  payload_changes:
    - source_path: $.body
      target_path: $.body_v2
      strategy: copy
```

The first implementation should keep this model explicit and declarative. The migrator should not guess how to transform payloads.

## Capacity Gate

The first Phase 2 capacity gate should be intentionally conservative.

Planner rules:

1. Compute source document footprint from live stats or bounded sampling.
2. Compute source index footprint from live index stats.
3. Estimate target document footprint.
   - For payload-compatible shadowing, this can be zero or near-zero additional document storage.
   - For payload rewrite shadowing, this includes the duplicated target payload.
4. Estimate target index footprint.
   - Use live source footprint as a baseline when the target is structurally similar.
   - Adjust for vector dimension, datatype, precision, and algorithm changes when those are present.
5. Compute reserve headroom as `max(operator reserve, 15 percent of configured memory)` when no stricter operator value is provided.
6. Compute `estimated_peak_overlap_footprint` as:
   - `source_docs + source_index + target_docs + target_index`
7. Return `READY` only if:
   - the migration diff is supported for Phase 2
   - any required transform plan is present and valid
   - available memory is greater than or equal to `estimated_peak_overlap_footprint + reserve`
8. Return `SCALE_REQUIRED` when the migration is supported but headroom is insufficient.
9. Return `MANUAL_REVIEW_REQUIRED` when the diff is ambiguous or live data is insufficient for a safe estimate.

This keeps the first shadow planner understandable and safe. More sophisticated estimators can come later if Phase 1 and early Phase 2 learnings justify them.

The planner should also report:

- estimated migration window
- estimated peak overlap footprint
- expected source-versus-target footprint delta after cutover
- whether the migration is `shadow_reindex` or `shadow_rewrite`

## Execution Flow

1. Capture the source snapshot and normalize requested changes.
2. Classify the migration as either:
   - `shadow_reindex` when the target schema can be built from the current payload
   - `shadow_rewrite` when a transform or backfill is needed
3. Load `platform_inventory.yaml`.
4. Load `transform_plan.yaml` when `shadow_rewrite` is required.
5. Compute the capacity gate result.
6. Stop if the result is not `READY`.
7. Create the shadow target for the current index only.
8. If `shadow_rewrite` is selected:
   - create the target keyspace
   - transform or backfill source documents into the target keyspace
9. Wait until the shadow index is ready.
10. Validate the shadow target.
11. Emit an operator cutover runbook.
12. Wait for operator confirmation that cutover is complete.
13. Retire the old index.
14. Retire old source payloads only when the plan explicitly says they are no longer needed.
15. Move to the next index only after the current index is finished.

The scheduler for Phase 2 is intentionally serial.

## Operator Actions

The operator is responsible for:

- supplying platform inventory
- supplying the transform or backfill plan when payload shape changes
- choosing the migration window
- scaling the database if the plan returns `SCALE_REQUIRED`
- switching application traffic to the shadow target
- confirming cutover before old index retirement
- monitoring the deployment during overlap

RedisVL should not attempt to perform these actions automatically in the first Phase 2 implementation.

Phase 2 should still emit structured benchmark outputs so operators can compare:

- estimated overlap duration versus actual overlap duration
- estimated capacity usage versus observed document and index stats
- memory savings or growth after algorithm, datatype, precision, dimension, or payload-shape changes
- query latency impact during shadow validation and overlap

## Blocked Scenarios

The initial Phase 2 plan still blocks:

- automatic scaling
- automatic traffic switching
- concurrent shadowing of multiple large indexes
- in-place destructive rewrites without a shadow target
- payload-shape-changing migrations without an explicit transform or backfill plan
- transform plans that do not define a deterministic target keyspace
- Active-Active specific workflows
- platform API integrations as a hard requirement

## Open Questions Deferred

These questions should stay deferred until after Phase 1 implementation:

- whether to add direct Redis Cloud or Redis Software API integrations
- whether to support checkpoint and resume across shadow runs
- whether alias-based cutover should be added later
- how transform hooks should be expressed beyond the initial declarative plan format
- whether re-embedding should be integrated directly or stay an operator-supplied preprocessing step
- how much historical benchmark data should influence ETA predictions
