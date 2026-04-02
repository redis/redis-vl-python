# Phase 2 Tests: `iterative_shadow`

## Clustered Test Setup

Phase 2 needs both automated planner coverage and manual clustered rehearsals.

Minimum clustered rehearsal setup:

- Redis Cloud or Redis Software deployment
- sharded database
- one source index large enough to make overlap meaningful
- representative application query set
- operator-supplied `platform_inventory.yaml`
- `transform_plan.yaml` for any vector or payload-shape-changing migration

The first manual scale rehearsal should stay focused on a single index, not a fleet-wide migration.

## Planner Acceptance

Automated planner tests should cover:

- supported shadow diff with sufficient headroom returns `READY`
- supported shadow diff with insufficient headroom returns `SCALE_REQUIRED`
- ambiguous or incomplete input returns `MANUAL_REVIEW_REQUIRED`
- vector datatype, precision, dimension, or algorithm changes require `shadow_rewrite`
- payload-shape-changing diffs stop before planning unless a valid transform plan is present

Planner acceptance is successful when the result is deterministic and the operator action list is clear.

## Unsafe Capacity Cases

Manual and automated coverage should include:

- insufficient available memory
- missing or invalid inventory fields
- conflicting operator reserve policy
- large source footprint with conservative reserve
- target footprint larger than source footprint because of dimension or payload expansion
- peak overlap estimate exceeds available headroom even when post-cutover memory would shrink

Unsafe capacity handling is correct when:

- the planner blocks the run
- no shadow index is created
- the report tells the operator what must change before retry

## Shadow Validation

Validation coverage should prove:

- shadow target reaches readiness before handoff
- schema matches the planned target
- transformed payload fields match the declared target shape when `shadow_rewrite` is used
- query checks pass before cutover
- old index is not retired before operator confirmation

This is the safety boundary for Phase 2.

## Benchmark Rehearsal

Phase 2 benchmarks should answer:

- how accurate the planner ETA was
- how long the old and shadow indexes overlapped
- how much query latency changed during overlap
- whether the capacity reserve was conservative enough
- how much memory or size changed after datatype, precision, dimension, algorithm, or payload-shape changes
- whether estimated peak overlap footprint matched observed overlap closely enough

Minimum manual benchmark coverage:

- one run where the planner returns `READY` and the migration completes
- one run where the planner returns `SCALE_REQUIRED`
- one run with representative read traffic during overlap
- one vector-shape or algorithm change such as `HNSW -> FLAT` or `FP32 -> FP16`
- one payload-shape-changing migration that requires transform or backfill

Every benchmark rehearsal should produce a structured benchmark report that can be compared against previous runs.

## Resume/Retry

The first Phase 2 implementation does not need fleet-grade checkpointing, but it does need basic retry behavior.

Required checks:

- planner can be rerun with the same inventory and produce the same decision
- failed shadow creation does not trigger cleanup of the old index
- operator can rerun the planned index only after fixing the blocking condition

If stronger checkpointing is needed later, it should become its own scoped follow-up rather than being absorbed into the first shadow implementation.

## Exit Criteria

Phase 2 should not move from planned to ready until:

- Phase 1 has been implemented and reviewed
- Phase 1 learnings have been written back into this workspace
- planner outcomes are covered by automated tests
- at least one manual clustered rehearsal has been designed in detail
- at least one benchmark rehearsal has been defined for a representative shadow migration
- at least one benchmark rehearsal has been defined for a vector or payload-shape-changing shadow migration
- the one-index-at-a-time execution rule is still preserved in the design
