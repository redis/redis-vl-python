# Phase 2 Tasks: `iterative_shadow`

## Task Template

Every Phase 2 task must document:

- `ID`
- `Status`
- `Goal`
- `Inputs`
- `Outputs`
- `Touchpoints`
- `Dependencies`
- `Acceptance Criteria`
- `Non-Goals`
- `Handoff Notes`

Phase 2 tasks are planned work only. They should not start until Phase 1 implementation is complete and learnings are folded back into this workspace.

## V2-T01

- `ID`: `V2-T01`
- `Status`: `Planned`
- `Goal`: Add the platform inventory model and parser used by the capacity-aware planner.
- `Inputs`: `platform_inventory.yaml`
- `Outputs`: validated inventory model
- `Touchpoints`: new `redisvl/migration/inventory.py`, `redisvl/migration/models.py`, `redisvl/cli/migrate.py`
- `Dependencies`: Phase 1 implementation complete
- `Acceptance Criteria`:
  - required inventory fields are validated
  - unsupported platform inventory shapes are rejected clearly
  - inventory values are available to the planner without CLI-specific parsing logic
- `Non-Goals`:
  - platform API calls
  - capacity math
  - shadow execution
- `Handoff Notes`: keep the inventory model platform-neutral enough to support both Redis Cloud and Redis Software.

## V2-T02

- `ID`: `V2-T02`
- `Status`: `Planned`
- `Goal`: Add the transform or backfill plan model and classify whether a migration is `shadow_reindex` or `shadow_rewrite`.
- `Inputs`: normalized diff classification, optional `transform_plan.yaml`
- `Outputs`: validated transform model and execution-mode classification
- `Touchpoints`: new `redisvl/migration/transforms.py`, `redisvl/migration/models.py`, `redisvl/migration/planner.py`
- `Dependencies`: `V2-T01`
- `Acceptance Criteria`:
  - payload-compatible migrations are classified as `shadow_reindex`
  - vector or payload-shape-changing migrations require `shadow_rewrite`
  - missing transform plans are rejected clearly when they are required
  - transform plans remain declarative and deterministic
- `Non-Goals`:
  - direct embedding generation
  - platform API calls
  - shadow execution
- `Handoff Notes`: keep the first transform model simple and explicit rather than inventing a generic transformation framework.

## V2-T03

- `ID`: `V2-T03`
- `Status`: `Planned`
- `Goal`: Implement the conservative capacity estimator and gate result classification.
- `Inputs`: source index stats, source document footprint, inventory model, normalized diff classification, optional transform model
- `Outputs`: `READY`, `SCALE_REQUIRED`, or `MANUAL_REVIEW_REQUIRED`
- `Touchpoints`: new `redisvl/migration/capacity.py`, `redisvl/migration/planner.py`
- `Dependencies`: `V2-T01`, `V2-T02`
- `Acceptance Criteria`:
  - source document and index footprint are computed consistently
  - target footprint estimates account for vector datatype, precision, dimension, algorithm, and payload-shape changes when those are present
  - reserve policy is applied consistently
  - supported diffs can produce `READY` or `SCALE_REQUIRED`
  - ambiguous inputs produce `MANUAL_REVIEW_REQUIRED`
- `Non-Goals`:
  - fine-grained shard placement modeling
  - automated scale actions
  - performance benchmarking as a separate subsystem
- `Handoff Notes`: keep the first estimator intentionally conservative and easy to inspect.

## V2-T04

- `ID`: `V2-T04`
- `Status`: `Planned`
- `Goal`: Extend the planner to support `iterative_shadow` for one index at a time.
- `Inputs`: source snapshot, normalized diff, inventory, transform model, capacity result
- `Outputs`: shadow migration plan and operator action list
- `Touchpoints`: `redisvl/migration/planner.py`, `redisvl/cli/migrate.py`
- `Dependencies`: `V2-T03`
- `Acceptance Criteria`:
  - supported vector and payload-shape changes can produce a valid shadow plan
  - non-`READY` capacity results block apply
  - plan artifact clearly identifies source, shadow target, target keyspace when present, and operator actions
  - plan artifact identifies whether the run is `shadow_reindex` or `shadow_rewrite`
  - plan format stays readable and deterministic
- `Non-Goals`:
  - multi-index concurrency
  - automatic cleanup
  - fleet scheduling
- `Handoff Notes`: preserve the same plan-first experience as Phase 1.

## V2-T05

- `ID`: `V2-T05`
- `Status`: `Planned`
- `Goal`: Implement shadow target creation, optional transform or backfill execution, readiness waiting, and validation hooks.
- `Inputs`: approved shadow migration plan
- `Outputs`: ready shadow index and validation state
- `Touchpoints`: new `redisvl/migration/shadow.py`, `redisvl/migration/executor.py`, `redisvl/migration/validation.py`
- `Dependencies`: `V2-T04`
- `Acceptance Criteria`:
  - only one index is processed at a time
  - shadow target creation follows the plan artifact
  - `shadow_rewrite` runs can build the target payload into the planned keyspace
  - readiness polling behaves deterministically
  - validation runs before cutover handoff
- `Non-Goals`:
  - automatic cutover
  - cross-index scheduling
  - platform autoscaling
- `Handoff Notes`: do not generalize this into a fleet scheduler in the first Phase 2 implementation.

## V2-T06

- `ID`: `V2-T06`
- `Status`: `Planned`
- `Goal`: Add validation reporting, benchmark reporting, operator handoff, cutover confirmation, and old-index retirement.
- `Inputs`: validated shadow plan and operator confirmation
- `Outputs`: post-cutover cleanup result and report
- `Touchpoints`: `redisvl/cli/migrate.py`, `redisvl/migration/reporting.py`, `redisvl/migration/executor.py`
- `Dependencies`: `V2-T05`
- `Acceptance Criteria`:
  - cutover remains operator-owned
  - cleanup does not run before operator confirmation
  - report captures cutover handoff, cleanup outcome, and source-versus-target footprint deltas
- `Non-Goals`:
  - alias management
  - application config mutation
  - rollback orchestration
- `Handoff Notes`: the CLI should guide the operator clearly, but it must not attempt to switch traffic itself.

## V2-T07

- `ID`: `V2-T07`
- `Status`: `Planned`
- `Goal`: Add future-facing tests and docs for clustered shadow migration planning.
- `Inputs`: completed Phase 2 planner and executor behavior
- `Outputs`: test coverage, manual scale rehearsal instructions, and updated planning docs
- `Touchpoints`: `tests/`, `local_docs/index_migrator`, `redisvl/cli`
- `Dependencies`: `V2-T04`, `V2-T05`, `V2-T06`
- `Acceptance Criteria`:
  - planner outcomes are covered in automated tests
  - benchmark, ETA, and memory-delta guidance are documented for manual cluster rehearsals
  - manual cluster rehearsal steps are documented
  - docs still reflect the shipped Phase 2 behavior accurately
- `Non-Goals`:
  - fleet-wide migration support
  - performance tuning beyond safety validation
  - platform-specific automation
- `Handoff Notes`: keep Phase 2 documentation grounded in the one-index-at-a-time rule.
