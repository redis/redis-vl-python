# Phase 1 Tests: `drop_recreate`

> **Status**: All test scenarios covered. See actual test files below.

## Actual Test Files

| File | Coverage |
| --- | --- |
| `tests/unit/test_migration_planner.py` | Planner, patch merge, diff classification, rename detection, quantization detection |
| `tests/unit/test_batch_migration.py` | Batch planner, batch executor, batch state management |
| `tests/unit/test_migration_wizard.py` | Wizard flow, interactive prompts, plan generation |
| `tests/integration/test_migration_comprehensive.py` | End-to-end: plan -> apply -> validate with live Redis |

## Original Test Matrix (mapped to implementation)

| ID | Scenario | Type | Status |
| --- | --- | --- | --- |
| V1-CI-01 | source snapshot loads live schema and stats | CI | Done (test_migration_planner) |
| V1-CI-02 | patch merge preserves unspecified config | CI | Done (test_migration_planner) |
| V1-CI-03 | blocked diff categories stop at `plan` | CI | Done (test_migration_planner) |
| V1-CI-04 | `plan` emits valid YAML artifact | CI | Done (test_migration_planner) |
| V1-CI-05 | quantization requires `--accept-data-loss` | CI | Done (test_migration_comprehensive) |
| V1-CI-06 | drop and recreate preserves documents | CI | Done (test_migration_comprehensive) |
| V1-CI-07 | readiness polling completes or times out | CI | Done (executor tests) |
| V1-CI-08 | `validate` emits a report on success | CI | Done (test_migration_comprehensive) |
| V1-CI-09 | `validate` emits a report on failure | CI | Done (test_migration_comprehensive) |
| V1-CI-10 | timing metrics are captured in reports | CI | Done (test_migration_comprehensive) |
| V1-MAN-01 | guided wizard produces the same plan model | Unit | Done (test_migration_wizard) |
| V1-MAN-02 | realistic rebuild on larger dataset | Manual | Done (benchmark report in 05_migration_benchmark_report.md) |
| V1-MAN-03 | benchmark rehearsal on representative workload | Manual | Done (benchmark report in 05_migration_benchmark_report.md) |

## Happy Path

The minimum automated happy path should cover:

- create a source index with existing documents
- generate `migration_plan.yaml` from `schema_patch.yaml`
- run `apply --plan migration_plan.yaml`
- wait for recreated index readiness
- run `validate`
- confirm schema match, doc count match, and zero indexing failure delta

Representative happy-path schema changes:

- add a tag field backed by existing JSON data
- remove a legacy numeric field from the index
- make an existing text field sortable

## Failure Paths

CI should cover at least:

- blocked diff because of vector dimension change
- blocked diff because of storage type change
- source snapshot mismatch between `plan` and `apply`
- recreate failure after drop
- validation failure because doc counts diverge
- readiness timeout
- missing required plan fields

Every failure path must prove:

- documents are not intentionally deleted by the migrator
- an actionable error is surfaced
- blocked vector and payload-shape diffs point the user to the Phase 2 migration path
- a `migration_report.yaml` can still be produced when the failure happens after `apply` starts

## Manual Smoke Test

Run a manual smoke test on a non-production Redis deployment:

1. Create an index with representative JSON documents.
2. Prepare a `schema_patch.yaml` that adds one non-vector field and removes one old field.
3. Run `rvl migrate plan`.
4. Confirm the plan includes the downtime warning and no blocked diffs.
5. Run `rvl migrate apply --plan migration_plan.yaml`.
6. Wait until readiness completes.
7. Run `rvl migrate validate --plan migration_plan.yaml`.
8. Confirm search behavior has resumed and the new schema is active.

Manual smoke test success means:

- the operator can understand the plan without reading code
- the index rebuild completes without deleting documents
- the report is sufficient to hand back to another operator

## Scale Sanity Check

Phase 1 does not need a cluster-wide stress harness, but it does need a basic scale sanity check.

Manual checks:

- run the flow on an index large enough to make polling and downtime visible
- confirm default key capture stays bounded
- confirm the tool does not attempt a full key manifest by default
- confirm console output still stays readable for a larger index

This is not a benchmark. The goal is to catch accidental implementation choices that make the MVP operationally unsafe on larger datasets.

## Benchmark Rehearsal

Phase 1 benchmarking should be lightweight and operationally useful.

Use a simple rehearsal driven by [03_benchmarking.md](./03_benchmarking.md):

1. Record a benchmark label and workload context.
2. Measure baseline query latency on a representative query set.
3. Run the migration on a realistic non-production index.
4. Record total migration duration, downtime duration, and readiness duration.
5. Record source and target document counts and index stats.
6. Record the observed source-versus-target index footprint delta.
7. Re-run the representative query set after migration.
8. Save a `benchmark_report.yaml`.

The first benchmark questions to answer are:

- how long does the rebuild take end-to-end
- how long is the index unavailable
- how many documents per second can the rebuild sustain
- how much query latency changes during and after the rebuild
- how much the recreated index footprint changes even for schema-only rebuilds
- whether the observed runtime is predictable enough for a maintenance window

## Release Gate

Phase 1 should not be considered ready until all of the following are true:

- all CI scenarios in the test matrix pass
- at least one manual smoke test passes
- at least one benchmark rehearsal has been documented on a representative dataset
- help text matches the spec
- the docs in `local_docs/index_migrator/` still match the shipped CLI behavior
- the release notes or implementation summary clearly state that `drop_recreate` is downtime-accepting
