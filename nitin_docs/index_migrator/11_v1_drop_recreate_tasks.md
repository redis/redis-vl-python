# Phase 1 Tasks: `drop_recreate`

## Task Template

Every Phase 1 task must document:

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

## V1-T01

- `ID`: `V1-T01`
- `Status`: `Ready`
- `Goal`: Build the source snapshot and artifact models used by `plan`, `apply`, and `validate`.
- `Inputs`: source index name, Redis connection settings, optional key sample limit
- `Outputs`: in-memory and YAML-serializable source snapshot, migration plan model, migration report model
- `Touchpoints`: `redisvl/index/index.py`, new `redisvl/migration/models.py`, new `redisvl/migration/snapshot.py`
- `Dependencies`: none
- `Acceptance Criteria`:
  - source schema can be loaded from a live index
  - source stats needed by the spec are captured
  - storage type, prefixes, key separator, and bounded key sample are recorded
  - models serialize to YAML without losing required fields
- `Non-Goals`:
  - full key manifest generation
  - capacity estimation
  - schema diff logic
- `Handoff Notes`: preserve the raw source schema as faithfully as possible so later diffing does not invent defaults.

## V1-T02

- `ID`: `V1-T02`
- `Status`: `Ready`
- `Goal`: Implement schema patch normalization, source-plus-patch merge, and supported-versus-blocked diff classification.
- `Inputs`: source schema snapshot, `schema_patch.yaml` or normalized target schema diff
- `Outputs`: merged target schema and diff classification result
- `Touchpoints`: new `redisvl/migration/patch.py`, new `redisvl/migration/diff.py`, `redisvl/schema`
- `Dependencies`: `V1-T01`
- `Acceptance Criteria`:
  - unspecified source config is preserved by default
  - blocked diff categories from the spec are rejected with actionable reasons
  - supported changes produce a deterministic merged target schema
  - `target_schema.yaml` input normalizes to the same patch model
- `Non-Goals`:
  - document rewrite planning
  - vector migration logic
  - shadow migration planning
- `Handoff Notes`: prefer an explicit allowlist of supported diff categories over a generic schema merge engine.

## V1-T03

- `ID`: `V1-T03`
- `Status`: `Ready`
- `Goal`: Add the `plan` command and plan artifact generation.
- `Inputs`: source index, connection settings, patch or target schema input
- `Outputs`: `migration_plan.yaml`, console summary
- `Touchpoints`: new `redisvl/cli/migrate.py`, `redisvl/cli/main.py`, `redisvl/migration/planner.py`
- `Dependencies`: `V1-T01`, `V1-T02`
- `Acceptance Criteria`:
  - `plan` emits the required YAML shape
  - blocked plans do not proceed to mutation
  - the console summary includes downtime warnings
  - the current plan format is stable enough for `apply` and `validate`
- `Non-Goals`:
  - interactive wizard flow
  - mutation against Redis
  - advanced report rendering
- `Handoff Notes`: make the plan file human-readable so operators can review it before running `apply`.

## V1-T04

- `ID`: `V1-T04`
- `Status`: `Ready`
- `Goal`: Add the guided `wizard` flow that emits the same plan artifact as `plan`.
- `Inputs`: source index, connection settings, interactive answers
- `Outputs`: normalized schema patch and `migration_plan.yaml`
- `Touchpoints`: new `redisvl/migration/wizard.py`, `redisvl/cli/migrate.py`
- `Dependencies`: `V1-T01`, `V1-T02`, `V1-T03`
- `Acceptance Criteria`:
  - wizard starts from the live source schema
  - wizard only offers supported MVP change categories
  - wizard emits the same plan structure as `plan`
  - unsupported requests are blocked during the flow
- `Non-Goals`:
  - platform inventory collection
  - free-form schema editing for blocked categories
  - shadow migration support
- `Handoff Notes`: keep prompts simple and linear; this is a guided assistant, not a general schema builder.

## V1-T05

- `ID`: `V1-T05`
- `Status`: `Ready`
- `Goal`: Implement `apply` for the `drop_recreate` strategy.
- `Inputs`: reviewed `migration_plan.yaml`, `--allow-downtime`
- `Outputs`: recreated index and execution status
- `Touchpoints`: new `redisvl/migration/executor.py`, `redisvl/cli/migrate.py`, `redisvl/index/index.py`
- `Dependencies`: `V1-T03`
- `Acceptance Criteria`:
  - `apply` refuses to run without `--allow-downtime`
  - source snapshot mismatch blocks execution
  - index drop preserves documents
  - recreated index uses the merged target schema
  - readiness polling stops on success or timeout
- `Non-Goals`:
  - automatic rollback
  - checkpointing
  - cutover orchestration
- `Handoff Notes`: use the simplest safe sequence possible; do not add background job machinery in the MVP.

## V1-T06

- `ID`: `V1-T06`
- `Status`: `Ready`
- `Goal`: Implement `validate` and `migration_report.yaml`.
- `Inputs`: `migration_plan.yaml`, live index state, optional query checks
- `Outputs`: validation result, report artifact, console summary
- `Touchpoints`: new `redisvl/migration/validation.py`, new `redisvl/migration/reporting.py`, `redisvl/cli/migrate.py`
- `Dependencies`: `V1-T01`, `V1-T03`, `V1-T05`
- `Acceptance Criteria`:
  - schema match is verified
  - doc count match is verified
  - indexing failure delta is captured
  - core timing metrics are captured in the report
  - optional query checks run deterministically
  - report artifact is emitted for both success and failure
- `Non-Goals`:
  - benchmark replay
  - observability integrations
  - automatic remediation
- `Handoff Notes`: keep the report format concise and stable so it can become the operator handoff artifact later.

## V1-T07

- `ID`: `V1-T07`
- `Status`: `Ready`
- `Goal`: Add Phase 1 tests and user-facing documentation for the new CLI flow.
- `Inputs`: completed planner, wizard, executor, and validator behavior
- `Outputs`: passing tests and concise usage docs
- `Touchpoints`: `tests/`, `redisvl/cli`, `nitin_docs/index_migrator`
- `Dependencies`: `V1-T03`, `V1-T04`, `V1-T05`, `V1-T06`
- `Acceptance Criteria`:
  - CI-friendly happy-path and failure-path tests exist
  - manual benchmark rehearsal guidance exists
  - manual smoke test instructions are captured in the test doc
  - help text matches the Phase 1 spec
  - the docs directory still points to the active truth
- `Non-Goals`:
  - Phase 2 implementation
  - platform API integrations
  - performance tuning beyond smoke coverage
- `Handoff Notes`: keep test coverage focused on correctness and operator safety, not on simulating every future migration shape.
