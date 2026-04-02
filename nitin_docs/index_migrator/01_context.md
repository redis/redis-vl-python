# Index Migrator Context

> **Status**: Phase 1 is complete and shipped (PRs #567-#572). This document is preserved as planning history.

## Problem Statement

RedisVL did not provide a first-class migration workflow for search index changes prior to this work.

Teams could create indexes, delete indexes, inspect index info, and load documents, but they needed ad hoc scripts and operational runbooks to handle schema evolution. This was risky when the index was large, shared by multiple applications, or deployed on clustered Redis Cloud or Redis Software.

The migration problem has three different shapes:

- A simpler index rebuild that preserves existing documents and recreates the index definition in place.
- A shadow migration over the same documents when the target schema can still be built from the current stored payload.
- A shadow migration with transform or backfill when vector dimensions, datatypes, precision, algorithms, or payload shape change and a new target payload must be built.

This workspace deliberately splits those shapes into phases. Phase 1 proved the plan-first migration workflow. Phase 2 exists to take on shadow migrations safely.

## Customer Requirements

The planning baseline for this work is:

- preserve existing documents during migration
- capture the previous index configuration before making changes
- apply only the requested schema changes
- preview the migration plan before execution
- support advanced vector migrations such as `HNSW -> FLAT`, `FP32 -> FP16`, vector dimension changes, and payload-shape-changing model or algorithm swaps
- estimate migration timing, memory impact, and operational impact using simple benchmark artifacts
- benchmark source-versus-target memory and size changes, including peak overlap footprint during shadow migrations
- support both guided and scripted workflows
- make downtime and disruption explicit
- support large datasets without defaulting to full-keyspace audits or fleet-wide orchestration
- keep the implementation understandable enough that another team can operate it safely

## Current RedisVL Capabilities

RedisVL already has useful primitives that should be reused instead of replaced:

- `SearchIndex.from_existing()` can reconstruct schema from a live index.
- `SearchIndex.delete(drop=False)` can remove the index structure without deleting documents.
- `SearchIndex.info()` can retrieve index stats used for planning and validation.
- Existing CLI commands already establish the connection and index lookup patterns the migrator can follow.

Phase 1 added the following (originally listed as missing):

- a migration planner (`MigrationPlanner`, `AsyncMigrationPlanner`, `BatchMigrationPlanner`)
- a schema diff classifier (in `planner.py`)
- a migration-specific CLI workflow (`rvl migrate` with 11 subcommands)
- a guided schema migration wizard (`MigrationWizard`)
- structured migration reports (`MigrationReport`, `MigrationValidation`, `MigrationBenchmarkSummary`)
- batch orchestration across indexes (`BatchMigrationExecutor`)
- vector quantization (e.g., FP32 -> FP16) with crash-safe reliability

Still not built (Phase 2 or future):

- capacity-aware orchestration with platform inventory
- transform or backfill planning for migrations that need new stored payloads

## Why Phase 1 Came First

Phase 1 was intentionally narrow because it gave the team an MVP that was both useful and low-risk:

- It preserves documents while changing only the index definition.
- It reuses current RedisVL primitives instead of introducing a separate migration runtime.
- It keeps operational ownership clear: RedisVL handles planning, execution, and validation for a single index, while the operator handles the migration window and downstream application expectations.
- It avoids the hardest problems for now: target-payload generation, shadow overlap estimation, cutover automation, and cluster-wide scheduling.

Phase 1 did not define the full migration goal. The harder shadow migrations are the reason Phase 2 exists.

The MVP proved the planning model, CLI shape, plan artifact, and validation/reporting flow. Notably, vector quantization (originally scoped for Phase 2) was pulled forward into Phase 1 during implementation because it could be done safely as an in-place rewrite without shadow indexes.

## Downtime and Disruption

Phase 1 accepts downtime for the migrated index.

Engineers need to plan for the following impacts:

- Search on the target index is unavailable between index drop and recreated index readiness.
- Query results can be partial or unstable while the recreated index is still completing its initial indexing pass.
- Reindexing uses shared database resources and can increase CPU, memory, and indexing pressure on the deployment.
- Shadow migrations can temporarily duplicate index structures and sometimes duplicate payloads as well, increasing peak memory requirements.
- Downstream applications need either a maintenance window, a degraded mode, or a clear operational pause during the rebuild.

The tooling does not hide these facts. The plan artifact and CLI output force the user to review the plan before applying a `drop_recreate` migration. (The original `--allow-downtime` flag was removed in favor of explicit plan review with `--accept-data-loss` only required for quantization.)

## Non-Goals

The following remain out of scope (not for the overall initiative, just for Phase 1):

- a generic migration framework for every schema evolution case
- automatic platform scaling
- automatic traffic cutover
- full key manifest capture by default
- document transforms or backfills that require new embeddings
- payload relocation to a new keyspace (shadow migrations)
- fully managed Redis Cloud or Redis Software integration
- automatic transform inference or automatic re-embedding

Note: Some items originally in this list were implemented during Phase 1:
- ~~concurrent migration of multiple large indexes~~ - batch mode was added (`rvl migrate batch-plan/batch-apply`)
- ~~field renames~~ - implemented via `rename_fields` in schema patch
- ~~prefix changes~~ - implemented via `index.prefix` in schema patch
- ~~vector datatype changes~~ - implemented as in-place quantization
