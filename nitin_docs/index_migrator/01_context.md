# Index Migrator Context

## Problem Statement

RedisVL does not currently provide a first-class migration workflow for search index changes.

Today, teams can create indexes, delete indexes, inspect index info, and load documents, but they still need ad hoc scripts and operational runbooks to handle schema evolution. This becomes risky when the index is large, shared by multiple applications, or deployed on clustered Redis Cloud or Redis Software.

The migration problem has three different shapes:

- A simpler index rebuild that preserves existing documents and recreates the index definition in place.
- A shadow migration over the same documents when the target schema can still be built from the current stored payload.
- A shadow migration with transform or backfill when vector dimensions, datatypes, precision, algorithms, or payload shape change and a new target payload must be built.

This workspace deliberately splits those shapes into phases instead of trying to solve everything in one design. Phase 1 proves the plan-first migration workflow. Phase 2 exists to take on the harder vector and payload-shape migrations safely.

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

RedisVL does not yet have:

- a migration planner
- a schema diff classifier
- a migration-specific CLI workflow
- a guided schema migration wizard
- structured migration reports
- capacity-aware orchestration across indexes
- transform or backfill planning for migrations that need new stored payloads

## Why Phase 1 Comes First

Phase 1 is intentionally narrow because it gives the team an MVP that is both useful and low-risk:

- It preserves documents while changing only the index definition.
- It reuses current RedisVL primitives instead of introducing a separate migration runtime.
- It keeps operational ownership clear: RedisVL handles planning, execution, and validation for a single index, while the operator handles the migration window and downstream application expectations.
- It avoids the hardest problems for now: target-payload generation, shadow overlap estimation, cutover automation, and cluster-wide scheduling.

Phase 1 does not define the full migration goal. The harder vector and payload-shape changes are the reason Phase 2 exists.

The MVP should prove the planning model, CLI shape, plan artifact, and validation/reporting flow before more advanced orchestration is attempted.

## Downtime and Disruption

Phase 1 accepts downtime for the migrated index.

Engineers need to plan for the following impacts:

- Search on the target index is unavailable between index drop and recreated index readiness.
- Query results can be partial or unstable while the recreated index is still completing its initial indexing pass.
- Reindexing uses shared database resources and can increase CPU, memory, and indexing pressure on the deployment.
- Shadow migrations can temporarily duplicate index structures and sometimes duplicate payloads as well, increasing peak memory requirements.
- Downstream applications need either a maintenance window, a degraded mode, or a clear operational pause during the rebuild.

The tooling should not hide these facts. The plan artifact and CLI output must force the user to acknowledge downtime before applying a `drop_recreate` migration.

## Non-Goals

The following are explicitly out of scope for Phase 1, not for the overall initiative:

- a generic migration framework for every schema evolution case
- automatic platform scaling
- automatic traffic cutover
- full key manifest capture by default
- document transforms or backfills in the MVP execution path
- payload relocation to a new keyspace in the MVP execution path
- concurrent migration of multiple large indexes
- fully managed Redis Cloud or Redis Software integration
- automatic transform inference or automatic re-embedding

The simplicity rules for this effort are:

- use existing RedisVL index introspection and lifecycle primitives
- do not design a generic migration framework for the MVP
- do not automate platform scaling
- do not automate traffic cutover
- do not require full key manifests by default
- require an explicit transform or backfill plan before Phase 2 handles payload-shape-changing migrations
