# Index Migrator Tickets

---

## Milestones

| Milestone | Theme | Stories |
|-----------|-------|---------|
| M1 | Plan and Execute Single-Index Schema Migrations | IM-01, IM-05 |
| M2 | Interactive Migration Wizard | IM-02 |
| M3 | Rename Indexes, Prefixes, and Fields | IM-06 |
| M4 | Async Execution and Batch Operations | IM-03, IM-04 |
| M5 | Validation Fixes, Integration Tests, and Documentation | IM-07, IM-08, IM-09, IM-10 |

---

## Completed

### IM-01: Plan, Execute, and Validate Document-Preserving Index Schema Migrations

**Status:** Done | **Commit:** `a3d534b` | **Milestone:** M1

**Story:** As a developer with an existing Redis index, I want to generate a reviewable migration plan, execute a safe drop-and-recreate, and validate the result, so that I can add/remove fields, change vector algorithms (FLAT/HNSW/SVS-VAMANA), change distance metrics (cosine/L2/IP), quantize vectors (float32 to float16/bfloat16/int8/uint8), and tune HNSW parameters (m, ef_construction, ef_runtime, epsilon) — all without losing documents.

**What This Delivers:**
- **Discovery**: `rvl migrate list` shows all indexes, `rvl migrate helper` explains capabilities
- **Planning**: MigrationPlanner generates a plan from a schema patch or target schema. Captures source snapshot, target schema, classifies changes as supported or blocked. Incompatible changes (dimension, storage type) are rejected at plan time.
- **Execution**: MigrationExecutor drops the index definition (not documents), re-encodes vectors if quantization is needed, and recreates the index with the merged schema.
- **Validation**: MigrationValidator confirms schema match, doc count parity, key sample existence, and functional query correctness post-migration.
- **Reporting**: Structured `migration_report.yaml` with per-phase timings, counts, benchmark summary, and warnings.

**Key Files:** `redisvl/migration/planner.py`, `executor.py`, `validation.py`, `models.py`

---

### IM-02: Build Migration Plans Interactively via Guided Wizard

**Status:** Done | **Commit:** `b06e949` | **Milestone:** M2

**Story:** As a developer unfamiliar with YAML schema syntax, I want a menu-driven wizard that walks me through adding, removing, updating, and renaming fields with real-time validation, so that I can build a correct migration plan without reading documentation.

**What This Delivers:**
- `rvl migrate wizard --index <name>` launches an interactive session
- Menus for: add field (text/tag/numeric/geo), remove field (any type, with vector warning), rename field, update field attributes (sortable, weight, no_stem, phonetic_matcher, separator, case_sensitive, index_missing, index_empty), update vector settings (algorithm, datatype, distance metric, all HNSW and SVS-VAMANA params), rename index, change prefix
- Shows current schema and previews changes before generating plan
- Outputs both `schema_patch.yaml` and `migration_plan.yaml`
- Validates choices against what's actually supported

**Key Files:** `redisvl/migration/wizard.py`

---

### IM-03: Execute Migrations Asynchronously for Large Indexes

**Status:** Done | **Commit:** `b559215` | **Milestone:** M4

**Story:** As a developer with a large index (1M+ vectors) in an async codebase, I want async migration planning, execution, and validation so that my application remains responsive and I don't block the event loop during long-running migrations.

**What This Delivers:**
- `AsyncMigrationPlanner`, `AsyncMigrationExecutor`, `AsyncMigrationValidator` classes with full feature parity
- `rvl migrate apply --async` CLI flag
- Same `MigrationPlan` model works for both sync and async
- Same plan format works for both sync and async

**Key Files:** `redisvl/migration/async_planner.py`, `async_executor.py`, `async_validation.py`

---

### IM-04: Migrate Multiple Indexes in a Single Batch with Failure Isolation and Resume

**Status:** Done | **Commit:** `61c6e80` | **Milestone:** M4

**Story:** As a platform operator with many indexes, I want to apply a shared schema patch to multiple indexes in one operation, choose whether to stop or continue on failure, and resume interrupted batches from a checkpoint, so that I can coordinate migrations during maintenance windows.

**What This Delivers:**
- `BatchMigrationPlanner` generates per-index plans from a shared patch
- `BatchMigrationExecutor` runs migrations sequentially with state persistence
- Failure policies: `fail_fast` (stop on first error), `continue_on_error` (skip and continue)
- CLI: `batch-plan`, `batch-apply`, `batch-resume`, `batch-status`
- `batch_state.yaml` checkpoint file for resume capability
- `BatchReport` with per-index status and aggregate summary

**Key Files:** `redisvl/migration/batch_planner.py`, `batch_executor.py`

---

### IM-05: Optimize Document Enumeration Using FT.AGGREGATE Cursors

**Status:** Done | **Commit:** `9561094` | **Milestone:** M1

**Story:** As a developer migrating a large index over a sparse keyspace, I want document enumeration to use the search index directly instead of SCAN, so that migration runs faster and only touches indexed keys.
```
FT.AGGREGATE idx "*" 
  LOAD 1 __key         # Get document key
  WITHCURSOR COUNT 500 # Cursor-based pagination
```

**What This Delivers:**
- Executor uses `FT.AGGREGATE ... WITHCURSOR COUNT <batch> LOAD 0` for key enumeration
- Falls back to SCAN only when `hash_indexing_failures > 0` (those docs wouldn't appear in aggregate)
- Pre-enumerates all keys before dropping index for reliable re-indexing
- CLI simplified: removed `--allow-downtime` flag (plan review is the safety mechanism)

**Key Files:** `redisvl/migration/executor.py`, `async_executor.py`

---

### IM-06: Rename Indexes, Change Key Prefixes, and Rename Fields Across Documents

**Status:** Done | **Commit:** pending | **Milestone:** M3

**Story:** As a developer, I want to rename my index, change its key prefix, or rename fields in my schema, so that I can refactor naming conventions without rebuilding from scratch.

**What This Delivers:**
- Index rename: drop old index, create new with same prefix (no document changes)
- Prefix change: `RENAME` command on every key (single-prefix indexes only, multi-prefix blocked)
- Field rename: `HSET`/`HDEL` for hash, `JSON.SET`/`JSON.DEL` for JSON, on every document
- Execution order: field renames, then key renames, then drop, then recreate
- `RenameOperations` model in migration plan
- Timing fields: `field_rename_duration_seconds`, `key_rename_duration_seconds`
- Warnings issued for expensive operations

**Key Files:** `redisvl/migration/models.py`, `planner.py`, `executor.py`, `async_executor.py`

**Spec:** `local_docs/index_migrator/30_rename_operations_spec.md`

---

### IM-07: Fix HNSW Parameter Parsing, Weight Normalization, and Algorithm Case Sensitivity

**Status:** Done | **Commit:** `ab8a017` | **Milestone:** M5

**Story:** As a developer, I want post-migration validation to correctly handle HNSW-specific parameters, weight normalization, and algorithm case sensitivity, so that validation doesn't produce false failures.

**What This Fixes:**
- HNSW-specific parameters (m, ef_construction) were not being parsed from `FT.INFO`, causing validation failures
- Weight int/float normalization mismatch (schema defines `1`, Redis returns `1.0`)
- Algorithm case sensitivity in wizard (schema stores `'hnsw'`, wizard compared to `'HNSW'`)

**Key Files:** `redisvl/redis/connection.py`, `redisvl/migration/utils.py`, `redisvl/migration/wizard.py`

---

### IM-08: Add Integration Tests for All Supported Migration Routes

**Status:** Done | **Commit:** `b3d88a0` | **Milestone:** M5

**Story:** As a maintainer, I want integration tests covering algorithm changes, quantization, distance metrics, HNSW tuning, and combined migrations, so that regressions are caught before release.

**What This Delivers:**
- 22 integration tests running full apply+validate against a live Redis instance
- Covers: 9 datatype routes, 4 distance metric routes, 5 HNSW tuning routes, 2 algorithm routes, 2 combined routes
- Tests require Redis 8.0+ for INT8/UINT8 datatypes
- Located in `tests/integration/test_migration_routes.py`

---

### IM-09: Update Migration Documentation to Reflect Rename, Batch, and Redis 8.0 Support

**Status:** Done | **Commit:** `d452eab` | **Milestone:** M5

**Story:** As a user, I want documentation that accurately reflects all supported migration operations, so that I can self-serve without guessing at capabilities.

**What This Delivers:**
- Updated `docs/concepts/index-migrations.md` to reflect prefix/field rename support
- Updated `docs/user_guide/how_to_guides/migrate-indexes.md` with Redis 8.0 requirements
- Added batch migration commands to CLI reference in `docs/user_guide/cli.ipynb`
- Removed prefix/field rename from "blocked" lists

---

### IM-10: Address PR Review Feedback for Correctness and Consistency

**Status:** Done | **Commit:** pending | **Milestone:** M5

**Story:** As a maintainer, I want code review issues addressed so that the migration engine is correct, consistent, and production-ready.

**What This Fixes:**
- `merge_patch()` now applies `rename_fields` to merged schema
- `BatchState.success_count` uses correct status string (`"succeeded"`)
- CLI helper text updated to show prefix/rename as supported
- Planner docstring updated to reflect current capabilities
- `batch_plan_path` stored in state for proper resume support
- Fixed `--output` to `--plan-out` in batch migration docs
- Fixed `--indexes` docs to use comma-separated format
- Added validation to block multi-prefix migrations
- Updated migration plan YAML example to match actual model
- Added `skipped_count` property and `[SKIP]` status display

**Key Files:** `redisvl/migration/planner.py`, `models.py`, `batch_executor.py`, `redisvl/cli/migrate.py`, `docs/user_guide/how_to_guides/migrate-indexes.md`

---

## Pending / Future

### IM-R1: Add Crash-Safe Quantization with Checkpoint Resume and Pre-Migration Snapshot

**Status:** Done | **Commit:** `30cc6c1` | **Priority:** High

**Story:** As a developer running vector quantization on a production index, I want the migration to be resumable if it crashes mid-quantization, so that I don't end up with a partially quantized index and no rollback path.

**Problem:**
The current quantization flow is: enumerate keys, drop index, quantize vectors in-place, recreate index, validate. If the process crashes during quantization, you're left with no index, a mix of float32 and float16 vectors, and no way to recover.

**What This Delivers:**
A four-layer reliability model. A pre-migration `BGSAVE` (run sequentially, waited to completion) provides full disaster recovery by restoring the RDB to pre-migration state. A checkpoint file on disk tracks which keys have been quantized, enabling resume from the exact failure point on retry. Each key conversion detects the vector dtype before converting, making it idempotent so already-converted keys are safely skipped on resume. A bounded undo buffer stores originals for only the current in-flight batch, allowing rollback of the batch that was in progress at crash time.

**Acceptance Criteria:**
1. Pre-migration `BGSAVE` is triggered and completes before any mutations begin
2. A checkpoint file records progress as each batch of keys is quantized
3. `rvl migrate apply --resume` picks up from the last checkpoint and completes the migration
4. Each key conversion is idempotent -- running the migration twice on the same key produces the correct result
5. If a batch fails mid-write, only that batch's vectors are rolled back using the bounded undo buffer
6. A disk space estimator function calculates projected RDB snapshot size, AOF growth, and total new disk required based on doc count, vector dimensions, source/target dtype, and AOF status. The estimator runs before any mutations and prints a human-readable summary. If available disk is below 80% of the estimate, the CLI prompts for confirmation. The estimator also supports a standalone dry-run mode via `rvl migrate estimate --plan plan.yaml`. See `local_docs/index_migrator/40_reliability_brainstorm.md` section "Pre-Migration Disk Space Estimator" for the full specification including inputs, outputs (DiskSpaceEstimate dataclass), calculation logic, CLI output format, integration points, and edge cases.

**Alternatives Considered:** Undo log (WAL-style), new-field-then-swap (side-write), shadow index (blue-green), streaming with bounded undo buffer. See `local_docs/index_migrator/40_reliability_brainstorm.md` for full analysis.

---

### IM-B1: Benchmark Float32 vs Float16 Quantization: Search Quality and Migration Performance at Scale

**Status:** Planned | **Priority:** High

**Story:** As a developer considering vector quantization to reduce memory, I want benchmarks measuring search quality degradation (precision, recall, F1) and migration performance (throughput, latency, memory savings) across realistic dataset sizes, so that I can make an informed decision about whether the memory-accuracy tradeoff is acceptable for my use case.

**Problem:**
We tell users they can quantize float32 vectors to float16 to cut memory in half, but we don't have published data showing what they actually lose in search quality or what they can expect in migration performance at different scales.

**What This Delivers:**
A benchmark script and published results using a real dataset (AG News with sentence-transformers embeddings) that measures two things across multiple dataset sizes (1K, 10K, 100K). For search quality: precision@K, recall@K, and F1@K comparing float32 (ground truth) vs float16 (post-migration) top-K nearest neighbor results. For migration performance: end-to-end duration, quantization throughput (vectors/second), index downtime, pre/post memory footprint, and query latency before and after (p50, p95, p99).

**Acceptance Criteria:**
1. Benchmark runs end-to-end against a local Redis instance with a single command
2. Uses a real public dataset with real embeddings (not synthetic random vectors)
3. Reports precision@K, recall@K, and F1@K for float32 vs float16 search results
4. Reports per-query statistics (mean, p50, p95, min, max) not just aggregates
5. Runs at multiple dataset sizes (at minimum 1K, 10K, 100K) to show how quality and performance scale
6. Reports memory savings (index size delta in MB) and migration throughput (docs/second)
7. Reports query latency before and after migration
8. Outputs a structured JSON report that can be compared across runs

**Note:** Benchmark script scaffolded at `tests/benchmarks/index_migrator_real_benchmark.py`.

---

### IM-11: Run Old and New Indexes in Parallel for Incompatible Changes with Operator-Controlled Cutover

**Status:** Future | **Priority:** Medium

**Story:** As a developer changing vector dimensions or storage type, I want to run old and new indexes in parallel until I'm confident in the new one, so that I can migrate without downtime and rollback if needed.

**Context:**
Some migrations cannot use `drop_recreate` because the stored data is incompatible (dimension changes, storage type changes, complex payload restructuring). Shadow migration creates a new index alongside the old one, copies/transforms documents, validates, then hands off cutover to the operator.

**What This Requires:**
- Capacity estimation (can Redis hold both indexes?)
- Shadow index creation
- Document copy with optional transform
- Progress tracking with resume
- Validation gate before cutover
- Operator handoff for cutover decision
- Cleanup of old index/keys after cutover

**Spec:** `local_docs/index_migrator/20_v2_iterative_shadow_spec.md`

---

### IM-12: Pipeline Vector Reads During Quantization to Reduce Round Trips on Large Datasets

**Status:** Backlog | **Priority:** Low

**Story:** As a developer migrating large datasets, I want quantization reads to be pipelined so that migration completes faster.

**Context:**
Current quantization implementation does O(N) round trips for reads (one `HGET` per key/field) while only pipelining writes. For large datasets this is slow.

**What This Requires:**
- Pipeline all reads in a batch before processing
- Use `transaction=False` for read pipeline
- Add JSON storage support (`JSON.GET`/`JSON.SET`) for JSON indexes

---

### IM-13: Wire ValidationPolicy Enforcement into Validators or Remove the Unused Model

**Status:** Backlog | **Priority:** Low

**Story:** As a developer, I want to skip certain validation checks (e.g., doc count) when I know they'll fail due to expected conditions.

**Context:**
`MigrationPlan.validation` (ValidationPolicy) exists in the model but is not enforced by validators. Schema/doc-count mismatches always produce errors.

**What This Requires:**
- Wire `ValidationPolicy.require_doc_count_match` into validators
- Add CLI flag to set policy during plan creation
- Or remove unused ValidationPolicy model

---

### IM-14: Clean Up Unused Imports and Linting Across the Codebase

**Status:** Backlog | **Priority:** Low

**Story:** As a maintainer, I want clean linting so that CI is reliable and code quality is consistent.

**Context:**
During development, pyflakes identified unused imports across the codebase. These were fixed in migration files but not committed for non-migration files to keep the PR focused.

**What This Requires:**
- Fix remaining unused imports (see `local_docs/issues/unused_imports_cleanup.md`)
- Update `.pylintrc` to remove deprecated Python 2/3 compat options
- Consider adding `check-lint` to the main `lint` target after cleanup

---

### IM-15: Use RENAMENX for Prefix Migrations to Fail Fast on Key Collisions

**Status:** Backlog | **Priority:** Low

**Story:** As a developer changing key prefixes, I want the migration to fail fast if target keys already exist, so I don't end up with a partially renamed keyspace.

**Context:**
Current implementation uses `RENAME` without checking if destination key exists. If a target key exists, RENAME will error and the pipeline may abort, leaving a partially-renamed keyspace.

**What This Requires:**
- Preflight check for key collisions or use `RENAMENX`
- Surface hard error rather than warning
- Consider rollback strategy

---

### IM-16: Auto-Detect AOF Status for Disk Space Estimation

**Status:** Backlog | **Priority:** Low

**Story:** As an operator running `rvl migrate estimate`, I want the disk space estimate to automatically detect whether AOF is enabled on the target Redis instance, so that AOF growth is included in the estimate without me needing to know or pass a flag.

**Context:**
The disk space estimator (`estimate_disk_space`) is a pure calculation that accepts `aof_enabled` as a parameter (default `False`). In CLI usage, this means AOF growth is never estimated unless the caller explicitly passes `aof_enabled=True`. The summary currently prints "not estimated (pass aof_enabled=True if AOF is on)" which is accurate but requires the operator to know their Redis config.

**What This Requires:**
- Add `--aof-enabled` flag to `rvl migrate estimate` CLI for offline/pure-calculation use
- During `rvl migrate apply`, read `CONFIG GET appendonly` from the live Redis connection and pass the result to `estimate_disk_space`
- Handle `CONFIG GET` failures gracefully (e.g. ACL restrictions) by falling back to the current "not estimated" behavior

---

## Summary

| Ticket | Title | Status |
|--------|-------|--------|
| IM-01 | Plan, Execute, and Validate Document-Preserving Index Schema Migrations | Done |
| IM-02 | Build Migration Plans Interactively via Guided Wizard | Done |
| IM-03 | Execute Migrations Asynchronously for Large Indexes | Done |
| IM-04 | Migrate Multiple Indexes in a Single Batch with Failure Isolation and Resume | Done |
| IM-05 | Optimize Document Enumeration Using FT.AGGREGATE Cursors | Done |
| IM-06 | Rename Indexes, Change Key Prefixes, and Rename Fields Across Documents | Done |
| IM-07 | Fix HNSW Parameter Parsing, Weight Normalization, and Algorithm Case Sensitivity | Done |
| IM-08 | Add Integration Tests for All Supported Migration Routes | Done |
| IM-09 | Update Migration Documentation to Reflect Rename, Batch, and Redis 8.0 Support | Done |
| IM-10 | Address PR Review Feedback for Correctness and Consistency | Done |
| IM-R1 | Add Crash-Safe Quantization with Checkpoint Resume and Pre-Migration Snapshot | Done |
| IM-B1 | Benchmark Float32 vs Float16 Quantization: Search Quality and Migration Performance at Scale | Planned |
| IM-11 | Run Old and New Indexes in Parallel for Incompatible Changes with Operator-Controlled Cutover | Future |
| IM-12 | Pipeline Vector Reads During Quantization to Reduce Round Trips on Large Datasets | Backlog |
| IM-13 | Wire ValidationPolicy Enforcement into Validators or Remove the Unused Model | Backlog |
| IM-14 | Clean Up Unused Imports and Linting Across the Codebase | Backlog |
| IM-15 | Use RENAMENX for Prefix Migrations to Fail Fast on Key Collisions | Backlog |
| IM-16 | Auto-Detect AOF Status for Disk Space Estimation | Backlog |
| IM-17 | Guard Against Connection Leaks in Long-Running Batch Migrations | Backlog |
| IM-18 | Optimize O(n^2) Checkpoint Serialization for Large Key Sets | Backlog |
| IM-19 | Add Redis Cluster Slot-Aware Key Distribution for Quantization Batches | Backlog |
| IM-20 | Add Pipelined Reads for Quantization to Reduce Per-Key Round Trips | Backlog |

