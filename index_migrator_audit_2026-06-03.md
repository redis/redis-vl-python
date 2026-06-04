2026-06-03

# Index Migrator Audit and Remediation Plan

## Scope

This audit covers the index migrator apply path, backup/checkpoint handling,
rollback behavior, multi-worker quantization, and user-visible backup reporting.
The goal is to make crash recovery deterministic before committing or pushing
the current migration changes.

## Resolution Status

All six audit findings were remediated in the current working tree.

| Finding | Result |
| --- | --- |
| Unsafe checkpoint phases | Fixed |
| Prefix changes plus quantization write to old keys | Fixed |
| Multi-worker crash resume unreachable from executor | Fixed |
| Multi-worker apply without `redis_url` fails after mutation | Fixed |
| JSON same-width datatype migrations blocked unnecessarily | Fixed |
| Backup reports list non-existent worker paths | Fixed |

Key decisions:

- `backup_dir` is mandatory for every migration apply path. If it is omitted,
  empty, or cannot be created/written, apply fails before source validation,
  enumeration, drop, rename, quantization, or index creation.
- Backups are retained after successful migrations. They are recovery and
  audit artifacts, not temporary scratch files.
- Single-worker backups and multi-worker manifests now model destructive
  migration boundaries explicitly so resume can make a deterministic choice
  from disk state plus live Redis state.
- Prefix-changing migrations store key-prefix mapping in the backup metadata.
  Quantization and rollback use that mapping to write to the current live keys.
- JSON datatype changes that do not rewrite stored bytes are treated as
  schema/index migrations; hash datatype changes that require byte rewrites
  still use the backup/quantization path.

## Findings and Fixes

### 1. Critical: checkpoint phases are unsafe at migration boundaries

Result: **Fixed**.

Decision and implementation:

- Added explicit phases for migration boundaries, including source-dropped,
  quantized, target-created, and validated states.
- Resume now checks both backup/manifest phase and live Redis schema state
  before choosing the next operation.
- A completed or quantized backup is not deleted simply because the target
  index is absent. It is used to recreate and validate the target when the
  source index is already gone.
- If the live source schema still matches after rollback/recovery, the stale
  completed backup is ignored and a fresh migration run starts.
- Sync and async executors use the same recovery model.

Current behavior:

- `VectorBackup.mark_dump_complete()` moves a backup from `dump` to `ready`, but
  `ready` only proves that vector backup data was written. It does not prove
  that the source index was dropped.
- Resume treats both `ready` and `active` as post-drop states and skips the drop
  step.
- `_quantize_from_backup()` marks the backup `completed` after vector writes,
  before the target index is recreated and validated.
- A resumed `completed` backup is deleted as stale if the target schema is not
  already live.

Impact:

- A crash after dump completion but before `source_index.delete(drop=False)` can
  resume by writing quantized vectors under the still-live source index.
- A crash after quantization but before target creation can resume, delete the
  completed backup as stale, and then fail because the source index is already
  gone.

Relevant code:

- `redisvl/migration/backup.py:159` marks `ready`.
- `redisvl/migration/backup.py:187` marks `completed`.
- `redisvl/migration/executor.py:756` handles `completed` by validating the
  live target or deleting the backup.
- `redisvl/migration/executor.py:911` resumes `ready`/`active` as though drop
  already happened.
- `redisvl/migration/executor.py:1445` marks `completed` before target create.
- Async mirrors the same behavior in `redisvl/migration/async_executor.py:635`
  and `redisvl/migration/async_executor.py:996`.

Fix:

1. Split the backup state machine into phases that describe destructive
   boundaries explicitly:
   - `dumping`
   - `dump_ready_source_live`
   - `index_dropped`
   - `quantizing`
   - `quantized`
   - `target_created`
   - `validated`
2. On resume, inspect live source and target schemas before choosing the next
   operation:
   - If the target index already matches the target schema, validate and report
     success.
   - If the source index still matches the source snapshot and the backup is
     dump-ready, drop the source index before quantizing.
   - If the source index is missing and the backup is dump-ready or quantizing,
     continue quantization and then create the target.
   - If the source index is missing and the backup is quantized, create and
     validate the target.
   - If the source index still matches the source snapshot and the backup is
     quantized/completed, treat the backup as stale after rollback and restart.
3. Do not delete a completed/quantized backup merely because the target schema
   is not live. Delete it only when the live source schema still matches the
   original snapshot, which indicates rollback or fresh setup.
4. Mirror the same state machine in `AsyncMigrationExecutor`.

Regression tests:

- Resume from `dump_ready_source_live` while the source index still exists must
  drop the source before quantization.
- Resume from `quantized` with the source index missing and target index absent
  must create and validate the target without deleting the backup.
- The same scenarios must pass for async apply.

### 2. Critical: prefix changes plus single-worker quantization write to old keys

Result: **Fixed**.

Decision and implementation:

- Backup headers now persist source and target key-prefix metadata when a
  migration changes prefixes.
- Backup quantization maps each backed-up key to the live destination key before
  writing converted vector bytes.
- Rollback uses the same mapping, so it restores original vector bytes onto the
  renamed live keys rather than recreating old-prefix keys.
- Added sync, async, and CLI rollback regression coverage for prefix-changing
  quantization migrations.

Current behavior:

- The single-worker backup captures original keys before key rename.
- The migration drops the index and renames keys from old prefix to new prefix.
- `keys_to_process` is updated to the new prefix, but the backup quantization
  path ignores that updated list and writes the converted vector bytes to the
  keys stored in the backup.
- Rollback restores backup vectors to the keys stored in the backup without
  prefix mapping.

Impact:

- A migration that changes both prefix and vector datatype can recreate
  old-prefix hash keys, leaving the renamed new-prefix documents unconverted.
- Rollback after a prefix-changing migration can restore vectors into old-prefix
  keys rather than the live renamed keys.

Relevant code:

- `redisvl/migration/executor.py:1060` dumps the old keys.
- `redisvl/migration/executor.py:1081` renames keys after dropping the index.
- `redisvl/migration/executor.py:1117` updates `keys_to_process` to new-prefix
  keys.
- `redisvl/migration/executor.py:1153` then quantizes from backup instead of
  the remapped key list.
- `redisvl/migration/executor.py:1431` writes converted vectors using the
  backup batch keys.
- `redisvl/cli/migrate.py:447` rollback restores to backup keys.
- Async mirrors the same behavior in `redisvl/migration/async_executor.py:916`
  through `redisvl/migration/async_executor.py:1000`.

Fix:

1. Store or derive a deterministic key mapping whenever a plan changes prefix.
2. Pass that mapping into `_quantize_from_backup()` so each backup key is mapped
   to its current live key before `pipeline_write_vectors()`.
3. Apply the same mapping in async backup quantization.
4. Extend rollback with prefix-aware restore behavior:
   - If the backup header contains a key mapping, restore to mapped live keys.
   - If no mapping exists, keep current behavior for backward compatibility.
5. Add a post-quantization guard for prefix-changing migrations that checks no
   old-prefix keys were recreated by quantization.

Regression tests:

- Sync single-worker prefix plus datatype migration must leave no old-prefix
  keys and must write target-width vectors under the new prefix.
- Async single-worker prefix plus datatype migration must do the same.
- Rollback from a prefix-changing backup must restore vector bytes on the live
  mapped keys.

### 3. High: multi-worker crash resume is not reachable from the executor

Result: **Fixed**.

Decision and implementation:

- Added a canonical multi-worker `.manifest` file written before destructive
  operations.
- The manifest records the canonical migration identity, key slices, actual
  worker shard paths, requested worker count, and phase.
- Executor resume now checks for either a single-worker backup or a
  multi-worker manifest before validating that the source index still exists.
- If the source index is gone and the manifest indicates remaining work, apply
  resumes through worker shard checkpoints and then creates/validates the
  target index.
- Sync executor support is covered by Redis-backed e2e tests. The async
  executor shares the same manifest model for multi-worker resume decisions.

Current behavior:

- Multi-worker quantization writes shard backup files such as
  `migration_backup_<index>_<hash>_worker0.header`.
- The executor only tries to load the canonical `backup_path`.
- After a post-drop crash, no canonical header may exist. The executor does not
  enter resume mode, then source schema validation fails because the source
  index was already dropped.

Impact:

- Worker-level checkpointing exists, but top-level crash resume for multi-worker
  apply can be unreachable after the destructive drop step.

Relevant code:

- `redisvl/migration/executor.py:746` loads only `VectorBackup.load(backup_path)`.
- `redisvl/migration/executor.py:802` validates the source schema when not
  resuming.
- `redisvl/migration/quantize.py:324` computes actual worker slices.
- `redisvl/migration/quantize.py:335` creates worker-specific backup paths.
- Batch migrations are affected indirectly because batch apply delegates to the
  same executor.

Fix:

1. Add a top-level multi-worker manifest written before any destructive step.
   The manifest should include:
   - source index name
   - source schema hash
   - target schema hash
   - worker shard backup paths
   - actual worker count
   - current phase
2. Teach executor resume to load either the canonical single-worker backup or
   the multi-worker manifest/shard set before source schema validation.
3. If the source index is gone and the manifest indicates quantization is
   incomplete, resume the multi-worker path from shard checkpoints.
4. Apply the same manifest logic to `AsyncMigrationExecutor`.

Regression tests:

- Multi-worker crash after drop must resume from shard backups without requiring
  a live source index.
- Batch migration with a multi-worker index must also resume through the shared
  executor path.

### 4. High: `num_workers > 1` with only `redis_client` fails after mutation

Result: **Fixed**.

Decision and implementation:

- Added early preflight checks in sync apply, async apply, and batch apply.
- `num_workers > 1` without `redis_url` now fails before any source lookup or
  destructive operation.
- Kept lower-level defensive checks in the worker path.
- Added sync, async, and batch regression coverage that verifies no index drop
  or checkpoint mutation happens before this configuration error.

Current behavior:

- Python callers can pass `redis_client=...`, `num_workers > 1`, and
  `redis_url=None`.
- The multi-worker path requires `redis_url`, but that check happens inside the
  quantization block after the source index has already been dropped and keys
  may have been renamed.
- CLI callers are less exposed because the CLI builds a Redis URL, but the
  Python API and batch paths are exposed.

Impact:

- An API caller can lose the source index definition before receiving the
  configuration error.

Relevant code:

- `redisvl/migration/executor.py:1077` drops the source index.
- `redisvl/migration/executor.py:1134` checks `redis_url` too late.
- Async mirrors this in `redisvl/migration/async_executor.py:919` and
  `redisvl/migration/async_executor.py:977`.

Fix:

1. Add an early preflight in sync, async, and batch apply:
   - If `num_workers > 1 and redis_url is None`, fail before source validation,
     enumeration, dump, drop, or rename.
2. Report the error through `MigrationReport.validation.errors` for report-based
   paths, and raise `ValueError` only in lower-level helper paths that do not
   return reports.
3. Keep the existing late guard as a defensive assertion.

Regression tests:

- `MigrationExecutor.apply(..., redis_client=client, redis_url=None,
  num_workers=2)` must fail before `SearchIndex.delete()` is called.
- Async apply must fail before `AsyncSearchIndex.delete()` is called.
- Batch apply must fail before applying any index in the batch.

### 5. Medium: same-width datatype guard blocks JSON migrations unnecessarily

Result: **Fixed**.

Decision and implementation:

- Scoped the same-width datatype guard to migrations that actually rewrite hash
  vector bytes.
- JSON same-width datatype changes now proceed as schema/index migrations and
  still validate/report the required backup directory.
- Hash same-width datatype changes remain blocked until a safe byte rewrite
  strategy exists.
- Added e2e coverage for JSON `float16` to `bfloat16` schema migration and unit
  coverage that hash same-width migrations remain blocked.

Current behavior:

- `needs_quantization` excludes JSON because JSON vectors are not rewritten.
- `has_same_width_quantization` is computed across all datatype changes and
  returned before the JSON distinction matters.

Impact:

- A JSON schema/datatype migration can be rejected even though the
  same-width hash byte rewrite risk does not apply to JSON storage.

Relevant code:

- `redisvl/migration/executor.py:861` excludes JSON from quantization.
- `redisvl/migration/executor.py:875` rejects same-width changes anyway.
- Async mirrors this in `redisvl/migration/async_executor.py:713` and
  `redisvl/migration/async_executor.py:727`.

Fix:

1. Scope the same-width unsupported guard to actual hash byte rewrites:
   `if needs_quantization and has_same_width_quantization`.
2. Keep JSON migrations as schema/index operations unless another JSON-specific
   rewrite path is added later.

Regression tests:

- JSON same-width datatype changes must not trigger the crash-safe resume
  rejection.
- Hash same-width datatype changes must still be rejected.

### 6. Low/Medium: backup reports can list non-existent worker paths

Result: **Fixed**.

Decision and implementation:

- Multi-worker backup paths are now reported from actual worker results or the
  manifest, not from the requested worker count.
- If fewer key slices are created than requested workers, reports list only the
  shard files that were actually created.
- Added regression coverage for `num_workers=8` over fewer keys.

Current behavior:

- Reports list backup paths using the requested `num_workers`.
- `split_keys()` may create fewer actual worker shards when there are fewer keys
  than requested workers.

Impact:

- A report can tell users to expect backup files that were never created.

Relevant code:

- `redisvl/migration/executor.py:867` builds report paths from requested worker
  count.
- `redisvl/migration/quantize.py:324` computes the actual worker count.

Fix:

1. Defer multi-worker `report.backup.backup_paths` until after the worker result
   is available.
2. Populate it from actual worker result paths or the multi-worker manifest.
3. Apply the same reporting behavior in async apply.

Regression tests:

- `num_workers=8` with two keys must report two shard backup paths, not eight.

## Recommended Implementation Order

1. Add early destructive-operation preflight checks.
2. Fix prefix-aware backup quantization and rollback key mapping.
3. Replace the backup phase model with explicit source-live, source-dropped,
   quantized, target-created, and validated phases.
4. Add multi-worker manifest detection and executor-level resume.
5. Scope the same-width datatype guard to non-JSON rewrites.
6. Correct backup path reporting from actual worker results.
7. Add the regression tests listed above for sync, async, batch, and CLI
   rollback behavior.

## Verification Target

Before considering the migrator ready, run at minimum:

```bash
uv run pytest \
  tests/unit/test_executor_backup_quantize.py \
  tests/unit/test_async_migration_executor.py \
  tests/unit/test_multi_worker_quantize.py \
  tests/unit/test_batch_migration.py \
  tests/integration/test_migration_v1.py \
  tests/integration/test_async_migration_v1.py \
  tests/integration/test_batch_migration_integration.py
```

Add focused tests for each regression above before relying on the current
backup/checkpointing behavior in production migrations.

## Verification Result

Completed on 2026-06-03:

- `uv run pytest ...` focused migrator unit and integration bundle:
  `304 passed`
- New Redis-backed e2e recovery suite:
  `6 passed`
- Combined migration integration suite:
  `88 passed`
- Migration notebook validation:
  `14 passed`
- Docs build:
  passed
- Ruff check and format check for touched migrator/test files:
  passed
- Targeted mypy for `redisvl/migration` and `redisvl/cli/migrate.py`:
  passed
- Synthetic migration benchmark smoke:
  sync and async runs passed

Broader full-repo tests were not clean in this environment because unrelated
optional dependencies are missing (`pydantic_settings`, `nltk`, `sql_redis`,
and `sentence_transformers`). Those failures do not route through the index
migrator changes audited here.
