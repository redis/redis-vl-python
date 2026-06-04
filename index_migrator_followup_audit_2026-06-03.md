2026-06-03

# Index Migrator Follow-Up Audit

## Scope

This follow-up audit re-checks the current post-remediation index migrator
implementation, focusing on edge cases introduced or left open by the first
backup/checkpointing fix set.

Reviewed areas:

- `VectorBackup` and `MultiWorkerBackupManifest` state transitions
- sync and async executor resume paths
- multi-worker shard resume behavior
- batch apply/resume checkpoint behavior
- rollback restore behavior
- report backup path accuracy
- user-facing docs versus implementation behavior

## Summary

The six findings from the first audit remain fixed for the covered happy paths
and tested crash windows. This follow-up audit found four additional resume and
checkpoint edge cases, and all four have now been fixed or explicitly decided.

The main decisions are:

- Existing multi-worker manifests are now authoritative for resume settings.
- Backup checkpoints now carry a migration identity and are rejected when they
  do not match the current plan.
- Worker shard resume offsets now come from the shard backup header, not from
  the retry command's current options.
- Empty migrations only report backup files that actually exist.

| ID | Severity | Finding | Status |
| --- | --- | --- | --- |
| F-01 | High | Multi-worker manifest resume is gated by the caller passing `num_workers > 1` again | Fixed |
| F-02 | High | Backups/manifests are not bound to the migration plan or target schema | Fixed |
| F-03 | Medium | Multi-worker partial dump resume can use the caller's new `batch_size` instead of checkpoint batch size | Fixed |
| F-04 | Low/Medium | Empty single-worker quantization can report a backup path that was never created | Fixed |

## Findings and Decisions

### F-01: Multi-worker manifest resume requires passing `num_workers > 1` again

Status: **Fixed**.

Prior behavior:

- The executor only loaded a `MultiWorkerBackupManifest` when the current apply
  call had `num_workers > 1`.
- The manifest itself already recorded `requested_workers`, `actual_workers`,
  `worker_backup_paths`, `key_slices`, and `phase`.
- If a multi-worker migration crashed after the source index was dropped, and a
  user resumed without passing `--workers` again, the executor did not load the
  manifest. It then saw no source index and reported that the live source schema
  no longer matched the plan.
- `batch-resume` examples omit `--workers`, and `BatchState` stores
  `backup_dir` but not `num_workers`.

Impact:

- Single-index resume worked when the user re-ran the exact same command with
  `--workers N`.
- Batch multi-worker resume was fragile because the documented resume command
  did not preserve `--workers`.
- The manifest contained enough information to recover, but the executor could
  fail to discover it.

Fix/decision:

1. `MigrationExecutor` and `AsyncMigrationExecutor` now load
   `MultiWorkerBackupManifest` whenever the canonical manifest exists,
   independent of the current `num_workers` argument.
2. Manifest resume now uses `manifest.requested_workers` and
   `manifest.batch_size`.
3. If a manifest requires worker orchestration and the caller only supplied an
   already-open Redis client, the executor fails before mutation with a clear
   error requiring `redis_url`.
4. No separate `BatchState` schema change was needed for this fix because
   per-index manifests are now the resume source of truth for worker count and
   batch size.

Verification:

- `test_multi_worker_manifest_resumes_without_num_workers_arg`
- `test_multi_worker_manifest_resume_after_drop_end_to_end`
- `test_multi_worker_requires_redis_url_before_loading_index`
- `test_async_multi_worker_requires_redis_url_before_loading_index`

### F-02: Backups/manifests are not bound to the migration plan or target schema

Status: **Fixed**.

Prior behavior:

- The canonical backup path was derived from `backup_dir` and source index name.
- Single-worker backup headers recorded `index_name`, vector field changes,
  batch progress, phase, and optional prefix mapping.
- Multi-worker manifests recorded index name, key slices, worker paths, phase,
  and optional prefix mapping.
- Neither checkpoint format stored or validated the source schema hash, target
  schema hash, plan hash, or requested datatype-change signature before resume.

Impact:

- If a user changed the migration plan but reused the same `backup_dir`, an old
  checkpoint for the same source index could be treated as resumable.
- A `ready` backup may be reusable when the source schema is identical and the
  new plan only changes target datatype, because it contains original bytes.
  But a `completed` or `quantized` checkpoint can be unsafe: the executor may
  skip quantization and create or validate a target schema that does not match
  the bytes already written to Redis.
- Validation should catch many bad outcomes through indexing failures or query
  checks, but by then the executor may have already created a new target index
  from a mismatched checkpoint.

Fix/decision:

1. `BackupHeader` and `MultiWorkerBackupManifest` now persist checkpoint
   identity fields:
   - source schema hash
   - target schema hash
   - datatype-change hash
   - plan hash
2. Sync and async executors now compute the current plan identity and compare it
   with an existing backup or manifest before deciding to skip dump, quantize,
   or create.
3. If the source index still matches the plan source schema and checkpoint
   identity differs, the checkpoint is treated as stale and the executor
   restarts from the live source.
4. If the source index is gone and checkpoint identity differs, the executor
   fails with manual recovery instructions instead of creating a target from a
   mismatched checkpoint.

Verification:

- `test_checkpoint_plan_mismatch_with_missing_source_fails_before_create`
- `test_ready_checkpoint_with_live_source_resumes_end_to_end`
- `test_completed_checkpoint_without_target_creates_target_end_to_end`
- Existing compatible-checkpoint resume tests continue to pass.

### F-03: Multi-worker partial dump resume can use the wrong batch size

Status: **Fixed**.

Prior behavior:

- `MultiWorkerBackupManifest` stored `batch_size`.
- Worker shard headers also stored their original `batch_size`.
- On manifest resume, the executor passed the current apply call's `batch_size`
  into `multi_worker_quantize` or `async_multi_worker_quantize`.
- `_worker_quantize` and `_async_worker_quantize` used the passed `batch_size`
  when resuming a shard in `dump` phase.

Impact:

- If a multi-worker migration crashed while a worker shard was still dumping
  and the resume command used a different `batch_size`, the worker could compute
  the wrong start offset for the remaining keys.
- This could skip keys or duplicate backup batches within a shard.
- Batch resume was especially exposed because the state file did not store
  `batch_size`, and the CLI defaults to `500` unless the user passes it again.

Fix/decision:

1. When resuming from a manifest, the executor passes
   `existing_manifest.batch_size` to worker orchestration.
2. Inside worker resume, if a shard backup exists, the worker uses
   `backup.header.batch_size` for resume offsets.
3. A conflicting retry-time `batch_size` is ignored for the existing shard's
   resume offset because the checkpoint header is the authoritative record of
   how the shard was originally written.

Verification:

- `test_sync_worker_resume_uses_backup_batch_size`
- `test_sync_worker_resumes_from_dump_phase`
- `test_async_worker_loads_existing_backup`
- `test_multi_worker_manifest_resume_after_drop_end_to_end`

### F-04: Empty single-worker quantization can report a non-existent backup path

Status: **Fixed**.

Prior behavior:

- For single-worker hash datatype migrations, `report.backup.backup_paths` was
  set to the canonical backup path before keys were enumerated.
- If the source index had zero documents, no vector backup file was created.
- The report could still list the canonical backup path even though no
  `.header` or `.data` file existed.

Impact:

- The migration itself should still succeed for an empty index.
- The report could mislead users and automation into expecting backup artifacts
  that were never written.

Fix/decision:

1. Single-worker `report.backup.backup_paths` is now populated only after
   creating or loading an actual backup.
2. Hash datatype migrations over empty indexes keep `backup_paths=[]`.

Verification:

- `test_empty_quantization_reports_no_backup_path`

## Verification Performed

This audit is based on static inspection of the current implementation plus the
fix verification run after remediation:

- focused sync/async recovery suite: `43 passed`
- broad migration unit and integration suite: `308 passed`
- focused unit resume suite: `41 passed`
- ruff check and ruff format check on touched files: passed
- targeted mypy over `redisvl/migration` and `redisvl/cli/migrate.py`: passed

The fix touched the sync executor, async executor, backup metadata model,
multi-worker quantization workers, and focused resume/recovery tests.
