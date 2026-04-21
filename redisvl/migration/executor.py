from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional

if TYPE_CHECKING:
    from redisvl.migration.backup import VectorBackup

from redis.cluster import RedisCluster
from redis.exceptions import ResponseError

from redisvl.index import SearchIndex
from redisvl.migration.models import (
    MigrationBenchmarkSummary,
    MigrationPlan,
    MigrationReport,
    MigrationTimings,
    MigrationValidation,
)
from redisvl.migration.planner import MigrationPlanner
from redisvl.migration.reliability import is_same_width_dtype_conversion
from redisvl.migration.utils import (
    build_scan_match_patterns,
    current_source_matches_snapshot,
    detect_aof_enabled,
    estimate_disk_space,
    get_schema_field_path,
    normalize_keys,
    timestamp_utc,
    wait_for_index_ready,
)
from redisvl.migration.validation import MigrationValidator
from redisvl.types import SyncRedisClient
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


class MigrationExecutor:
    def __init__(self, validator: Optional[MigrationValidator] = None):
        self.validator = validator or MigrationValidator()

    def _enumerate_indexed_keys(
        self,
        client: SyncRedisClient,
        index_name: str,
        batch_size: int = 1000,
        key_separator: str = ":",
    ) -> Generator[str, None, None]:
        """Enumerate document keys using FT.AGGREGATE with SCAN fallback.

        Uses FT.AGGREGATE WITHCURSOR for efficient enumeration when the index
        has no indexing failures. Falls back to SCAN if:
        - Index has hash_indexing_failures > 0 (would miss failed docs)
        - FT.AGGREGATE command fails for any reason

        Args:
            client: Redis client
            index_name: Name of the index to enumerate
            batch_size: Number of keys per batch
            key_separator: Separator between prefix and key ID

        Yields:
            Document keys as strings
        """
        # Check for indexing failures - if any, fall back to SCAN
        try:
            info = client.ft(index_name).info()
            failures = int(info.get("hash_indexing_failures", 0) or 0)
            if failures > 0:
                logger.warning(
                    f"Index '{index_name}' has {failures} indexing failures. "
                    "Using SCAN for complete enumeration."
                )
                yield from self._enumerate_with_scan(
                    client, index_name, batch_size, key_separator
                )
                return
        except Exception as e:
            logger.warning(f"Failed to check index info: {e}. Using SCAN fallback.")
            yield from self._enumerate_with_scan(
                client, index_name, batch_size, key_separator
            )
            return

        # Try FT.AGGREGATE enumeration
        try:
            yield from self._enumerate_with_aggregate(client, index_name, batch_size)
        except ResponseError as e:
            logger.warning(
                f"FT.AGGREGATE failed: {e}. Falling back to SCAN enumeration."
            )
            yield from self._enumerate_with_scan(
                client, index_name, batch_size, key_separator
            )

    def _enumerate_with_aggregate(
        self,
        client: SyncRedisClient,
        index_name: str,
        batch_size: int = 1000,
    ) -> Generator[str, None, None]:
        """Enumerate keys using FT.AGGREGATE WITHCURSOR.

        More efficient than SCAN for sparse indexes (only returns indexed docs).
        Requires LOAD 1 __key to retrieve document keys.

        Note: FT.AGGREGATE cursors expire after ~5 minutes of idle time on the
        server side.  If the caller processes a batch slowly (e.g. performing
        heavy per-key work between reads), a subsequent FT.CURSOR READ will
        fail with a ``Cursor not found`` error.  This is caught and re-raised
        so the caller (_enumerate_indexed_keys) can fall back to SCAN.
        """
        cursor_id: Optional[int] = None

        try:
            # Initial aggregate call with LOAD 1 __key (not LOAD 0!)
            # Use MAXIDLE to extend the server-side cursor idle timeout.
            # Default Redis cursor idle timeout is 300 000 ms (5 min);
            # we request the maximum allowed (300 000 ms).
            result = client.execute_command(
                "FT.AGGREGATE",
                index_name,
                "*",
                "LOAD",
                "1",
                "__key",
                "WITHCURSOR",
                "COUNT",
                str(batch_size),
                "MAXIDLE",
                "300000",
            )

            while True:
                results_data, cursor_id = result

                # Extract keys from results (skip first element which is count)
                for item in results_data[1:]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        key = item[1]
                        yield key.decode() if isinstance(key, bytes) else str(key)

                # Check if done (cursor_id == 0)
                if cursor_id == 0:
                    break

                # Read next batch.  The cursor may have expired if the caller
                # took longer than MAXIDLE between reads — let the
                # ResponseError propagate so the caller can fall back to SCAN.
                result = client.execute_command(
                    "FT.CURSOR",
                    "READ",
                    index_name,
                    str(cursor_id),
                    "COUNT",
                    str(batch_size),
                )
        finally:
            # Clean up cursor if interrupted
            if cursor_id and cursor_id != 0:
                try:
                    client.execute_command(
                        "FT.CURSOR", "DEL", index_name, str(cursor_id)
                    )
                except Exception:
                    pass  # Cursor may have expired

    def _enumerate_with_scan(
        self,
        client: SyncRedisClient,
        index_name: str,
        batch_size: int = 1000,
        key_separator: str = ":",
    ) -> Generator[str, None, None]:
        """Enumerate keys using SCAN with prefix matching.

        Fallback method that scans all keys matching the index prefix.
        Less efficient but more complete (includes failed-to-index docs).
        """
        # Get prefix from index info
        try:
            info = client.ft(index_name).info()
            # Handle both dict and list formats from FT.INFO
            if isinstance(info, dict):
                prefixes = info.get("index_definition", {}).get("prefixes", [])
            else:
                # List format - find index_definition
                prefixes = []
                for i, item in enumerate(info):
                    if item == b"index_definition" or item == "index_definition":
                        defn = info[i + 1]
                        if isinstance(defn, dict):
                            prefixes = defn.get("prefixes", [])
                        elif isinstance(defn, list):
                            for j, d in enumerate(defn):
                                if d in (b"prefixes", "prefixes") and j + 1 < len(defn):
                                    prefixes = defn[j + 1]
                        break
            normalized_prefixes = [
                p.decode() if isinstance(p, bytes) else str(p) for p in prefixes
            ]
        except Exception as e:
            logger.warning(f"Failed to get prefix from index info: {e}")
            normalized_prefixes = []

        seen_keys: set[str] = set()
        for match_pattern in build_scan_match_patterns(
            normalized_prefixes, key_separator
        ):
            cursor = 0
            while True:
                cursor, keys = client.scan(  # type: ignore[misc]
                    cursor=cursor,
                    match=match_pattern,
                    count=batch_size,
                )
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else str(key)
                    if key_str not in seen_keys:
                        seen_keys.add(key_str)
                        yield key_str

                if cursor == 0:
                    break

    def _rename_keys(
        self,
        client: SyncRedisClient,
        keys: List[str],
        old_prefix: str,
        new_prefix: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Rename keys from old prefix to new prefix.

        Uses RENAMENX to avoid overwriting existing destination keys.
        Raises on collision to prevent silent data loss.

        For Redis Cluster, RENAME/RENAMENX fails with CROSSSLOT errors when
        old and new keys hash to different slots.  In that case we fall back
        to DUMP/RESTORE/DEL per key, which works across slots.

        Args:
            client: Redis client
            keys: List of keys to rename
            old_prefix: Current prefix (e.g., "doc:")
            new_prefix: New prefix (e.g., "article:")
            progress_callback: Optional callback(done, total)

        Returns:
            Number of keys successfully renamed
        """
        is_cluster = isinstance(client, RedisCluster)
        if is_cluster:
            return self._rename_keys_cluster(
                client, keys, old_prefix, new_prefix, progress_callback
            )
        return self._rename_keys_standalone(
            client, keys, old_prefix, new_prefix, progress_callback
        )

    def _rename_keys_standalone(
        self,
        client: SyncRedisClient,
        keys: List[str],
        old_prefix: str,
        new_prefix: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Rename keys using pipelined RENAMENX (standalone Redis only)."""
        renamed = 0
        total = len(keys)
        pipeline_size = 100
        collisions: List[str] = []
        successfully_renamed: List[tuple] = []  # (old_key, new_key) for recovery info

        for i in range(0, total, pipeline_size):
            batch = keys[i : i + pipeline_size]
            pipe = client.pipeline(transaction=False)
            batch_key_pairs: List[tuple] = []  # (old_key, new_key)

            for key in batch:
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix) :]
                else:
                    logger.warning(
                        f"Key '{key}' does not start with prefix '{old_prefix}'"
                    )
                    continue
                pipe.renamenx(key, new_key)
                batch_key_pairs.append((key, new_key))

            try:
                results = pipe.execute()
                for j, r in enumerate(results):
                    if r is True or r == 1:
                        renamed += 1
                        successfully_renamed.append(batch_key_pairs[j])
                    else:
                        collisions.append(batch_key_pairs[j][1])
            except Exception as e:
                logger.warning(f"Error in rename batch: {e}")
                raise

            # Fail fast on collisions to avoid partial renames across batches.
            if collisions:
                raise RuntimeError(
                    f"Prefix rename aborted after {renamed} successful rename(s): "
                    f"{len(collisions)} destination key(s) already exist "
                    f"(first 5: {collisions[:5]}). This would overwrite existing data. "
                    f"Remove conflicting keys or choose a different prefix. "
                    f"Note: {renamed} key(s) were already renamed from "
                    f"'{old_prefix}*' to '{new_prefix}*' and must be reversed "
                    f"manually if you want to retry."
                )

            if progress_callback:
                progress_callback(min(i + pipeline_size, total), total)

        return renamed

    def _rename_keys_cluster(
        self,
        client: SyncRedisClient,
        keys: List[str],
        old_prefix: str,
        new_prefix: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Rename keys using batched DUMP/RESTORE/DEL for Redis Cluster.

        RENAME/RENAMENX raises CROSSSLOT errors when source and destination
        hash to different slots.  DUMP/RESTORE works across slots.

        Batches DUMP+PTTL reads and RESTORE+DEL writes in groups of
        ``pipeline_size`` to reduce per-key round-trip overhead.
        """
        renamed = 0
        total = len(keys)
        pipeline_size = 100

        for i in range(0, total, pipeline_size):
            batch = keys[i : i + pipeline_size]

            # Build (key, new_key) pairs for this batch
            pairs = []
            for key in batch:
                if not key.startswith(old_prefix):
                    logger.warning(
                        "Key '%s' does not start with prefix '%s'", key, old_prefix
                    )
                    continue
                new_key = new_prefix + key[len(old_prefix) :]
                pairs.append((key, new_key))

            if not pairs:
                continue

            # Phase 1: Check destination keys don't exist (batched)
            check_pipe = client.pipeline(transaction=False)
            for _, new_key in pairs:
                check_pipe.exists(new_key)
            exists_results = check_pipe.execute()
            for (_, new_key), exists in zip(pairs, exists_results):
                if exists:
                    raise RuntimeError(
                        f"Prefix rename aborted after {renamed} successful rename(s): "
                        f"destination key '{new_key}' already exists. "
                        f"Remove conflicting keys or choose a different prefix."
                    )

            # Phase 2: DUMP + PTTL all source keys (batched — 1 RTT)
            dump_pipe = client.pipeline(transaction=False)
            for key, _ in pairs:
                dump_pipe.dump(key)
                dump_pipe.pttl(key)
            dump_results = dump_pipe.execute()

            # Phase 3: RESTORE + DEL (batched — 1 RTT)
            restore_pipe = client.pipeline(transaction=False)
            valid_pairs = []
            for idx, (key, new_key) in enumerate(pairs):
                dumped = dump_results[idx * 2]
                ttl = int(dump_results[idx * 2 + 1])  # type: ignore[arg-type]
                if dumped is None:
                    logger.warning("Key '%s' does not exist, skipping", key)
                    continue
                restore_ttl = max(ttl, 0)
                restore_pipe.restore(new_key, restore_ttl, dumped, replace=False)  # type: ignore[arg-type]
                restore_pipe.delete(key)
                valid_pairs.append((key, new_key))

            if valid_pairs:
                restore_pipe.execute()
                renamed += len(valid_pairs)

            if progress_callback:
                progress_callback(min(i + pipeline_size, total), total)

        if progress_callback:
            progress_callback(total, total)

        return renamed

    def _rename_field_in_hash(
        self,
        client: SyncRedisClient,
        keys: List[str],
        old_name: str,
        new_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Rename a field in hash documents.

        For each document:
        1. HGET key old_name -> value
        2. HSET key new_name value
        3. HDEL key old_name
        """
        renamed = 0
        total = len(keys)
        pipeline_size = 100

        for i in range(0, total, pipeline_size):
            batch = keys[i : i + pipeline_size]

            # First, get old field values AND check if destination exists
            pipe = client.pipeline(transaction=False)
            for key in batch:
                pipe.hget(key, old_name)
                pipe.hexists(key, new_name)
            raw_results = pipe.execute()
            # Interleaved: [hget_0, hexists_0, hget_1, hexists_1, ...]
            values = raw_results[0::2]
            dest_exists = raw_results[1::2]

            # Now set new field and delete old
            pipe = client.pipeline(transaction=False)
            batch_ops = 0
            for key, value, exists in zip(batch, values, dest_exists):
                if value is not None:
                    if exists:
                        logger.warning(
                            "Field '%s' already exists in key '%s'; "
                            "overwriting with value from '%s'",
                            new_name,
                            key,
                            old_name,
                        )
                    pipe.hset(key, new_name, value)
                    pipe.hdel(key, old_name)
                    batch_ops += 1

            try:
                pipe.execute()
                # Count by number of keys that had old field values,
                # not by HSET return (HSET returns 0 for existing field updates)
                renamed += batch_ops
            except Exception as e:
                logger.warning(f"Error in field rename batch: {e}")
                raise

            if progress_callback:
                progress_callback(min(i + pipeline_size, total), total)

        return renamed

    def _rename_field_in_json(
        self,
        client: SyncRedisClient,
        keys: List[str],
        old_path: str,
        new_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Rename a field in JSON documents.

        For each document:
        1. JSON.GET key old_path -> value
        2. JSON.SET key new_path value
        3. JSON.DEL key old_path
        """
        renamed = 0
        total = len(keys)
        pipeline_size = 100

        for i in range(0, total, pipeline_size):
            batch = keys[i : i + pipeline_size]

            # First, get all old field values
            pipe = client.pipeline(transaction=False)
            for key in batch:
                pipe.json().get(key, old_path)
            values = pipe.execute()

            # Now set new field and delete old
            # JSONPath GET returns results as a list; unwrap single-element
            # results to preserve the original document shape.
            # Missing paths return None or [] depending on Redis version.
            pipe = client.pipeline(transaction=False)
            batch_ops = 0
            for key, value in zip(batch, values):
                if value is None or value == []:
                    continue
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                pipe.json().set(key, new_path, value)
                pipe.json().delete(key, old_path)
                batch_ops += 1
            try:
                pipe.execute()
                # Count by number of keys that had old field values,
                # not by JSON.SET return value
                renamed += batch_ops
            except Exception as e:
                logger.warning(f"Error in JSON field rename batch: {e}")
                raise

            if progress_callback:
                progress_callback(min(i + pipeline_size, total), total)

        return renamed

    def apply(
        self,
        plan: MigrationPlan,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
        query_check_file: Optional[str] = None,
        progress_callback: Optional[Callable[[str, Optional[str]], None]] = None,
        backup_dir: Optional[str] = None,
        batch_size: int = 500,
        num_workers: int = 1,
        keep_backup: bool = False,
    ) -> MigrationReport:
        """Apply a migration plan.

        Executes the migration phases in order: enumerate → dump → drop →
        key-renames → quantize → create → index → validate.

        **Single-worker mode** (default): original vectors are read from Redis
        and backed up to disk *before* the index is dropped, then converted
        and written back after the drop. This provides the strongest
        crash-safety: if the process dies after drop, the complete backup is
        already on disk for manual rollback.

        **Multi-worker mode** (``num_workers > 1``): for performance, the dump
        and quantize phases are fused — each worker reads its key shard,
        writes the original to its backup shard, converts, and writes the
        quantized vector back, all *after* the index drop. This avoids a
        redundant full read pass but means the backup may be incomplete if
        the process crashes mid-quantize. A re-run with the same
        ``backup_dir`` will detect partial backups and resume from where it
        left off.

        Args:
            plan: The migration plan to apply (from ``MigrationPlanner.create_plan``).
            redis_url: Redis connection URL (e.g. ``"redis://localhost:6379"``).
                Required when *num_workers* > 1 so each worker can open its own
                connection. Mutually exclusive with *redis_client* for the
                multi-worker path.
            redis_client: Optional existing Redis client. Ignored when
                *num_workers* > 1.
            query_check_file: Optional YAML file containing post-migration
                queries to verify search results.
            progress_callback: Optional ``callback(step, detail)`` invoked
                during each migration phase.

                * *step*: phase name (``"enumerate"``, ``"dump"``, ``"drop"``,
                  ``"quantize"``, ``"create"``, ``"index"``, ``"validate"``)
                * *detail*: human-readable progress string
                  (e.g. ``"1000/5000 docs"``) or ``None``
            backup_dir: Directory for vector backup files. When provided,
                original vectors are saved to disk before mutation, enabling
                crash-safe resume (re-run the same command) and manual rollback.
                Required when *num_workers* > 1.  Disk usage is approximately
                ``num_docs × dims × bytes_per_element`` (e.g. ~2.9 GB for 1 M
                768-dim float32 vectors).
            batch_size: Number of keys per Redis pipeline batch (default 500).
                Controls the granularity of pipelined ``HGET``/``HSET`` calls.
                Larger batches reduce round-trips but increase per-batch memory.
                Values between 200 and 1000 are typical.
            num_workers: Number of parallel quantization workers (default 1).
                Each worker opens its own Redis connection and writes to its own
                backup-file shard. Requires *backup_dir* and *redis_url*.
                Parallelism improves throughput for high-dimensional vectors
                where conversion is CPU-bound. For low-dimensional vectors
                (≤ 256 dims), a single worker is often faster because the
                per-worker overhead (process spawning, extra connections)
                outweighs the parallelism benefit. Diminishing returns above
                4–8 workers on a single Redis instance.
            keep_backup: If ``True``, retain backup files after a successful
                migration. Default ``False`` (auto-cleanup on success). Useful
                for post-migration auditing or manual rollback.

        Returns:
            MigrationReport: Outcome including timing breakdown, validation
            results, and any warnings or manual actions.
        """
        started_at = timestamp_utc()
        started = time.perf_counter()

        report = MigrationReport(
            source_index=plan.source.index_name,
            target_index=plan.merged_target_schema["index"]["name"],
            result="failed",
            started_at=started_at,
            finished_at=started_at,
            warnings=list(plan.warnings),
        )

        if not plan.diff_classification.supported:
            report.validation.errors.extend(plan.diff_classification.blocked_reasons)
            report.manual_actions.append(
                "This change requires document migration, which is not yet supported."
            )
            report.finished_at = timestamp_utc()
            return report

        # Check if we are resuming from a backup file (post-crash).
        # New migration order: enumerate → field-renames → DUMP → DROP
        # → key-renames → QUANTIZE → CREATE.
        # The backup file stores original vectors and tracks progress.
        # If a backup file exists, we can determine exactly where the
        # previous run stopped and resume from there.
        from redisvl.migration.backup import VectorBackup

        resuming_from_backup = False
        existing_backup: Optional[VectorBackup] = None
        backup_path: Optional[str] = None

        if backup_dir:
            # Sanitize index name for filesystem with hash suffix to avoid
            # collisions between distinct names that sanitize identically
            # (e.g., "a/b" and "a:b" both become "a_b").
            safe_name = (
                plan.source.index_name.replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )
            name_hash = hashlib.sha256(plan.source.index_name.encode()).hexdigest()[:8]
            backup_path = str(
                Path(backup_dir) / f"migration_backup_{safe_name}_{name_hash}"
            )
            existing_backup = VectorBackup.load(backup_path)

            # Fallback: probe for legacy backup filename (pre-hash naming)
            if existing_backup is None:
                legacy_path = str(Path(backup_dir) / f"migration_backup_{safe_name}")
                legacy_backup = VectorBackup.load(legacy_path)
                if legacy_backup is not None:
                    logger.info(
                        "Found legacy backup at %s (pre-hash naming), using it",
                        legacy_path,
                    )
                    existing_backup = legacy_backup
                    backup_path = legacy_path

            if existing_backup is not None:
                if existing_backup.header.index_name != plan.source.index_name:
                    logger.warning(
                        "Backup index '%s' does not match plan index '%s', ignoring",
                        existing_backup.header.index_name,
                        plan.source.index_name,
                    )
                    existing_backup = None
                elif existing_backup.header.phase == "completed":
                    # Previous run completed quantization. Index may need recreating.
                    resuming_from_backup = True
                    logger.info(
                        "Backup at %s is completed; skipping to index creation",
                        backup_path,
                    )
                elif existing_backup.header.phase in ("active", "ready"):
                    # Crash after dump (possibly after drop). Resume.
                    resuming_from_backup = True
                    logger.info(
                        "Backup at %s found (phase=%s), resuming migration",
                        backup_path,
                        existing_backup.header.phase,
                    )
                elif existing_backup.header.phase == "dump":
                    # Crash during dump — index should still be alive.
                    # For simplicity, remove partial backup and restart.
                    logger.info(
                        "Partial dump found at %s, restarting dump",
                        backup_path,
                    )
                    Path(backup_path + ".header").unlink(missing_ok=True)
                    Path(backup_path + ".data").unlink(missing_ok=True)
                    existing_backup = None

        resuming = resuming_from_backup

        if not resuming:
            if not current_source_matches_snapshot(
                plan.source.index_name,
                plan.source.schema_snapshot,
                redis_url=redis_url,
                redis_client=redis_client,
            ):
                report.validation.errors.append(
                    "The current live source schema no longer matches the saved source snapshot."
                )
                report.manual_actions.append(
                    "Re-run `rvl migrate plan` to refresh the migration plan before applying."
                )
                report.finished_at = timestamp_utc()
                return report

            source_index = SearchIndex.from_existing(
                plan.source.index_name,
                redis_url=redis_url,
                redis_client=redis_client,
            )
        else:
            # Source index was dropped before crash; reconstruct from snapshot
            # to get a valid SearchIndex with a Redis client attached.
            source_index = SearchIndex.from_dict(
                plan.source.schema_snapshot,
                redis_url=redis_url,
                redis_client=redis_client,
            )

        target_index = SearchIndex.from_dict(
            plan.merged_target_schema,
            redis_url=redis_url,
            redis_client=redis_client,
        )

        enumerate_duration = 0.0
        drop_duration = 0.0
        quantize_duration = 0.0
        field_rename_duration = 0.0
        key_rename_duration = 0.0
        recreate_duration = 0.0
        indexing_duration = 0.0
        target_info: Dict[str, Any] = {}
        docs_quantized = 0
        keys_to_process: List[str] = []
        storage_type = plan.source.keyspace.storage_type

        # Check if we need to re-encode vectors for datatype changes
        datatype_changes = MigrationPlanner.get_vector_datatype_changes(
            plan.source.schema_snapshot,
            plan.merged_target_schema,
            rename_operations=plan.rename_operations,
        )

        # Check for rename operations
        rename_ops = plan.rename_operations
        has_prefix_change = rename_ops.change_prefix is not None
        has_field_renames = bool(rename_ops.rename_fields)
        needs_quantization = bool(datatype_changes) and storage_type != "json"
        needs_enumeration = needs_quantization or has_prefix_change or has_field_renames
        has_same_width_quantization = any(
            is_same_width_dtype_conversion(change["source"], change["target"])
            for change in datatype_changes.values()
        )

        if backup_dir and has_same_width_quantization:
            report.validation.errors.append(
                "Crash-safe resume is not supported for same-width datatype "
                "changes (float16<->bfloat16 or int8<->uint8)."
            )
            report.manual_actions.append(
                "Re-run without --backup-dir for same-width vector conversions, or "
                "split the migration to avoid same-width datatype changes."
            )
            report.finished_at = timestamp_utc()
            return report

        def _notify(step: str, detail: Optional[str] = None) -> None:
            if progress_callback:
                progress_callback(step, detail)

        try:
            client = source_index._redis_client
            aof_enabled = detect_aof_enabled(client)
            disk_estimate = estimate_disk_space(plan, aof_enabled=aof_enabled)
            if disk_estimate.has_quantization:
                logger.info(
                    "Disk space estimate: RDB ~%d bytes, AOF ~%d bytes, total ~%d bytes",
                    disk_estimate.rdb_snapshot_disk_bytes,
                    disk_estimate.aof_growth_bytes,
                    disk_estimate.total_new_disk_bytes,
                )
            report.disk_space_estimate = disk_estimate

            if resuming_from_backup and existing_backup is not None:
                # Resume from backup file. The backup has the key list
                # and original vectors — no enumeration or SCAN needed.
                if existing_backup.header.phase == "completed":
                    # Quantize already done, skip to CREATE
                    _notify("enumerate", "skipped (resume from backup)")
                    _notify("drop", "skipped (already dropped)")
                    _notify("quantize", "skipped (already completed)")
                elif existing_backup.header.phase in ("active", "ready"):
                    _notify("enumerate", "skipped (resume from backup)")
                    _notify("drop", "skipped (already dropped)")

                    # Remap datatype_changes if field renames happened
                    effective_changes = datatype_changes
                    if has_field_renames:
                        field_rename_map = {
                            fr.old_name: fr.new_name for fr in rename_ops.rename_fields
                        }
                        effective_changes = {
                            field_rename_map.get(k, k): v
                            for k, v in datatype_changes.items()
                        }

                    _notify("quantize", "Resuming vector re-encoding from backup...")
                    quantize_started = time.perf_counter()
                    docs_quantized = self._quantize_from_backup(
                        client=client,
                        backup=existing_backup,
                        datatype_changes=effective_changes,
                        progress_callback=lambda done, total: _notify(
                            "quantize", f"{done:,}/{total:,} docs"
                        ),
                    )
                    quantize_duration = round(time.perf_counter() - quantize_started, 3)
                    _notify(
                        "quantize",
                        f"done ({docs_quantized:,} docs in {quantize_duration}s)",
                    )

                # Key prefix renames may not have happened before the crash
                # (they run after index drop in the normal path). Re-apply
                # idempotently — RENAME is a no-op if old == new or key
                # was already renamed.
                if has_prefix_change:
                    # Collect keys from backup to know what to rename
                    resume_keys = []
                    for batch_keys, _ in existing_backup.iter_batches():
                        resume_keys.extend(batch_keys)
                    if resume_keys:
                        old_prefix = plan.source.keyspace.prefixes[0]
                        new_prefix = rename_ops.change_prefix
                        assert new_prefix is not None
                        _notify("key_rename", "Renaming keys (resume)...")
                        key_rename_started = time.perf_counter()
                        renamed_count = self._rename_keys(
                            client,
                            resume_keys,
                            old_prefix,
                            new_prefix,
                            progress_callback=lambda done, total: _notify(
                                "key_rename", f"{done:,}/{total:,} keys"
                            ),
                        )
                        key_rename_duration = round(
                            time.perf_counter() - key_rename_started, 3
                        )
                        _notify(
                            "key_rename",
                            f"done ({renamed_count:,} keys in {key_rename_duration}s)",
                        )
            else:
                # Normal (non-resume) path
                # STEP 1: Enumerate keys BEFORE any modifications
                if needs_enumeration:
                    _notify("enumerate", "Enumerating indexed documents...")
                    enumerate_started = time.perf_counter()
                    keys_to_process = list(
                        self._enumerate_indexed_keys(
                            client,
                            plan.source.index_name,
                            batch_size=1000,
                            key_separator=plan.source.keyspace.key_separator,
                        )
                    )
                    keys_to_process = normalize_keys(keys_to_process)
                    enumerate_duration = round(
                        time.perf_counter() - enumerate_started, 3
                    )
                    _notify(
                        "enumerate",
                        f"found {len(keys_to_process):,} documents ({enumerate_duration}s)",
                    )

                # STEP 2: Field renames (before dropping index)
                if has_field_renames and keys_to_process:
                    _notify("field_rename", "Renaming fields in documents...")
                    field_rename_started = time.perf_counter()
                    for field_rename in rename_ops.rename_fields:
                        if storage_type == "json":
                            old_path = get_schema_field_path(
                                plan.source.schema_snapshot, field_rename.old_name
                            )
                            new_path = get_schema_field_path(
                                plan.merged_target_schema, field_rename.new_name
                            )
                            if not old_path or not new_path or old_path == new_path:
                                continue
                            self._rename_field_in_json(
                                client,
                                keys_to_process,
                                old_path,
                                new_path,
                                progress_callback=lambda done, total: _notify(
                                    "field_rename",
                                    f"{field_rename.old_name} -> {field_rename.new_name}: {done:,}/{total:,}",
                                ),
                            )
                        else:
                            self._rename_field_in_hash(
                                client,
                                keys_to_process,
                                field_rename.old_name,
                                field_rename.new_name,
                                progress_callback=lambda done, total: _notify(
                                    "field_rename",
                                    f"{field_rename.old_name} -> {field_rename.new_name}: {done:,}/{total:,}",
                                ),
                            )
                    field_rename_duration = round(
                        time.perf_counter() - field_rename_started, 3
                    )
                    _notify("field_rename", f"done ({field_rename_duration}s)")

                # STEP 3: Dump original vectors to backup file (before drop)
                # For multi-worker, dump happens inside multi_worker_quantize
                # after the drop, so we skip the separate dump step.
                dump_duration = 0.0
                active_backup = None
                use_multi_worker = num_workers > 1 and backup_dir is not None
                if (
                    needs_quantization
                    and keys_to_process
                    and backup_path
                    and not use_multi_worker
                ):
                    # Single-worker dump before drop
                    effective_changes = datatype_changes
                    if has_field_renames:
                        field_rename_map = {
                            fr.old_name: fr.new_name for fr in rename_ops.rename_fields
                        }
                        effective_changes = {
                            field_rename_map.get(k, k): v
                            for k, v in datatype_changes.items()
                        }
                    _notify("dump", "Backing up original vectors...")
                    dump_started = time.perf_counter()
                    active_backup = self._dump_vectors(
                        client=client,
                        index_name=plan.source.index_name,
                        keys=keys_to_process,
                        datatype_changes=effective_changes,
                        backup_path=backup_path,
                        batch_size=batch_size,
                        progress_callback=lambda done, total: _notify(
                            "dump", f"{done:,}/{total:,} docs"
                        ),
                    )
                    dump_duration = round(time.perf_counter() - dump_started, 3)
                    _notify("dump", f"done ({dump_duration}s)")

                # STEP 4: Drop the index
                _notify("drop", "Dropping index definition...")
                drop_started = time.perf_counter()
                source_index.delete(drop=False)
                drop_duration = round(time.perf_counter() - drop_started, 3)
                _notify("drop", f"done ({drop_duration}s)")

                # STEP 5: Key renames (after drop, before recreate)
                if has_prefix_change and keys_to_process:
                    _notify("key_rename", "Renaming keys...")
                    key_rename_started = time.perf_counter()
                    old_prefix = plan.source.keyspace.prefixes[0]
                    new_prefix = rename_ops.change_prefix
                    assert new_prefix is not None
                    renamed_count = self._rename_keys(
                        client,
                        keys_to_process,
                        old_prefix,
                        new_prefix,
                        progress_callback=lambda done, total: _notify(
                            "key_rename", f"{done:,}/{total:,} keys"
                        ),
                    )
                    key_rename_duration = round(
                        time.perf_counter() - key_rename_started, 3
                    )
                    _notify(
                        "key_rename",
                        f"done ({renamed_count:,} keys in {key_rename_duration}s)",
                    )

                # STEP 6: Quantize vectors
                if needs_quantization and keys_to_process:
                    effective_changes = datatype_changes
                    if has_field_renames:
                        field_rename_map = {
                            fr.old_name: fr.new_name for fr in rename_ops.rename_fields
                        }
                        effective_changes = {
                            field_rename_map.get(k, k): v
                            for k, v in datatype_changes.items()
                        }

                    # Update key references if prefix changed
                    if has_prefix_change and rename_ops.change_prefix:
                        old_prefix = plan.source.keyspace.prefixes[0]
                        new_prefix = rename_ops.change_prefix
                        keys_to_process = [
                            (
                                new_prefix + k[len(old_prefix) :]
                                if k.startswith(old_prefix)
                                else k
                            )
                            for k in keys_to_process
                        ]

                    if use_multi_worker:
                        # Multi-worker path: dump + quantize in parallel
                        from redisvl.migration.quantize import multi_worker_quantize

                        if backup_dir is None:
                            raise ValueError(
                                "--backup-dir is required when using --workers > 1"
                            )
                        if redis_url is None:
                            raise ValueError(
                                "redis_url is required when using num_workers > 1"
                            )
                        _notify(
                            "quantize",
                            f"Re-encoding vectors ({num_workers} workers)...",
                        )
                        quantize_started = time.perf_counter()
                        mw_result = multi_worker_quantize(
                            redis_url=redis_url,
                            keys=keys_to_process,
                            datatype_changes=effective_changes,
                            backup_dir=backup_dir,
                            index_name=plan.source.index_name,
                            num_workers=num_workers,
                            batch_size=batch_size,
                        )
                        docs_quantized = mw_result.total_docs_quantized
                    elif active_backup:
                        # Single-worker backup path
                        _notify("quantize", "Re-encoding vectors from backup...")
                        quantize_started = time.perf_counter()
                        docs_quantized = self._quantize_from_backup(
                            client=client,
                            backup=active_backup,
                            datatype_changes=effective_changes,
                            progress_callback=lambda done, total: _notify(
                                "quantize", f"{done:,}/{total:,} docs"
                            ),
                        )
                    else:
                        # No backup dir — direct pipeline read + write
                        from redisvl.migration.quantize import (
                            convert_vectors,
                            pipeline_read_vectors,
                            pipeline_write_vectors,
                        )

                        _notify("quantize", "Re-encoding vectors...")
                        quantize_started = time.perf_counter()
                        docs_quantized = 0
                        total = len(keys_to_process)
                        for batch_start in range(0, total, batch_size):
                            batch_keys = keys_to_process[
                                batch_start : batch_start + batch_size
                            ]
                            originals = pipeline_read_vectors(
                                client, batch_keys, effective_changes
                            )
                            converted = convert_vectors(originals, effective_changes)
                            if converted:
                                pipeline_write_vectors(client, converted)
                            docs_quantized += len(converted) if converted else 0
                            if progress_callback:
                                _notify(
                                    "quantize",
                                    f"{docs_quantized:,}/{total:,} docs",
                                )
                    quantize_duration = round(time.perf_counter() - quantize_started, 3)
                    _notify(
                        "quantize",
                        f"done ({docs_quantized:,} docs in {quantize_duration}s)",
                    )
                    report.warnings.append(
                        f"Re-encoded {docs_quantized} documents for vector quantization: "
                        f"{datatype_changes}"
                    )
                elif datatype_changes and storage_type == "json":
                    _notify(
                        "quantize", "skipped (JSON vectors are re-indexed on recreate)"
                    )

            _notify("create", "Creating index with new schema...")
            recreate_started = time.perf_counter()
            target_index.create()
            recreate_duration = round(time.perf_counter() - recreate_started, 3)
            _notify("create", f"done ({recreate_duration}s)")

            _notify("index", "Waiting for re-indexing...")

            def _index_progress(indexed: int, total: int, pct: float) -> None:
                _notify("index", f"{indexed:,}/{total:,} docs ({pct:.0f}%)")

            target_info, indexing_duration = wait_for_index_ready(
                target_index, progress_callback=_index_progress
            )
            _notify("index", f"done ({indexing_duration}s)")

            _notify("validate", "Validating migration...")
            validation, target_info, validation_duration = self.validator.validate(
                plan,
                redis_url=redis_url,
                redis_client=redis_client,
                query_check_file=query_check_file,
            )
            _notify("validate", f"done ({validation_duration}s)")
            report.validation = validation
            total_duration = round(time.perf_counter() - started, 3)
            report.timings = MigrationTimings(
                total_migration_duration_seconds=total_duration,
                drop_duration_seconds=drop_duration,
                quantize_duration_seconds=(
                    quantize_duration if quantize_duration else None
                ),
                field_rename_duration_seconds=(
                    field_rename_duration if field_rename_duration else None
                ),
                key_rename_duration_seconds=(
                    key_rename_duration if key_rename_duration else None
                ),
                recreate_duration_seconds=recreate_duration,
                initial_indexing_duration_seconds=indexing_duration,
                validation_duration_seconds=validation_duration,
                downtime_duration_seconds=round(
                    drop_duration
                    + field_rename_duration
                    + key_rename_duration
                    + quantize_duration
                    + recreate_duration
                    + indexing_duration,
                    3,
                ),
            )
            report.benchmark_summary = self._build_benchmark_summary(
                plan,
                target_info,
                report.timings,
            )
            report.result = "succeeded" if not validation.errors else "failed"
            if validation.errors:
                report.manual_actions.append(
                    "Review validation errors before treating the migration as complete."
                )
        except Exception as exc:
            total_duration = round(time.perf_counter() - started, 3)
            report.timings = MigrationTimings(
                total_migration_duration_seconds=total_duration,
                drop_duration_seconds=drop_duration or None,
                quantize_duration_seconds=quantize_duration or None,
                field_rename_duration_seconds=field_rename_duration or None,
                key_rename_duration_seconds=key_rename_duration or None,
                recreate_duration_seconds=recreate_duration or None,
                initial_indexing_duration_seconds=indexing_duration or None,
                downtime_duration_seconds=(
                    round(
                        drop_duration
                        + field_rename_duration
                        + key_rename_duration
                        + quantize_duration
                        + recreate_duration
                        + indexing_duration,
                        3,
                    )
                    if drop_duration
                    or field_rename_duration
                    or key_rename_duration
                    or quantize_duration
                    or recreate_duration
                    or indexing_duration
                    else None
                ),
            )
            report.validation = MigrationValidation(
                errors=[f"Migration execution failed: {exc}"]
            )
            report.manual_actions.extend(
                [
                    "Inspect the Redis index state before retrying.",
                    "If the source index was dropped, recreate it from the saved migration plan.",
                ]
            )
        finally:
            report.finished_at = timestamp_utc()

        # Auto-cleanup backup files on success
        if backup_dir and not keep_backup and report.result == "succeeded":
            self._cleanup_backup_files(backup_dir, plan.source.index_name)

        return report

    def _cleanup_backup_files(self, backup_dir: str, index_name: str) -> None:
        """Remove backup files after successful migration.

        Only removes files with the exact extensions produced by VectorBackup
        (.header and .data), avoiding accidental deletion of unrelated files
        that happen to share the same prefix.
        """
        safe_name = index_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        name_hash = hashlib.sha256(index_name.encode()).hexdigest()[:8]
        base_prefix = f"migration_backup_{safe_name}_{name_hash}"
        # Exact suffixes written by VectorBackup
        known_suffixes = (".header", ".data")
        backup_dir_path = Path(backup_dir)

        for entry in backup_dir_path.iterdir():
            if not entry.is_file():
                continue
            name = entry.name
            # Match: base_prefix exactly, or base_prefix + shard suffix
            # e.g., migration_backup_myidx.header
            #        migration_backup_myidx_shard_0.header
            if not name.startswith(base_prefix):
                continue
            # Check that the file ends with a known extension
            if not any(name.endswith(s) for s in known_suffixes):
                continue
            # Verify the character after the prefix is either a dot or underscore
            # (prevents matching migration_backup_myidx2.header)
            remainder = name[len(base_prefix) :]
            if remainder and remainder[0] not in (".", "_"):
                continue
            try:
                entry.unlink()
                logger.debug("Removed backup file: %s", entry)
            except OSError as e:
                logger.warning("Failed to remove backup file %s: %s", entry, e)

    # ------------------------------------------------------------------
    # Two-phase quantization: dump originals → convert from backup
    # ------------------------------------------------------------------

    def _dump_vectors(
        self,
        client: Any,
        index_name: str,
        keys: List[str],
        datatype_changes: Dict[str, Dict[str, Any]],
        backup_path: str,
        batch_size: int = 500,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> "VectorBackup":
        """Phase 1: Pipeline-read original vectors and write to backup file.

        Runs BEFORE index drop — the index is still alive.
        No Redis state is modified.

        Args:
            client: Redis client
            index_name: Name of the source index
            keys: Pre-enumerated list of document keys
            datatype_changes: {field_name: {"source", "target", "dims"}}
            backup_path: Path prefix for backup files
            batch_size: Keys per pipeline batch
            progress_callback: Optional callback(docs_done, total_docs)

        Returns:
            VectorBackup in "ready" phase (dump complete)
        """
        from redisvl.migration.backup import VectorBackup
        from redisvl.migration.quantize import pipeline_read_vectors

        backup = VectorBackup.create(
            path=backup_path,
            index_name=index_name,
            fields=datatype_changes,
            batch_size=batch_size,
        )

        total = len(keys)
        for batch_idx in range(0, total, batch_size):
            batch_keys = keys[batch_idx : batch_idx + batch_size]
            originals = pipeline_read_vectors(client, batch_keys, datatype_changes)
            backup.write_batch(batch_idx // batch_size, batch_keys, originals)
            if progress_callback:
                progress_callback(min(batch_idx + batch_size, total), total)

        backup.mark_dump_complete()
        return backup

    def _quantize_from_backup(
        self,
        client: Any,
        backup: "VectorBackup",
        datatype_changes: Dict[str, Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Phase 2: Read originals from backup file, convert, pipeline-write.

        Runs AFTER index drop. Reads from local disk, not Redis.
        Tracks progress via backup header for crash-safe resume.

        Args:
            client: Redis client
            backup: VectorBackup in "ready" or "active" phase
            datatype_changes: {field_name: {"source", "target", "dims"}}
            progress_callback: Optional callback(docs_done, total_docs)

        Returns:
            Number of documents quantized
        """
        from redisvl.migration.quantize import convert_vectors, pipeline_write_vectors

        if backup.header.phase == "ready":
            backup.start_quantize()

        docs_quantized = 0
        start_batch = backup.header.quantize_completed_batches
        docs_done = start_batch * backup.header.batch_size

        for batch_idx, (batch_keys, originals) in enumerate(
            backup.iter_remaining_batches()
        ):
            actual_batch_idx = start_batch + batch_idx
            converted = convert_vectors(originals, datatype_changes)
            if converted:
                pipeline_write_vectors(client, converted)
            backup.mark_batch_quantized(actual_batch_idx)
            docs_quantized += len(batch_keys)
            docs_done += len(batch_keys)
            if progress_callback:
                total = backup.header.dump_completed_batches * backup.header.batch_size
                progress_callback(docs_done, total)

        backup.mark_complete()
        return docs_quantized

    def _build_benchmark_summary(
        self,
        plan: MigrationPlan,
        target_info: dict,
        timings: MigrationTimings,
    ) -> MigrationBenchmarkSummary:
        source_index_size = float(
            plan.source.stats_snapshot.get("vector_index_sz_mb", 0) or 0
        )
        target_index_size = float(target_info.get("vector_index_sz_mb", 0) or 0)
        source_num_docs = int(plan.source.stats_snapshot.get("num_docs", 0) or 0)
        indexed_per_second = None
        indexing_time = timings.initial_indexing_duration_seconds
        if indexing_time and indexing_time > 0:
            indexed_per_second = round(source_num_docs / indexing_time, 3)

        return MigrationBenchmarkSummary(
            documents_indexed_per_second=indexed_per_second,
            source_index_size_mb=round(source_index_size, 3),
            target_index_size_mb=round(target_index_size, 3),
            index_size_delta_mb=round(target_index_size - source_index_size, 3),
        )
