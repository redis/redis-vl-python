from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from redisvl.migration.backup import VectorBackup

from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.exceptions import ResponseError

from redisvl.index import AsyncSearchIndex
from redisvl.migration.async_planner import AsyncMigrationPlanner
from redisvl.migration.async_validation import AsyncMigrationValidator
from redisvl.migration.models import (
    MigrationBenchmarkSummary,
    MigrationPlan,
    MigrationReport,
    MigrationTimings,
    MigrationValidation,
)
from redisvl.migration.reliability import (
    BatchUndoBuffer,
    QuantizationCheckpoint,
    async_trigger_bgsave_and_wait,
    is_already_quantized,
    is_same_width_dtype_conversion,
)
from redisvl.migration.utils import (
    build_scan_match_patterns,
    estimate_disk_space,
    get_schema_field_path,
    normalize_keys,
    timestamp_utc,
)
from redisvl.redis.utils import array_to_buffer, buffer_to_array
from redisvl.types import AsyncRedisClient
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


class AsyncMigrationExecutor:
    """Async migration executor for document-preserving drop/recreate flows.

    This is the async version of MigrationExecutor. It uses AsyncSearchIndex
    and async Redis operations for better performance on large indexes,
    especially during vector quantization.
    """

    def __init__(self, validator: Optional[AsyncMigrationValidator] = None):
        self.validator = validator or AsyncMigrationValidator()

    async def _detect_aof_enabled(self, client: Any) -> bool:
        """Best-effort detection of whether AOF is enabled on the live Redis."""
        try:
            info = await client.info("persistence")
            if isinstance(info, dict) and "aof_enabled" in info:
                return bool(int(info["aof_enabled"]))
        except Exception:
            logger.debug("Could not read Redis INFO persistence for AOF detection.")

        try:
            config = await client.config_get("appendonly")
            if isinstance(config, dict):
                value = config.get("appendonly")
                if value is not None:
                    return str(value).lower() in {"yes", "1", "true", "on"}
        except Exception:
            logger.debug("Could not read Redis CONFIG GET appendonly.")

        return False

    async def _enumerate_indexed_keys(
        self,
        client: AsyncRedisClient,
        index_name: str,
        batch_size: int = 1000,
        key_separator: str = ":",
    ) -> AsyncGenerator[str, None]:
        """Async version: Enumerate document keys using FT.AGGREGATE with SCAN fallback.

        Uses FT.AGGREGATE WITHCURSOR for efficient enumeration when the index
        has no indexing failures. Falls back to SCAN if:
        - Index has hash_indexing_failures > 0 (would miss failed docs)
        - FT.AGGREGATE command fails for any reason
        """
        # Check for indexing failures - if any, fall back to SCAN
        try:
            info = await client.ft(index_name).info()
            failures = int(info.get("hash_indexing_failures", 0) or 0)
            if failures > 0:
                logger.warning(
                    f"Index '{index_name}' has {failures} indexing failures. "
                    "Using SCAN for complete enumeration."
                )
                async for key in self._enumerate_with_scan(
                    client, index_name, batch_size, key_separator
                ):
                    yield key
                return
        except Exception as e:
            logger.warning(f"Failed to check index info: {e}. Using SCAN fallback.")
            async for key in self._enumerate_with_scan(
                client, index_name, batch_size, key_separator
            ):
                yield key
            return

        # Try FT.AGGREGATE enumeration
        try:
            async for key in self._enumerate_with_aggregate(
                client, index_name, batch_size
            ):
                yield key
        except ResponseError as e:
            logger.warning(
                f"FT.AGGREGATE failed: {e}. Falling back to SCAN enumeration."
            )
            async for key in self._enumerate_with_scan(
                client, index_name, batch_size, key_separator
            ):
                yield key

    async def _enumerate_with_aggregate(
        self,
        client: AsyncRedisClient,
        index_name: str,
        batch_size: int = 1000,
    ) -> AsyncGenerator[str, None]:
        """Async version: Enumerate keys using FT.AGGREGATE WITHCURSOR.

        Uses MAXIDLE to extend the server-side cursor idle timeout (default
        ~5 min).  If the cursor still expires, the ResponseError propagates
        so the caller can fall back to SCAN.
        """
        cursor_id: Optional[int] = None

        try:
            # Initial aggregate call with LOAD 1 __key
            result = await client.execute_command(
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

                # Extract keys from results
                for item in results_data[1:]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        key = item[1]
                        yield key.decode() if isinstance(key, bytes) else str(key)

                if cursor_id == 0:
                    break

                result = await client.execute_command(
                    "FT.CURSOR",
                    "READ",
                    index_name,
                    str(cursor_id),
                    "COUNT",
                    str(batch_size),
                )
        finally:
            if cursor_id and cursor_id != 0:
                try:
                    await client.execute_command(
                        "FT.CURSOR", "DEL", index_name, str(cursor_id)
                    )
                except Exception:
                    pass

    async def _enumerate_with_scan(
        self,
        client: AsyncRedisClient,
        index_name: str,
        batch_size: int = 1000,
        key_separator: str = ":",
    ) -> AsyncGenerator[str, None]:
        """Async version: Enumerate keys using SCAN with prefix matching."""
        # Get prefix from index info
        try:
            info = await client.ft(index_name).info()
            if isinstance(info, dict):
                prefixes = info.get("index_definition", {}).get("prefixes", [])
            else:
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
            cursor: int = 0
            while True:
                cursor, keys = await client.scan(
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

    async def _rename_keys(
        self,
        client: AsyncRedisClient,
        keys: List[str],
        old_prefix: str,
        new_prefix: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Async version: Rename keys from old prefix to new prefix.

        Uses RENAMENX for standalone Redis.  For Redis Cluster, falls back
        to DUMP/RESTORE/DEL to avoid CROSSSLOT errors.
        """
        is_cluster = isinstance(client, AsyncRedisCluster)
        if is_cluster:
            return await self._rename_keys_cluster(
                client, keys, old_prefix, new_prefix, progress_callback
            )
        return await self._rename_keys_standalone(
            client, keys, old_prefix, new_prefix, progress_callback
        )

    async def _rename_keys_standalone(
        self,
        client: AsyncRedisClient,
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
        successfully_renamed: List[tuple] = []

        for i in range(0, total, pipeline_size):
            batch = keys[i : i + pipeline_size]
            pipe = client.pipeline(transaction=False)
            batch_key_pairs: List[tuple] = []

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
                results = await pipe.execute()
                for j, r in enumerate(results):
                    if r is True or r == 1:
                        renamed += 1
                        successfully_renamed.append(batch_key_pairs[j])
                    else:
                        collisions.append(batch_key_pairs[j][1])
            except Exception as e:
                logger.warning(f"Error in rename batch: {e}")
                raise

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

    async def _rename_keys_cluster(
        self,
        client: AsyncRedisClient,
        keys: List[str],
        old_prefix: str,
        new_prefix: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Rename keys using DUMP/RESTORE/DEL for Redis Cluster.

        RENAME/RENAMENX raises CROSSSLOT errors when source and destination
        hash to different slots.  DUMP/RESTORE works across slots.
        """
        renamed = 0
        total = len(keys)

        for idx, key in enumerate(keys):
            if not key.startswith(old_prefix):
                logger.warning(f"Key '{key}' does not start with prefix '{old_prefix}'")
                continue
            new_key = new_prefix + key[len(old_prefix) :]

            if await client.exists(new_key):
                raise RuntimeError(
                    f"Prefix rename aborted after {renamed} successful rename(s): "
                    f"destination key '{new_key}' already exists. "
                    f"Remove conflicting keys or choose a different prefix."
                )

            dumped = await client.dump(key)
            if dumped is None:
                logger.warning(f"Key '{key}' does not exist, skipping")
                continue
            ttl = await client.pttl(key)
            restore_ttl = max(ttl, 0)
            await client.restore(new_key, restore_ttl, dumped, replace=False)
            await client.delete(key)
            renamed += 1

            if progress_callback and (idx + 1) % 100 == 0:
                progress_callback(idx + 1, total)

        if progress_callback:
            progress_callback(total, total)

        return renamed

    async def _rename_field_in_hash(
        self,
        client: AsyncRedisClient,
        keys: List[str],
        old_name: str,
        new_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Async version: Rename a field in hash documents."""
        renamed = 0
        total = len(keys)
        pipeline_size = 100

        for i in range(0, total, pipeline_size):
            batch = keys[i : i + pipeline_size]

            # Get old field values AND check if destination exists
            pipe = client.pipeline(transaction=False)
            for key in batch:
                pipe.hget(key, old_name)
                pipe.hexists(key, new_name)
            raw_results = await pipe.execute()
            # Interleaved: [hget_0, hexists_0, hget_1, hexists_1, ...]
            values = raw_results[0::2]
            dest_exists = raw_results[1::2]

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
                await pipe.execute()
                # Count by number of keys that had old field values,
                # not by HSET return (HSET returns 0 for existing field updates)
                renamed += batch_ops
            except Exception as e:
                logger.warning(f"Error in field rename batch: {e}")
                raise

            if progress_callback:
                progress_callback(min(i + pipeline_size, total), total)

        return renamed

    async def _rename_field_in_json(
        self,
        client: AsyncRedisClient,
        keys: List[str],
        old_path: str,
        new_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Async version: Rename a field in JSON documents."""
        renamed = 0
        total = len(keys)
        pipeline_size = 100

        for i in range(0, total, pipeline_size):
            batch = keys[i : i + pipeline_size]

            pipe = client.pipeline(transaction=False)
            for key in batch:
                pipe.json().get(key, old_path)
            values = await pipe.execute()

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
                await pipe.execute()
                # Count by number of keys that had old field values,
                # not by JSON.SET return value
                renamed += batch_ops
            except Exception as e:
                logger.warning(f"Error in JSON field rename batch: {e}")
                raise

            if progress_callback:
                progress_callback(min(i + pipeline_size, total), total)

        return renamed

    async def apply(
        self,
        plan: MigrationPlan,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedisClient] = None,
        query_check_file: Optional[str] = None,
        progress_callback: Optional[Callable[[str, Optional[str]], None]] = None,
        checkpoint_path: Optional[str] = None,
        backup_dir: Optional[str] = None,
        batch_size: int = 500,
    ) -> MigrationReport:
        """Apply a migration plan asynchronously.

        Args:
            plan: The migration plan to apply.
            redis_url: Redis connection URL.
            redis_client: Optional existing async Redis client.
            query_check_file: Optional file with query checks.
            progress_callback: Optional callback(step, detail) for progress updates.
            checkpoint_path: Deprecated — use backup_dir instead.
            backup_dir: Directory for vector backup files.
            batch_size: Keys per pipeline batch (default 500).
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
        from redisvl.migration.backup import VectorBackup

        resuming_from_backup = False
        existing_backup: Optional[VectorBackup] = None
        backup_path: Optional[str] = None

        if backup_dir:
            safe_name = (
                plan.source.index_name.replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )
            backup_path = str(Path(backup_dir) / f"migration_backup_{safe_name}")
            existing_backup = VectorBackup.load(backup_path)

            if existing_backup is not None:
                if existing_backup.header.index_name != plan.source.index_name:
                    existing_backup = None
                elif existing_backup.header.phase == "completed":
                    resuming_from_backup = True
                elif existing_backup.header.phase in ("active", "ready"):
                    resuming_from_backup = True
                elif existing_backup.header.phase == "dump":
                    Path(backup_path + ".header").unlink(missing_ok=True)
                    Path(backup_path + ".data").unlink(missing_ok=True)
                    existing_backup = None

        # Legacy checkpoint support
        resuming_from_checkpoint = False
        if checkpoint_path and not backup_dir:
            existing_checkpoint = QuantizationCheckpoint.load(checkpoint_path)
            if existing_checkpoint is not None:
                if existing_checkpoint.index_name != plan.source.index_name:
                    Path(checkpoint_path).unlink(missing_ok=True)
                elif existing_checkpoint.status == "completed":
                    source_still_exists = (
                        await self._async_current_source_matches_snapshot(
                            plan.source.index_name,
                            plan.source.schema_snapshot,
                            redis_url=redis_url,
                            redis_client=redis_client,
                        )
                    )
                    if source_still_exists:
                        Path(checkpoint_path).unlink(missing_ok=True)
                    else:
                        resuming_from_checkpoint = True
                else:
                    resuming_from_checkpoint = True

        resuming = resuming_from_backup or resuming_from_checkpoint

        if not resuming:
            if not await self._async_current_source_matches_snapshot(
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

            source_index = await AsyncSearchIndex.from_existing(
                plan.source.index_name,
                redis_url=redis_url,
                redis_client=redis_client,
            )
        else:
            # Source index was dropped before crash; reconstruct from snapshot
            # to get a valid AsyncSearchIndex with a Redis client attached.
            source_index = AsyncSearchIndex.from_dict(
                plan.source.schema_snapshot,
                redis_url=redis_url,
                redis_client=redis_client,
            )

        target_index = AsyncSearchIndex.from_dict(
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

        datatype_changes = AsyncMigrationPlanner.get_vector_datatype_changes(
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

        if checkpoint_path and has_same_width_quantization:
            report.validation.errors.append(
                "Crash-safe resume is not supported for same-width datatype "
                "changes (float16<->bfloat16 or int8<->uint8)."
            )
            report.manual_actions.append(
                "Re-run without --resume for same-width vector conversions, or "
                "split the migration to avoid same-width datatype changes."
            )
            report.finished_at = timestamp_utc()
            return report

        def _notify(step: str, detail: Optional[str] = None) -> None:
            if progress_callback:
                progress_callback(step, detail)

        try:
            client = await source_index._get_client()
            if client is None:
                raise ValueError("Failed to get Redis client from source index")
            aof_enabled = await self._detect_aof_enabled(client)
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
                if existing_backup.header.phase == "completed":
                    _notify("enumerate", "skipped (resume from backup)")
                    _notify("drop", "skipped (already dropped)")
                    _notify("quantize", "skipped (already completed)")
                elif existing_backup.header.phase in ("active", "ready"):
                    _notify("enumerate", "skipped (resume from backup)")
                    _notify("drop", "skipped (already dropped)")
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
                    docs_quantized = await self._quantize_from_backup(
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
            elif resuming_from_checkpoint:
                # Legacy checkpoint resume path
                if needs_enumeration:
                    _notify("enumerate", "Enumerating documents via SCAN (resume)...")
                    enumerate_started = time.perf_counter()
                    prefixes = list(plan.source.keyspace.prefixes)
                    if has_prefix_change and rename_ops.change_prefix:
                        prefixes = [rename_ops.change_prefix]
                    seen_keys: set[str] = set()
                    for match_pattern in build_scan_match_patterns(
                        prefixes, plan.source.keyspace.key_separator
                    ):
                        cursor_val: int = 0
                        while True:
                            cursor_val, scanned = await client.scan(  # type: ignore[misc]
                                cursor=cursor_val,
                                match=match_pattern,
                                count=1000,
                            )
                            for k in scanned:
                                key = k.decode() if isinstance(k, bytes) else str(k)
                                if key not in seen_keys:
                                    seen_keys.add(key)
                                    keys_to_process.append(key)
                            if cursor_val == 0:
                                break
                    keys_to_process = normalize_keys(keys_to_process)
                    enumerate_duration = round(
                        time.perf_counter() - enumerate_started, 3
                    )
                    _notify(
                        "enumerate",
                        f"found {len(keys_to_process):,} documents ({enumerate_duration}s)",
                    )
                _notify("drop", "skipped (already dropped)")
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
                    _notify("quantize", "Re-encoding vectors...")
                    quantize_started = time.perf_counter()
                    docs_quantized = await self._async_quantize_vectors(
                        source_index,
                        effective_changes,
                        keys_to_process,
                        progress_callback=lambda done, total: _notify(
                            "quantize", f"{done:,}/{total:,} docs"
                        ),
                        checkpoint_path=checkpoint_path,
                    )
                    quantize_duration = round(time.perf_counter() - quantize_started, 3)
                    _notify(
                        "quantize",
                        f"done ({docs_quantized:,} docs in {quantize_duration}s)",
                    )
            else:
                # Normal (non-resume) path
                if needs_enumeration:
                    _notify("enumerate", "Enumerating indexed documents...")
                    enumerate_started = time.perf_counter()
                    keys_to_process = [
                        key
                        async for key in self._enumerate_indexed_keys(
                            client,
                            plan.source.index_name,
                            batch_size=1000,
                            key_separator=plan.source.keyspace.key_separator,
                        )
                    ]
                    keys_to_process = normalize_keys(keys_to_process)
                    enumerate_duration = round(
                        time.perf_counter() - enumerate_started, 3
                    )
                    _notify(
                        "enumerate",
                        f"found {len(keys_to_process):,} documents ({enumerate_duration}s)",
                    )

                # Field renames
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
                            await self._rename_field_in_json(
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
                            await self._rename_field_in_hash(
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

                # Dump original vectors to backup file (before drop)
                active_backup = None
                if needs_quantization and keys_to_process and backup_path:
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
                    active_backup = await self._dump_vectors(
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

                # Drop the index
                _notify("drop", "Dropping index definition...")
                drop_started = time.perf_counter()
                await source_index.delete(drop=False)
                drop_duration = round(time.perf_counter() - drop_started, 3)
                _notify("drop", f"done ({drop_duration}s)")

                # Key renames
                if has_prefix_change and keys_to_process:
                    _notify("key_rename", "Renaming keys...")
                    key_rename_started = time.perf_counter()
                    old_prefix = plan.source.keyspace.prefixes[0]
                    new_prefix = rename_ops.change_prefix
                    assert new_prefix is not None
                    renamed_count = await self._rename_keys(
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

                # Quantize vectors
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
                    if active_backup:
                        _notify("quantize", "Re-encoding vectors from backup...")
                        quantize_started = time.perf_counter()
                        docs_quantized = await self._quantize_from_backup(
                            client=client,
                            backup=active_backup,
                            datatype_changes=effective_changes,
                            progress_callback=lambda done, total: _notify(
                                "quantize", f"{done:,}/{total:,} docs"
                            ),
                        )
                    else:
                        _notify("quantize", "Re-encoding vectors...")
                        quantize_started = time.perf_counter()
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
                            keys_to_process = normalize_keys(keys_to_process)
                        docs_quantized = await self._async_quantize_vectors(
                            source_index,
                            effective_changes,
                            keys_to_process,
                            progress_callback=lambda done, total: _notify(
                                "quantize", f"{done:,}/{total:,} docs"
                            ),
                            checkpoint_path=checkpoint_path,
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
            await target_index.create()
            recreate_duration = round(time.perf_counter() - recreate_started, 3)
            _notify("create", f"done ({recreate_duration}s)")

            _notify("index", "Waiting for re-indexing...")

            def _index_progress(indexed: int, total: int, pct: float) -> None:
                _notify("index", f"{indexed:,}/{total:,} docs ({pct:.0f}%)")

            target_info, indexing_duration = await self._async_wait_for_index_ready(
                target_index, progress_callback=_index_progress
            )
            _notify("index", f"done ({indexing_duration}s)")

            _notify("validate", "Validating migration...")
            validation, target_info, validation_duration = (
                await self.validator.validate(
                    plan,
                    redis_url=redis_url,
                    redis_client=redis_client,
                    query_check_file=query_check_file,
                )
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

        return report

    # ------------------------------------------------------------------
    # Two-phase quantization: dump originals → convert from backup
    # ------------------------------------------------------------------

    async def _dump_vectors(
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

        Async version. Runs BEFORE index drop.
        """
        from redisvl.migration.backup import VectorBackup

        backup = VectorBackup.create(
            path=backup_path,
            index_name=index_name,
            fields=datatype_changes,
            batch_size=batch_size,
        )

        total = len(keys)
        field_names = list(datatype_changes.keys())

        for batch_start in range(0, total, batch_size):
            batch_keys = keys[batch_start : batch_start + batch_size]

            # Pipelined async reads
            pipe = client.pipeline(transaction=False)
            call_order: List[tuple] = []
            for key in batch_keys:
                for field_name in field_names:
                    pipe.hget(key, field_name)
                    call_order.append((key, field_name))
            results = await pipe.execute()

            # Reassemble
            originals: Dict[str, Dict[str, bytes]] = {}
            for (key, field_name), value in zip(call_order, results):
                if value is not None:
                    if key not in originals:
                        originals[key] = {}
                    originals[key][field_name] = value

            backup.write_batch(batch_start // batch_size, batch_keys, originals)
            if progress_callback:
                progress_callback(min(batch_start + batch_size, total), total)

        backup.mark_dump_complete()
        return backup

    async def _quantize_from_backup(
        self,
        client: Any,
        backup: "VectorBackup",
        datatype_changes: Dict[str, Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Phase 2: Read originals from backup file, convert, pipeline-write.

        Async version. Runs AFTER index drop.
        """
        from redisvl.migration.quantize import convert_vectors

        if backup.header.phase == "ready":
            backup.start_quantize()

        docs_quantized = 0
        docs_done = backup.header.quantize_completed_batches * backup.header.batch_size

        for batch_idx, (batch_keys, originals) in enumerate(
            backup.iter_remaining_batches()
        ):
            actual_batch_idx = backup.header.quantize_completed_batches + batch_idx
            converted = convert_vectors(originals, datatype_changes)
            if converted:
                pipe = client.pipeline(transaction=False)
                for key, fields in converted.items():
                    for field_name, data in fields.items():
                        pipe.hset(key, field_name, data)
                await pipe.execute()
            backup.mark_batch_quantized(actual_batch_idx)
            docs_quantized += len(batch_keys)
            docs_done += len(batch_keys)
            if progress_callback:
                total = backup.header.dump_completed_batches * backup.header.batch_size
                progress_callback(docs_done, total)

        backup.mark_complete()
        return docs_quantized

    async def _async_quantize_vectors(
        self,
        source_index: AsyncSearchIndex,
        datatype_changes: Dict[str, Dict[str, Any]],
        keys: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        checkpoint_path: Optional[str] = None,
    ) -> int:
        """Re-encode vectors in documents for datatype changes (quantization).

        Uses pre-enumerated keys (from _enumerate_indexed_keys) to process
        only the documents that were in the index, avoiding full keyspace scan.
        Includes idempotent skip (already-quantized vectors), bounded undo
        buffer for per-batch rollback, and optional checkpointing for resume.

        Args:
            source_index: The source AsyncSearchIndex (already dropped but client available)
            datatype_changes: Dict mapping field_name -> {"source", "target", "dims"}
            keys: Pre-enumerated list of document keys to process
            progress_callback: Optional callback(docs_done, total_docs)
            checkpoint_path: Optional path for checkpoint file (enables resume)

        Returns:
            Number of documents quantized
        """
        client = await source_index._get_client()
        if client is None:
            raise ValueError("Failed to get Redis client from source index")

        total_keys = len(keys)
        docs_processed = 0
        docs_quantized = 0
        skipped = 0
        batch_size = 500

        # Load or create checkpoint for resume support
        checkpoint: Optional[QuantizationCheckpoint] = None
        if checkpoint_path:
            checkpoint = QuantizationCheckpoint.load(checkpoint_path)
            if checkpoint:
                # Validate checkpoint matches current migration BEFORE
                # checking completion status to avoid skipping quantization
                # for an unrelated completed checkpoint.
                if checkpoint.index_name != source_index.name:
                    raise ValueError(
                        f"Checkpoint index '{checkpoint.index_name}' does not match "
                        f"source index '{source_index.name}'. "
                        f"Use the correct checkpoint file or remove it to start fresh."
                    )
                # Skip if checkpoint shows a completed migration
                if checkpoint.status == "completed":
                    logger.info(
                        "Checkpoint already marked as completed for index '%s'. "
                        "Skipping quantization. Remove the checkpoint file to force re-run.",
                        checkpoint.index_name,
                    )
                    return 0
                if checkpoint.total_keys != total_keys:
                    if checkpoint.processed_keys:
                        current_keys = set(keys)
                        missing_processed = [
                            key
                            for key in checkpoint.processed_keys
                            if key not in current_keys
                        ]
                        if missing_processed or total_keys < checkpoint.total_keys:
                            raise ValueError(
                                f"Checkpoint total_keys={checkpoint.total_keys} does not match "
                                f"the current key set ({total_keys}). "
                                "Use the correct checkpoint file or remove it to start fresh."
                            )
                        logger.warning(
                            "Checkpoint total_keys=%d differs from current key set size=%d. "
                            "Proceeding because all legacy processed keys are present.",
                            checkpoint.total_keys,
                            total_keys,
                        )
                    else:
                        raise ValueError(
                            f"Checkpoint total_keys={checkpoint.total_keys} does not match "
                            f"the current key set ({total_keys}). "
                            "Use the correct checkpoint file or remove it to start fresh."
                        )
                remaining = checkpoint.get_remaining_keys(keys)
                logger.info(
                    "Resuming from checkpoint: %d/%d keys already processed",
                    total_keys - len(remaining),
                    total_keys,
                )
                docs_processed = total_keys - len(remaining)
                keys = remaining
                total_keys_for_progress = total_keys
            else:
                checkpoint = QuantizationCheckpoint(
                    index_name=source_index.name,
                    total_keys=total_keys,
                    checkpoint_path=checkpoint_path,
                )
                checkpoint.save()
                total_keys_for_progress = total_keys
        else:
            total_keys_for_progress = total_keys

        remaining_keys = len(keys)

        for i in range(0, remaining_keys, batch_size):
            batch = keys[i : i + batch_size]
            pipe = client.pipeline(transaction=False)
            undo = BatchUndoBuffer()
            keys_updated_in_batch: set[str] = set()

            try:
                for key in batch:
                    for field_name, change in datatype_changes.items():
                        field_data: bytes | None = await client.hget(key, field_name)  # type: ignore[misc,assignment]
                        if not field_data:
                            continue

                        # Idempotent: skip if already converted to target dtype
                        dims = change.get("dims", 0)
                        if dims and is_already_quantized(
                            field_data, dims, change["source"], change["target"]
                        ):
                            skipped += 1
                            continue

                        undo.store(key, field_name, field_data)
                        array = buffer_to_array(field_data, change["source"])
                        new_bytes = array_to_buffer(array, change["target"])
                        pipe.hset(key, field_name, new_bytes)  # type: ignore[arg-type]
                        keys_updated_in_batch.add(key)

                if keys_updated_in_batch:
                    await pipe.execute()
            except Exception:
                logger.warning(
                    "Batch %d failed, rolling back %d entries",
                    i // batch_size,
                    undo.size,
                )
                rollback_pipe = client.pipeline()
                await undo.async_rollback(rollback_pipe)
                if checkpoint:
                    checkpoint.save()
                raise
            finally:
                undo.clear()

            docs_quantized += len(keys_updated_in_batch)
            docs_processed += len(batch)

            if checkpoint:
                # Record all keys in batch (including skipped) so they
                # are not re-scanned on resume
                checkpoint.record_batch(batch)
                checkpoint.save()

            if progress_callback:
                progress_callback(docs_processed, total_keys_for_progress)

        if checkpoint:
            checkpoint.mark_complete()
            checkpoint.save()

        if skipped:
            logger.info("Skipped %d already-quantized vector fields", skipped)
        logger.info(
            "Quantized %d documents across %d fields",
            docs_quantized,
            len(datatype_changes),
        )
        return docs_quantized

    async def _async_wait_for_index_ready(
        self,
        index: AsyncSearchIndex,
        *,
        timeout_seconds: int = 1800,
        poll_interval_seconds: float = 0.5,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> tuple[Dict[str, Any], float]:
        """Wait for index to finish indexing all documents (async version)."""
        start = time.perf_counter()
        deadline = start + timeout_seconds
        latest_info = await index.info()

        stable_ready_checks: Optional[int] = None
        while time.perf_counter() < deadline:
            ready = False
            latest_info = await index.info()
            indexing = latest_info.get("indexing")
            percent_indexed = latest_info.get("percent_indexed")

            if percent_indexed is not None or indexing is not None:
                pct = float(percent_indexed) if percent_indexed is not None else None
                is_indexing = bool(indexing)
                if pct is not None:
                    ready = pct >= 1.0 and not is_indexing
                else:
                    # percent_indexed missing but indexing flag present:
                    # treat as ready when indexing flag is falsy (0 / False).
                    ready = not is_indexing
                if progress_callback:
                    total_docs = int(latest_info.get("num_docs", 0))
                    display_pct = pct if pct is not None else (1.0 if ready else 0.0)
                    indexed_docs = int(total_docs * display_pct)
                    progress_callback(indexed_docs, total_docs, display_pct * 100)
            else:
                current_docs = latest_info.get("num_docs")
                if current_docs is None:
                    ready = True
                else:
                    if stable_ready_checks is None:
                        stable_ready_checks = int(current_docs)
                        await asyncio.sleep(poll_interval_seconds)
                        continue
                    current = int(current_docs)
                    if current == stable_ready_checks:
                        ready = True
                    else:
                        # num_docs changed; update baseline and keep waiting
                        stable_ready_checks = current

            if ready:
                return latest_info, round(time.perf_counter() - start, 3)

            await asyncio.sleep(poll_interval_seconds)

        raise TimeoutError(
            f"Index {index.schema.index.name} did not become ready within {timeout_seconds} seconds"
        )

    async def _async_current_source_matches_snapshot(
        self,
        index_name: str,
        expected_schema: Dict[str, Any],
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedisClient] = None,
    ) -> bool:
        """Check if current source schema matches the snapshot (async version)."""
        from redisvl.migration.utils import schemas_equal

        try:
            current_index = await AsyncSearchIndex.from_existing(
                index_name,
                redis_url=redis_url,
                redis_client=redis_client,
            )
        except Exception:
            # Index no longer exists (e.g. already dropped during migration)
            return False
        return schemas_equal(current_index.schema.to_dict(), expected_schema)

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
