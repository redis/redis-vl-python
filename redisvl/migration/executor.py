from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Generator, List, Optional

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
from redisvl.migration.reliability import (
    BatchUndoBuffer,
    QuantizationCheckpoint,
    is_already_quantized,
    trigger_bgsave_and_wait,
)
from redisvl.migration.utils import (
    current_source_matches_snapshot,
    estimate_disk_space,
    timestamp_utc,
    wait_for_index_ready,
)
from redisvl.migration.validation import MigrationValidator
from redisvl.redis.utils import array_to_buffer, buffer_to_array
from redisvl.types import SyncRedisClient

logger = logging.getLogger(__name__)


class MigrationExecutor:
    def __init__(self, validator: Optional[MigrationValidator] = None):
        self.validator = validator or MigrationValidator()

    def _enumerate_indexed_keys(
        self,
        client: SyncRedisClient,
        index_name: str,
        batch_size: int = 1000,
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
                yield from self._enumerate_with_scan(client, index_name, batch_size)
                return
        except Exception as e:
            logger.warning(f"Failed to check index info: {e}. Using SCAN fallback.")
            yield from self._enumerate_with_scan(client, index_name, batch_size)
            return

        # Try FT.AGGREGATE enumeration
        try:
            yield from self._enumerate_with_aggregate(client, index_name, batch_size)
        except ResponseError as e:
            logger.warning(
                f"FT.AGGREGATE failed: {e}. Falling back to SCAN enumeration."
            )
            yield from self._enumerate_with_scan(client, index_name, batch_size)

    def _enumerate_with_aggregate(
        self,
        client: SyncRedisClient,
        index_name: str,
        batch_size: int = 1000,
    ) -> Generator[str, None, None]:
        """Enumerate keys using FT.AGGREGATE WITHCURSOR.

        More efficient than SCAN for sparse indexes (only returns indexed docs).
        Requires LOAD 1 __key to retrieve document keys.
        """
        cursor_id: Optional[int] = None

        try:
            # Initial aggregate call with LOAD 1 __key (not LOAD 0!)
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

                # Read next batch
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
            prefix = prefixes[0] if prefixes else ""
            if isinstance(prefix, bytes):
                prefix = prefix.decode()
        except Exception as e:
            logger.warning(f"Failed to get prefix from index info: {e}")
            prefix = ""

        if not prefix:
            logger.warning("No prefix found for index, SCAN may return unexpected keys")

        cursor = 0
        while True:
            cursor, keys = client.scan(  # type: ignore[misc]
                cursor=cursor,
                match=f"{prefix}*" if prefix else "*",
                count=batch_size,
            )
            for key in keys:
                yield key.decode() if isinstance(key, bytes) else str(key)

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

        Args:
            client: Redis client
            keys: List of keys to rename
            old_prefix: Current prefix (e.g., "doc:")
            new_prefix: New prefix (e.g., "article:")
            progress_callback: Optional callback(done, total)

        Returns:
            Number of keys successfully renamed
        """
        renamed = 0
        total = len(keys)
        pipeline_size = 100  # Process in batches

        for i in range(0, total, pipeline_size):
            batch = keys[i : i + pipeline_size]
            pipe = client.pipeline(transaction=False)

            for key in batch:
                # Compute new key name
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix) :]
                else:
                    # Key doesn't match expected prefix, skip
                    logger.warning(
                        f"Key '{key}' does not start with prefix '{old_prefix}'"
                    )
                    continue
                pipe.rename(key, new_key)

            try:
                results = pipe.execute()
                renamed += sum(1 for r in results if r is True or r == "OK")
            except Exception as e:
                logger.warning(f"Error in rename batch: {e}")

            if progress_callback:
                progress_callback(min(i + pipeline_size, total), total)

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

            # First, get all old field values
            pipe = client.pipeline(transaction=False)
            for key in batch:
                pipe.hget(key, old_name)
            values = pipe.execute()

            # Now set new field and delete old
            pipe = client.pipeline(transaction=False)
            for key, value in zip(batch, values):
                if value is not None:
                    pipe.hset(key, new_name, value)
                    pipe.hdel(key, old_name)

            try:
                results = pipe.execute()
                # Count successful HSET operations (every other result)
                renamed += sum(1 for j, r in enumerate(results) if j % 2 == 0 and r)
            except Exception as e:
                logger.warning(f"Error in field rename batch: {e}")

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
            pipe = client.pipeline(transaction=False)
            for key, value in zip(batch, values):
                if value is not None:
                    pipe.json().set(key, new_path, value)
                    pipe.json().delete(key, old_path)

            try:
                results = pipe.execute()
                # Count successful JSON.SET operations (every other result)
                renamed += sum(1 for j, r in enumerate(results) if j % 2 == 0 and r)
            except Exception as e:
                logger.warning(f"Error in JSON field rename batch: {e}")

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
        checkpoint_path: Optional[str] = None,
    ) -> MigrationReport:
        """Apply a migration plan.

        Args:
            plan: The migration plan to apply.
            redis_url: Redis connection URL.
            redis_client: Optional existing Redis client.
            query_check_file: Optional file with query checks.
            progress_callback: Optional callback(step, detail) for progress updates.
                step: Current step name (e.g., "drop", "quantize", "create", "index", "validate")
                detail: Optional detail string (e.g., "1000/5000 docs (20%)")
            checkpoint_path: Optional path for quantization checkpoint file.
                When provided, enables crash-safe resume for vector re-encoding.
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

        # Check if we need to re-encode vectors for datatype changes
        datatype_changes = MigrationPlanner.get_vector_datatype_changes(
            plan.source.schema_snapshot, plan.merged_target_schema
        )

        # Check for rename operations
        rename_ops = plan.rename_operations
        has_prefix_change = bool(rename_ops.change_prefix)
        has_field_renames = bool(rename_ops.rename_fields)
        needs_enumeration = datatype_changes or has_prefix_change or has_field_renames

        # Compute disk space estimate before any mutations
        disk_estimate = estimate_disk_space(plan)
        if disk_estimate.has_quantization:
            logger.info(
                "Disk space estimate: RDB ~%d bytes, AOF ~%d bytes, total ~%d bytes",
                disk_estimate.rdb_snapshot_disk_bytes,
                disk_estimate.aof_growth_bytes,
                disk_estimate.total_new_disk_bytes,
            )
        report.disk_space_estimate = disk_estimate

        def _notify(step: str, detail: Optional[str] = None) -> None:
            if progress_callback:
                progress_callback(step, detail)

        try:
            client = source_index._redis_client
            storage_type = plan.source.keyspace.storage_type

            # STEP 1: Enumerate keys BEFORE any modifications
            # Needed for: quantization, prefix change, or field renames
            if needs_enumeration:
                _notify("enumerate", "Enumerating indexed documents...")
                enumerate_started = time.perf_counter()
                keys_to_process = list(
                    self._enumerate_indexed_keys(
                        client, plan.source.index_name, batch_size=1000
                    )
                )
                enumerate_duration = round(time.perf_counter() - enumerate_started, 3)
                _notify(
                    "enumerate",
                    f"found {len(keys_to_process):,} documents ({enumerate_duration}s)",
                )

            # BGSAVE safety net: snapshot data before mutations begin
            if needs_enumeration and keys_to_process:
                _notify("bgsave", "Triggering BGSAVE safety snapshot...")
                try:
                    trigger_bgsave_and_wait(client)
                    _notify("bgsave", "done")
                except Exception as e:
                    logger.warning("BGSAVE safety snapshot failed: %s", e)
                    _notify("bgsave", f"skipped ({e})")

            # STEP 2: Field renames (before dropping index, while docs are still indexed)
            if has_field_renames and keys_to_process:
                _notify("field_rename", "Renaming fields in documents...")
                field_rename_started = time.perf_counter()
                for field_rename in rename_ops.rename_fields:
                    if storage_type == "json":
                        # For JSON, use JSON paths
                        old_path = f"$.{field_rename.old_name}"
                        new_path = f"$.{field_rename.new_name}"
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
                        # For HASH, use field names directly
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

            # STEP 3: Drop the index
            _notify("drop", "Dropping index definition...")
            drop_started = time.perf_counter()
            source_index.delete(drop=False)
            drop_duration = round(time.perf_counter() - drop_started, 3)
            _notify("drop", f"done ({drop_duration}s)")

            # STEP 4: Key renames (after drop, before recreate)
            if has_prefix_change and keys_to_process:
                _notify("key_rename", "Renaming keys...")
                key_rename_started = time.perf_counter()
                old_prefix = plan.source.keyspace.prefixes[0]
                new_prefix = rename_ops.change_prefix
                assert new_prefix is not None  # For type checker
                renamed_count = self._rename_keys(
                    client,
                    keys_to_process,
                    old_prefix,
                    new_prefix,
                    progress_callback=lambda done, total: _notify(
                        "key_rename", f"{done:,}/{total:,} keys"
                    ),
                )
                key_rename_duration = round(time.perf_counter() - key_rename_started, 3)
                _notify(
                    "key_rename",
                    f"done ({renamed_count:,} keys in {key_rename_duration}s)",
                )

            # STEP 5: Re-encode vectors using pre-enumerated keys
            if datatype_changes and keys_to_process:
                _notify("quantize", "Re-encoding vectors...")
                quantize_started = time.perf_counter()
                # If we renamed keys, update keys_to_process to new names
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
                docs_quantized = self._quantize_vectors(
                    source_index,
                    datatype_changes,
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

        return report

    def _quantize_vectors(
        self,
        source_index: SearchIndex,
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
            source_index: The source SearchIndex (already dropped but client available)
            datatype_changes: Dict mapping field_name -> {"source", "target", "dims"}
            keys: Pre-enumerated list of document keys to process
            progress_callback: Optional callback(docs_done, total_docs)
            checkpoint_path: Optional path for checkpoint file (enables resume)

        Returns:
            Number of documents processed
        """
        client = source_index._redis_client
        total_keys = len(keys)
        docs_processed = 0
        skipped = 0
        batch_size = 500

        # Load or create checkpoint for resume support
        checkpoint: Optional[QuantizationCheckpoint] = None
        if checkpoint_path:
            checkpoint = QuantizationCheckpoint.load(checkpoint_path)
            if checkpoint:
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
                total_keys_for_progress = total_keys
        else:
            total_keys_for_progress = total_keys

        remaining_keys = len(keys)

        for i in range(0, remaining_keys, batch_size):
            batch = keys[i : i + batch_size]
            pipe = client.pipeline()
            undo = BatchUndoBuffer()
            keys_updated_in_batch: set[str] = set()

            try:
                for key in batch:
                    for field_name, change in datatype_changes.items():
                        field_data: bytes | None = client.hget(key, field_name)  # type: ignore[misc,assignment]
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
                    pipe.execute()
            except Exception:
                logger.warning(
                    "Batch %d failed, rolling back %d entries",
                    i // batch_size,
                    undo.size,
                )
                rollback_pipe = client.pipeline()
                undo.rollback(rollback_pipe)
                if checkpoint:
                    checkpoint.save()
                raise
            finally:
                undo.clear()

            docs_processed += len(keys_updated_in_batch)

            if checkpoint:
                checkpoint.record_batch(list(keys_updated_in_batch))
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
            docs_processed,
            len(datatype_changes),
        )
        return docs_processed

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
