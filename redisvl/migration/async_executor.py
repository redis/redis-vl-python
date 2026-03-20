from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

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
from redisvl.migration.utils import timestamp_utc
from redisvl.redis.utils import array_to_buffer, buffer_to_array
from redisvl.types import AsyncRedisClient

logger = logging.getLogger(__name__)


class AsyncMigrationExecutor:
    """Async migration executor for document-preserving drop/recreate flows.

    This is the async version of MigrationExecutor. It uses AsyncSearchIndex
    and async Redis operations for better performance on large indexes,
    especially during vector quantization.
    """

    def __init__(self, validator: Optional[AsyncMigrationValidator] = None):
        self.validator = validator or AsyncMigrationValidator()

    async def apply(
        self,
        plan: MigrationPlan,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedisClient] = None,
        query_check_file: Optional[str] = None,
        progress_callback: Optional[Callable[[str, Optional[str]], None]] = None,
    ) -> MigrationReport:
        """Apply a migration plan asynchronously.

        Args:
            plan: The migration plan to apply.
            redis_url: Redis connection URL.
            redis_client: Optional existing async Redis client.
            query_check_file: Optional file with query checks.
            progress_callback: Optional callback(step, detail) for progress updates.
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
        target_index = AsyncSearchIndex.from_dict(
            plan.merged_target_schema,
            redis_url=redis_url,
            redis_client=redis_client,
        )

        drop_duration = 0.0
        quantize_duration = 0.0
        recreate_duration = 0.0
        indexing_duration = 0.0
        target_info: Dict[str, Any] = {}
        docs_quantized = 0

        datatype_changes = AsyncMigrationPlanner.get_vector_datatype_changes(
            plan.source.schema_snapshot, plan.merged_target_schema
        )

        def _notify(step: str, detail: Optional[str] = None) -> None:
            if progress_callback:
                progress_callback(step, detail)

        try:
            _notify("drop", "Dropping index definition...")
            drop_started = time.perf_counter()
            await source_index.delete(drop=False)
            drop_duration = round(time.perf_counter() - drop_started, 3)
            _notify("drop", f"done ({drop_duration}s)")

            if datatype_changes:
                _notify("quantize", "Re-encoding vectors...")
                quantize_started = time.perf_counter()
                docs_quantized = await self._async_quantize_vectors(
                    source_index,
                    datatype_changes,
                    plan,
                    progress_callback=lambda done, total: _notify(
                        "quantize", f"{done:,}/{total:,} docs"
                    ),
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
                recreate_duration_seconds=recreate_duration,
                initial_indexing_duration_seconds=indexing_duration,
                validation_duration_seconds=validation_duration,
                downtime_duration_seconds=round(
                    drop_duration
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
                recreate_duration_seconds=recreate_duration or None,
                initial_indexing_duration_seconds=indexing_duration or None,
                downtime_duration_seconds=(
                    round(
                        drop_duration
                        + quantize_duration
                        + recreate_duration
                        + indexing_duration,
                        3,
                    )
                    if drop_duration
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

    async def _async_quantize_vectors(
        self,
        source_index: AsyncSearchIndex,
        datatype_changes: Dict[str, Dict[str, str]],
        plan: MigrationPlan,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Re-encode vectors in documents for datatype changes (quantization).

        This is the async version that uses async pipeline operations for
        better performance on large indexes.
        """
        client = source_index._redis_client
        if client is None:
            raise ValueError("Failed to get Redis client from source index")

        prefix = plan.source.schema_snapshot["index"]["prefix"]
        storage_type = (
            plan.source.schema_snapshot["index"].get("storage_type", "hash").lower()
        )
        estimated_total = int(plan.source.stats_snapshot.get("num_docs", 0) or 0)

        docs_processed = 0
        batch_size = 500
        cursor: int = 0

        while True:
            cursor, keys = await client.scan(
                cursor=cursor,
                match=f"{prefix}*",
                count=batch_size,
            )

            if keys:
                pipe = client.pipeline()
                keys_to_update = []

                for key in keys:
                    if storage_type == "hash":
                        for field_name, change in datatype_changes.items():
                            # hget returns bytes for binary data
                            field_data: bytes | None = await client.hget(key, field_name)  # type: ignore[misc,assignment]
                            if field_data:
                                # field_data is bytes from Redis
                                array = buffer_to_array(field_data, change["source"])
                                new_bytes = array_to_buffer(array, change["target"])
                                pipe.hset(
                                    key, field_name, new_bytes  # type: ignore[arg-type]
                                )
                                keys_to_update.append(key)
                    else:
                        logger.warning(
                            f"JSON storage quantization for key {key} - "
                            "vectors stored as arrays may not need re-encoding"
                        )

                if keys_to_update:
                    await pipe.execute()
                    docs_processed += len(set(keys_to_update))
                    if progress_callback:
                        progress_callback(docs_processed, estimated_total)

            if cursor == 0:
                break

        logger.info(f"Quantized {docs_processed} documents: {datatype_changes}")
        return docs_processed

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

        stable_ready_checks = 0
        while time.perf_counter() < deadline:
            latest_info = await index.info()
            indexing = latest_info.get("indexing")
            percent_indexed = latest_info.get("percent_indexed")

            if percent_indexed is not None or indexing is not None:
                ready = float(percent_indexed or 0) >= 1.0 and not bool(indexing)
                if progress_callback:
                    total_docs = int(latest_info.get("num_docs", 0))
                    pct = float(percent_indexed or 0)
                    indexed_docs = int(total_docs * pct)
                    progress_callback(indexed_docs, total_docs, pct * 100)
            else:
                current_docs = latest_info.get("num_docs")
                if current_docs is None:
                    ready = True
                else:
                    if stable_ready_checks == 0:
                        stable_ready_checks = int(current_docs)
                        await asyncio.sleep(poll_interval_seconds)
                        continue
                    ready = int(current_docs) == stable_ready_checks

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

        current_index = await AsyncSearchIndex.from_existing(
            index_name,
            redis_url=redis_url,
            redis_client=redis_client,
        )
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
