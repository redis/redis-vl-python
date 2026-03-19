from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from redisvl.index import SearchIndex
from redisvl.migration.models import (
    MigrationBenchmarkSummary,
    MigrationPlan,
    MigrationReport,
    MigrationTimings,
    MigrationValidation,
)
from redisvl.migration.planner import MigrationPlanner
from redisvl.migration.utils import (
    current_source_matches_snapshot,
    timestamp_utc,
    wait_for_index_ready,
)
from redisvl.migration.validation import MigrationValidator
from redisvl.redis.utils import array_to_buffer, buffer_to_array
from redisvl.schema import StorageType

logger = logging.getLogger(__name__)


class MigrationExecutor:
    def __init__(self, validator: Optional[MigrationValidator] = None):
        self.validator = validator or MigrationValidator()

    def apply(
        self,
        plan: MigrationPlan,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
        query_check_file: Optional[str] = None,
        progress_callback: Optional[Callable[[str, Optional[str]], None]] = None,
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

        drop_duration = 0.0
        quantize_duration = 0.0
        recreate_duration = 0.0
        indexing_duration = 0.0
        target_info: Dict[str, Any] = {}
        docs_quantized = 0

        # Check if we need to re-encode vectors for datatype changes
        datatype_changes = MigrationPlanner.get_vector_datatype_changes(
            plan.source.schema_snapshot, plan.merged_target_schema
        )

        def _notify(step: str, detail: Optional[str] = None) -> None:
            if progress_callback:
                progress_callback(step, detail)

        try:
            _notify("drop", "Dropping index definition...")
            drop_started = time.perf_counter()
            source_index.delete(drop=False)
            drop_duration = round(time.perf_counter() - drop_started, 3)
            _notify("drop", f"done ({drop_duration}s)")

            # Re-encode vectors if datatype changes are needed
            if datatype_changes:
                _notify("quantize", "Re-encoding vectors...")
                quantize_started = time.perf_counter()
                docs_quantized = self._quantize_vectors(
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

    def _quantize_vectors(
        self,
        source_index: SearchIndex,
        datatype_changes: Dict[str, Dict[str, str]],
        plan: MigrationPlan,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Re-encode vectors in documents for datatype changes (quantization).

        This iterates over all documents matching the index prefix and converts
        vector fields from source datatype to target datatype.

        Args:
            source_index: The source SearchIndex (already dropped but client available)
            datatype_changes: Dict mapping field_name -> {"source": dtype, "target": dtype}
            plan: The migration plan containing schema info
            progress_callback: Optional callback(docs_done, total_docs)

        Returns:
            Number of documents processed
        """
        client = source_index._redis_client
        prefix = plan.source.schema_snapshot["index"]["prefix"]
        storage_type = (
            plan.source.schema_snapshot["index"].get("storage_type", "hash").lower()
        )

        # Get estimated total from source stats
        estimated_total = int(plan.source.stats_snapshot.get("num_docs", 0) or 0)

        # Get vector field dimensions for validation
        field_dims: Dict[str, int] = {}
        for field in plan.source.schema_snapshot.get("fields", []):
            if field.get("type") == "vector" and field["name"] in datatype_changes:
                field_dims[field["name"]] = field.get("attrs", {}).get("dims", 0)

        docs_processed = 0
        batch_size = 500
        cursor = 0

        while True:
            cursor, keys = client.scan(
                cursor=cursor,
                match=f"{prefix}*",
                count=batch_size,
            )

            if keys:
                pipe = client.pipeline()
                keys_to_update = []

                for key in keys:
                    if storage_type == "hash":
                        # Read all vector fields that need conversion
                        for field_name, change in datatype_changes.items():
                            field_data = client.hget(key, field_name)
                            if field_data:
                                # Convert: source dtype -> array -> target dtype -> bytes
                                array = buffer_to_array(field_data, change["source"])
                                new_bytes = array_to_buffer(array, change["target"])
                                pipe.hset(key, field_name, new_bytes)
                                keys_to_update.append(key)
                    else:
                        # JSON storage - vectors stored as arrays, need different handling
                        logger.warning(
                            f"JSON storage quantization for key {key} - "
                            "vectors stored as arrays may not need re-encoding"
                        )

                if keys_to_update:
                    pipe.execute()
                    docs_processed += len(set(keys_to_update))
                    if progress_callback:
                        progress_callback(docs_processed, estimated_total)

            if cursor == 0:
                break

        logger.info(f"Quantized {docs_processed} documents: {datatype_changes}")
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
