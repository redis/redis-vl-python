from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class FieldUpdate(BaseModel):
    """Partial field update for schema patch inputs."""

    name: str
    type: Optional[str] = None
    path: Optional[str] = None
    attrs: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def merge_options_into_attrs(self) -> "FieldUpdate":
        if self.options:
            merged_attrs = dict(self.attrs)
            merged_attrs.update(self.options)
            self.attrs = merged_attrs
            self.options = {}
        return self


class FieldRename(BaseModel):
    """Field rename specification for schema patch inputs."""

    old_name: str
    new_name: str


class SchemaPatchChanges(BaseModel):
    add_fields: List[Dict[str, Any]] = Field(default_factory=list)
    remove_fields: List[str] = Field(default_factory=list)
    update_fields: List[FieldUpdate] = Field(default_factory=list)
    rename_fields: List[FieldRename] = Field(default_factory=list)
    index: Dict[str, Any] = Field(default_factory=dict)


class SchemaPatch(BaseModel):
    version: int = 1
    changes: SchemaPatchChanges = Field(default_factory=SchemaPatchChanges)


class KeyspaceSnapshot(BaseModel):
    storage_type: str
    prefixes: List[str]
    key_separator: str
    key_sample: List[str] = Field(default_factory=list)


class SourceSnapshot(BaseModel):
    index_name: str
    schema_snapshot: Dict[str, Any]
    stats_snapshot: Dict[str, Any]
    keyspace: KeyspaceSnapshot


class DiffClassification(BaseModel):
    supported: bool
    blocked_reasons: List[str] = Field(default_factory=list)


class ValidationPolicy(BaseModel):
    require_doc_count_match: bool = True
    require_schema_match: bool = True


class RenameOperations(BaseModel):
    """Tracks which rename operations are required for a migration."""

    rename_index: Optional[str] = None  # New index name if renaming
    change_prefix: Optional[str] = None  # New prefix if changing
    rename_fields: List[FieldRename] = Field(default_factory=list)

    @property
    def has_operations(self) -> bool:
        return bool(self.rename_index or self.change_prefix or self.rename_fields)


class MigrationPlan(BaseModel):
    version: int = 1
    mode: str = "drop_recreate"
    source: SourceSnapshot
    requested_changes: Dict[str, Any]
    merged_target_schema: Dict[str, Any]
    diff_classification: DiffClassification
    rename_operations: RenameOperations = Field(default_factory=RenameOperations)
    warnings: List[str] = Field(default_factory=list)
    validation: ValidationPolicy = Field(default_factory=ValidationPolicy)


class QueryCheckResult(BaseModel):
    name: str
    passed: bool
    details: Optional[str] = None


class MigrationValidation(BaseModel):
    schema_match: bool = False
    doc_count_match: bool = False
    key_sample_exists: bool = False
    indexing_failures_delta: int = 0
    query_checks: List[QueryCheckResult] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class MigrationTimings(BaseModel):
    total_migration_duration_seconds: Optional[float] = None
    drop_duration_seconds: Optional[float] = None
    quantize_duration_seconds: Optional[float] = None
    field_rename_duration_seconds: Optional[float] = None
    key_rename_duration_seconds: Optional[float] = None
    recreate_duration_seconds: Optional[float] = None
    initial_indexing_duration_seconds: Optional[float] = None
    validation_duration_seconds: Optional[float] = None
    downtime_duration_seconds: Optional[float] = None


class MigrationBenchmarkSummary(BaseModel):
    documents_indexed_per_second: Optional[float] = None
    source_index_size_mb: Optional[float] = None
    target_index_size_mb: Optional[float] = None
    index_size_delta_mb: Optional[float] = None


class MigrationReport(BaseModel):
    version: int = 1
    mode: str = "drop_recreate"
    source_index: str
    target_index: str
    result: str
    started_at: str
    finished_at: str
    timings: MigrationTimings = Field(default_factory=MigrationTimings)
    validation: MigrationValidation = Field(default_factory=MigrationValidation)
    benchmark_summary: MigrationBenchmarkSummary = Field(
        default_factory=MigrationBenchmarkSummary
    )
    disk_space_estimate: Optional["DiskSpaceEstimate"] = None
    warnings: List[str] = Field(default_factory=list)
    manual_actions: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Disk Space Estimation
# -----------------------------------------------------------------------------

# Bytes per element for each vector datatype
DTYPE_BYTES: Dict[str, int] = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
}

# AOF protocol overhead per HSET command (RESP framing)
AOF_HSET_OVERHEAD_BYTES = 114
# JSON.SET has slightly larger RESP framing
AOF_JSON_SET_OVERHEAD_BYTES = 140
# RDB compression ratio for pseudo-random vector data (compresses poorly)
RDB_COMPRESSION_RATIO = 0.95


class VectorFieldEstimate(BaseModel):
    """Per-field disk space breakdown for a single vector field."""

    field_name: str
    dims: int
    source_dtype: str
    target_dtype: str
    source_bytes_per_doc: int
    target_bytes_per_doc: int


class DiskSpaceEstimate(BaseModel):
    """Pre-migration estimate of disk and memory costs.

    Produced by estimate_disk_space() as a pure calculation from the migration
    plan. No Redis mutations are performed.
    """

    # Index metadata
    index_name: str
    doc_count: int
    storage_type: str = "hash"

    # Per-field breakdowns
    vector_fields: List[VectorFieldEstimate] = Field(default_factory=list)

    # Aggregate vector data sizes
    total_source_vector_bytes: int = 0
    total_target_vector_bytes: int = 0

    # RDB snapshot cost (BGSAVE before migration)
    rdb_snapshot_disk_bytes: int = 0
    rdb_cow_memory_if_concurrent_bytes: int = 0

    # AOF growth cost (only if aof_enabled is True)
    aof_enabled: bool = False
    aof_growth_bytes: int = 0

    # Totals
    total_new_disk_bytes: int = 0
    memory_savings_after_bytes: int = 0

    @property
    def has_quantization(self) -> bool:
        return len(self.vector_fields) > 0

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        if not self.has_quantization:
            return "No vector quantization in this migration. No additional disk space required."

        lines = [
            "Pre-migration disk space estimate:",
            f"  Index: {self.index_name} ({self.doc_count:,} documents)",
        ]
        for vf in self.vector_fields:
            lines.append(
                f"  Vector field '{vf.field_name}': {vf.dims} dims, "
                f"{vf.source_dtype} -> {vf.target_dtype}"
            )

        lines.append("")
        lines.append(
            f"  RDB snapshot (BGSAVE):        ~{_format_bytes(self.rdb_snapshot_disk_bytes)}"
        )
        if self.aof_enabled:
            lines.append(
                f"  AOF growth (appendonly=yes):  ~{_format_bytes(self.aof_growth_bytes)}"
            )
        else:
            lines.append(
                "  AOF growth:                  not estimated (pass aof_enabled=True if AOF is on)"
            )
        lines.append(
            f"  Total new disk required:      ~{_format_bytes(self.total_new_disk_bytes)}"
        )
        lines.append("")
        lines.append(
            f"  Post-migration memory savings: ~{_format_bytes(self.memory_savings_after_bytes)} "
            f"({self._savings_pct()}% reduction)"
        )
        return "\n".join(lines)

    def _savings_pct(self) -> int:
        if self.total_source_vector_bytes == 0:
            return 0
        return round(
            100 * self.memory_savings_after_bytes / self.total_source_vector_bytes
        )


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.2f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} bytes"


# -----------------------------------------------------------------------------
# Batch Migration Models
# -----------------------------------------------------------------------------


class BatchIndexEntry(BaseModel):
    """Entry for a single index in a batch migration plan."""

    name: str
    applicable: bool = True
    skip_reason: Optional[str] = None


class BatchPlan(BaseModel):
    """Plan for migrating multiple indexes with a shared patch."""

    version: int = 1
    batch_id: str
    mode: str = "drop_recreate"
    failure_policy: str = "fail_fast"  # or "continue_on_error"
    requires_quantization: bool = False
    shared_patch: SchemaPatch
    indexes: List[BatchIndexEntry] = Field(default_factory=list)
    created_at: str

    @property
    def applicable_count(self) -> int:
        return sum(1 for idx in self.indexes if idx.applicable)

    @property
    def skipped_count(self) -> int:
        return sum(1 for idx in self.indexes if not idx.applicable)


class BatchIndexState(BaseModel):
    """State of a single index in batch execution."""

    name: str
    status: str  # pending, in_progress, success, failed, skipped
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    error: Optional[str] = None
    report_path: Optional[str] = None


class BatchState(BaseModel):
    """Checkpoint state for batch migration execution."""

    batch_id: str
    plan_path: str
    started_at: str
    updated_at: str
    completed: List[BatchIndexState] = Field(default_factory=list)
    current_index: Optional[str] = None
    remaining: List[str] = Field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for idx in self.completed if idx.status == "succeeded")

    @property
    def failed_count(self) -> int:
        return sum(1 for idx in self.completed if idx.status == "failed")

    @property
    def skipped_count(self) -> int:
        return sum(1 for idx in self.completed if idx.status == "skipped")

    @property
    def is_complete(self) -> bool:
        return len(self.remaining) == 0 and self.current_index is None


class BatchReportSummary(BaseModel):
    """Summary statistics for batch migration."""

    total_indexes: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_seconds: float = 0.0


class BatchIndexReport(BaseModel):
    """Report for a single index in batch execution."""

    name: str
    status: str  # success, failed, skipped
    duration_seconds: Optional[float] = None
    docs_migrated: Optional[int] = None
    report_path: Optional[str] = None
    error: Optional[str] = None


class BatchReport(BaseModel):
    """Final report for batch migration execution."""

    version: int = 1
    batch_id: str
    status: str  # completed, partial_failure, failed
    summary: BatchReportSummary = Field(default_factory=BatchReportSummary)
    indexes: List[BatchIndexReport] = Field(default_factory=list)
    started_at: str
    completed_at: str
