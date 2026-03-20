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


class SchemaPatchChanges(BaseModel):
    add_fields: List[Dict[str, Any]] = Field(default_factory=list)
    remove_fields: List[str] = Field(default_factory=list)
    update_fields: List[FieldUpdate] = Field(default_factory=list)
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


class MigrationPlan(BaseModel):
    version: int = 1
    mode: str = "drop_recreate"
    source: SourceSnapshot
    requested_changes: Dict[str, Any]
    merged_target_schema: Dict[str, Any]
    diff_classification: DiffClassification
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
    warnings: List[str] = Field(default_factory=list)
    manual_actions: List[str] = Field(default_factory=list)


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
        return sum(1 for idx in self.completed if idx.status == "success")

    @property
    def failed_count(self) -> int:
        return sum(1 for idx in self.completed if idx.status == "failed")

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
