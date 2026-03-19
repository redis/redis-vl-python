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
