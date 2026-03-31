"""Batch migration executor with checkpointing and resume support."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from redisvl.migration.executor import MigrationExecutor
from redisvl.migration.models import (
    BatchIndexReport,
    BatchIndexState,
    BatchPlan,
    BatchReport,
    BatchReportSummary,
    BatchState,
)
from redisvl.migration.planner import MigrationPlanner
from redisvl.migration.utils import timestamp_utc, write_yaml
from redisvl.redis.connection import RedisConnectionFactory


class BatchMigrationExecutor:
    """Executor for batch migration of multiple indexes.

    Supports:
    - Sequential execution (one index at a time)
    - Checkpointing for resume after failure
    - Configurable failure policies (fail_fast, continue_on_error)
    """

    def __init__(self, executor: Optional[MigrationExecutor] = None):
        self._single_executor = executor or MigrationExecutor()
        self._planner = MigrationPlanner()

    def apply(
        self,
        batch_plan: BatchPlan,
        *,
        batch_plan_path: Optional[str] = None,
        state_path: str = "batch_state.yaml",
        report_dir: str = "./reports",
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> BatchReport:
        """Execute batch migration with checkpointing.

        Args:
            batch_plan: The batch plan to execute.
            batch_plan_path: Path to the batch plan file (stored in state for resume).
            state_path: Path to checkpoint state file.
            report_dir: Directory for per-index reports.
            redis_url: Redis connection URL.
            redis_client: Existing Redis client.
            progress_callback: Optional callback(index_name, position, total, status).

        Returns:
            BatchReport with results for all indexes.
        """
        # Get Redis client
        client = redis_client
        if client is None:
            if not redis_url:
                raise ValueError("Must provide either redis_url or redis_client")
            client = RedisConnectionFactory.get_redis_connection(redis_url=redis_url)

        # Ensure report directory exists
        report_path = Path(report_dir).resolve()
        report_path.mkdir(parents=True, exist_ok=True)

        # Initialize or load state
        state = self._init_or_load_state(batch_plan, state_path, batch_plan_path)
        started_at = state.started_at
        batch_start_time = time.perf_counter()

        # Get applicable indexes
        applicable_indexes = [idx for idx in batch_plan.indexes if idx.applicable]
        total = len(applicable_indexes)

        # Process each remaining index
        for position, index_name in enumerate(state.remaining[:], start=1):
            state.current_index = index_name
            state.updated_at = timestamp_utc()
            self._write_state(state, state_path)

            if progress_callback:
                progress_callback(index_name, position, total, "starting")

            # Find the index entry
            index_entry = next(
                (idx for idx in batch_plan.indexes if idx.name == index_name), None
            )
            if not index_entry or not index_entry.applicable:
                # Skip non-applicable indexes
                state.remaining.remove(index_name)
                state.completed.append(
                    BatchIndexState(
                        name=index_name,
                        status="skipped",
                        completed_at=timestamp_utc(),
                    )
                )
                continue

            # Execute migration for this index
            index_state = self._migrate_single_index(
                index_name=index_name,
                batch_plan=batch_plan,
                report_dir=report_path,
                redis_client=client,
            )

            # Update state
            state.remaining.remove(index_name)
            state.completed.append(index_state)
            state.current_index = None
            state.updated_at = timestamp_utc()
            self._write_state(state, state_path)

            if progress_callback:
                progress_callback(index_name, position, total, index_state.status)

            # Check failure policy
            if (
                index_state.status == "failed"
                and batch_plan.failure_policy == "fail_fast"
            ):
                # Mark remaining as skipped
                for remaining_name in state.remaining[:]:
                    state.remaining.remove(remaining_name)
                    state.completed.append(
                        BatchIndexState(
                            name=remaining_name,
                            status="skipped",
                            completed_at=timestamp_utc(),
                        )
                    )
                state.updated_at = timestamp_utc()
                self._write_state(state, state_path)
                break

        # Build final report
        total_duration = time.perf_counter() - batch_start_time
        return self._build_batch_report(batch_plan, state, started_at, total_duration)

    def resume(
        self,
        state_path: str,
        *,
        batch_plan_path: Optional[str] = None,
        retry_failed: bool = False,
        report_dir: str = "./reports",
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> BatchReport:
        """Resume batch migration from checkpoint.

        Args:
            state_path: Path to checkpoint state file.
            batch_plan_path: Path to batch plan (uses state.plan_path if not provided).
            retry_failed: If True, retry previously failed indexes.
            report_dir: Directory for per-index reports.
            redis_url: Redis connection URL.
            redis_client: Existing Redis client.
            progress_callback: Optional callback(index_name, position, total, status).
        """
        state = self._load_state(state_path)
        plan_path = batch_plan_path or state.plan_path
        if not plan_path or not plan_path.strip():
            raise ValueError(
                "No batch plan path available. Provide batch_plan_path explicitly, "
                "or ensure the checkpoint state contains a valid plan_path."
            )
        batch_plan = self._load_batch_plan(plan_path)

        # Optionally retry failed indexes
        if retry_failed:
            failed_names = [
                idx.name for idx in state.completed if idx.status == "failed"
            ]
            state.remaining = failed_names + state.remaining
            state.completed = [idx for idx in state.completed if idx.status != "failed"]
            # Write updated state back to file so apply() picks up the changes
            self._write_state(state, state_path)

        # Re-run apply with the updated state
        return self.apply(
            batch_plan,
            state_path=state_path,
            report_dir=report_dir,
            redis_url=redis_url,
            redis_client=redis_client,
            progress_callback=progress_callback,
        )

    def _migrate_single_index(
        self,
        *,
        index_name: str,
        batch_plan: BatchPlan,
        report_dir: Path,
        redis_client: Any,
    ) -> BatchIndexState:
        """Execute migration for a single index."""
        try:
            # Create migration plan for this index
            plan = self._planner.create_plan_from_patch(
                index_name,
                schema_patch=batch_plan.shared_patch,
                redis_client=redis_client,
            )

            # Execute migration
            report = self._single_executor.apply(
                plan,
                redis_client=redis_client,
            )

            # Write individual report
            report_file = report_dir / f"{index_name}_report.yaml"
            write_yaml(report.model_dump(exclude_none=True), str(report_file))

            return BatchIndexState(
                name=index_name,
                status="succeeded" if report.result == "succeeded" else "failed",
                completed_at=timestamp_utc(),
                report_path=str(report_file),
                error=report.validation.errors[0] if report.validation.errors else None,
            )

        except Exception as e:
            return BatchIndexState(
                name=index_name,
                status="failed",
                completed_at=timestamp_utc(),
                error=str(e),
            )

    def _init_or_load_state(
        self,
        batch_plan: BatchPlan,
        state_path: str,
        batch_plan_path: Optional[str] = None,
    ) -> BatchState:
        """Initialize new state or load existing checkpoint."""
        path = Path(state_path).resolve()
        if path.exists():
            loaded = self._load_state(state_path)
            # Validate that loaded state matches the current batch plan
            if loaded.batch_id and loaded.batch_id != batch_plan.batch_id:
                raise ValueError(
                    f"Checkpoint state batch_id '{loaded.batch_id}' does not match "
                    f"current batch plan '{batch_plan.batch_id}'. "
                    "Remove the stale state file or use a different state_path."
                )
            return loaded

        # Create new state with plan_path for resume support
        applicable_names = [idx.name for idx in batch_plan.indexes if idx.applicable]
        return BatchState(
            batch_id=batch_plan.batch_id,
            plan_path=str(Path(batch_plan_path).resolve()) if batch_plan_path else "",
            started_at=timestamp_utc(),
            updated_at=timestamp_utc(),
            remaining=applicable_names,
            completed=[],
            current_index=None,
        )

    def _write_state(self, state: BatchState, state_path: str) -> None:
        """Write checkpoint state to file."""
        path = Path(state_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(state.model_dump(exclude_none=True), f, sort_keys=False)

    def _load_state(self, state_path: str) -> BatchState:
        """Load checkpoint state from file."""
        path = Path(state_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return BatchState.model_validate(data)

    def _load_batch_plan(self, plan_path: str) -> BatchPlan:
        """Load batch plan from file."""
        path = Path(plan_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Batch plan not found: {plan_path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return BatchPlan.model_validate(data)

    def _build_batch_report(
        self,
        batch_plan: BatchPlan,
        state: BatchState,
        started_at: str,
        total_duration: float,
    ) -> BatchReport:
        """Build final batch report from state."""
        index_reports = []
        succeeded = 0
        failed = 0
        skipped = 0

        for idx_state in state.completed:
            index_reports.append(
                BatchIndexReport(
                    name=idx_state.name,
                    status=idx_state.status,
                    report_path=idx_state.report_path,
                    error=idx_state.error,
                )
            )
            if idx_state.status == "succeeded":
                succeeded += 1
            elif idx_state.status == "failed":
                failed += 1
            else:
                skipped += 1

        # Add non-applicable indexes as skipped
        for idx in batch_plan.indexes:
            if not idx.applicable:
                index_reports.append(
                    BatchIndexReport(
                        name=idx.name,
                        status="skipped",
                        error=idx.skip_reason,
                    )
                )
                skipped += 1

        # Determine overall status
        if failed == 0 and len(state.remaining) == 0:
            status = "completed"
        elif succeeded > 0:
            status = "partial_failure"
        else:
            status = "failed"

        return BatchReport(
            batch_id=batch_plan.batch_id,
            status=status,
            started_at=started_at,
            completed_at=timestamp_utc(),
            summary=BatchReportSummary(
                total_indexes=len(batch_plan.indexes),
                successful=succeeded,
                failed=failed,
                skipped=skipped,
                total_duration_seconds=round(total_duration, 3),
            ),
            indexes=index_reports,
        )
