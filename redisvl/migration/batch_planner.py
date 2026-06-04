"""Batch migration planner for migrating multiple indexes with a shared patch."""

from __future__ import annotations

import fnmatch
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

import redis.exceptions
import yaml

from redisvl.index import SearchIndex
from redisvl.migration.models import BatchIndexEntry, BatchPlan, SchemaPatch
from redisvl.migration.planner import MigrationPlanner
from redisvl.migration.utils import (
    find_overlapping_index_groups,
    list_indexes,
    normalize_prefixes,
    timestamp_utc,
)
from redisvl.redis.connection import RedisConnectionFactory


class BatchMigrationPlanner:
    """Planner for batch migration of multiple indexes with a shared patch.

    The batch planner applies a single SchemaPatch to multiple indexes,
    checking applicability for each index based on field name matching.
    """

    def __init__(self):
        self._single_planner = MigrationPlanner()

    def create_batch_plan(
        self,
        *,
        indexes: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        indexes_file: Optional[str] = None,
        schema_patch_path: str,
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
        failure_policy: str = "fail_fast",
    ) -> BatchPlan:
        # --- NEW: validate failure_policy early ---
        """Create a batch migration plan for multiple indexes.

        Args:
            indexes: Explicit list of index names.
            pattern: Glob pattern to match index names (e.g., "*_idx").
            indexes_file: Path to file with index names (one per line).
            schema_patch_path: Path to shared schema patch YAML file.
            redis_url: Redis connection URL.
            redis_client: Existing Redis client.
            failure_policy: "fail_fast" or "continue_on_error".

        Returns:
            BatchPlan with shared patch and per-index applicability.
        """
        _VALID_FAILURE_POLICIES = {"fail_fast", "continue_on_error"}
        if failure_policy not in _VALID_FAILURE_POLICIES:
            raise ValueError(
                f"Invalid failure_policy '{failure_policy}'. "
                f"Must be one of: {sorted(_VALID_FAILURE_POLICIES)}"
            )

        # Get Redis client
        client = redis_client
        if client is None:
            if not redis_url:
                raise ValueError("Must provide either redis_url or redis_client")
            client = RedisConnectionFactory.get_redis_connection(redis_url=redis_url)

        # Resolve index list
        index_names = self._resolve_index_names(
            indexes=indexes,
            pattern=pattern,
            indexes_file=indexes_file,
            redis_client=client,
        )

        if not index_names:
            raise ValueError("No indexes found matching the specified criteria")

        # Load shared patch
        shared_patch = self._single_planner.load_schema_patch(schema_patch_path)

        # Check applicability for each index
        batch_entries: List[BatchIndexEntry] = []
        applicable_prefixes: List[Tuple[str, List[str]]] = []
        requires_quantization = False

        for index_name in index_names:
            entry, has_quantization, prefixes = self._check_index_applicability(
                index_name=index_name,
                shared_patch=shared_patch,
                redis_client=client,
            )
            batch_entries.append(entry)
            if has_quantization:
                requires_quantization = True
            if entry.applicable:
                applicable_prefixes.append((index_name, prefixes))

        # Refuse plan creation when applicable indexes share keyspace.
        # Overlapping indexes cause double-mutation of the same keys during
        # sequential batch execution (e.g., double-quantization of vectors).
        overlaps = find_overlapping_index_groups(applicable_prefixes)
        if overlaps:
            raise ValueError(self._format_overlap_error(overlaps))

        batch_id = f"batch_{uuid.uuid4().hex[:12]}"

        return BatchPlan(
            batch_id=batch_id,
            mode="drop_recreate",
            failure_policy=failure_policy,
            requires_quantization=requires_quantization,
            shared_patch=shared_patch,
            indexes=batch_entries,
            created_at=timestamp_utc(),
        )

    def _resolve_index_names(
        self,
        *,
        indexes: Optional[List[str]],
        pattern: Optional[str],
        indexes_file: Optional[str],
        redis_client: Any,
    ) -> List[str]:
        """Resolve index names from explicit list, pattern, or file."""
        sources = sum([bool(indexes), bool(pattern), bool(indexes_file)])
        if sources == 0:
            raise ValueError("Must provide one of: indexes, pattern, or indexes_file")
        if sources > 1:
            raise ValueError("Provide only one of: indexes, pattern, or indexes_file")

        if indexes:
            # Deduplicate while preserving order
            return list(dict.fromkeys(indexes))

        if indexes_file:
            return self._load_indexes_from_file(indexes_file)

        # Pattern matching -- pattern is guaranteed non-None at this point
        assert pattern is not None, "pattern must be set when reaching fnmatch"
        all_indexes = list_indexes(redis_client=redis_client)
        matched = [idx for idx in all_indexes if fnmatch.fnmatch(idx, pattern)]
        return sorted(matched)

    def _load_indexes_from_file(self, file_path: str) -> List[str]:
        """Load index names from a file (one per line)."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Indexes file not found: {file_path}")

        with open(path, "r") as f:
            lines = f.readlines()

        return [
            stripped
            for line in lines
            if (stripped := line.strip()) and not stripped.startswith("#")
        ]

    def _check_index_applicability(
        self,
        *,
        index_name: str,
        shared_patch: SchemaPatch,
        redis_client: Any,
    ) -> Tuple[BatchIndexEntry, bool, List[str]]:
        """Check if the shared patch can be applied to a specific index.

        Returns:
            Tuple of (BatchIndexEntry, requires_quantization, prefixes).
            ``prefixes`` is the list of key prefixes the index is bound to,
            or an empty list when the index could not be loaded.
        """
        try:
            index = SearchIndex.from_existing(index_name, redis_client=redis_client)
            schema_dict = index.schema.to_dict()
            field_names = {f["name"] for f in schema_dict.get("fields", [])}
            prefixes = normalize_prefixes(schema_dict.get("index", {}).get("prefix"))

            # Build a set of field names that includes rename targets so
            # that update_fields referencing the NEW name of a renamed field
            # are considered applicable.
            rename_target_names = {
                fr.new_name for fr in shared_patch.changes.rename_fields
            }
            effective_field_names = field_names | rename_target_names

            # Check that all update_fields exist in this index (or are rename targets)
            missing_fields = []
            for field_update in shared_patch.changes.update_fields:
                if field_update.name not in effective_field_names:
                    missing_fields.append(field_update.name)

            if missing_fields:
                return (
                    BatchIndexEntry(
                        name=index_name,
                        applicable=False,
                        skip_reason=f"Missing fields: {', '.join(missing_fields)}",
                    ),
                    False,
                    prefixes,
                )

            # Validate rename targets don't collide with each other or
            # existing fields (after accounting for the source being renamed away)
            if shared_patch.changes.rename_fields:
                rename_targets = [
                    fr.new_name for fr in shared_patch.changes.rename_fields
                ]
                rename_sources = {
                    fr.old_name for fr in shared_patch.changes.rename_fields
                }
                seen_targets: dict[str, int] = {}
                for t in rename_targets:
                    seen_targets[t] = seen_targets.get(t, 0) + 1
                duplicates = [t for t, c in seen_targets.items() if c > 1]
                if duplicates:
                    return (
                        BatchIndexEntry(
                            name=index_name,
                            applicable=False,
                            skip_reason=f"Rename targets collide: {', '.join(duplicates)}",
                        ),
                        False,
                        prefixes,
                    )
                # Check if any rename target already exists and isn't itself being renamed away
                collisions = [
                    t
                    for t in rename_targets
                    if t in field_names and t not in rename_sources
                ]
                if collisions:
                    return (
                        BatchIndexEntry(
                            name=index_name,
                            applicable=False,
                            skip_reason=f"Rename targets already exist: {', '.join(collisions)}",
                        ),
                        False,
                        prefixes,
                    )

            # Check that add_fields don't already exist.
            # Fields being renamed away free their name for new additions.
            rename_sources = {fr.old_name for fr in shared_patch.changes.rename_fields}
            post_rename_fields = (field_names - rename_sources) | rename_target_names
            existing_adds: list[str] = []
            for field in shared_patch.changes.add_fields:
                field_name = field.get("name")
                if field_name and field_name in post_rename_fields:
                    existing_adds.append(field_name)

            if existing_adds:
                return (
                    BatchIndexEntry(
                        name=index_name,
                        applicable=False,
                        skip_reason=f"Fields already exist: {', '.join(existing_adds)}",
                    ),
                    False,
                    prefixes,
                )

            # Try creating a plan to check for blocked changes
            plan = self._single_planner.create_plan_from_patch(
                index_name,
                schema_patch=shared_patch,
                redis_client=redis_client,
            )

            if not plan.diff_classification.supported:
                return (
                    BatchIndexEntry(
                        name=index_name,
                        applicable=False,
                        skip_reason=(
                            plan.diff_classification.blocked_reasons[0]
                            if plan.diff_classification.blocked_reasons
                            else "Unsupported changes"
                        ),
                    ),
                    False,
                    prefixes,
                )

            # Detect quantization from the plan we already created
            has_quantization = bool(
                MigrationPlanner.get_vector_datatype_changes(
                    plan.source.schema_snapshot,
                    plan.merged_target_schema,
                    rename_operations=plan.rename_operations,
                )
            )

            return (
                BatchIndexEntry(name=index_name, applicable=True),
                has_quantization,
                prefixes,
            )

        except (
            ConnectionError,
            OSError,
            TimeoutError,
            redis.exceptions.ConnectionError,
        ):
            # Infrastructure failures should propagate, not be silently
            # treated as "not applicable".
            raise
        except Exception as e:
            return (
                BatchIndexEntry(
                    name=index_name,
                    applicable=False,
                    skip_reason=str(e),
                ),
                False,
                [],
            )

    @staticmethod
    def _format_overlap_error(
        overlaps: List[Tuple[str, str, List[Tuple[str, str]]]],
    ) -> str:
        """Build a human-readable error for overlapping index prefixes."""
        lines = [
            "Refusing to create batch plan: overlapping indexes detected.",
            "",
            "Multiple indexes in the batch share Redis key prefixes. Running a",
            "batch migration over overlapping indexes can mutate the same keys",
            "more than once (e.g., double-quantization of vectors), corrupting",
            "the underlying data.",
            "",
            "Conflicts:",
        ]
        for name_a, name_b, pairs in overlaps:
            pretty_pairs = ", ".join(f"'{pa}' <-> '{pb}'" for pa, pb in pairs)
            lines.append(f"  - {name_a} <-> {name_b}: {pretty_pairs}")
        lines.extend(
            [
                "",
                "Resolve by migrating overlapping indexes one at a time, or by",
                "narrowing the batch to a set of indexes with disjoint prefixes.",
            ]
        )
        return "\n".join(lines)

    def write_batch_plan(self, batch_plan: BatchPlan, path: str) -> None:
        """Write batch plan to YAML file."""
        plan_path = Path(path).resolve()
        with open(plan_path, "w") as f:
            yaml.safe_dump(batch_plan.model_dump(exclude_none=True), f, sort_keys=False)
