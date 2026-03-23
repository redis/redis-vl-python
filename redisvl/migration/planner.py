from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from redisvl.index import SearchIndex
from redisvl.migration.models import (
    DiffClassification,
    FieldRename,
    KeyspaceSnapshot,
    MigrationPlan,
    RenameOperations,
    SchemaPatch,
    SourceSnapshot,
)
from redisvl.redis.connection import supports_svs
from redisvl.schema.schema import IndexSchema


class MigrationPlanner:
    """Migration planner for document-preserving drop/recreate flows.

    The `drop_recreate` mode drops the index definition and recreates it with
    a new schema. Documents remain untouched in Redis.

    This means:
    - Index-only changes work (algorithm, distance metric, tuning params)
    - Document-dependent changes fail (the index expects data in a format
      that doesn't match what's stored)

    Document-dependent changes (not supported):
    - Vector dimensions: stored vectors have wrong number of dimensions
    - Prefix/keyspace: documents are at keys the new index won't scan
    - Field rename: documents store data under the old field name
    - Storage type: documents are in hash format but index expects JSON
    """

    def __init__(self, key_sample_limit: int = 10):
        self.key_sample_limit = key_sample_limit

    def create_plan(
        self,
        index_name: str,
        *,
        redis_url: Optional[str] = None,
        schema_patch_path: Optional[str] = None,
        target_schema_path: Optional[str] = None,
        redis_client: Optional[Any] = None,
    ) -> MigrationPlan:
        if not schema_patch_path and not target_schema_path:
            raise ValueError(
                "Must provide either --schema-patch or --target-schema for migration planning"
            )
        if schema_patch_path and target_schema_path:
            raise ValueError(
                "Provide only one of --schema-patch or --target-schema for migration planning"
            )

        snapshot = self.snapshot_source(
            index_name,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        source_schema = IndexSchema.from_dict(snapshot.schema_snapshot)

        if schema_patch_path:
            schema_patch = self.load_schema_patch(schema_patch_path)
        else:
            # target_schema_path is guaranteed non-None here due to validation above
            assert target_schema_path is not None
            schema_patch = self.normalize_target_schema_to_patch(
                source_schema, target_schema_path
            )

        return self.create_plan_from_patch(
            index_name,
            schema_patch=schema_patch,
            redis_url=redis_url,
            redis_client=redis_client,
        )

    def create_plan_from_patch(
        self,
        index_name: str,
        *,
        schema_patch: SchemaPatch,
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
    ) -> MigrationPlan:
        snapshot = self.snapshot_source(
            index_name,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        source_schema = IndexSchema.from_dict(snapshot.schema_snapshot)
        merged_target_schema = self.merge_patch(source_schema, schema_patch)

        # Extract rename operations first
        rename_operations, rename_warnings = self._extract_rename_operations(
            source_schema, schema_patch
        )

        # Classify diff with awareness of rename operations
        diff_classification = self.classify_diff(
            source_schema, schema_patch, merged_target_schema, rename_operations
        )

        # Build warnings list
        warnings = ["Index downtime is required"]
        warnings.extend(rename_warnings)

        # Check for SVS-VAMANA in target schema and add appropriate warnings
        svs_warnings = self._check_svs_vamana_requirements(
            merged_target_schema,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        warnings.extend(svs_warnings)

        return MigrationPlan(
            source=snapshot,
            requested_changes=schema_patch.model_dump(exclude_none=True),
            merged_target_schema=merged_target_schema.to_dict(),
            diff_classification=diff_classification,
            rename_operations=rename_operations,
            warnings=warnings,
        )

    def snapshot_source(
        self,
        index_name: str,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
    ) -> SourceSnapshot:
        index = SearchIndex.from_existing(
            index_name,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        schema_dict = index.schema.to_dict()
        stats_snapshot = index.info()
        prefixes = index.schema.index.prefix
        prefix_list = prefixes if isinstance(prefixes, list) else [prefixes]

        return SourceSnapshot(
            index_name=index_name,
            schema_snapshot=schema_dict,
            stats_snapshot=stats_snapshot,
            keyspace=KeyspaceSnapshot(
                storage_type=index.schema.index.storage_type.value,
                prefixes=prefix_list,
                key_separator=index.schema.index.key_separator,
                key_sample=self._sample_keys(
                    client=index.client,
                    prefixes=prefix_list,
                    key_separator=index.schema.index.key_separator,
                ),
            ),
        )

    def load_schema_patch(self, schema_patch_path: str) -> SchemaPatch:
        patch_path = Path(schema_patch_path).resolve()
        if not patch_path.exists():
            raise FileNotFoundError(
                f"Schema patch file {schema_patch_path} does not exist"
            )

        with open(patch_path, "r") as f:
            patch_data = yaml.safe_load(f) or {}
        return SchemaPatch.model_validate(patch_data)

    def normalize_target_schema_to_patch(
        self, source_schema: IndexSchema, target_schema_path: str
    ) -> SchemaPatch:
        target_schema = IndexSchema.from_yaml(target_schema_path)
        source_dict = source_schema.to_dict()
        target_dict = target_schema.to_dict()

        changes: Dict[str, Any] = {
            "add_fields": [],
            "remove_fields": [],
            "update_fields": [],
            "index": {},
        }

        source_fields = {field["name"]: field for field in source_dict["fields"]}
        target_fields = {field["name"]: field for field in target_dict["fields"]}

        for field_name, target_field in target_fields.items():
            if field_name not in source_fields:
                changes["add_fields"].append(target_field)
            elif source_fields[field_name] != target_field:
                changes["update_fields"].append(target_field)

        for field_name in source_fields:
            if field_name not in target_fields:
                changes["remove_fields"].append(field_name)

        for index_key, target_value in target_dict["index"].items():
            source_value = source_dict["index"].get(index_key)
            if source_value != target_value:
                changes["index"][index_key] = target_value

        return SchemaPatch.model_validate({"version": 1, "changes": changes})

    def merge_patch(
        self, source_schema: IndexSchema, schema_patch: SchemaPatch
    ) -> IndexSchema:
        schema_dict = deepcopy(source_schema.to_dict())
        changes = schema_patch.changes
        fields_by_name = {
            field["name"]: deepcopy(field) for field in schema_dict["fields"]
        }

        for field_name in changes.remove_fields:
            fields_by_name.pop(field_name, None)

        for field_update in changes.update_fields:
            if field_update.name not in fields_by_name:
                raise ValueError(
                    f"Cannot update field '{field_update.name}' because it does not exist in the source schema"
                )
            existing_field = fields_by_name[field_update.name]
            if field_update.type is not None:
                existing_field["type"] = field_update.type
            if field_update.path is not None:
                existing_field["path"] = field_update.path
            if field_update.attrs:
                merged_attrs = dict(existing_field.get("attrs", {}))
                merged_attrs.update(field_update.attrs)
                existing_field["attrs"] = merged_attrs

        for field in changes.add_fields:
            field_name = field["name"]
            if field_name in fields_by_name:
                raise ValueError(
                    f"Cannot add field '{field_name}' because it already exists in the source schema"
                )
            fields_by_name[field_name] = deepcopy(field)

        schema_dict["fields"] = list(fields_by_name.values())
        schema_dict["index"].update(changes.index)
        return IndexSchema.from_dict(schema_dict)

    def _extract_rename_operations(
        self,
        source_schema: IndexSchema,
        schema_patch: SchemaPatch,
    ) -> Tuple[RenameOperations, List[str]]:
        """Extract rename operations from the patch and generate warnings.

        Returns:
            Tuple of (RenameOperations, warnings list)
        """
        source_dict = source_schema.to_dict()
        changes = schema_patch.changes
        warnings: List[str] = []

        # Index rename
        rename_index: Optional[str] = None
        if "name" in changes.index:
            new_name = changes.index["name"]
            old_name = source_dict["index"].get("name")
            if new_name != old_name:
                rename_index = new_name
                warnings.append(
                    f"Index rename: '{old_name}' -> '{new_name}' (index-only change, no document migration needed)"
                )

        # Prefix change
        change_prefix: Optional[str] = None
        if "prefix" in changes.index:
            new_prefix = changes.index["prefix"]
            old_prefix = source_dict["index"].get("prefix")
            if new_prefix != old_prefix:
                change_prefix = new_prefix
                warnings.append(
                    f"Prefix change: '{old_prefix}' -> '{new_prefix}' "
                    "(requires RENAME for all keys, may be slow for large datasets)"
                )

        # Field renames from explicit rename_fields
        rename_fields: List[FieldRename] = list(changes.rename_fields)
        for field_rename in rename_fields:
            warnings.append(
                f"Field rename: '{field_rename.old_name}' -> '{field_rename.new_name}' "
                "(requires read/write for all documents, may be slow for large datasets)"
            )

        return (
            RenameOperations(
                rename_index=rename_index,
                change_prefix=change_prefix,
                rename_fields=rename_fields,
            ),
            warnings,
        )

    def _check_svs_vamana_requirements(
        self,
        target_schema: IndexSchema,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
    ) -> List[str]:
        """Check SVS-VAMANA requirements and return warnings.

        Checks:
        1. If target uses SVS-VAMANA, verify Redis version supports it
        2. Add Intel hardware warning for LVQ/LeanVec optimizations
        """
        warnings: List[str] = []
        target_dict = target_schema.to_dict()

        # Check if any vector field uses SVS-VAMANA
        uses_svs = False
        uses_compression = False
        compression_type = None

        for field in target_dict.get("fields", []):
            if field.get("type") != "vector":
                continue
            attrs = field.get("attrs", {})
            algo = attrs.get("algorithm", "").upper()
            if algo == "SVS-VAMANA":
                uses_svs = True
                compression = attrs.get("compression", "")
                if compression:
                    uses_compression = True
                    compression_type = compression

        if not uses_svs:
            return warnings

        # Check Redis version support
        try:
            if redis_client:
                client = redis_client
            elif redis_url:
                from redis import Redis

                client = Redis.from_url(redis_url)
            else:
                client = None

            if client and not supports_svs(client):
                warnings.append(
                    "SVS-VAMANA requires Redis >= 8.2.0 and Redis Search >= 2.8.10. "
                    "The target Redis instance may not support this algorithm. "
                    "Migration will fail at apply time if requirements are not met."
                )
        except Exception:
            # If we can't check, add a general warning
            warnings.append(
                "SVS-VAMANA requires Redis >= 8.2.0 and Redis Search >= 2.8.10. "
                "Verify your Redis instance supports this algorithm before applying."
            )

        # Intel hardware warning for compression
        if uses_compression:
            warnings.append(
                f"SVS-VAMANA with {compression_type} compression: "
                "LVQ and LeanVec optimizations require Intel hardware with AVX-512 support. "
                "On non-Intel platforms or Redis Open Source, these fall back to basic "
                "8-bit scalar quantization with reduced performance benefits."
            )
        else:
            warnings.append(
                "SVS-VAMANA: For optimal performance, Intel hardware with AVX-512 support "
                "is recommended. LVQ/LeanVec compression options provide additional memory "
                "savings on supported hardware."
            )

        return warnings

    def classify_diff(
        self,
        source_schema: IndexSchema,
        schema_patch: SchemaPatch,
        merged_target_schema: IndexSchema,
        rename_operations: Optional[RenameOperations] = None,
    ) -> DiffClassification:
        blocked_reasons: List[str] = []
        changes = schema_patch.changes
        source_dict = source_schema.to_dict()
        target_dict = merged_target_schema.to_dict()

        # Check which rename operations are being handled
        has_index_rename = rename_operations and rename_operations.rename_index
        has_prefix_change = rename_operations and rename_operations.change_prefix
        has_field_renames = (
            rename_operations and len(rename_operations.rename_fields) > 0
        )
        renamed_field_names = set()
        if has_field_renames and rename_operations:
            renamed_field_names = {
                fr.old_name for fr in rename_operations.rename_fields
            }

        for index_key, target_value in changes.index.items():
            source_value = source_dict["index"].get(index_key)
            if source_value == target_value:
                continue
            if index_key == "name":
                # Index rename is now supported - skip blocking if we have rename_operations
                if not has_index_rename:
                    blocked_reasons.append(
                        "Changing the index name requires document migration (not yet supported)."
                    )
            elif index_key == "prefix":
                # Prefix change is now supported
                if not has_prefix_change:
                    blocked_reasons.append(
                        "Changing index prefixes requires document migration (not yet supported)."
                    )
            elif index_key == "key_separator":
                blocked_reasons.append(
                    "Changing the key separator requires document migration (not yet supported)."
                )
            elif index_key == "storage_type":
                blocked_reasons.append(
                    "Changing the storage type requires document migration (not yet supported)."
                )

        source_fields = {field["name"]: field for field in source_dict["fields"]}
        target_fields = {field["name"]: field for field in target_dict["fields"]}

        for field in changes.add_fields:
            if field["type"] == "vector":
                blocked_reasons.append(
                    f"Adding vector field '{field['name']}' requires document migration (not yet supported)."
                )

        for field_update in changes.update_fields:
            source_field = source_fields[field_update.name]
            target_field = target_fields[field_update.name]
            source_type = source_field["type"]
            target_type = target_field["type"]

            if source_type != target_type:
                blocked_reasons.append(
                    f"Changing field '{field_update.name}' type from {source_type} to {target_type} is not supported by drop_recreate."
                )
                continue

            source_path = source_field.get("path")
            target_path = target_field.get("path")
            if source_path != target_path:
                blocked_reasons.append(
                    f"Changing field '{field_update.name}' path from {source_path} to {target_path} is not supported by drop_recreate."
                )
                continue

            if target_type == "vector" and source_field != target_field:
                # Check for document-dependent changes that are not yet supported
                vector_blocked = self._classify_vector_field_change(
                    source_field, target_field
                )
                blocked_reasons.extend(vector_blocked)

        # Detect possible field renames only if not explicitly provided
        if not has_field_renames:
            blocked_reasons.extend(
                self._detect_possible_field_renames(source_fields, target_fields)
            )

        return DiffClassification(
            supported=len(blocked_reasons) == 0,
            blocked_reasons=self._dedupe(blocked_reasons),
        )

    def write_plan(self, plan: MigrationPlan, plan_out: str) -> None:
        plan_path = Path(plan_out).resolve()
        with open(plan_path, "w") as f:
            yaml.safe_dump(plan.model_dump(exclude_none=True), f, sort_keys=False)

    def _sample_keys(
        self, *, client: Any, prefixes: List[str], key_separator: str
    ) -> List[str]:
        key_sample: List[str] = []
        if client is None or self.key_sample_limit <= 0:
            return key_sample

        for prefix in prefixes:
            if len(key_sample) >= self.key_sample_limit:
                break
            match_pattern = (
                f"{prefix}*"
                if prefix.endswith(key_separator)
                else f"{prefix}{key_separator}*"
            )
            cursor = 0
            while True:
                cursor, keys = client.scan(
                    cursor=cursor,
                    match=match_pattern,
                    count=max(self.key_sample_limit, 10),
                )
                for key in keys:
                    decoded_key = key.decode() if isinstance(key, bytes) else str(key)
                    if decoded_key not in key_sample:
                        key_sample.append(decoded_key)
                    if len(key_sample) >= self.key_sample_limit:
                        return key_sample
                if cursor == 0:
                    break
        return key_sample

    def _detect_possible_field_renames(
        self,
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        blocked_reasons: List[str] = []
        added_fields = [
            field for name, field in target_fields.items() if name not in source_fields
        ]
        removed_fields = [
            field for name, field in source_fields.items() if name not in target_fields
        ]

        for removed_field in removed_fields:
            for added_field in added_fields:
                if self._fields_match_except_name(removed_field, added_field):
                    blocked_reasons.append(
                        f"Possible field rename from '{removed_field['name']}' to '{added_field['name']}' is not supported by drop_recreate."
                    )
        return blocked_reasons

    @staticmethod
    def _classify_vector_field_change(
        source_field: Dict[str, Any], target_field: Dict[str, Any]
    ) -> List[str]:
        """Classify vector field changes as supported or blocked for drop_recreate.

        Index-only changes (allowed with drop_recreate):
            - algorithm (FLAT -> HNSW -> SVS-VAMANA)
            - distance_metric (COSINE, L2, IP)
            - initial_cap
            - Algorithm tuning: m, ef_construction, ef_runtime, epsilon, block_size,
              graph_max_degree, construction_window_size, search_window_size, etc.

        Quantization changes (allowed with drop_recreate, requires vector re-encoding):
            - datatype (float32 -> float16, etc.) - executor will re-encode vectors

        Document-dependent changes (blocked, not yet supported):
            - dims (vectors stored with wrong number of dimensions)
        """
        blocked_reasons: List[str] = []
        field_name = source_field.get("name", "unknown")
        source_attrs = source_field.get("attrs", {})
        target_attrs = target_field.get("attrs", {})

        # Document-dependent properties (not yet supported)
        if source_attrs.get("dims") != target_attrs.get("dims"):
            blocked_reasons.append(
                f"Changing vector field '{field_name}' dims from {source_attrs.get('dims')} "
                f"to {target_attrs.get('dims')} requires document migration (not yet supported). "
                "Vectors are stored with incompatible dimensions."
            )

        # Datatype changes are now ALLOWED - executor will re-encode vectors
        # before recreating the index

        # All other vector changes are index-only and allowed
        return blocked_reasons

    @staticmethod
    def get_vector_datatype_changes(
        source_schema: Dict[str, Any], target_schema: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        """Identify vector fields that need datatype conversion (quantization).

        Returns:
            Dict mapping field_name -> {"source": source_dtype, "target": target_dtype}
        """
        changes: Dict[str, Dict[str, str]] = {}
        source_fields = {f["name"]: f for f in source_schema.get("fields", [])}
        target_fields = {f["name"]: f for f in target_schema.get("fields", [])}

        for name, source_field in source_fields.items():
            if source_field.get("type") != "vector":
                continue
            target_field = target_fields.get(name)
            if not target_field or target_field.get("type") != "vector":
                continue

            source_dtype = source_field.get("attrs", {}).get("datatype", "float32")
            target_dtype = target_field.get("attrs", {}).get("datatype", "float32")

            if source_dtype != target_dtype:
                changes[name] = {"source": source_dtype, "target": target_dtype}

        return changes

    @staticmethod
    def _fields_match_except_name(
        source_field: Dict[str, Any], target_field: Dict[str, Any]
    ) -> bool:
        comparable_source = {k: v for k, v in source_field.items() if k != "name"}
        comparable_target = {k: v for k, v in target_field.items() if k != "name"}
        return comparable_source == comparable_target

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        deduped: List[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
