from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml

from redisvl.migration.models import FieldUpdate, SchemaPatch, SchemaPatchChanges
from redisvl.migration.planner import MigrationPlanner
from redisvl.migration.utils import list_indexes, write_yaml
from redisvl.schema.schema import IndexSchema

SUPPORTED_FIELD_TYPES = ["text", "tag", "numeric", "geo"]
UPDATABLE_FIELD_TYPES = ["text", "tag", "numeric", "geo", "vector"]


class MigrationWizard:
    def __init__(self, planner: Optional[MigrationPlanner] = None):
        self.planner = planner or MigrationPlanner()

    def run(
        self,
        *,
        index_name: Optional[str] = None,
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
        existing_patch_path: Optional[str] = None,
        plan_out: str = "migration_plan.yaml",
        patch_out: Optional[str] = None,
        target_schema_out: Optional[str] = None,
    ):
        resolved_index_name = self._resolve_index_name(
            index_name=index_name,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        snapshot = self.planner.snapshot_source(
            resolved_index_name,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        source_schema = IndexSchema.from_dict(snapshot.schema_snapshot)

        print(f"Building a migration plan for index '{resolved_index_name}'")
        self._print_source_schema(source_schema.to_dict())

        # Load existing patch if provided
        existing_changes = None
        if existing_patch_path:
            existing_changes = self._load_existing_patch(existing_patch_path)

        schema_patch = self._build_patch(
            source_schema.to_dict(), existing_changes=existing_changes
        )
        plan = self.planner.create_plan_from_patch(
            resolved_index_name,
            schema_patch=schema_patch,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        self.planner.write_plan(plan, plan_out)

        if patch_out:
            write_yaml(schema_patch.model_dump(exclude_none=True), patch_out)
        if target_schema_out:
            write_yaml(plan.merged_target_schema, target_schema_out)

        return plan

    def _load_existing_patch(self, patch_path: str) -> SchemaPatchChanges:
        from redisvl.migration.utils import load_yaml

        data = load_yaml(patch_path)
        patch = SchemaPatch.model_validate(data)
        print(f"Loaded existing patch from {patch_path}")
        print(f"  Add fields: {len(patch.changes.add_fields)}")
        print(f"  Update fields: {len(patch.changes.update_fields)}")
        print(f"  Remove fields: {len(patch.changes.remove_fields)}")
        return patch.changes

    def _resolve_index_name(
        self,
        *,
        index_name: Optional[str],
        redis_url: Optional[str],
        redis_client: Optional[Any],
    ) -> str:
        if index_name:
            return index_name

        indexes = list_indexes(redis_url=redis_url, redis_client=redis_client)
        if not indexes:
            raise ValueError("No indexes found in Redis")

        print("Available indexes:")
        for position, name in enumerate(indexes, start=1):
            print(f"{position}. {name}")

        while True:
            choice = input("Select an index by number or name: ").strip()
            if choice in indexes:
                return choice
            if choice.isdigit():
                offset = int(choice) - 1
                if 0 <= offset < len(indexes):
                    return indexes[offset]
            print("Invalid selection. Please try again.")

    def _build_patch(
        self,
        source_schema: Dict[str, Any],
        existing_changes: Optional[SchemaPatchChanges] = None,
    ) -> SchemaPatch:
        if existing_changes:
            changes = existing_changes
        else:
            changes = SchemaPatchChanges()
        done = False
        while not done:
            print("\nChoose an action:")
            print("1. Add field        (text, tag, numeric, geo)")
            print("2. Update field     (sortable, weight, separator, vector config)")
            print("3. Remove field")
            print("4. Preview patch    (show pending changes as YAML)")
            print("5. Finish")
            action = input("Enter a number: ").strip()

            if action == "1":
                field = self._prompt_add_field(source_schema)
                if field:
                    changes.add_fields.append(field)
            elif action == "2":
                update = self._prompt_update_field(source_schema)
                if update:
                    changes.update_fields.append(update)
            elif action == "3":
                field_name = self._prompt_remove_field(source_schema)
                if field_name:
                    changes.remove_fields.append(field_name)
            elif action == "4":
                print(
                    yaml.safe_dump(
                        {"version": 1, "changes": changes.model_dump()}, sort_keys=False
                    )
                )
            elif action == "5":
                done = True
            else:
                print("Invalid action. Please choose 1-5.")

        return SchemaPatch(version=1, changes=changes)

    def _prompt_add_field(
        self, source_schema: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        field_name = input("Field name: ").strip()
        existing_names = {field["name"] for field in source_schema["fields"]}
        if not field_name:
            print("Field name is required.")
            return None
        if field_name in existing_names:
            print(f"Field '{field_name}' already exists in the source schema.")
            return None

        field_type = self._prompt_from_choices(
            "Field type",
            SUPPORTED_FIELD_TYPES,
            block_message="Vector fields cannot be added (requires embedding all documents). Only text, tag, numeric, and geo are supported.",
        )
        if not field_type:
            return None

        field: Dict[str, Any] = {"name": field_name, "type": field_type}
        storage_type = source_schema["index"]["storage_type"]
        if storage_type == "json":
            print("  JSON path: location in document where this field is stored")
            path = (
                input(f"JSON path [default $.{field_name}]: ").strip()
                or f"$.{field_name}"
            )
            field["path"] = path

        attrs = self._prompt_common_attrs(field_type)
        if attrs:
            field["attrs"] = attrs
        return field

    def _prompt_update_field(
        self, source_schema: Dict[str, Any]
    ) -> Optional[FieldUpdate]:
        fields = [
            field
            for field in source_schema["fields"]
            if field["type"] in UPDATABLE_FIELD_TYPES
        ]
        if not fields:
            print("No updatable fields are available.")
            return None

        print("Updatable fields:")
        for position, field in enumerate(fields, start=1):
            print(f"{position}. {field['name']} ({field['type']})")

        choice = input("Select a field to update by number or name: ").strip()
        selected: Optional[Dict[str, Any]] = None
        for position, field in enumerate(fields, start=1):
            if choice == str(position) or choice == field["name"]:
                selected = field
                break
        if not selected:
            print("Invalid field selection.")
            return None

        if selected["type"] == "vector":
            attrs = self._prompt_vector_attrs(selected)
        else:
            attrs = self._prompt_common_attrs(selected["type"], allow_blank=True)
        if not attrs:
            print("No changes collected.")
            return None
        return FieldUpdate(name=selected["name"], attrs=attrs)

    def _prompt_remove_field(self, source_schema: Dict[str, Any]) -> Optional[str]:
        removable_fields = [
            field["name"]
            for field in source_schema["fields"]
            if field["type"] != "vector"
        ]
        if not removable_fields:
            print("No removable Phase 1 fields are available.")
            return None

        print("Removable fields:")
        for position, field_name in enumerate(removable_fields, start=1):
            print(f"{position}. {field_name}")

        choice = input("Select a field to remove by number or name: ").strip()
        if choice in removable_fields:
            return choice
        if choice.isdigit():
            offset = int(choice) - 1
            if 0 <= offset < len(removable_fields):
                return removable_fields[offset]
        print("Invalid field selection.")
        return None

    def _prompt_common_attrs(
        self, field_type: str, allow_blank: bool = False
    ) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}

        # Sortable - available for all non-vector types
        print("  Sortable: enables sorting and aggregation on this field")
        sortable = self._prompt_bool("Sortable", allow_blank=allow_blank)
        if sortable is not None:
            attrs["sortable"] = sortable

        # Index missing - available for all types (requires Redis Search 2.10+)
        print(
            "  Index missing: enables ismissing() queries for documents without this field"
        )
        index_missing = self._prompt_bool("Index missing", allow_blank=allow_blank)
        if index_missing is not None:
            attrs["index_missing"] = index_missing

        # Type-specific attributes
        if field_type == "text":
            self._prompt_text_attrs(attrs, allow_blank)
        elif field_type == "tag":
            self._prompt_tag_attrs(attrs, allow_blank)
        elif field_type == "numeric":
            self._prompt_numeric_attrs(attrs, allow_blank, sortable)

        # No index - only meaningful with sortable
        if sortable or (allow_blank and attrs.get("sortable")):
            print("  No index: store field for sorting only, not searchable")
            no_index = self._prompt_bool("No index", allow_blank=allow_blank)
            if no_index is not None:
                attrs["no_index"] = no_index

        return attrs

    def _prompt_text_attrs(self, attrs: Dict[str, Any], allow_blank: bool) -> None:
        """Prompt for text field specific attributes."""
        # No stem
        print(
            "  Disable stemming: prevents word variations (running/runs) from matching"
        )
        no_stem = self._prompt_bool("Disable stemming", allow_blank=allow_blank)
        if no_stem is not None:
            attrs["no_stem"] = no_stem

        # Weight
        print("  Weight: relevance multiplier for full-text search (default: 1.0)")
        weight_input = input("Weight [leave blank for default]: ").strip()
        if weight_input:
            try:
                weight = float(weight_input)
                if weight > 0:
                    attrs["weight"] = weight
                else:
                    print("Weight must be positive.")
            except ValueError:
                print("Invalid weight value.")

        # Index empty (requires Redis Search 2.10+)
        print("  Index empty: enables searching for empty string values")
        index_empty = self._prompt_bool("Index empty", allow_blank=allow_blank)
        if index_empty is not None:
            attrs["index_empty"] = index_empty

        # UNF (only if sortable)
        if attrs.get("sortable"):
            print("  UNF: preserve original form (no lowercasing) for sorting")
            unf = self._prompt_bool("UNF (un-normalized form)", allow_blank=allow_blank)
            if unf is not None:
                attrs["unf"] = unf

    def _prompt_tag_attrs(self, attrs: Dict[str, Any], allow_blank: bool) -> None:
        """Prompt for tag field specific attributes."""
        # Separator
        print("  Separator: character that splits multiple values (default: comma)")
        separator = input("Separator [leave blank to keep existing/default]: ").strip()
        if separator:
            attrs["separator"] = separator

        # Case sensitive
        print("  Case sensitive: match tags with exact case (default: false)")
        case_sensitive = self._prompt_bool("Case sensitive", allow_blank=allow_blank)
        if case_sensitive is not None:
            attrs["case_sensitive"] = case_sensitive

        # Index empty (requires Redis Search 2.10+)
        print("  Index empty: enables searching for empty tag values")
        index_empty = self._prompt_bool("Index empty", allow_blank=allow_blank)
        if index_empty is not None:
            attrs["index_empty"] = index_empty

    def _prompt_numeric_attrs(
        self, attrs: Dict[str, Any], allow_blank: bool, sortable: Optional[bool]
    ) -> None:
        """Prompt for numeric field specific attributes."""
        # UNF (only if sortable)
        if sortable or attrs.get("sortable"):
            print("  UNF: preserve exact numeric representation for sorting")
            unf = self._prompt_bool("UNF (un-normalized form)", allow_blank=allow_blank)
            if unf is not None:
                attrs["unf"] = unf

    def _prompt_vector_attrs(self, field: Dict[str, Any]) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        current = field.get("attrs", {})
        field_name = field["name"]

        print(f"Current vector config for '{field_name}':")
        print(f"  algorithm: {current.get('algorithm', 'HNSW')}")
        print(f"  datatype: {current.get('datatype', 'float32')}")
        print(f"  distance_metric: {current.get('distance_metric', 'cosine')}")
        print(f"  dims: {current.get('dims')} (cannot be changed)")
        if current.get("algorithm", "HNSW") == "HNSW":
            print(f"  m: {current.get('m', 16)}")
            print(f"  ef_construction: {current.get('ef_construction', 200)}")

        print("\nLeave blank to keep current value.")

        # Algorithm
        print(
            "  Algorithm: vector search method (FLAT=brute force, HNSW=graph, SVS-VAMANA=compressed graph)"
        )
        algo = (
            input(f"Algorithm [current: {current.get('algorithm', 'HNSW')}]: ")
            .strip()
            .upper()
            .replace("_", "-")  # Normalize SVS_VAMANA to SVS-VAMANA
        )
        if algo and algo in ("FLAT", "HNSW", "SVS-VAMANA"):
            attrs["algorithm"] = algo

        # Datatype (quantization) - show algorithm-specific options
        effective_algo = attrs.get(
            "algorithm", current.get("algorithm", "HNSW")
        ).upper()
        if effective_algo == "SVS-VAMANA":
            # SVS-VAMANA only supports float16, float32
            print(
                "  Datatype for SVS-VAMANA: float16, float32 "
                "(float16 reduces memory by ~50%)"
            )
            valid_datatypes = ("float16", "float32")
        else:
            # FLAT/HNSW support: float16, float32, bfloat16, float64, int8, uint8
            print(
                "  Datatype: float16, float32, bfloat16, float64, int8, uint8\n"
                "            (float16 reduces memory ~50%, int8/uint8 reduce ~75%)"
            )
            valid_datatypes = (
                "float16",
                "float32",
                "bfloat16",
                "float64",
                "int8",
                "uint8",
            )
        datatype = (
            input(f"Datatype [current: {current.get('datatype', 'float32')}]: ")
            .strip()
            .lower()
        )
        if datatype and datatype in valid_datatypes:
            attrs["datatype"] = datatype

        # Distance metric
        print("  Distance metric: how similarity is measured (cosine, l2, ip)")
        metric = (
            input(
                f"Distance metric [current: {current.get('distance_metric', 'cosine')}]: "
            )
            .strip()
            .lower()
        )
        if metric and metric in ("cosine", "l2", "ip"):
            attrs["distance_metric"] = metric

        # Algorithm-specific params (effective_algo already computed above)
        if effective_algo == "HNSW":
            print(
                "  M: number of connections per node (higher=better recall, more memory)"
            )
            m_input = input(f"M [current: {current.get('m', 16)}]: ").strip()
            if m_input and m_input.isdigit():
                attrs["m"] = int(m_input)

            print(
                "  EF_CONSTRUCTION: build-time search depth (higher=better recall, slower build)"
            )
            ef_input = input(
                f"EF_CONSTRUCTION [current: {current.get('ef_construction', 200)}]: "
            ).strip()
            if ef_input and ef_input.isdigit():
                attrs["ef_construction"] = int(ef_input)

        elif effective_algo == "SVS-VAMANA":
            print(
                "  GRAPH_MAX_DEGREE: max edges per node (higher=better recall, more memory)"
            )
            gmd_input = input(
                f"GRAPH_MAX_DEGREE [current: {current.get('graph_max_degree', 40)}]: "
            ).strip()
            if gmd_input and gmd_input.isdigit():
                attrs["graph_max_degree"] = int(gmd_input)

            print("  COMPRESSION: optional vector compression for memory savings")
            print("    Options: LVQ4, LVQ8, LVQ4x4, LVQ4x8, LeanVec4x8, LeanVec8x8")
            compression = input("COMPRESSION [leave blank for none]: ").strip().upper()
            if compression and compression in (
                "LVQ4",
                "LVQ8",
                "LVQ4X4",
                "LVQ4X8",
                "LEANVEC4X8",
                "LEANVEC8X8",
            ):
                attrs["compression"] = compression

        return attrs

    def _prompt_bool(self, label: str, allow_blank: bool = False) -> Optional[bool]:
        suffix = " [y/n]" if not allow_blank else " [y/n/skip]"
        while True:
            value = input(f"{label}{suffix}: ").strip().lower()
            if value in ("y", "yes"):
                return True
            if value in ("n", "no"):
                return False
            if allow_blank and value in ("", "skip", "s"):
                return None
            if not allow_blank and value == "":
                return False
            print("Please answer y, n, or skip.")

    def _prompt_from_choices(
        self,
        label: str,
        choices: List[str],
        *,
        block_message: str,
    ) -> Optional[str]:
        print(f"{label} options: {', '.join(choices)}")
        value = input(f"{label}: ").strip().lower()
        if value not in choices:
            print(block_message)
            return None
        return value

    def _print_source_schema(self, schema_dict: Dict[str, Any]) -> None:
        print("Current schema:")
        print(f"- Index name: {schema_dict['index']['name']}")
        print(f"- Storage type: {schema_dict['index']['storage_type']}")
        for field in schema_dict["fields"]:
            path = field.get("path")
            suffix = f" path={path}" if path else ""
            print(f"  - {field['name']} ({field['type']}){suffix}")
