from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml

from redisvl.migration.models import (
    FieldRename,
    FieldUpdate,
    SchemaPatch,
    SchemaPatchChanges,
)
from redisvl.migration.planner import MigrationPlanner
from redisvl.migration.utils import list_indexes, write_yaml
from redisvl.schema.schema import IndexSchema

SUPPORTED_FIELD_TYPES = ["text", "tag", "numeric", "geo"]
UPDATABLE_FIELD_TYPES = ["text", "tag", "numeric", "geo", "vector"]


class MigrationWizard:
    def __init__(self, planner: Optional[MigrationPlanner] = None):
        self.planner = planner or MigrationPlanner()
        self._existing_sortable: bool = False

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

        # Guard: the wizard does not support indexes with multiple prefixes.
        prefixes = source_schema.index.prefix
        if isinstance(prefixes, list) and len(prefixes) > 1:
            raise ValueError(
                f"Index '{resolved_index_name}' has multiple prefixes "
                f"({prefixes}). The migration wizard only supports single-prefix "
                "indexes. Use the planner API directly for multi-prefix indexes."
            )

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
        print(f"  Rename fields: {len(patch.changes.rename_fields)}")
        if patch.changes.index:
            print(f"  Index changes: {list(patch.changes.index.keys())}")
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

    @staticmethod
    def _filter_staged_adds(
        working_schema: Dict[str, Any], staged_add_names: set
    ) -> Dict[str, Any]:
        """Return a copy of working_schema with staged-add fields removed.

        This prevents staged additions from appearing in update/rename
        candidate lists.
        """
        import copy

        filtered = copy.deepcopy(working_schema)
        filtered["fields"] = [
            f for f in filtered["fields"] if f["name"] not in staged_add_names
        ]
        return filtered

    def _apply_staged_changes(
        self,
        source_schema: Dict[str, Any],
        changes: SchemaPatchChanges,
    ) -> Dict[str, Any]:
        """Build a working copy of source_schema reflecting staged changes.

        This ensures subsequent prompts show the current state of the schema
        after renames, removes, and adds have been queued.
        """
        import copy

        working = copy.deepcopy(source_schema)

        # Apply removes
        removed_names = set(changes.remove_fields)
        working["fields"] = [
            f for f in working["fields"] if f["name"] not in removed_names
        ]

        # Apply renames.  Apply each rename sequentially so that chained
        # renames (A→B, B→C) are handled correctly even if they weren't
        # collapsed at input time.
        rename_map = {r.old_name: r.new_name for r in changes.rename_fields}
        for r in changes.rename_fields:
            for field in working["fields"]:
                if field["name"] == r.old_name:
                    field["name"] = r.new_name
                    break

        # Apply updates (reflect attribute changes in working schema).
        # Resolve update names through the rename map so that updates staged
        # before a rename (referencing the old name) still match.
        update_map = {}
        for u in changes.update_fields:
            resolved = rename_map.get(u.name, u.name)
            update_map[resolved] = u
        for field in working["fields"]:
            if field["name"] in update_map:
                upd = update_map[field["name"]]
                if upd.attrs:
                    field.setdefault("attrs", {}).update(upd.attrs)
                if upd.type:
                    field["type"] = upd.type

        # Apply adds
        for added in changes.add_fields:
            working["fields"].append(added)

        # Apply index-level changes (name, prefix) so preview reflects them
        if changes.index:
            for key, value in changes.index.items():
                working["index"][key] = value

        return working

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
            # Refresh working schema to reflect staged changes
            working_schema = self._apply_staged_changes(source_schema, changes)

            print("\nChoose an action:")
            print("1. Add field        (text, tag, numeric, geo)")
            print("2. Update field     (sortable, weight, separator, vector config)")
            print("3. Remove field")
            print("4. Rename field     (rename field in all documents)")
            print("5. Rename index     (change index name)")
            print("6. Change prefix    (rename all keys)")
            print("7. Preview patch    (show pending changes as YAML)")
            print("8. Finish")
            action = input("Enter a number: ").strip()

            if action == "1":
                field = self._prompt_add_field(working_schema)
                if field:
                    staged_names = {f["name"] for f in changes.add_fields}
                    if field["name"] in staged_names:
                        print(
                            f"Field '{field['name']}' is already staged for addition."
                        )
                    else:
                        changes.add_fields.append(field)
            elif action == "2":
                # Filter out staged additions from update candidates
                staged_add_names = {f["name"] for f in changes.add_fields}
                update_schema = self._filter_staged_adds(
                    working_schema, staged_add_names
                )
                update = self._prompt_update_field(update_schema)
                if update:
                    # Merge with existing update for same field if present
                    existing = next(
                        (u for u in changes.update_fields if u.name == update.name),
                        None,
                    )
                    if existing:
                        if update.attrs:
                            existing.attrs = {**(existing.attrs or {}), **update.attrs}
                        if update.type:
                            existing.type = update.type
                    else:
                        changes.update_fields.append(update)
            elif action == "3":
                field_name = self._prompt_remove_field(working_schema)
                if field_name:
                    # If removing a staged-add, cancel the add instead of
                    # appending to remove_fields
                    staged_add_names = {f["name"] for f in changes.add_fields}
                    if field_name in staged_add_names:
                        changes.add_fields = [
                            f for f in changes.add_fields if f["name"] != field_name
                        ]
                        print(f"Cancelled staged addition of '{field_name}'.")
                    else:
                        changes.remove_fields.append(field_name)
                        # Also remove any queued updates or renames for this field.
                        # Check both old_name and new_name so that:
                        #  - renames FROM this field are dropped (old_name match)
                        #  - renames TO this field are dropped (new_name match)
                        # Also drop updates referencing either the field itself or
                        # any pre-rename name that mapped to it.
                        rename_aliases = {field_name}
                        for r in changes.rename_fields:
                            if r.new_name == field_name:
                                rename_aliases.add(r.old_name)
                            if r.old_name == field_name:
                                rename_aliases.add(r.new_name)
                        changes.update_fields = [
                            u
                            for u in changes.update_fields
                            if u.name not in rename_aliases
                        ]
                        changes.rename_fields = [
                            r
                            for r in changes.rename_fields
                            if r.old_name != field_name and r.new_name != field_name
                        ]
            elif action == "4":
                # Filter out staged additions from rename candidates
                staged_add_names = {f["name"] for f in changes.add_fields}
                rename_schema = self._filter_staged_adds(
                    working_schema, staged_add_names
                )
                field_rename = self._prompt_rename_field(rename_schema)
                if field_rename:
                    # Check rename target doesn't collide with staged additions
                    # or staged removals
                    staged_remove_names = set(changes.remove_fields)
                    if field_rename.new_name in staged_add_names:
                        print(
                            f"Cannot rename to '{field_rename.new_name}': "
                            "a field with that name is already staged for addition."
                        )
                    elif field_rename.new_name in staged_remove_names:
                        print(
                            f"Cannot rename to '{field_rename.new_name}': "
                            "a field with that name is staged for removal."
                        )
                    else:
                        # Collapse chained renames: if there's an existing
                        # rename X→Y and the user now renames Y→Z, collapse
                        # into a single X→Z rename.
                        collapsed = False
                        for ridx, prev_rename in enumerate(changes.rename_fields):
                            if prev_rename.new_name == field_rename.old_name:
                                changes.rename_fields[ridx] = FieldRename(
                                    old_name=prev_rename.old_name,
                                    new_name=field_rename.new_name,
                                )
                                collapsed = True
                                break
                        if not collapsed:
                            changes.rename_fields.append(field_rename)
            elif action == "5":
                new_name = self._prompt_rename_index(working_schema)
                if new_name:
                    changes.index["name"] = new_name
            elif action == "6":
                new_prefix = self._prompt_change_prefix(working_schema)
                if new_prefix:
                    changes.index["prefix"] = new_prefix
            elif action == "7":
                print(
                    yaml.safe_dump(
                        {
                            "version": 1,
                            "changes": changes.model_dump(exclude_none=True),
                        },
                        sort_keys=False,
                    )
                )
            elif action == "8":
                done = True
            else:
                print("Invalid action. Please choose 1-8.")

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
            attrs = self._prompt_common_attrs(
                selected["type"],
                allow_blank=True,
                existing_attrs=selected.get("attrs"),
            )
        if not attrs:
            print("No changes collected.")
            return None
        return FieldUpdate(name=selected["name"], attrs=attrs)

    def _prompt_remove_field(self, source_schema: Dict[str, Any]) -> Optional[str]:
        removable_fields = [field["name"] for field in source_schema["fields"]]
        if not removable_fields:
            print("No fields available to remove.")
            return None

        print("Removable fields:")
        for position, field in enumerate(source_schema["fields"], start=1):
            field_type = field["type"]
            warning = " [WARNING: vector field]" if field_type == "vector" else ""
            print(f"{position}. {field['name']} ({field_type}){warning}")

        choice = input("Select a field to remove by number or name: ").strip()
        selected_name: Optional[str] = None
        if choice in removable_fields:
            selected_name = choice
        elif choice.isdigit():
            offset = int(choice) - 1
            if 0 <= offset < len(removable_fields):
                selected_name = removable_fields[offset]

        if not selected_name:
            print("Invalid field selection.")
            return None

        # Check if it's a vector field and require confirmation
        selected_field = next(
            (f for f in source_schema["fields"] if f["name"] == selected_name), None
        )
        if selected_field and selected_field["type"] == "vector":
            print(
                f"\n  WARNING: Removing vector field '{selected_name}' will:\n"
                "    - Remove it from the search index\n"
                "    - Leave vector data in documents (wasted storage)\n"
                "    - Require re-embedding if you want to restore it later"
            )
            confirm = input("Type 'yes' to confirm removal: ").strip().lower()
            if confirm != "yes":
                print("Cancelled.")
                return None

        return selected_name

    def _prompt_rename_field(
        self, source_schema: Dict[str, Any]
    ) -> Optional[FieldRename]:
        """Prompt user to rename a field in all documents."""
        fields = source_schema["fields"]
        if not fields:
            print("No fields available to rename.")
            return None

        print("Fields available for renaming:")
        for position, field in enumerate(fields, start=1):
            print(f"{position}. {field['name']} ({field['type']})")

        choice = input("Select a field to rename by number or name: ").strip()
        selected: Optional[Dict[str, Any]] = None
        for position, field in enumerate(fields, start=1):
            if choice == str(position) or choice == field["name"]:
                selected = field
                break
        if not selected:
            print("Invalid field selection.")
            return None

        old_name = selected["name"]
        print(f"Renaming field '{old_name}'")
        print(
            "  Warning: This will modify all documents to rename the field. "
            "This is an expensive operation for large datasets."
        )
        new_name = input("New field name: ").strip()
        if not new_name:
            print("New field name is required.")
            return None
        if new_name == old_name:
            print("New name is the same as the old name.")
            return None

        existing_names = {f["name"] for f in fields}
        if new_name in existing_names:
            print(f"Field '{new_name}' already exists.")
            return None

        return FieldRename(old_name=old_name, new_name=new_name)

    def _prompt_rename_index(self, source_schema: Dict[str, Any]) -> Optional[str]:
        """Prompt user to rename the index."""
        current_name = source_schema["index"]["name"]
        print(f"Current index name: {current_name}")
        print(
            "  Note: This only changes the index name. "
            "Documents and keys are unchanged."
        )
        new_name = input("New index name: ").strip()
        if not new_name:
            print("New index name is required.")
            return None
        if new_name == current_name:
            print("New name is the same as the current name.")
            return None
        return new_name

    def _prompt_change_prefix(self, source_schema: Dict[str, Any]) -> Optional[str]:
        """Prompt user to change the key prefix."""
        current_prefix = source_schema["index"]["prefix"]
        print(f"Current prefix: {current_prefix}")
        print(
            "  Warning: This will RENAME all keys from the old prefix to the new prefix. "
            "This is an expensive operation for large datasets."
        )
        new_prefix = input("New prefix: ").strip()
        if not new_prefix:
            print("New prefix is required.")
            return None
        if new_prefix == current_prefix:
            print("New prefix is the same as the current prefix.")
            return None
        return new_prefix

    def _prompt_common_attrs(
        self,
        field_type: str,
        allow_blank: bool = False,
        existing_attrs: Optional[Dict[str, Any]] = None,
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

        # Index empty - index documents where field value is empty string
        print(
            "  Index empty: enables isempty() queries for documents with empty string values"
        )
        index_empty = self._prompt_bool("Index empty", allow_blank=allow_blank)
        if index_empty is not None:
            attrs["index_empty"] = index_empty

        # Track whether the field was already sortable so that type-specific
        # prompt helpers (text UNF, numeric UNF) can offer dependent prompts
        # even when the user leaves sortable blank during an update.
        self._existing_sortable = (existing_attrs or {}).get("sortable", False)

        # Type-specific attributes
        if field_type == "text":
            self._prompt_text_attrs(attrs, allow_blank)
        elif field_type == "tag":
            self._prompt_tag_attrs(attrs, allow_blank)
        elif field_type == "numeric":
            self._prompt_numeric_attrs(attrs, allow_blank, sortable)

        # No index - only meaningful with sortable.
        # When updating (allow_blank), also check the existing field's sortable
        # state so we offer dependent prompts even if the user left sortable blank.
        # But if sortable was explicitly set to False, skip dependent prompts.
        _existing_sortable = self._existing_sortable
        if sortable or (
            sortable is None
            and allow_blank
            and (_existing_sortable or attrs.get("sortable"))
        ):
            print("  No index: store field for sorting only, not searchable")
            no_index = self._prompt_bool("No index", allow_blank=allow_blank)
            if no_index is not None:
                attrs["no_index"] = no_index

        # When explicitly disabling sortable on a previously-sortable field,
        # clear sortable-dependent attributes that are no longer meaningful.
        # UNF and no_index are only used with sortable; leaving them set would
        # be confusing even though Redis technically allows it.
        if sortable is False and _existing_sortable:
            if "unf" not in attrs:
                attrs["unf"] = False
            if "no_index" not in attrs:
                attrs["no_index"] = False

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

        # Phonetic matcher
        print(
            "  Phonetic matcher: enables phonetic matching (e.g., 'dm:en' for Metaphone)"
        )
        phonetic = input("Phonetic matcher [leave blank for none]: ").strip()
        if phonetic:
            attrs["phonetic_matcher"] = phonetic

        # UNF (only if sortable – skip if sortable was explicitly set to False)
        if attrs.get("sortable") or (
            attrs.get("sortable") is not False and self._existing_sortable
        ):
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

    def _prompt_numeric_attrs(
        self, attrs: Dict[str, Any], allow_blank: bool, sortable: Optional[bool]
    ) -> None:
        """Prompt for numeric field specific attributes."""
        # UNF (only if sortable – skip if sortable was explicitly set to False)
        if sortable or (
            sortable is not False and (attrs.get("sortable") or self._existing_sortable)
        ):
            print("  UNF: preserve exact numeric representation for sorting")
            unf = self._prompt_bool("UNF (un-normalized form)", allow_blank=allow_blank)
            if unf is not None:
                attrs["unf"] = unf

    def _prompt_vector_attrs(self, field: Dict[str, Any]) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        current = field.get("attrs", {})
        field_name = field["name"]

        print(f"Current vector config for '{field_name}':")
        current_algo = current.get("algorithm", "hnsw").upper()
        print(f"  algorithm: {current_algo}")
        print(f"  datatype: {current.get('datatype', 'float32')}")
        print(f"  distance_metric: {current.get('distance_metric', 'cosine')}")
        print(f"  dims: {current.get('dims')} (cannot be changed)")
        if current_algo == "HNSW":
            print(f"  m: {current.get('m', 16)}")
            print(f"  ef_construction: {current.get('ef_construction', 200)}")

        print("\nLeave blank to keep current value.")

        # Algorithm
        print(
            "  Algorithm: vector search method (FLAT=brute force, HNSW=graph, SVS-VAMANA=compressed graph)"
        )
        algo = (
            input(f"Algorithm [current: {current_algo}]: ")
            .strip()
            .upper()
            .replace("_", "-")  # Normalize SVS_VAMANA to SVS-VAMANA
        )
        if algo and algo in ("FLAT", "HNSW", "SVS-VAMANA") and algo != current_algo:
            attrs["algorithm"] = algo

        # Datatype (quantization) - show algorithm-specific options
        effective_algo = attrs.get("algorithm", current_algo)
        valid_datatypes: tuple[str, ...]
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
        current_datatype = current.get("datatype", "float32")
        # If switching to SVS-VAMANA and current datatype is incompatible,
        # require the user to pick a valid one.
        force_datatype = (
            effective_algo == "SVS-VAMANA" and current_datatype not in valid_datatypes
        )
        if force_datatype:
            print(
                f"  Current datatype '{current_datatype}' is not compatible with SVS-VAMANA. "
                "You must select a valid datatype."
            )
        datatype = input(f"Datatype [current: {current_datatype}]: ").strip().lower()
        if datatype and datatype in valid_datatypes:
            attrs["datatype"] = datatype
        elif force_datatype:
            # Default to float32 when user skips but current dtype is incompatible
            print("  Defaulting to float32 for SVS-VAMANA compatibility.")
            attrs["datatype"] = "float32"

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

            print(
                "  EF_RUNTIME: query-time search depth (higher=better recall, slower queries)"
            )
            ef_runtime_input = input(
                f"EF_RUNTIME [current: {current.get('ef_runtime', 10)}]: "
            ).strip()
            if ef_runtime_input and ef_runtime_input.isdigit():
                ef_runtime_val = int(ef_runtime_input)
                if ef_runtime_val > 0:
                    attrs["ef_runtime"] = ef_runtime_val

            print(
                "  EPSILON: relative factor for range queries (0.0-1.0, lower=more accurate)"
            )
            epsilon_input = input(
                f"EPSILON [current: {current.get('epsilon', 0.01)}]: "
            ).strip()
            if epsilon_input:
                try:
                    epsilon_val = float(epsilon_input)
                    if 0.0 <= epsilon_val <= 1.0:
                        attrs["epsilon"] = epsilon_val
                    else:
                        print("    Epsilon must be between 0.0 and 1.0, ignoring.")
                except ValueError:
                    print("    Invalid epsilon value, ignoring.")

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
            print(
                "    Note: LVQ/LeanVec optimizations require Intel hardware with AVX-512"
            )
            compression_input = (
                input("COMPRESSION [leave blank for none]: ").strip().upper()
            )
            # Map input to correct enum case (CompressionType expects exact case)
            compression_map = {
                "LVQ4": "LVQ4",
                "LVQ8": "LVQ8",
                "LVQ4X4": "LVQ4x4",
                "LVQ4X8": "LVQ4x8",
                "LEANVEC4X8": "LeanVec4x8",
                "LEANVEC8X8": "LeanVec8x8",
            }
            compression = compression_map.get(compression_input)
            if compression:
                attrs["compression"] = compression

                # Prompt for REDUCE if LeanVec compression is selected
                if compression.startswith("LeanVec"):
                    dims = current.get("dims", 0)
                    recommended = dims // 2 if dims > 0 else None
                    print(
                        f"  REDUCE: dimensionality reduction for LeanVec (must be < {dims})"
                    )
                    if recommended:
                        print(
                            f"    Recommended: {recommended} (dims/2 for balanced performance)"
                        )
                    reduce_input = input(f"REDUCE [leave blank to skip]: ").strip()
                    if reduce_input and reduce_input.isdigit():
                        reduce_val = int(reduce_input)
                        if reduce_val > 0 and reduce_val < dims:
                            attrs["reduce"] = reduce_val
                        else:
                            print(
                                f"    Invalid: reduce must be > 0 and < {dims}, ignoring."
                            )

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
            hint = "y, n, or skip" if allow_blank else "y or n"
            print(f"Please answer {hint}.")

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
