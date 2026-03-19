from fnmatch import fnmatch

import yaml

from redisvl.migration import MigrationPlanner
from redisvl.schema.schema import IndexSchema


class DummyClient:
    def __init__(self, keys):
        self.keys = keys

    def scan(self, cursor=0, match=None, count=None):
        matched = []
        for key in self.keys:
            decoded_key = key.decode() if isinstance(key, bytes) else str(key)
            if match is None or fnmatch(decoded_key, match):
                matched.append(key)
        return 0, matched


class DummyIndex:
    def __init__(self, schema, stats, keys):
        self.schema = schema
        self._stats = stats
        self._client = DummyClient(keys)

    @property
    def client(self):
        return self._client

    def info(self):
        return self._stats


def _make_source_schema():
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "docs",
                "prefix": "docs",
                "key_separator": ":",
                "storage_type": "json",
            },
            "fields": [
                {
                    "name": "title",
                    "type": "text",
                    "path": "$.title",
                    "attrs": {"sortable": False},
                },
                {
                    "name": "price",
                    "type": "numeric",
                    "path": "$.price",
                    "attrs": {"sortable": True},
                },
                {
                    "name": "embedding",
                    "type": "vector",
                    "path": "$.embedding",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


def test_create_plan_from_schema_patch_preserves_unspecified_config(
    monkeypatch, tmp_path
):
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(
        source_schema,
        {"num_docs": 2, "indexing": False},
        [b"docs:1", b"docs:2", b"docs:3"],
    )
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    patch_path = tmp_path / "schema_patch.yaml"
    patch_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "changes": {
                    "add_fields": [
                        {
                            "name": "category",
                            "type": "tag",
                            "path": "$.category",
                            "attrs": {"separator": ","},
                        }
                    ],
                    "remove_fields": ["price"],
                    "update_fields": [
                        {
                            "name": "title",
                            "options": {"sortable": True},
                        }
                    ],
                },
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner(key_sample_limit=2)
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        schema_patch_path=str(patch_path),
    )

    assert plan.diff_classification.supported is True
    assert plan.source.index_name == "docs"
    assert plan.source.keyspace.storage_type == "json"
    assert plan.source.keyspace.prefixes == ["docs"]
    assert plan.source.keyspace.key_separator == ":"
    assert plan.source.keyspace.key_sample == ["docs:1", "docs:2"]
    assert plan.warnings == ["Index downtime is required"]

    merged_fields = {
        field["name"]: field for field in plan.merged_target_schema["fields"]
    }
    assert plan.merged_target_schema["index"]["prefix"] == "docs"
    assert merged_fields["title"]["attrs"]["sortable"] is True
    assert "price" not in merged_fields
    assert merged_fields["category"]["type"] == "tag"

    plan_path = tmp_path / "migration_plan.yaml"
    planner.write_plan(plan, str(plan_path))
    written_plan = yaml.safe_load(plan_path.read_text())
    assert written_plan["mode"] == "drop_recreate"
    assert written_plan["validation"]["require_doc_count_match"] is True
    assert written_plan["diff_classification"]["supported"] is True


def test_target_schema_vector_datatype_change_is_allowed(monkeypatch, tmp_path):
    """Changing vector datatype (quantization) is allowed - executor will re-encode."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {
                        "name": "title",
                        "type": "text",
                        "path": "$.title",
                        "attrs": {"sortable": False},
                    },
                    {
                        "name": "price",
                        "type": "numeric",
                        "path": "$.price",
                        "attrs": {"sortable": True},
                    },
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "flat",  # Same algorithm
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float16",  # Changed from float32
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    # Datatype change (quantization) should now be ALLOWED
    assert plan.diff_classification.supported is True
    assert len(plan.diff_classification.blocked_reasons) == 0

    # Verify datatype changes are detected for the executor
    datatype_changes = MigrationPlanner.get_vector_datatype_changes(
        plan.source.schema_snapshot, plan.merged_target_schema
    )
    assert "embedding" in datatype_changes
    assert datatype_changes["embedding"]["source"] == "float32"
    assert datatype_changes["embedding"]["target"] == "float16"


def test_target_schema_vector_algorithm_change_is_allowed(monkeypatch, tmp_path):
    """Changing vector algorithm is allowed (index-only change)."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {
                        "name": "title",
                        "type": "text",
                        "path": "$.title",
                        "attrs": {"sortable": False},
                    },
                    {
                        "name": "price",
                        "type": "numeric",
                        "path": "$.price",
                        "attrs": {"sortable": True},
                    },
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "hnsw",  # Changed from flat
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",  # Same datatype
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    # Algorithm change should be ALLOWED
    assert plan.diff_classification.supported is True
    assert len(plan.diff_classification.blocked_reasons) == 0


# =============================================================================
# BLOCKED CHANGES (Document-Dependent) - require iterative_shadow
# =============================================================================


def test_target_schema_prefix_change_is_blocked(monkeypatch, tmp_path):
    """Prefix change is blocked: documents are at wrong keys."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs_v2",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": source_schema.to_dict()["fields"],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is False
    assert any(
        "prefix" in reason.lower() and "iterative_shadow" in reason
        for reason in plan.diff_classification.blocked_reasons
    )


def test_key_separator_change_is_blocked(monkeypatch, tmp_path):
    """Key separator change is blocked: document keys don't match new pattern."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": "/",  # Changed from ":"
                    "storage_type": "json",
                },
                "fields": source_schema.to_dict()["fields"],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is False
    assert any(
        "key_separator" in reason.lower() or "separator" in reason.lower()
        for reason in plan.diff_classification.blocked_reasons
    )


def test_storage_type_change_is_blocked(monkeypatch, tmp_path):
    """Storage type change is blocked: documents are in wrong format."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "hash",  # Changed from "json"
                },
                "fields": [
                    {"name": "title", "type": "text", "attrs": {"sortable": False}},
                    {"name": "price", "type": "numeric", "attrs": {"sortable": True}},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is False
    assert any(
        "storage" in reason.lower()
        for reason in plan.diff_classification.blocked_reasons
    )


def test_vector_dimension_change_is_blocked(monkeypatch, tmp_path):
    """Vector dimension change is blocked: stored vectors have wrong size."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {
                        "name": "title",
                        "type": "text",
                        "path": "$.title",
                        "attrs": {"sortable": False},
                    },
                    {
                        "name": "price",
                        "type": "numeric",
                        "path": "$.price",
                        "attrs": {"sortable": True},
                    },
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 768,  # Changed from 3
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is False
    assert any(
        "dims" in reason and "iterative_shadow" in reason
        for reason in plan.diff_classification.blocked_reasons
    )


def test_field_path_change_is_blocked(monkeypatch, tmp_path):
    """JSON path change is blocked: stored data is at wrong path."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {
                        "name": "title",
                        "type": "text",
                        "path": "$.metadata.title",  # Changed from $.title
                        "attrs": {"sortable": False},
                    },
                    {
                        "name": "price",
                        "type": "numeric",
                        "path": "$.price",
                        "attrs": {"sortable": True},
                    },
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is False
    assert any(
        "path" in reason.lower() for reason in plan.diff_classification.blocked_reasons
    )


def test_field_type_change_is_blocked(monkeypatch, tmp_path):
    """Field type change is blocked: index expects different data format."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {
                        "name": "title",
                        "type": "tag",  # Changed from text
                        "path": "$.title",
                    },
                    {
                        "name": "price",
                        "type": "numeric",
                        "path": "$.price",
                        "attrs": {"sortable": True},
                    },
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is False
    assert any(
        "type" in reason.lower() for reason in plan.diff_classification.blocked_reasons
    )


def test_field_rename_is_detected_and_blocked(monkeypatch, tmp_path):
    """Field rename is blocked: stored data uses old field name."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {
                        "name": "document_title",  # Renamed from "title"
                        "type": "text",
                        "path": "$.title",
                        "attrs": {"sortable": False},
                    },
                    {
                        "name": "price",
                        "type": "numeric",
                        "path": "$.price",
                        "attrs": {"sortable": True},
                    },
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is False
    assert any(
        "rename" in reason.lower()
        for reason in plan.diff_classification.blocked_reasons
    )


# =============================================================================
# ALLOWED CHANGES (Index-Only)
# =============================================================================


def test_add_non_vector_field_is_allowed(monkeypatch, tmp_path):
    """Adding a non-vector field is allowed."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    patch_path = tmp_path / "schema_patch.yaml"
    patch_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "changes": {
                    "add_fields": [
                        {"name": "category", "type": "tag", "path": "$.category"}
                    ]
                },
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        schema_patch_path=str(patch_path),
    )

    assert plan.diff_classification.supported is True


def test_remove_field_is_allowed(monkeypatch, tmp_path):
    """Removing a field from the index is allowed."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    patch_path = tmp_path / "schema_patch.yaml"
    patch_path.write_text(
        yaml.safe_dump(
            {"version": 1, "changes": {"remove_fields": ["price"]}},
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        schema_patch_path=str(patch_path),
    )

    assert plan.diff_classification.supported is True


def test_change_field_sortable_is_allowed(monkeypatch, tmp_path):
    """Changing field sortable option is allowed."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    patch_path = tmp_path / "schema_patch.yaml"
    patch_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "changes": {
                    "update_fields": [{"name": "title", "options": {"sortable": True}}]
                },
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        schema_patch_path=str(patch_path),
    )

    assert plan.diff_classification.supported is True


def test_change_vector_distance_metric_is_allowed(monkeypatch, tmp_path):
    """Changing vector distance metric is allowed (index-only)."""
    source_schema = _make_source_schema()
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {
                        "name": "title",
                        "type": "text",
                        "path": "$.title",
                        "attrs": {"sortable": False},
                    },
                    {
                        "name": "price",
                        "type": "numeric",
                        "path": "$.price",
                        "attrs": {"sortable": True},
                    },
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "L2",  # Changed from cosine
                            "datatype": "float32",
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is True
    assert len(plan.diff_classification.blocked_reasons) == 0


def test_change_hnsw_tuning_params_is_allowed(monkeypatch, tmp_path):
    """Changing HNSW tuning parameters is allowed (index-only)."""
    source_schema = IndexSchema.from_dict(
        {
            "index": {
                "name": "docs",
                "prefix": "docs",
                "key_separator": ":",
                "storage_type": "json",
            },
            "fields": [
                {
                    "name": "embedding",
                    "type": "vector",
                    "path": "$.embedding",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                        "m": 16,
                        "ef_construction": 200,
                    },
                },
            ],
        }
    )
    dummy_index = DummyIndex(source_schema, {"num_docs": 2}, [b"docs:1"])
    monkeypatch.setattr(
        "redisvl.migration.planner.SearchIndex.from_existing",
        lambda *args, **kwargs: dummy_index,
    )

    target_schema_path = tmp_path / "target_schema.yaml"
    target_schema_path.write_text(
        yaml.safe_dump(
            {
                "index": {
                    "name": "docs",
                    "prefix": "docs",
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {
                        "name": "embedding",
                        "type": "vector",
                        "path": "$.embedding",
                        "attrs": {
                            "algorithm": "hnsw",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                            "m": 32,  # Changed from 16
                            "ef_construction": 400,  # Changed from 200
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    planner = MigrationPlanner()
    plan = planner.create_plan(
        "docs",
        redis_url="redis://localhost:6379",
        target_schema_path=str(target_schema_path),
    )

    assert plan.diff_classification.supported is True
    assert len(plan.diff_classification.blocked_reasons) == 0
