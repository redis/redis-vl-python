import uuid

import yaml

from redisvl.index import SearchIndex
from redisvl.migration import MigrationExecutor, MigrationPlanner, MigrationValidator
from redisvl.migration.utils import load_migration_plan, schemas_equal
from redisvl.redis.utils import array_to_buffer


def test_drop_recreate_plan_apply_validate_flow(redis_url, worker_id, tmp_path):
    unique_id = str(uuid.uuid4())[:8]
    index_name = f"migration_v1_{worker_id}_{unique_id}"
    prefix = f"migration_v1:{worker_id}:{unique_id}"

    source_index = SearchIndex.from_dict(
        {
            "index": {
                "name": index_name,
                "prefix": prefix,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {"name": "title", "type": "text"},
                {"name": "price", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        },
        redis_url=redis_url,
    )

    docs = [
        {
            "doc_id": "1",
            "title": "alpha",
            "price": 1,
            "category": "news",
            "embedding": array_to_buffer([0.1, 0.2, 0.3], "float32"),
        },
        {
            "doc_id": "2",
            "title": "beta",
            "price": 2,
            "category": "sports",
            "embedding": array_to_buffer([0.2, 0.1, 0.4], "float32"),
        },
    ]

    source_index.create(overwrite=True)
    source_index.load(docs, id_field="doc_id")

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
                            "attrs": {"separator": ","},
                        }
                    ],
                    "remove_fields": ["price"],
                    "update_fields": [{"name": "title", "attrs": {"sortable": True}}],
                },
            },
            sort_keys=False,
        )
    )

    plan_path = tmp_path / "migration_plan.yaml"
    planner = MigrationPlanner()
    plan = planner.create_plan(
        index_name,
        redis_url=redis_url,
        schema_patch_path=str(patch_path),
    )
    assert plan.diff_classification.supported is True
    planner.write_plan(plan, str(plan_path))

    query_check_path = tmp_path / "query_checks.yaml"
    query_check_path.write_text(
        yaml.safe_dump({"fetch_ids": ["1", "2"]}, sort_keys=False)
    )

    executor = MigrationExecutor()
    report = executor.apply(
        load_migration_plan(str(plan_path)),
        redis_url=redis_url,
        query_check_file=str(query_check_path),
    )

    assert report.result == "succeeded"
    assert report.validation.schema_match is True
    assert report.validation.doc_count_match is True
    assert report.validation.key_sample_exists is True
    assert report.validation.indexing_failures_delta == 0
    assert not report.validation.errors
    assert report.benchmark_summary.documents_indexed_per_second is not None

    live_index = SearchIndex.from_existing(index_name, redis_url=redis_url)
    assert schemas_equal(live_index.schema.to_dict(), plan.merged_target_schema)

    validator = MigrationValidator()
    validation, _target_info, _duration = validator.validate(
        load_migration_plan(str(plan_path)),
        redis_url=redis_url,
        query_check_file=str(query_check_path),
    )
    assert validation.schema_match is True
    assert validation.doc_count_match is True
    assert validation.key_sample_exists is True
    assert not validation.errors

    live_index.delete(drop=True)
