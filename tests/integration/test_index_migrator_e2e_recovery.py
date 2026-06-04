"""End-to-end recovery tests for index migrator backup/checkpoint behavior."""

import sys
import uuid
from pathlib import Path

import pytest
import yaml

from redisvl.cli.migrate import Migrate
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.migration import (
    AsyncMigrationExecutor,
    AsyncMigrationPlanner,
    MigrationExecutor,
    MigrationPlanner,
)
from redisvl.migration.backup import MultiWorkerBackupManifest, VectorBackup
from redisvl.migration.executor import _checkpoint_identity, _resolve_backup_path
from redisvl.migration.quantize import build_worker_backup_paths, split_keys
from redisvl.redis.utils import array_to_buffer


def _uid(worker_id: str) -> str:
    return f"{worker_id}_{uuid.uuid4().hex[:8]}"


def _hash_schema(
    index_name: str,
    prefix: str,
    *,
    datatype: str = "float32",
    dims: int = 3,
) -> dict:
    return {
        "index": {
            "name": index_name,
            "prefix": prefix,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "doc_id", "type": "tag"},
            {"name": "title", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": dims,
                    "distance_metric": "cosine",
                    "datatype": datatype,
                },
            },
        ],
    }


def _json_schema(
    index_name: str,
    prefix: str,
    *,
    datatype: str = "float16",
    dims: int = 4,
) -> dict:
    return {
        "index": {
            "name": index_name,
            "prefix": prefix,
            "storage_type": "json",
        },
        "fields": [
            {"name": "doc_id", "type": "tag", "path": "$.doc_id"},
            {"name": "title", "type": "text", "path": "$.title"},
            {
                "name": "embedding",
                "type": "vector",
                "path": "$.embedding",
                "attrs": {
                    "algorithm": "flat",
                    "dims": dims,
                    "distance_metric": "cosine",
                    "datatype": datatype,
                },
            },
        ],
    }


def _hash_docs(count: int = 3) -> list[dict]:
    return [
        {
            "doc_id": str(i),
            "title": f"doc {i}",
            "embedding": array_to_buffer(
                [0.1 + i * 0.01, 0.2 + i * 0.01, 0.3 + i * 0.01],
                "float32",
            ),
        }
        for i in range(1, count + 1)
    ]


def _json_docs(count: int = 2) -> list[dict]:
    return [
        {
            "doc_id": str(i),
            "title": f"json doc {i}",
            "embedding": [
                0.1 + i * 0.01,
                0.2 + i * 0.01,
                0.3 + i * 0.01,
                0.4 + i * 0.01,
            ],
        }
        for i in range(1, count + 1)
    ]


def _write_patch(tmp_path, index_name: str, changes: dict) -> str:
    patch_path = tmp_path / f"{index_name}_patch.yaml"
    patch_path.write_text(
        yaml.safe_dump({"version": 1, "changes": changes}, sort_keys=False)
    )
    return str(patch_path)


def _create_plan(redis_url: str, tmp_path, index_name: str, changes: dict):
    planner = MigrationPlanner()
    return planner.create_plan(
        index_name,
        redis_url=redis_url,
        schema_patch_path=_write_patch(tmp_path, index_name, changes),
    )


async def _create_async_plan(redis_url: str, tmp_path, index_name: str, changes: dict):
    planner = AsyncMigrationPlanner()
    return await planner.create_plan(
        index_name,
        redis_url=redis_url,
        schema_patch_path=_write_patch(tmp_path, index_name, changes),
    )


def _datatype_changes(plan) -> dict:
    return MigrationPlanner.get_vector_datatype_changes(
        plan.source.schema_snapshot,
        plan.merged_target_schema,
        rename_operations=plan.rename_operations,
    )


def _checkpoint_identity_for_plan(plan) -> dict:
    return _checkpoint_identity(plan, _datatype_changes(plan))


def _key(prefix: str, doc_id: str) -> str:
    return f"{prefix}:{doc_id}"


def _delete_index(redis_url: str, index_name: str) -> None:
    try:
        SearchIndex.from_existing(index_name, redis_url=redis_url).delete(drop=True)
    except Exception:
        pass


def _delete_prefix(client, prefix: str) -> None:
    keys = list(client.scan_iter(match=f"{prefix}*"))
    if keys:
        client.delete(*keys)


def _cleanup(redis_url: str, client, index_name: str, *prefixes: str) -> None:
    _delete_index(redis_url, index_name)
    for prefix in prefixes:
        _delete_prefix(client, prefix)


def _assert_vector_size(client, key: str, expected_len: int) -> bytes:
    raw = client.hget(key, "embedding")
    assert raw is not None, f"missing embedding at {key}"
    assert len(raw) == expected_len
    return raw


def test_ready_checkpoint_with_live_source_resumes_end_to_end(
    redis_url, client, worker_id, tmp_path
):
    uid = _uid(worker_id)
    index_name = f"e2e_ready_{uid}"
    prefix = f"e2e_ready:{uid}"
    backup_dir = tmp_path / "backups"
    docs = _hash_docs()

    source_index = SearchIndex.from_dict(
        _hash_schema(index_name, prefix), redis_url=redis_url
    )
    source_index.create(overwrite=True)
    source_index.load(docs, id_field="doc_id")

    try:
        backup_dir.mkdir()
        plan = _create_plan(
            redis_url,
            tmp_path,
            index_name,
            {
                "update_fields": [
                    {"name": "embedding", "attrs": {"datatype": "float16"}}
                ]
            },
        )
        keys = [_key(prefix, doc["doc_id"]) for doc in docs]
        backup_path = _resolve_backup_path(str(backup_dir), index_name)
        executor = MigrationExecutor()
        backup = executor._dump_vectors(
            client=client,
            index_name=index_name,
            keys=keys,
            datatype_changes=_datatype_changes(plan),
            backup_path=backup_path,
            batch_size=1,
            checkpoint_identity=_checkpoint_identity_for_plan(plan),
        )
        assert backup.header.phase == "ready"
        assert source_index.info()["num_docs"] == len(docs)

        report = executor.apply(plan, redis_url=redis_url, backup_dir=str(backup_dir))

        assert report.result == "succeeded", report.validation.errors
        reloaded = VectorBackup.load(backup_path)
        assert reloaded is not None
        assert reloaded.header.phase == "validated"
        for doc in docs:
            _assert_vector_size(client, _key(prefix, doc["doc_id"]), 3 * 2)
    finally:
        _cleanup(redis_url, client, index_name, prefix)


def test_completed_checkpoint_without_target_creates_target_end_to_end(
    redis_url, client, worker_id, tmp_path
):
    uid = _uid(worker_id)
    index_name = f"e2e_completed_{uid}"
    prefix = f"e2e_completed:{uid}"
    backup_dir = tmp_path / "backups"
    docs = _hash_docs()

    source_index = SearchIndex.from_dict(
        _hash_schema(index_name, prefix), redis_url=redis_url
    )
    source_index.create(overwrite=True)
    source_index.load(docs, id_field="doc_id")

    try:
        backup_dir.mkdir()
        plan = _create_plan(
            redis_url,
            tmp_path,
            index_name,
            {
                "update_fields": [
                    {"name": "embedding", "attrs": {"datatype": "float16"}}
                ]
            },
        )
        keys = [_key(prefix, doc["doc_id"]) for doc in docs]
        backup_path = _resolve_backup_path(str(backup_dir), index_name)
        executor = MigrationExecutor()
        backup = executor._dump_vectors(
            client=client,
            index_name=index_name,
            keys=keys,
            datatype_changes=_datatype_changes(plan),
            backup_path=backup_path,
            batch_size=1,
            checkpoint_identity=_checkpoint_identity_for_plan(plan),
        )
        source_index.delete(drop=False)
        executor._quantize_from_backup(
            client=client,
            backup=backup,
            datatype_changes=_datatype_changes(plan),
        )
        assert VectorBackup.load(backup_path).header.phase == "completed"  # type: ignore[union-attr]

        report = executor.apply(plan, redis_url=redis_url, backup_dir=str(backup_dir))

        assert report.result == "succeeded", report.validation.errors
        reloaded = VectorBackup.load(backup_path)
        assert reloaded is not None
        assert reloaded.header.phase == "validated"
        SearchIndex.from_existing(index_name, redis_url=redis_url)
        for doc in docs:
            _assert_vector_size(client, _key(prefix, doc["doc_id"]), 3 * 2)
    finally:
        _cleanup(redis_url, client, index_name, prefix)


def test_prefix_quantization_and_cli_rollback_restore_new_keys_end_to_end(
    redis_url, client, worker_id, tmp_path, monkeypatch
):
    uid = _uid(worker_id)
    index_name = f"e2e_prefix_{uid}"
    old_prefix = f"e2e_prefix_old:{uid}"
    new_prefix = f"e2e_prefix_new:{uid}"
    backup_dir = tmp_path / "backups"
    docs = _hash_docs()

    source_index = SearchIndex.from_dict(
        _hash_schema(index_name, old_prefix), redis_url=redis_url
    )
    source_index.create(overwrite=True)
    source_index.load(docs, id_field="doc_id")
    original_bytes = {
        _key(old_prefix, doc["doc_id"]): client.hget(
            _key(old_prefix, doc["doc_id"]), "embedding"
        )
        for doc in docs
    }

    try:
        plan = _create_plan(
            redis_url,
            tmp_path,
            index_name,
            {
                "index": {"prefix": new_prefix},
                "update_fields": [
                    {"name": "embedding", "attrs": {"datatype": "float16"}}
                ],
            },
        )

        report = MigrationExecutor().apply(
            plan,
            redis_url=redis_url,
            backup_dir=str(backup_dir),
        )

        assert report.result == "succeeded", report.validation.errors
        assert report.backup is not None
        assert len(report.backup.backup_paths) == 1
        backup = VectorBackup.load(report.backup.backup_paths[0])
        assert backup is not None
        assert backup.header.key_prefix == {"source": old_prefix, "target": new_prefix}

        for doc in docs:
            old_key = _key(old_prefix, doc["doc_id"])
            new_key = _key(new_prefix, doc["doc_id"])
            assert client.exists(old_key) == 0
            assert client.exists(new_key) == 1
            _assert_vector_size(client, new_key, 3 * 2)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rvl",
                "migrate",
                "rollback",
                "--backup-dir",
                str(backup_dir),
                "--index",
                index_name,
                "--yes",
                "--url",
                redis_url,
            ],
        )
        Migrate.__new__(Migrate).rollback()

        for doc in docs:
            old_key = _key(old_prefix, doc["doc_id"])
            new_key = _key(new_prefix, doc["doc_id"])
            assert client.exists(old_key) == 0
            assert client.hget(new_key, "embedding") == original_bytes[old_key]
    finally:
        _cleanup(redis_url, client, index_name, old_prefix, new_prefix)


def test_multi_worker_manifest_resume_after_drop_end_to_end(
    redis_url, client, worker_id, tmp_path
):
    uid = _uid(worker_id)
    index_name = f"e2e_multi_{uid}"
    prefix = f"e2e_multi:{uid}"
    backup_dir = tmp_path / "backups"
    docs = _hash_docs(count=4)

    source_index = SearchIndex.from_dict(
        _hash_schema(index_name, prefix), redis_url=redis_url
    )
    source_index.create(overwrite=True)
    source_index.load(docs, id_field="doc_id")

    try:
        backup_dir.mkdir()
        plan = _create_plan(
            redis_url,
            tmp_path,
            index_name,
            {
                "update_fields": [
                    {"name": "embedding", "attrs": {"datatype": "float16"}}
                ]
            },
        )
        keys = [_key(prefix, doc["doc_id"]) for doc in docs]
        backup_path = _resolve_backup_path(str(backup_dir), index_name)
        key_slices = split_keys(keys, 8)
        worker_backup_paths = build_worker_backup_paths(
            str(backup_dir), index_name, len(key_slices)
        )
        manifest = MultiWorkerBackupManifest.create(
            backup_path,
            index_name=index_name,
            batch_size=1,
            requested_workers=8,
            key_slices=key_slices,
            worker_backup_paths=worker_backup_paths,
            **_checkpoint_identity_for_plan(plan),
        )
        source_index.delete(drop=False)
        manifest.mark_index_dropped()

        report = MigrationExecutor().apply(
            plan,
            redis_url=redis_url,
            backup_dir=str(backup_dir),
        )

        assert report.result == "succeeded", report.validation.errors
        assert report.backup is not None
        assert len(report.backup.backup_paths) == len(key_slices)
        assert all(
            Path(path + ".header").is_file() for path in report.backup.backup_paths
        )
        assert all(
            Path(path + ".data").is_file() for path in report.backup.backup_paths
        )
        reloaded = MultiWorkerBackupManifest.load(backup_path)
        assert reloaded is not None
        assert reloaded.phase == "validated"
        for doc in docs:
            _assert_vector_size(client, _key(prefix, doc["doc_id"]), 3 * 2)
    finally:
        _cleanup(redis_url, client, index_name, prefix)


@pytest.mark.asyncio
async def test_async_prefix_quantization_no_old_keys_end_to_end(
    redis_url, client, worker_id, tmp_path
):
    uid = _uid(worker_id)
    index_name = f"e2e_async_prefix_{uid}"
    old_prefix = f"e2e_async_old:{uid}"
    new_prefix = f"e2e_async_new:{uid}"
    backup_dir = tmp_path / "backups"
    docs = _hash_docs()

    source_index = AsyncSearchIndex.from_dict(
        _hash_schema(index_name, old_prefix), redis_url=redis_url
    )
    await source_index.create(overwrite=True)
    await source_index.load(docs, id_field="doc_id")

    try:
        plan = await _create_async_plan(
            redis_url,
            tmp_path,
            index_name,
            {
                "index": {"prefix": new_prefix},
                "update_fields": [
                    {"name": "embedding", "attrs": {"datatype": "float16"}}
                ],
            },
        )

        report = await AsyncMigrationExecutor().apply(
            plan,
            redis_url=redis_url,
            backup_dir=str(backup_dir),
        )

        assert report.result == "succeeded", report.validation.errors
        for doc in docs:
            assert client.exists(_key(old_prefix, doc["doc_id"])) == 0
            _assert_vector_size(client, _key(new_prefix, doc["doc_id"]), 3 * 2)
    finally:
        _cleanup(redis_url, client, index_name, old_prefix, new_prefix)


def test_json_same_width_datatype_change_is_schema_only_end_to_end(
    redis_url, client, worker_id, tmp_path
):
    uid = _uid(worker_id)
    index_name = f"e2e_json_same_width_{uid}"
    prefix = f"e2e_json_same_width:{uid}"
    backup_dir = tmp_path / "backups"
    docs = _json_docs()

    index = SearchIndex.from_dict(
        _json_schema(index_name, prefix, datatype="float16"),
        redis_url=redis_url,
    )
    index.create(overwrite=True)
    for doc in docs:
        client.json().set(_key(prefix, doc["doc_id"]), "$", doc)

    try:
        plan = _create_plan(
            redis_url,
            tmp_path,
            index_name,
            {
                "update_fields": [
                    {"name": "embedding", "attrs": {"datatype": "bfloat16"}}
                ]
            },
        )

        report = MigrationExecutor().apply(
            plan,
            redis_url=redis_url,
            backup_dir=str(backup_dir),
        )

        assert report.result == "succeeded", report.validation.errors
        assert report.backup is not None
        assert report.backup.backup_paths == []
        assert list(backup_dir.glob("*.header")) == []
        assert list(backup_dir.glob("*.data")) == []
        live_index = SearchIndex.from_existing(index_name, redis_url=redis_url)
        vector_field = live_index.schema.fields.get("embedding")
        assert vector_field is not None
        datatype = getattr(
            vector_field.attrs.datatype, "value", vector_field.attrs.datatype
        )
        assert str(datatype).lower() == "bfloat16"
    finally:
        _cleanup(redis_url, client, index_name, prefix)
