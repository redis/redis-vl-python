import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.mcp.tools.list_indexes import list_indexes
from redisvl.redis.connection import is_version_gte
from redisvl.schema import IndexSchema
from tests.conftest import (
    get_redis_version_async,
    mcp_binding_index,
    mcp_binding_vectorizer,
)


class FakeVectorizer:
    def __init__(self, model: str, dims: int = 3, **kwargs):
        self.model = model
        self.dims = dims
        self.kwargs = kwargs


class FailingAsyncCloseVectorizer(FakeVectorizer):
    async def aclose(self):
        raise RuntimeError("vectorizer close failed")


@pytest.fixture
async def existing_index(async_client, worker_id):
    created_indexes = []

    async def factory(
        *,
        index_name: str,
        storage_type: str = "hash",
        vector_path: str | None = None,
    ) -> AsyncSearchIndex:
        fields = [{"name": "content", "type": "text"}]
        vector_field = {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "flat",
                "dims": 3,
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        }
        if storage_type == "json":
            fields[0]["path"] = "$.content"
            vector_field["path"] = vector_path or "$.embedding"

        fields.append(vector_field)
        schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": f"{index_name}-{worker_id}",
                    "prefix": f"{index_name}:{worker_id}",
                    "storage_type": storage_type,
                },
                "fields": fields,
            }
        )
        index = AsyncSearchIndex(schema=schema, redis_client=async_client)
        await index.create(overwrite=True, drop=True)
        created_indexes.append(index)
        return index

    yield factory

    for index in created_indexes:
        try:
            await index.delete(drop=True)
        except Exception:
            pass


@pytest.fixture
def mcp_config_path(tmp_path: Path, redis_url: str):
    def factory(
        *,
        redis_name: str,
        vector_dims: int = 3,
        schema_overrides: dict[str, Any] | None = None,
        runtime_overrides: dict[str, Any] | None = None,
        search: dict[str, Any] | None = None,
        include_vectorizer: bool = True,
    ) -> str:
        runtime = {
            "text_field_name": "content",
            "vector_field_name": "embedding",
            "default_embed_text_field": "content",
        }
        if runtime_overrides:
            runtime.update(runtime_overrides)

        binding = {
            "redis_name": redis_name,
            "search": search or {"type": "vector"},
            "runtime": runtime,
        }
        if include_vectorizer:
            binding["vectorizer"] = {
                "class": "FakeVectorizer",
                "model": "fake-model",
                "dims": vector_dims,
            }

        config = {
            "server": {"redis_url": redis_url},
            "indexes": {"knowledge": binding},
        }
        if schema_overrides is not None:
            config["indexes"]["knowledge"]["schema_overrides"] = schema_overrides

        config_path = tmp_path / f"{redis_name}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(config_path)

    return factory


@pytest.mark.asyncio
async def test_server_startup_success(monkeypatch, existing_index, mcp_config_path):
    index = await existing_index(index_name="mcp-startup")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name))
    )

    await server.startup()

    started_index = mcp_binding_index(server)
    vectorizer = mcp_binding_vectorizer(server)

    assert await started_index.exists() is True
    assert started_index.schema.index.name == index.name
    assert vectorizer.dims == 3

    await server.shutdown()


@pytest.mark.asyncio
async def test_server_startup_succeeds_for_fulltext_without_vectorizer(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(
        index_name="mcp-startup-fulltext",
        storage_type="hash",
    )
    original_build_vectorizer = RedisVLMCPServer._build_vectorizer
    build_vectorizer_called = False

    def tracked_build_vectorizer(binding):
        nonlocal build_vectorizer_called
        build_vectorizer_called = True
        return original_build_vectorizer(binding)

    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    monkeypatch.setattr(
        RedisVLMCPServer,
        "_build_vectorizer",
        staticmethod(tracked_build_vectorizer),
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                search={"type": "fulltext", "params": {"stopwords": None}},
                runtime_overrides={
                    "vector_field_name": None,
                    "default_embed_text_field": None,
                },
                include_vectorizer=False,
            )
        )
    )

    await server.startup()

    started_index = mcp_binding_index(server)
    assert await started_index.exists() is True
    assert build_vectorizer_called is False
    with pytest.raises(RuntimeError, match="vectorizer is not configured"):
        mcp_binding_vectorizer(server)

    await server.shutdown()


@pytest.mark.asyncio
async def test_server_fails_when_hybrid_config_requires_native_runtime(
    monkeypatch, existing_index, mcp_config_path, async_client
):
    redis_version = await get_redis_version_async(async_client)
    if is_version_gte(redis_version, "8.4.0"):
        pytest.skip(f"Redis version {redis_version} supports native hybrid search")

    index = await existing_index(index_name="mcp-native-required")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                search={
                    "type": "hybrid",
                    "params": {
                        "stopwords": None,
                        "vector_search_method": "KNN",
                        "knn_ef_runtime": 150,
                    },
                },
            )
        )
    )

    with pytest.raises(ValueError, match="knn_ef_runtime"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_fails_when_configured_index_is_missing(
    monkeypatch, mcp_config_path, worker_id
):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=f"missing-{worker_id}"))
    )

    with pytest.raises(ValueError, match="does not exist"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_uses_schema_overrides_when_inspection_is_incomplete(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-overrides")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    original_info = AsyncSearchIndex._info

    async def incomplete_info(name, redis_client):
        info = await original_info(name, redis_client)
        for field in info["attributes"]:
            if "VECTOR" in field:
                del field[6:]
        return info

    monkeypatch.setattr(
        "redisvl.mcp.server.AsyncSearchIndex._info",
        staticmethod(incomplete_info),
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                schema_overrides={
                    "fields": [
                        {
                            "name": "embedding",
                            "type": "vector",
                            "attrs": {
                                "algorithm": "flat",
                                "dims": 3,
                                "datatype": "float32",
                                "distance_metric": "cosine",
                            },
                        }
                    ]
                },
            )
        )
    )

    await server.startup()

    started_index = mcp_binding_index(server)
    assert started_index.schema.fields["embedding"].attrs.dims == 3

    await server.shutdown()


@pytest.mark.asyncio
async def test_server_fails_on_conflicting_schema_override(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(
        index_name="mcp-conflict",
        storage_type="json",
        vector_path="$.embedding",
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                schema_overrides={
                    "fields": [
                        {
                            "name": "embedding",
                            "type": "vector",
                            "path": "$.other_embedding",
                        }
                    ]
                },
            )
        )
    )

    with pytest.raises(ValueError, match="cannot change discovered field path"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_fails_fast_on_vector_dimension_mismatch(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-dims")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name, vector_dims=8))
    )

    with pytest.raises(ValueError, match="Vectorizer dims"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_startup_failure_disconnects_index(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-startup-failure")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    original_disconnect = AsyncSearchIndex.disconnect
    disconnect_called = False

    async def tracked_disconnect(self):
        nonlocal disconnect_called
        disconnect_called = True
        await original_disconnect(self)

    monkeypatch.setattr(
        "redisvl.mcp.server.AsyncSearchIndex.disconnect",
        tracked_disconnect,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name, vector_dims=8))
    )

    with pytest.raises(ValueError, match="Vectorizer dims"):
        await server.startup()

    assert disconnect_called is True


@pytest.mark.asyncio
async def test_server_shutdown_disconnects_owned_client(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-shutdown")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name))
    )

    await server.startup()
    started_index = mcp_binding_index(server)

    assert started_index.client is not None

    await server.shutdown()

    assert started_index.client is None


@pytest.mark.asyncio
async def test_server_get_index_fails_after_shutdown(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-get-index-after-shutdown")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name))
    )

    await server.startup()
    await server.shutdown()

    with pytest.raises(RuntimeError, match="has not been started"):
        mcp_binding_index(server)


@pytest.mark.asyncio
async def test_server_shutdown_disconnects_index_when_vectorizer_close_fails(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-shutdown-failure")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FailingAsyncCloseVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name))
    )

    await server.startup()
    started_index = mcp_binding_index(server)

    # Teardown is best-effort: a failing vectorizer close is logged and
    # swallowed rather than aborting teardown, so the index is still
    # disconnected and its Redis connection cannot leak.
    await server.shutdown()

    assert started_index.client is None

    with pytest.raises(RuntimeError, match="has not been started"):
        mcp_binding_vectorizer(server)


@pytest.mark.asyncio
async def test_run_guarded_allows_admitted_request_to_finish_during_shutdown(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-guarded-shutdown-drain")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                runtime_overrides={"max_concurrency": 2, "request_timeout_seconds": 5},
            )
        )
    )

    await server.startup()
    started_index = mcp_binding_index(server)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def guarded_operation():
        entered.set()
        await release.wait()
        return "done"

    operation_task = asyncio.create_task(
        server.run_guarded(
            "drain-during-shutdown", guarded_operation(), timeout_seconds=5
        )
    )
    await entered.wait()

    shutdown_task = asyncio.create_task(server.shutdown())
    await asyncio.sleep(0)

    assert shutdown_task.done() is False
    assert started_index.client is not None

    release.set()

    assert await operation_task == "done"
    await shutdown_task

    assert started_index.client is None


@pytest.mark.asyncio
async def test_run_guarded_rejects_new_requests_after_shutdown_begins(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-guarded-stop-reject-new")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                runtime_overrides={"max_concurrency": 2, "request_timeout_seconds": 5},
            )
        )
    )

    await server.startup()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def guarded_operation():
        entered.set()
        await release.wait()
        return "done"

    active_task = asyncio.create_task(
        server.run_guarded(
            "active-during-shutdown", guarded_operation(), timeout_seconds=5
        )
    )
    await entered.wait()

    shutdown_task = asyncio.create_task(server.shutdown())
    await asyncio.sleep(0)

    future = asyncio.get_running_loop().create_future()
    future.set_result("later")
    with pytest.raises(RuntimeError, match="not running"):
        await server.run_guarded("reject-after-stop", future, timeout_seconds=5)

    release.set()
    assert await active_task == "done"
    await shutdown_task


@pytest.mark.asyncio
async def test_run_guarded_rejects_requests_waiting_on_semaphore_when_shutdown_starts(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-guarded-stop-queued")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                runtime_overrides={"max_concurrency": 1, "request_timeout_seconds": 5},
            )
        )
    )

    await server.startup()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    async def first_operation():
        first_entered.set()
        await release_first.wait()
        return "first"

    async def second_operation():
        second_started.set()
        return "second"

    first_task = asyncio.create_task(
        server.run_guarded("first-op", first_operation(), timeout_seconds=5)
    )
    await first_entered.wait()

    second_task = asyncio.create_task(
        server.run_guarded("second-op", second_operation(), timeout_seconds=5)
    )
    await asyncio.sleep(0)

    shutdown_task = asyncio.create_task(server.shutdown())
    await asyncio.sleep(0)

    release_first.set()

    assert await first_task == "first"
    with pytest.raises(RuntimeError, match="not running"):
        await second_task

    assert second_started.is_set() is False
    await shutdown_task


@pytest.fixture
def multi_index_config_path(tmp_path: Path, redis_url: str):
    def factory(bindings: dict[str, dict[str, Any]]) -> str:
        config = {"server": {"redis_url": redis_url}, "indexes": bindings}
        config_path = tmp_path / "multi-index.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(config_path)

    return factory


def _binding_config(redis_name: str, *, read_only: bool = False) -> dict[str, Any]:
    return {
        "redis_name": redis_name,
        "read_only": read_only,
        "vectorizer": {"class": "FakeVectorizer", "model": "fake-model", "dims": 3},
        "search": {"type": "vector"},
        "runtime": {
            "text_field_name": "content",
            "vector_field_name": "embedding",
            "default_embed_text_field": "content",
        },
    }


@pytest.mark.asyncio
async def test_server_starts_with_multiple_bindings(
    monkeypatch, existing_index, multi_index_config_path
):
    knowledge = await existing_index(index_name="mcp-multi-knowledge")
    tickets = await existing_index(index_name="mcp-multi-tickets")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=multi_index_config_path(
                {
                    "knowledge": _binding_config(knowledge.name),
                    "tickets": _binding_config(tickets.name, read_only=True),
                }
            )
        )
    )

    await server.startup()

    try:
        assert sorted(server._bindings) == ["knowledge", "tickets"]

        knowledge_rt = server.resolve_binding("knowledge")
        tickets_rt = server.resolve_binding("tickets")

        # Each binding is inspected and initialized independently.
        assert knowledge_rt.index.schema.index.name == knowledge.name
        assert tickets_rt.index.schema.index.name == tickets.name
        assert knowledge_rt.index is not tickets_rt.index

        # Per-index write availability is respected.
        assert knowledge_rt.effective_read_only is False
        assert tickets_rt.effective_read_only is True

        # An omitted index is ambiguous when multiple bindings are configured.
        with pytest.raises(RedisVLMCPError) as excinfo:
            server.resolve_binding(None)
        assert excinfo.value.code == MCPErrorCode.INVALID_REQUEST
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_server_global_read_only_overrides_all_bindings(
    monkeypatch, existing_index, multi_index_config_path
):
    knowledge = await existing_index(index_name="mcp-multi-ro-knowledge")
    tickets = await existing_index(index_name="mcp-multi-ro-tickets")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=multi_index_config_path(
                {
                    "knowledge": _binding_config(knowledge.name),
                    "tickets": _binding_config(tickets.name, read_only=True),
                }
            ),
            read_only=True,
        )
    )

    await server.startup()

    try:
        # Global read-only forces effective write availability false everywhere.
        assert server.resolve_binding("knowledge").effective_read_only is True
        assert server.resolve_binding("tickets").effective_read_only is True
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_server_startup_fails_when_one_binding_is_invalid(
    monkeypatch, existing_index, multi_index_config_path
):
    knowledge = await existing_index(index_name="mcp-multi-invalid")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=multi_index_config_path(
                {
                    "knowledge": _binding_config(knowledge.name),
                    "missing": _binding_config("nonexistent-index-name"),
                }
            )
        )
    )

    with pytest.raises(ValueError, match="does not exist"):
        await server.startup()

    assert server._lifecycle_state.name == "STOPPED"
    assert server._bindings == {}


@pytest.mark.asyncio
async def test_list_indexes_derives_fields_from_inspected_schema(
    monkeypatch, existing_index, multi_index_config_path
):
    knowledge = await existing_index(index_name="mcp-list-knowledge")
    tickets = await existing_index(index_name="mcp-list-tickets")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=multi_index_config_path(
                {
                    # Vector binding: content is the embed source.
                    "knowledge": {
                        "redis_name": knowledge.name,
                        "description": "Product docs",
                        "vectorizer": {
                            "class": "FakeVectorizer",
                            "model": "fake-model",
                            "dims": 3,
                        },
                        "search": {"type": "vector"},
                        "runtime": {
                            "text_field_name": "content",
                            "vector_field_name": "embedding",
                            "default_embed_text_field": "content",
                            "max_limit": 25,
                        },
                    },
                    # Fulltext binding: no embed source, read-only.
                    "tickets": {
                        "redis_name": tickets.name,
                        "read_only": True,
                        "search": {"type": "fulltext"},
                        "runtime": {"text_field_name": "content"},
                    },
                }
            )
        )
    )

    await server.startup()

    try:
        result = list_indexes(server)
        indexes = {entry["id"]: entry for entry in result["indexes"]}

        # Both bindings are discoverable; redis_name is never leaked.
        assert set(indexes) == {"knowledge", "tickets"}
        for entry in indexes.values():
            assert "redis_name" not in entry
            assert knowledge.name not in entry.values()
            assert tickets.name not in entry.values()

        # Fields come from the inspected schema. The vector field is always
        # omitted; the embed-source field is omitted only where configured.
        knowledge_fields = {f["name"] for f in indexes["knowledge"]["fields"]}
        tickets_fields = {f["name"] for f in indexes["tickets"]["fields"]}
        assert "embedding" not in knowledge_fields
        assert "embedding" not in tickets_fields
        assert "content" not in knowledge_fields  # embed source omitted
        assert "content" in tickets_fields  # no embed source configured

        # Per-index write policy and explicit limits are reflected.
        assert indexes["knowledge"]["upsert_available"] is True
        assert indexes["tickets"]["upsert_available"] is False
        assert indexes["knowledge"]["limits"] == {"max_limit": 25}
        assert "limits" not in indexes["tickets"]
        assert indexes["knowledge"]["description"] == "Product docs"
    finally:
        await server.shutdown()
