from pathlib import Path

import pytest

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings


class FakeVectorizer:
    def __init__(self, model: str, dims: int = 3, **kwargs):
        self.model = model
        self.dims = dims
        self.kwargs = kwargs


class FailingAsyncCloseVectorizer(FakeVectorizer):
    async def aclose(self):
        raise RuntimeError("vectorizer close failed")


@pytest.fixture
def mcp_config_path(tmp_path: Path, redis_url: str, worker_id: str):
    def factory(
        *, index_name: str, index_mode: str = "create_if_missing", vector_dims: int = 3
    ):
        config_path = tmp_path / f"{index_name}.yaml"
        config_path.write_text(
            f"""
redis_url: {redis_url}
index:
  name: {index_name}
  prefix: doc
  storage_type: hash
fields:
  - name: content
    type: text
  - name: embedding
    type: vector
    attrs:
      algorithm: flat
      dims: 3
      distance_metric: cosine
      datatype: float32
vectorizer:
  class: FakeVectorizer
  model: fake-model
  dims: {vector_dims}
runtime:
  index_mode: {index_mode}
  text_field_name: content
  vector_field_name: embedding
  default_embed_field: content
""".strip(),
            encoding="utf-8",
        )
        return str(config_path)

    return factory


@pytest.mark.asyncio
async def test_server_startup_success(monkeypatch, mcp_config_path, worker_id):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    settings = MCPSettings(
        config=mcp_config_path(index_name=f"mcp-startup-{worker_id}")
    )
    server = RedisVLMCPServer(settings)

    await server.startup()

    index = await server.get_index()
    vectorizer = await server.get_vectorizer()

    assert await index.exists() is True
    assert vectorizer.dims == 3

    await server.shutdown()


@pytest.mark.asyncio
async def test_server_validate_only_missing_index(
    monkeypatch, mcp_config_path, worker_id
):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    settings = MCPSettings(
        config=mcp_config_path(
            index_name=f"mcp-missing-{worker_id}",
            index_mode="validate_only",
        )
    )
    server = RedisVLMCPServer(settings)

    with pytest.raises(ValueError, match="does not exist"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_create_if_missing_is_idempotent(
    monkeypatch, mcp_config_path, worker_id
):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    config_path = mcp_config_path(index_name=f"mcp-idempotent-{worker_id}")
    first = RedisVLMCPServer(MCPSettings(config=config_path))
    second = RedisVLMCPServer(MCPSettings(config=config_path))

    await first.startup()
    await first.shutdown()
    await second.startup()

    assert await (await second.get_index()).exists() is True

    await second.shutdown()


@pytest.mark.asyncio
async def test_server_fails_fast_on_vector_dimension_mismatch(
    monkeypatch, mcp_config_path, worker_id
):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    settings = MCPSettings(
        config=mcp_config_path(
            index_name=f"mcp-dims-{worker_id}",
            vector_dims=8,
        )
    )
    server = RedisVLMCPServer(settings)

    with pytest.raises(ValueError, match="Vectorizer dims"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_startup_failure_disconnects_index(
    monkeypatch, mcp_config_path, worker_id
):
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
    settings = MCPSettings(
        config=mcp_config_path(
            index_name=f"mcp-startup-failure-{worker_id}",
            vector_dims=8,
        )
    )
    server = RedisVLMCPServer(settings)

    with pytest.raises(ValueError, match="Vectorizer dims"):
        await server.startup()

    assert disconnect_called is True


@pytest.mark.asyncio
async def test_server_shutdown_disconnects_owned_client(
    monkeypatch, mcp_config_path, worker_id
):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    settings = MCPSettings(
        config=mcp_config_path(index_name=f"mcp-shutdown-{worker_id}")
    )
    server = RedisVLMCPServer(settings)

    await server.startup()
    index = await server.get_index()

    assert index.client is not None

    await server.shutdown()

    assert index.client is None


@pytest.mark.asyncio
async def test_server_shutdown_disconnects_index_when_vectorizer_close_fails(
    monkeypatch, mcp_config_path, worker_id
):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FailingAsyncCloseVectorizer,
    )
    settings = MCPSettings(
        config=mcp_config_path(index_name=f"mcp-shutdown-failure-{worker_id}")
    )
    server = RedisVLMCPServer(settings)

    await server.startup()
    index = await server.get_index()

    with pytest.raises(RuntimeError, match="vectorizer close failed"):
        await server.shutdown()

    assert index.client is None
