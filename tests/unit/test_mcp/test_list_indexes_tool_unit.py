from types import SimpleNamespace
from typing import Any

import pytest

from redisvl.mcp.config import MCPConfig
from redisvl.mcp.runtime import BindingRuntime
from redisvl.mcp.tools.list_indexes import list_indexes, register_list_indexes_tool
from redisvl.schema import IndexSchema


def _schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "docs-index",
                "prefix": "doc",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "title", "type": "text"},
                {"name": "content", "type": "text"},
                {"name": "category", "type": "tag"},
                {"name": "rating", "type": "numeric"},
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
        }
    )


def _binding_runtime(
    binding_id: str = "knowledge",
    *,
    runtime: dict[str, Any] | None = None,
    description: str | None = None,
    read_only: bool = False,
    effective_read_only: bool = False,
    schema: IndexSchema | None = None,
) -> BindingRuntime:
    runtime_config = {
        "vector_field_name": "embedding",
        "default_embed_text_field": "content",
    }
    if runtime:
        runtime_config.update(runtime)

    binding_dict: dict[str, Any] = {
        "redis_name": f"{binding_id}-redis-name",
        "read_only": read_only,
        "vectorizer": {"class": "FakeVectorizer", "model": "test-model"},
        "search": {"type": "vector"},
        "runtime": runtime_config,
    }
    if description is not None:
        binding_dict["description"] = description

    config = MCPConfig.model_validate(
        {
            "server": {"redis_url": "redis://localhost:6379"},
            "indexes": {binding_id: binding_dict},
        }
    )
    return BindingRuntime(
        binding_id=binding_id,
        binding=config.indexes[binding_id],
        index=SimpleNamespace(),
        schema=schema or _schema(),
        vectorizer=None,
        supports_native_hybrid_search=False,
        effective_read_only=effective_read_only,
    )


class FakeServer:
    def __init__(self, bindings: list[BindingRuntime]):
        self._bindings = {rt.binding_id: rt for rt in bindings}
        self.mcp_settings = SimpleNamespace()
        self.auth_config = None
        self._auth_enabled = False
        self.registered_tools: list[dict[str, Any]] = []

    def tool(self, name=None, description=None, **kwargs):
        def decorator(fn):
            self.registered_tools.append(
                {"name": name, "description": description, "fn": fn}
            )
            return fn

        return decorator


def test_list_indexes_minimal_single_binding():
    server = FakeServer([_binding_runtime()])

    result = list_indexes(server)

    assert result == {
        "indexes": [
            {
                "id": "knowledge",
                "upsert_available": True,
                "fields": [
                    {"name": "title", "type": "text"},
                    {"name": "category", "type": "tag"},
                    {"name": "rating", "type": "numeric"},
                ],
            }
        ]
    }


def test_list_indexes_omits_vector_and_embed_source_fields():
    server = FakeServer([_binding_runtime()])

    fields = list_indexes(server)["indexes"][0]["fields"]
    field_names = [field["name"] for field in fields]

    # embedding is the vector field; content is the default embed-source field.
    assert "embedding" not in field_names
    assert "content" not in field_names


def test_list_indexes_includes_description_when_configured():
    server = FakeServer([_binding_runtime(description="Product docs and runbooks")])

    entry = list_indexes(server)["indexes"][0]

    assert entry["description"] == "Product docs and runbooks"


def test_list_indexes_omits_description_when_absent():
    server = FakeServer([_binding_runtime()])

    assert "description" not in list_indexes(server)["indexes"][0]


def test_list_indexes_upsert_available_reflects_effective_read_only():
    server = FakeServer(
        [
            _binding_runtime("knowledge", effective_read_only=False),
            _binding_runtime("tickets", read_only=True, effective_read_only=True),
        ]
    )

    indexes = {entry["id"]: entry for entry in list_indexes(server)["indexes"]}

    assert indexes["knowledge"]["upsert_available"] is True
    assert indexes["tickets"]["upsert_available"] is False


def test_list_indexes_includes_limits_only_when_explicitly_configured():
    server = FakeServer(
        [
            _binding_runtime(
                "explicit",
                runtime={"max_limit": 25, "max_upsert_records": 64},
            ),
            _binding_runtime("defaults"),
        ]
    )

    indexes = {entry["id"]: entry for entry in list_indexes(server)["indexes"]}

    assert indexes["explicit"]["limits"] == {
        "max_limit": 25,
        "max_upsert_records": 64,
    }
    assert "limits" not in indexes["defaults"]


def test_list_indexes_includes_only_the_explicitly_set_limit():
    server = FakeServer([_binding_runtime(runtime={"max_limit": 25})])

    entry = list_indexes(server)["indexes"][0]

    assert entry["limits"] == {"max_limit": 25}


def test_list_indexes_never_exposes_redis_name():
    server = FakeServer([_binding_runtime()])

    entry = list_indexes(server)["indexes"][0]

    assert "redis_name" not in entry
    assert "knowledge-redis-name" not in entry.values()


def test_list_indexes_preserves_binding_order():
    server = FakeServer(
        [
            _binding_runtime("knowledge"),
            _binding_runtime("tickets"),
        ]
    )

    ids = [entry["id"] for entry in list_indexes(server)["indexes"]]

    assert ids == ["knowledge", "tickets"]


@pytest.mark.asyncio
async def test_register_list_indexes_tool_is_read_only_and_callable():
    server = FakeServer([_binding_runtime()])

    register_list_indexes_tool(server)

    assert len(server.registered_tools) == 1
    tool = server.registered_tools[0]
    assert tool["name"] == "list-indexes"
    assert tool["description"]

    result = await tool["fn"]()
    assert result == list_indexes(server)
