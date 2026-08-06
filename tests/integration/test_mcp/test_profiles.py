from pathlib import Path

import pytest
import yaml

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.schema import IndexSchema


@pytest.fixture
async def profile_index(async_client, worker_id):
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": f"mcp-profile-{worker_id}",
                "prefix": f"mcp-profile:{worker_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "category", "type": "tag"},
                {"name": "rating", "type": "numeric"},
            ],
        }
    )
    index = AsyncSearchIndex(schema=schema, redis_client=async_client)
    await index.create(overwrite=True, drop=True)

    await index.load(
        [
            {
                "id": f"prof:{worker_id}:1",
                "content": "science article about planets",
                "category": "science",
                "rating": 5,
            },
            {
                "id": f"prof:{worker_id}:2",
                "content": "medical science and health",
                "category": "health",
                "rating": 4,
            },
            {
                "id": f"prof:{worker_id}:3",
                "content": "science of sports performance",
                "category": "sports",
                "rating": 2,
            },
        ]
    )

    yield index

    await index.delete(drop=True)


@pytest.fixture
def profile_config_path(tmp_path: Path, redis_url: str):
    def factory(
        redis_name: str,
        custom_tools: list[dict],
        *,
        builtin_tools: dict | None = None,
    ) -> str:
        server: dict = {"redis_url": redis_url}
        if builtin_tools is not None:
            server["builtin_tools"] = builtin_tools

        config = {
            "server": server,
            "indexes": {
                "knowledge": {
                    "redis_name": redis_name,
                    "search": {"type": "fulltext"},
                    "runtime": {
                        "text_field_name": "content",
                        "default_limit": 10,
                        "max_limit": 20,
                    },
                }
            },
            "custom_tools": custom_tools,
        }
        config_path = tmp_path / f"{redis_name}-profiles.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(config_path)

    return factory


@pytest.fixture
async def started_server(profile_index, profile_config_path):
    servers: list[RedisVLMCPServer] = []

    async def started(custom_tools: list[dict], **kwargs) -> RedisVLMCPServer:
        server = RedisVLMCPServer(
            MCPSettings(
                config=profile_config_path(
                    profile_index.schema.index.name, custom_tools, **kwargs
                )
            )
        )
        await server.startup()
        servers.append(server)
        return server

    yield started

    for server in servers:
        await server.shutdown()


async def _tool(server: RedisVLMCPServer, name: str):
    """Return the registered callable for a tool name."""
    tool = await server.get_tool(name)
    assert tool is not None, f"tool '{name}' is not registered"
    return tool.fn


async def _tool_names(server: RedisVLMCPServer) -> set[str]:
    """Return every registered tool name."""
    return {tool.name for tool in await server.list_tools()}


_SCIENCE_ONLY = {
    "name": "science-search",
    "description": "Search science records.",
    "lock": {"filter": {"field": "category", "op": "eq", "value": "science"}},
}


@pytest.mark.asyncio
async def test_profile_locked_filter_restricts_results_in_redis(started_server):
    server = await started_server([_SCIENCE_ONLY])

    tool = await _tool(server, "science-search")
    response = await tool(query="science")

    # All three documents match the text query, so only the locked filter can
    # account for a single result coming back.
    assert len(response["results"]) == 1
    assert response["results"][0]["record"]["category"] == "science"


@pytest.mark.asyncio
async def test_profile_caller_filter_cannot_escape_the_locked_scope(started_server):
    server = await started_server([_SCIENCE_ONLY])

    tool = await _tool(server, "science-search")
    response = await tool(
        query="science",
        filter={"field": "category", "op": "eq", "value": "sports"},
    )

    # The sports document exists and matches the text query, but asking for it
    # intersects with the locked category rather than replacing it.
    assert response["results"] == []


@pytest.mark.asyncio
async def test_profile_caller_or_filter_stays_inside_the_locked_scope(started_server):
    server = await started_server([_SCIENCE_ONLY])

    tool = await _tool(server, "science-search")
    response = await tool(
        query="science",
        filter={
            "or": [
                {"field": "category", "op": "eq", "value": "sports"},
                {"field": "category", "op": "eq", "value": "health"},
            ]
        },
    )

    # An OR of two other categories cannot widen past the locked AND, so nothing
    # comes back rather than the sports and health documents.
    assert response["results"] == []


@pytest.mark.asyncio
async def test_profile_locked_return_fields_restrict_the_projection(started_server):
    server = await started_server(
        [
            {
                "name": "redacted-search",
                "description": "Search records with a redacted projection.",
                "lock": {"return_fields": ["content"]},
            }
        ]
    )

    tool = await _tool(server, "redacted-search")
    response = await tool(query="science")

    assert response["results"]
    for result in response["results"]:
        # `category` and `rating` exist on every document but are not projected.
        assert set(result["record"]) == {"content"}


@pytest.mark.asyncio
async def test_disabled_builtin_is_not_registered_alongside_a_profile(started_server):
    server = await started_server(
        [_SCIENCE_ONLY], builtin_tools={"search-records": "disabled"}
    )

    registered = await _tool_names(server)

    # The curated tool replaces the generic one rather than sitting beside it.
    assert "science-search" in registered
    assert "search-records" not in registered
    assert "list-indexes" in registered


@pytest.mark.asyncio
async def test_startup_fails_when_a_locked_filter_names_an_unknown_field(
    started_server,
):
    with pytest.raises(RedisVLMCPError) as exc_info:
        await started_server(
            [
                {
                    "name": "broken-search",
                    "description": "Search records.",
                    "lock": {"filter": {"field": "missing", "op": "eq", "value": "x"}},
                }
            ]
        )

    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER


@pytest.mark.asyncio
async def test_profile_advertises_only_its_exposed_arguments(started_server):
    server = await started_server(
        [
            {
                "name": "narrow-search",
                "description": "Search science records.",
                "lock": {
                    "filter": {"field": "category", "op": "eq", "value": "science"},
                    "return_fields": ["content"],
                },
                "params": {"offset": {"expose": False}},
            }
        ]
    )

    tool = await server.get_tool("narrow-search")
    schema = tool.parameters

    # This is the actual security boundary: a locked or hidden argument is not
    # merely ignored, it is never published, and a compliant client cannot send
    # one because extras are refused. Asserted against real FastMCP rather than
    # a fake, since the schema is derived by the library from the signature.
    assert set(schema["properties"]) == {"query", "limit", "filter"}
    # A set, not a list: `required` ordering is a pydantic implementation detail,
    # and only its membership is behavior. `fastmcp` is pinned `>=2.0.0` with no
    # upper bound, so a minor bump must not break this on presentation alone.
    assert set(schema["required"]) == {"query"}
    assert schema["additionalProperties"] is False

    # A raw string filter has no safe composition with a locked filter, so the
    # object form is the only shape offered. Optional[dict] is encoded as an
    # `anyOf` union by current pydantic, but a nullable type list is the other
    # legal JSON Schema spelling of the same thing -- accept either, since only
    # the advertised type set is the behavior under test.
    filter_schema = schema["properties"]["filter"]
    if "anyOf" in filter_schema:
        filter_types = {entry.get("type") for entry in filter_schema["anyOf"]}
    else:
        declared = filter_schema["type"]
        filter_types = set(declared) if isinstance(declared, list) else {declared}
    assert filter_types == {"object", "null"}


@pytest.mark.asyncio
async def test_profile_text_filter_cannot_inject_past_the_locked_filter(started_server):
    server = await started_server([_SCIENCE_ONLY])

    tool = await _tool(server, "science-search")
    response = await tool(
        query="science",
        filter={
            "field": "content",
            "op": "like",
            "value": "zzznomatch) | (@category:{sports}",
        },
    )

    # Unescaped, this payload closes the text clause and the trailing `|` splits
    # the query into a union, returning documents the lock excludes. Escaped, it
    # is a literal that simply matches nothing.
    assert response["results"] == []


@pytest.mark.asyncio
async def test_profile_caps_an_omitted_limit_against_real_results(started_server):
    server = await started_server(
        [
            {
                "name": "one-result-search",
                "description": "Search records, one at a time.",
                "params": {"limit": {"expose": True, "max": 1}},
            }
        ]
    )

    tool = await _tool(server, "one-result-search")
    response = await tool(query="science")

    # Three documents match the query and the binding default is 10, so only the
    # cap can explain a single result.
    assert len(response["results"]) == 1


@pytest.mark.asyncio
async def test_profile_offset_past_the_locked_result_set_returns_nothing(
    started_server,
):
    server = await started_server([_SCIENCE_ONLY])

    tool = await _tool(server, "science-search")
    unpaged = await tool(query="science")
    paged = await tool(query="science", offset=1, limit=2)

    # The lock leaves exactly one of the three matching documents, so paging past
    # it has to come back empty. If offset were applied before the locked filter
    # -- or if the filter were applied only to the first page -- this would return
    # documents the lock excludes.
    assert len(unpaged["results"]) == 1
    assert paged["results"] == []
