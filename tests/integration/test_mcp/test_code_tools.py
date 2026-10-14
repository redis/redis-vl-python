"""Code tools registered with ``server.code_tool``, against a real Redis.

The unit suites own the validation and wrapping matrices. What lives here is
what depends on a started server: that a tool's handle reaches the real index,
that a writing tool really writes, that startup refuses what the config cannot
hold, and that the concurrency cap governs each Redis call without deadlocking
a tool that makes several.
"""

import asyncio
from pathlib import Path

import pytest
import yaml

from redisvl.index import AsyncSearchIndex
from redisvl.mcp import ToolContext
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema

fastmcp = pytest.importorskip(
    "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
)
from fastmcp.server.auth.providers.jwt import RSAKeyPair


@pytest.fixture
async def tickets_index(async_client, worker_id):
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": f"mcp-code-tools-{worker_id}",
                "prefix": f"mcp-code-tools:{worker_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "status", "type": "tag"},
            ],
        }
    )
    index = AsyncSearchIndex(schema=schema, redis_client=async_client)
    await index.create(overwrite=True, drop=True)
    await index.load(
        [
            {"id": "t1", "content": "refund delayed", "status": "open"},
            {"id": "t2", "content": "refund issued", "status": "resolved"},
            {"id": "t3", "content": "login failure", "status": "open"},
        ],
        id_field="id",
    )
    yield index
    await index.delete(drop=True)


@pytest.fixture
def config_path(tmp_path: Path, redis_url: str):
    def factory(
        redis_name: str,
        *,
        read_only: bool = False,
        max_concurrency: int = 16,
        custom_tools: list[dict] | None = None,
        auth: dict | None = None,
    ) -> str:
        server: dict = {"redis_url": redis_url}
        if auth is not None:
            server["auth"] = auth
            server["builtin_tools"] = {
                "search-records": "disabled",
                "upsert-records": "disabled",
            }
        config = {
            "server": server,
            "indexes": {
                "tickets": {
                    "redis_name": redis_name,
                    "read_only": read_only,
                    "search": {"type": "fulltext"},
                    "runtime": {
                        "text_field_name": "content",
                        "max_concurrency": max_concurrency,
                    },
                }
            },
            "custom_tools": custom_tools or [],
        }
        path = tmp_path / f"{redis_name}-code-tools.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(path)

    return factory


@pytest.fixture
async def server_for(tickets_index, config_path):
    servers: list[RedisVLMCPServer] = []

    def build(**kwargs) -> RedisVLMCPServer:
        server = RedisVLMCPServer(
            MCPSettings(config=config_path(tickets_index.schema.index.name, **kwargs))
        )
        servers.append(server)
        return server

    yield build

    for server in servers:
        await server.shutdown()


async def _call(server: RedisVLMCPServer, name: str, **kwargs):
    tool = await server.get_tool(name)
    assert tool is not None, f"tool '{name}' is not registered"
    return await tool.fn(**kwargs)


async def test_a_read_tool_queries_the_bound_index(server_for):
    server = server_for()

    @server.code_tool()
    async def open_tickets(ctx: ToolContext, limit: int = 10) -> dict:
        """List open support tickets."""
        query = FilterQuery(
            filter_expression=Tag("status") == "open",
            return_fields=["content"],
            num_results=limit,
        )
        return {"tickets": sorted(r["content"] for r in await ctx.index.query(query))}

    await server.startup()

    assert await _call(server, "open-tickets") == {
        "tickets": ["login failure", "refund delayed"]
    }


async def test_a_writing_tool_writes_through_its_handle(server_for):
    server = server_for()

    @server.code_tool(writes=True)
    async def close_ticket(ctx: ToolContext, ticket_id: str) -> dict:
        """Mark a support ticket resolved."""
        record = await ctx.index.fetch(ticket_id)
        await ctx.index.load([{**record, "status": "resolved"}], id_field="id")
        return {"status": (await ctx.index.fetch(ticket_id))["status"]}

    await server.startup()

    assert await _call(server, "close-ticket", ticket_id="t1") == {"status": "resolved"}


async def test_a_code_tool_on_a_tenant_scoped_index_fails_startup(server_for):
    key = RSAKeyPair.generate()
    server = server_for(
        auth={
            "type": "jwt",
            "public_key": key.public_key,
            "issuer": "https://auth.acme.example/",
            "audience": "api://redisvl-mcp",
        },
        custom_tools=[
            {
                "name": "tenant-tickets",
                "description": "Search this tenant's tickets.",
                "lock": {
                    "inject": [{"field": "status", "from": "claim", "claim": "org"}]
                },
            }
        ],
    )
    # Startup only checks for a token source on HTTP; naming one keeps the
    # refusal under test the code-tool one.
    server._transport = "streamable-http"

    @server.code_tool()
    async def all_tickets(ctx: ToolContext) -> dict:
        """List every ticket."""
        return {}

    with pytest.raises(ValueError, match="code tools 'all-tickets' reach it"):
        await server.startup()


async def test_each_redis_call_takes_the_cap_without_deadlocking_the_tool(
    server_for,
):
    # With a cap of one, a tool holding the cap across its whole body would
    # block its own second call. The cap is per Redis call, so two concurrent
    # calls of a tool that queries twice each still finish.
    server = server_for(max_concurrency=1)

    @server.code_tool()
    async def ticket_counts(ctx: ToolContext) -> dict:
        """Count open and resolved tickets."""
        counts = {}
        for status in ("open", "resolved"):
            query = FilterQuery(filter_expression=Tag("status") == status)
            counts[status] = len(await ctx.index.query(query))
        return counts

    await server.startup()

    results = await asyncio.wait_for(
        asyncio.gather(_call(server, "ticket-counts"), _call(server, "ticket-counts")),
        timeout=10,
    )

    assert results == [{"open": 2, "resolved": 1}] * 2
