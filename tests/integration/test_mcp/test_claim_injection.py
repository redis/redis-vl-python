"""End-to-end tests for claim-injected tenant scoping against a real Redis.

Two tenants share one index, separated by an ``org_id`` tag. Tokens are real
RS256 JWTs minted with FastMCP's ``RSAKeyPair`` and verified by the server's own
configured verifier, so the claims the profile reads are the ones a verified
request would carry -- only the request-context lookup is substituted.
"""

from pathlib import Path

import pytest
import yaml

fastmcp = pytest.importorskip(
    "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
)
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.schema import IndexSchema

ISSUER = "https://auth.acme.example/"
AUDIENCE = "api://redisvl-mcp"
ORG_CLAIM = "https://acme.example/org"
TOOL = "search-customer-kb"

_PROFILE = {
    "name": TOOL,
    "description": "Search this customer's knowledge base.",
    "lock": {"inject": [{"field": "org_id", "from": "claim", "claim": ORG_CLAIM}]},
}


@pytest.fixture(scope="module")
def key() -> RSAKeyPair:
    return RSAKeyPair.generate()


@pytest.fixture
async def tenant_index(async_client, worker_id):
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": f"mcp-tenants-{worker_id}",
                "prefix": f"mcp-tenants:{worker_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "org_id", "type": "tag", "attrs": {"case_sensitive": True}},
                {"name": "category", "type": "tag"},
                {"name": "notes", "type": "text"},
                # NOINDEX is only accepted alongside SORTABLE.
                {
                    "name": "shadow",
                    "type": "tag",
                    "attrs": {"no_index": True, "sortable": True},
                },
            ],
        }
    )
    index = AsyncSearchIndex(schema=schema, redis_client=async_client)
    await index.create(overwrite=True, drop=True)
    # Identical content across tenants, so only the tenant clause can separate
    # them -- a query that leaks would match all four.
    await index.load(
        [
            {
                "id": "a1",
                "content": "refund policy",
                "org_id": "acme",
                "category": "billing",
            },
            {
                "id": "a2",
                "content": "refund window",
                "org_id": "acme",
                "category": "billing",
            },
            {
                "id": "v1",
                "content": "refund policy",
                "org_id": "victim",
                "category": "billing",
            },
            {
                "id": "v2",
                "content": "refund window",
                "org_id": "victim",
                "category": "billing",
            },
        ],
        id_field="id",
    )
    yield index
    await index.delete(drop=True)


@pytest.fixture
def config_path(tmp_path: Path, redis_url: str, key: RSAKeyPair):
    def factory(redis_name: str, custom_tools: list[dict], *, auth: bool = True) -> str:
        server: dict = {"redis_url": redis_url}
        if auth:
            server["auth"] = {
                "type": "jwt",
                "public_key": key.public_key,
                "issuer": ISSUER,
                "audience": AUDIENCE,
            }
        config = {
            "server": server,
            "indexes": {
                "kb": {
                    "redis_name": redis_name,
                    "search": {"type": "fulltext"},
                    "runtime": {"text_field_name": "content"},
                }
            },
            "custom_tools": custom_tools,
        }
        path = tmp_path / f"{redis_name}-inject.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(path)

    return factory


@pytest.fixture
async def started(tenant_index, config_path):
    servers: list[RedisVLMCPServer] = []

    async def start(custom_tools=(_PROFILE,), **kwargs) -> RedisVLMCPServer:
        server = RedisVLMCPServer(
            MCPSettings(
                config=config_path(
                    tenant_index.schema.index.name, list(custom_tools), **kwargs
                )
            )
        )
        await server.startup()
        servers.append(server)
        return server

    yield start

    for server in servers:
        await server.shutdown()


async def _as_caller(monkeypatch, server, key, claims):
    """Verify a real token through the server's verifier and make it current."""
    token = key.create_token(
        subject="user-42",
        issuer=ISSUER,
        audience=AUDIENCE,
        additional_claims=claims,
    )
    access = await server.auth.verify_token(token)
    assert access is not None, "the server's verifier rejected the minted token"
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: access, raising=False
    )


async def _search(server, **kwargs):
    tool = await server.get_tool(TOOL)
    assert tool is not None
    return await tool.fn(query="refund", **kwargs)


def _orgs(result) -> set:
    return {hit["record"]["org_id"] for hit in result["results"]}


async def test_a_tenants_token_returns_only_that_tenants_documents(
    started, key, monkeypatch
):
    server = await started()
    await _as_caller(monkeypatch, server, key, {ORG_CLAIM: "acme"})

    result = await _search(server)

    assert len(result["results"]) == 2
    assert _orgs(result) == {"acme"}


@pytest.mark.parametrize(
    "caller_filter, expected",
    [
        # Naming the other tenant ANDs with the injected one: nothing matches.
        pytest.param({"field": "org_id", "op": "eq", "value": "victim"}, 0, id="swap"),
        # The `or` stays nested inside the AND, so it narrows within acme rather
        # than hoisting to the top level and reaching victim's documents.
        pytest.param(
            {
                "or": [
                    {"field": "org_id", "op": "eq", "value": "victim"},
                    {"field": "category", "op": "eq", "value": "billing"},
                ]
            },
            2,
            id="or-widening",
        ),
        pytest.param(
            {"not": {"field": "org_id", "op": "eq", "value": "acme"}},
            0,
            id="negation",
        ),
    ],
)
async def test_a_caller_filter_cannot_widen_or_escape_the_tenant(
    started, key, monkeypatch, caller_filter, expected
):
    server = await started()
    await _as_caller(monkeypatch, server, key, {ORG_CLAIM: "acme"})

    result = await _search(server, filter=caller_filter)

    # An exact count, not just "no victim rows": a subset check alone would
    # pass on an empty result for the wrong reason.
    assert len(result["results"]) == expected
    assert _orgs(result) <= {"acme"}


async def test_the_injected_field_is_absent_from_the_advertised_schema(started):
    server = await started()
    tool = await server.get_tool(TOOL)

    assert "org_id" not in tool.parameters["properties"]
    assert tool.parameters["additionalProperties"] is False
    # Nor is it named in the description's field hints.
    assert "org_id" not in tool.description
    assert "category(tag)" in tool.description


@pytest.mark.parametrize(
    "claims",
    [
        pytest.param({}, id="absent"),
        pytest.param({ORG_CLAIM: ""}, id="empty"),
        pytest.param({ORG_CLAIM: "   "}, id="whitespace"),
        pytest.param({ORG_CLAIM: ["acme", "victim"]}, id="array"),
        pytest.param({ORG_CLAIM: {"id": "acme"}}, id="object"),
        pytest.param({ORG_CLAIM: "acme|victim"}, id="pipe"),
    ],
)
async def test_an_unusable_claim_is_refused_before_any_query(
    started, key, monkeypatch, claims
):
    server = await started()
    await _as_caller(monkeypatch, server, key, claims)

    queried: list = []
    index = server.resolve_binding("kb").index
    original = index.query

    async def spy(query):
        queried.append(query)
        return await original(query)

    monkeypatch.setattr(index, "query", spy)

    with pytest.raises(RedisVLMCPError) as exc:
        await _search(server)

    assert exc.value.code == MCPErrorCode.FORBIDDEN
    assert queried == []


@pytest.mark.parametrize(
    "inject_field, auth, expected",
    [
        pytest.param("org_id", False, "authentication is not enabled", id="auth-off"),
        pytest.param("missing", True, "unknown field 'missing'", id="field-absent"),
        pytest.param("notes", True, "requires a tag field", id="field-text"),
        pytest.param("shadow", True, "NOINDEX", id="field-noindex"),
    ],
)
async def test_startup_refuses_an_injection_profile_that_cannot_scope(
    tenant_index, config_path, inject_field, auth, expected
):
    profile = {
        **_PROFILE,
        "lock": {
            "inject": [{"field": inject_field, "from": "claim", "claim": ORG_CLAIM}]
        },
    }
    server = RedisVLMCPServer(
        MCPSettings(
            config=config_path(tenant_index.schema.index.name, [profile], auth=auth)
        )
    )
    with pytest.raises(ValueError, match=expected):
        await server.startup()
