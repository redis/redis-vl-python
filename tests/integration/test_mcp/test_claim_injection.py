"""End-to-end tests for claim-injected tenant scoping against a real Redis.

Tenants share one index, separated by an ``org_id`` tag. Tokens are real RS256
JWTs minted with FastMCP's ``RSAKeyPair`` and verified by the server's own
configured verifier, so the claims the profile reads are the ones a verified
request would carry -- only the request-context lookup is substituted.

The unit suites own the exhaustive matrices. What lives here is what depends on
real Redis behaviour: how the merged query actually matches, how a value Redis
parses specially is handled, and what startup does against an inspected index.
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

# Both built-ins reach the index without the tenant scope, so a server with an
# injecting profile refuses to start unless they are off.
_SCOPED_BUILTINS = {"search-records": "disabled", "upsert-records": "disabled"}


def _doc(doc_id: str, org_id: str, content: str = "refund policy") -> dict:
    return {"id": doc_id, "content": content, "org_id": org_id, "category": "billing"}


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
    # them: a query that leaked would match every document below.
    await index.load(
        [
            _doc("a1", "acme"),
            _doc("a2", "acme", "refund window"),
            _doc("v1", "victim"),
            _doc("v2", "victim", "refund window"),
            # An Auth0-shaped subject beside the two halves it must not be
            # confused with, so a `|` claim that unioned would match three.
            _doc("p1", "auth0|64f1c2"),
            _doc("p2", "auth0"),
            _doc("p3", "64f1c2"),
        ],
        id_field="id",
    )
    yield index
    await index.delete(drop=True)


@pytest.fixture
def config_path(tmp_path: Path, redis_url: str, key: RSAKeyPair):
    def factory(
        redis_name: str,
        custom_tools: list[dict],
        *,
        auth: bool = True,
        builtin_tools: dict = _SCOPED_BUILTINS,
        read_only: bool = False,
    ) -> str:
        server: dict = {"redis_url": redis_url, "builtin_tools": builtin_tools}
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
                    "read_only": read_only,
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


def _orgs(result) -> list:
    return sorted(hit["record"]["org_id"] for hit in result["results"])


async def test_a_tenants_token_returns_only_that_tenants_documents(
    started, key, monkeypatch
):
    server = await started()
    await _as_caller(monkeypatch, server, key, {ORG_CLAIM: "acme"})

    assert _orgs(await _search(server)) == ["acme", "acme"]


async def test_a_pipe_in_the_claim_is_one_identifier_not_a_union(
    started, key, monkeypatch
):
    # Redis matches the escaped `\|` literally, so the claim reaches exactly its
    # own document and neither of the halves a union would also have matched.
    server = await started()
    await _as_caller(monkeypatch, server, key, {ORG_CLAIM: "auth0|64f1c2"})

    assert _orgs(await _search(server)) == ["auth0|64f1c2"]


@pytest.mark.parametrize(
    "caller_filter, expected",
    [
        # Naming the other tenant ANDs with the injected one: nothing matches.
        pytest.param({"field": "org_id", "op": "eq", "value": "victim"}, [], id="swap"),
        # The `or` stays nested inside the AND, so it narrows within acme rather
        # than hoisting to the top level and reaching victim's documents.
        pytest.param(
            {
                "or": [
                    {"field": "org_id", "op": "eq", "value": "victim"},
                    {"field": "category", "op": "eq", "value": "billing"},
                ]
            },
            ["acme", "acme"],
            id="or-widening",
        ),
        pytest.param(
            {"not": {"field": "org_id", "op": "eq", "value": "acme"}},
            [],
            id="negation",
        ),
    ],
)
async def test_a_caller_filter_cannot_widen_or_escape_the_tenant(
    started, key, monkeypatch, caller_filter, expected
):
    server = await started()
    await _as_caller(monkeypatch, server, key, {ORG_CLAIM: "acme"})

    # Exact, not merely "no victim rows": a subset check alone would pass on an
    # empty result for the wrong reason.
    assert _orgs(await _search(server, filter=caller_filter)) == expected


async def test_the_description_does_not_name_the_injected_field(started):
    server = await started()
    tool = await server.get_tool(TOOL)

    assert "org_id" not in tool.description
    assert "category(tag)" in tool.description


@pytest.mark.parametrize(
    "claims",
    [
        # The shape that genuinely unions, refused on type.
        pytest.param({ORG_CLAIM: ["acme", "victim"]}, id="array"),
        # Without the refusal this matched the tenant `acme` on Redis 8.4: the
        # query parser splits a tag term on control characters.
        pytest.param({ORG_CLAIM: "\x01acme"}, id="control-character"),
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
    "custom_tools, kwargs, expected",
    [
        pytest.param(
            [_PROFILE], {"auth": False}, "authentication is not enabled", id="auth-off"
        ),
        # Inspection has to preserve NOINDEX for this to be caught.
        pytest.param(
            [
                {
                    **_PROFILE,
                    "lock": {
                        "inject": [
                            {"field": "shadow", "from": "claim", "claim": ORG_CLAIM}
                        ]
                    },
                }
            ],
            {},
            "NOINDEX",
            id="field-noindex",
        ),
        pytest.param(
            [_PROFILE],
            {"builtin_tools": {"upsert-records": "disabled"}},
            "through search-records",
            id="unscoped-search",
        ),
        pytest.param(
            [_PROFILE],
            {"builtin_tools": {"search-records": "disabled"}},
            "through upsert-records",
            id="unscoped-write",
        ),
        pytest.param(
            [_PROFILE, {"name": "open-search", "description": "Search the kb."}],
            {},
            "custom tool 'open-search'",
            id="unscoped-profile",
        ),
    ],
)
async def test_startup_refuses_an_injection_profile_that_cannot_hold(
    tenant_index, config_path, custom_tools, kwargs, expected
):
    server = RedisVLMCPServer(
        MCPSettings(
            config=config_path(tenant_index.schema.index.name, custom_tools, **kwargs)
        )
    )
    with pytest.raises(ValueError, match=expected):
        await server.startup()


async def test_a_read_only_index_may_keep_upsert_enabled(started):
    # Writes to a read-only binding are refused per call, so upsert-records is
    # no route to it.
    server = await started(builtin_tools={"search-records": "disabled"}, read_only=True)
    assert await server.get_tool(TOOL) is not None
