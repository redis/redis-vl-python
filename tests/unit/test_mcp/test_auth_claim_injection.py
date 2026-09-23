"""Unit tests for auth-claim tenant injection.

The property under test is the one the feature exists to provide: *a client
presenting a validly-signed token cannot make the model widen or escape the
tenant scope carried in that token.* Every case below is a route someone could
take to an unscoped or a cross-tenant query.
"""

from dataclasses import dataclass

import pytest

# These tests monkeypatch fastmcp.server.dependencies.get_access_token, which
# imports fastmcp; skip the module when the optional extra is absent.
pytest.importorskip("fastmcp", reason="fastmcp not installed (install redisvl[mcp])")

from redisvl.mcp.auth import (
    build_injected_filter,
    resolve_injected_claim,
    validate_inject_against_schema,
)
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.schema import IndexSchema


@dataclass(frozen=True)
class _Spec:
    """Stands in for the configuration model Stack 02 adds.

    The security core takes a spec list rather than a config object, so this
    double is the whole contract: a field name and a claim name.
    """

    field: str
    claim: str


class _AccessToken:
    def __init__(self, claims=None):
        self.claims = claims or {}


def _schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {"name": "kb", "prefix": "kb", "storage_type": "hash"},
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "org_id", "type": "tag", "attrs": {"case_sensitive": True}},
                {"name": "region", "type": "tag", "attrs": {"case_sensitive": True}},
                {"name": "rating", "type": "numeric"},
                {"name": "shadow", "type": "tag", "attrs": {"no_index": True}},
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


def _token(monkeypatch, claims):
    """Install a request-scoped token, or none at all when claims is None."""
    token = None if claims is None else _AccessToken(claims)
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: token, raising=False
    )
    return token


# --- resolve_injected_claim ------------------------------------------------


def test_a_verified_claim_resolves_to_its_value(monkeypatch):
    _token(monkeypatch, {"org_id": "acme"})
    assert resolve_injected_claim("org_id", tool_name="search_kb") == "acme"


def test_a_tokenless_request_is_refused_rather_than_passed_through(monkeypatch):
    # The scope gate returns early here, reading a missing token as "stdio, so
    # no gate applies". Injection must invert that: the same exit would attach
    # no tenant clause and query every tenant.
    _token(monkeypatch, None)
    with pytest.raises(RedisVLMCPError) as exc:
        resolve_injected_claim("org_id", tool_name="search_kb")
    assert exc.value.code == MCPErrorCode.FORBIDDEN
    assert exc.value.retryable is False


@pytest.mark.parametrize(
    "claims",
    [
        pytest.param({}, id="no-claims"),
        pytest.param({"tenant": "acme"}, id="claim-misspelled"),
        pytest.param({"org_id": None}, id="null"),
        pytest.param({"org_id": ""}, id="empty"),
        pytest.param({"org_id": "   "}, id="whitespace-only"),
        pytest.param({"org_id": " acme"}, id="left-padded"),
        pytest.param({"org_id": "acme "}, id="right-padded"),
        pytest.param({"org_id": ["acme", "victim"]}, id="list"),
        pytest.param({"org_id": {"id": "acme"}}, id="dict"),
        pytest.param({"org_id": True}, id="bool"),
        pytest.param({"org_id": 42}, id="int"),
        pytest.param({"org_id": "acme|victim"}, id="pipe"),
    ],
)
def test_an_unusable_claim_is_refused(monkeypatch, claims):
    _token(monkeypatch, claims)
    with pytest.raises(RedisVLMCPError) as exc:
        resolve_injected_claim("org_id", tool_name="search_kb")
    assert exc.value.code == MCPErrorCode.FORBIDDEN
    assert exc.value.retryable is False


def test_the_refusal_names_the_claim_and_the_tool(monkeypatch):
    # A misspelled claim name is otherwise an opaque permanent failure. The
    # disclosure is free: a JWT is signed, not encrypted, so a client holding a
    # valid token can already read its own claim names.
    _token(monkeypatch, {})
    with pytest.raises(RedisVLMCPError) as exc:
        resolve_injected_claim("org_id", tool_name="search_kb")
    assert "org_id" in str(exc.value)
    assert "search_kb" in str(exc.value)


# --- build_injected_filter -------------------------------------------------


def test_one_entry_scopes_the_query_to_the_claim(monkeypatch):
    _token(monkeypatch, {"org_id": "acme"})
    expression = build_injected_filter(
        [_Spec("org_id", "org_id")], _schema(), tool_name="search_kb"
    )
    assert str(expression) == "@org_id:{acme}"


def test_two_entries_and_together(monkeypatch):
    _token(monkeypatch, {"org_id": "acme", "region": "eu"})
    expression = build_injected_filter(
        [_Spec("org_id", "org_id"), _Spec("region", "region")],
        _schema(),
        tool_name="search_kb",
    )
    rendered = str(expression)
    assert "@org_id:{acme}" in rendered
    assert "@region:{eu}" in rendered
    assert " | " not in rendered


def test_one_unusable_claim_refuses_the_whole_request(monkeypatch):
    # Not "narrow by the entries that resolved" -- a partially applied scope is
    # a wider scope than the one configured.
    _token(monkeypatch, {"org_id": "acme"})
    with pytest.raises(RedisVLMCPError) as exc:
        build_injected_filter(
            [_Spec("org_id", "org_id"), _Spec("region", "region")],
            _schema(),
            tool_name="search_kb",
        )
    assert exc.value.code == MCPErrorCode.FORBIDDEN


def test_no_entries_refuses_rather_than_matching_everything(monkeypatch):
    _token(monkeypatch, {"org_id": "acme"})
    with pytest.raises(RedisVLMCPError) as exc:
        build_injected_filter([], _schema(), tool_name="search_kb")
    assert exc.value.code == MCPErrorCode.INTERNAL_ERROR


def test_a_field_that_stopped_being_an_indexed_tag_is_refused(monkeypatch):
    # Startup validation rejects these, so reaching here means the bound schema
    # changed under a tool that was already registered.
    _token(monkeypatch, {"org_id": "acme"})
    for field in ("rating", "shadow", "absent"):
        with pytest.raises(RedisVLMCPError) as exc:
            build_injected_filter(
                [_Spec(field, "org_id")], _schema(), tool_name="search_kb"
            )
        assert exc.value.code == MCPErrorCode.FORBIDDEN


def test_an_injected_clause_cannot_be_elided_by_the_intersection(monkeypatch):
    # `Tag(f) == ""` renders as the match-all `*`, and an intersection drops a
    # `*` operand -- so `locked & injected` would render as `locked` alone with
    # the tenant clause silently gone. The guard therefore runs on the injected
    # clause by itself, before any combination.
    from redisvl.query.filter import Tag

    injected = Tag("org_id") == ""
    locked = Tag("status") == "resolved"
    assert str(locked & injected) == str(locked)

    _token(monkeypatch, {"org_id": ""})
    with pytest.raises(RedisVLMCPError):
        build_injected_filter(
            [_Spec("org_id", "org_id")], _schema(), tool_name="search_kb"
        )


def test_a_clause_that_renders_match_all_is_refused_even_so(monkeypatch):
    """The second, independent layer over the claim reader's own empty check.

    Today the reader rejects an empty claim first, so this guard cannot fire
    through any real token -- substituting the reader is the only way to reach
    it, which is precisely the future it insures against: a reader that admits
    a value ``Tag`` renders as the wildcard. Without it, `locked & injected`
    would render as `locked` alone and the request would run unscoped.
    """
    monkeypatch.setattr(
        "redisvl.mcp.auth.resolve_injected_claim",
        lambda claim, *, tool_name: "",
    )
    with pytest.raises(RedisVLMCPError) as exc:
        build_injected_filter(
            [_Spec("org_id", "org_id")], _schema(), tool_name="search_kb"
        )
    assert exc.value.code == MCPErrorCode.FORBIDDEN
    assert "matches every document" in str(exc.value)


def test_injected_pipe_cannot_union_across_tenants(monkeypatch):
    """Canary for the rendering property, not for a character class.

    ``Tag`` escapes ``|`` inside a single value as of 0.27.1. That character
    class has already changed once, so this asserts the property the guarantee
    actually rests on -- an injected value cannot produce a clause that matches
    a tenant other than the one named -- rather than asserting which characters
    are in which set.
    """
    from redisvl.query.filter import Tag

    # Property one: a scalar value carrying `|` is escaped, so it is content
    # rather than structure.
    assert str(Tag("org_id") == "acme|victim") == "@org_id:{acme\\|victim}"

    # Property two: a list value is *not* escaped into one value -- it renders
    # a genuine union. This is why the claim reader rejects on type, and why a
    # character scan of the rendered output would not close it.
    assert str(Tag("org_id") == ["acme", "victim"]) == "@org_id:{acme|victim}"

    # So neither shape can reach a query through injection.
    for value in ("acme|victim", ["acme", "victim"]):
        _token(monkeypatch, {"org_id": value})
        with pytest.raises(RedisVLMCPError) as exc:
            build_injected_filter(
                [_Spec("org_id", "org_id")], _schema(), tool_name="search_kb"
            )
        assert exc.value.code == MCPErrorCode.FORBIDDEN


# --- validate_inject_against_schema ----------------------------------------


def test_a_valid_injected_field_passes_validation():
    validate_inject_against_schema(
        [_Spec("org_id", "org_id"), _Spec("region", "region")],
        _schema(),
        profile_name="search_kb",
    )


@pytest.mark.parametrize(
    "field, expected",
    [
        pytest.param("absent", "unknown field", id="absent"),
        pytest.param("content", "requires a tag field", id="text"),
        pytest.param("rating", "requires a tag field", id="numeric"),
        pytest.param("embedding", "requires a tag field", id="vector"),
        pytest.param("shadow", "NOINDEX", id="no-index"),
    ],
)
def test_an_unusable_injected_field_fails_startup(field, expected):
    with pytest.raises(ValueError) as exc:
        validate_inject_against_schema(
            [_Spec(field, "org_id")], _schema(), profile_name="search_kb"
        )
    assert expected in str(exc.value)
    assert "search_kb" in str(exc.value)
