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
        # Each of these matched a different tenant on Redis 8.4: the query
        # parser splits a tag term on control characters and the backtick.
        pytest.param({"org_id": "\x01acme"}, id="leading-control"),
        pytest.param({"org_id": "acme\x7f"}, id="trailing-delete"),
        pytest.param({"org_id": "acme\tcorp"}, id="interior-tab"),
        pytest.param({"org_id": "acme\x00"}, id="nul"),
        pytest.param({"org_id": "acme`"}, id="backtick"),
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
        [_Spec("org_id", "org_id")], tool_name="search_kb"
    )
    assert str(expression) == "@org_id:{acme}"


def test_two_entries_and_together(monkeypatch):
    _token(monkeypatch, {"org_id": "acme", "region": "eu"})
    expression = build_injected_filter(
        [_Spec("org_id", "org_id"), _Spec("region", "region")],
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
            tool_name="search_kb",
        )
    assert exc.value.code == MCPErrorCode.FORBIDDEN


def test_no_entries_refuses_rather_than_matching_everything(monkeypatch):
    _token(monkeypatch, {"org_id": "acme"})
    with pytest.raises(RedisVLMCPError) as exc:
        build_injected_filter([], tool_name="search_kb")
    assert exc.value.code == MCPErrorCode.INTERNAL_ERROR


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
        build_injected_filter([_Spec("org_id", "org_id")], tool_name="search_kb")


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
        build_injected_filter([_Spec("org_id", "org_id")], tool_name="search_kb")
    assert exc.value.code == MCPErrorCode.FORBIDDEN
    assert "matches every document" in str(exc.value)


def test_a_pipe_in_a_claim_is_content_not_a_union(monkeypatch):
    """Canary for the rendering property the guarantee rests on.

    An identifier such as an Auth0 ``sub`` carries ``|``. ``Tag`` escapes it
    inside a single value, so the claim scopes to exactly its own documents;
    the integration suite proves Redis matches the escaped form literally. A
    list claim is the shape that genuinely unions, and is refused on type.
    """
    from redisvl.query.filter import Tag

    _token(monkeypatch, {"org_id": "auth0|64f1c2"})
    expression = build_injected_filter(
        [_Spec("org_id", "org_id")], tool_name="search_kb"
    )
    assert str(expression) == "@org_id:{auth0\\|64f1c2}"

    # The structure a list produces is a real union, indistinguishable from a
    # legitimate one by inspecting the output -- hence the type check.
    assert str(Tag("org_id") == ["acme", "victim"]) == "@org_id:{acme|victim}"
    _token(monkeypatch, {"org_id": ["acme", "victim"]})
    with pytest.raises(RedisVLMCPError) as exc:
        build_injected_filter([_Spec("org_id", "org_id")], tool_name="search_kb")
    assert exc.value.code == MCPErrorCode.FORBIDDEN


def test_a_combination_that_loses_a_clause_is_refused(monkeypatch):
    # The last check before a query runs. `format_expression` elides a `*`
    # operand by design; if any future change made it drop a real one, the
    # query would be scoped by fewer clauses than were configured.
    from redisvl.query.filter import FilterExpression

    _token(monkeypatch, {"org_id": "acme", "region": "eu"})
    monkeypatch.setattr(
        FilterExpression,
        "format_expression",
        staticmethod(lambda left, right, operator_str: str(left)),
    )
    with pytest.raises(RedisVLMCPError) as exc:
        build_injected_filter(
            [_Spec("org_id", "org_id"), _Spec("region", "region")],
            tool_name="search_kb",
        )
    assert exc.value.code == MCPErrorCode.INTERNAL_ERROR
    assert "lost the clauses" in str(exc.value)


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


def test_validation_warns_when_an_injected_tag_folds_case(caplog):
    schema = _schema()
    schema.fields["org_id"].attrs.case_sensitive = False
    with caplog.at_level("WARNING", logger="redisvl.mcp.auth"):
        validate_inject_against_schema(
            [_Spec("org_id", "org_id")], schema, profile_name="search_kb"
        )
    assert any("not CASESENSITIVE" in record.message for record in caplog.records)


def test_validation_does_not_warn_for_a_case_sensitive_tag(caplog):
    with caplog.at_level("WARNING", logger="redisvl.mcp.auth"):
        validate_inject_against_schema(
            [_Spec("org_id", "org_id")], _schema(), profile_name="search_kb"
        )
    assert not any("CASESENSITIVE" in record.message for record in caplog.records)
