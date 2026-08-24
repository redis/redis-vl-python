"""Every SCAN pattern built from a caller-supplied name escapes it.

Hermetic: these capture the pattern handed to Redis instead of needing a
server. ``match_pattern`` itself is tested in ``test_utils.py``.
"""

import asyncio

import pytest

from redisvl.extensions.cache.base import BaseCache
from redisvl.extensions.router.semantic import SemanticRouter
from redisvl.migration.utils import build_scan_match_patterns

# Unescaped, "cache[ab]" matches "cachea"/"cacheb" but not itself.
COLLIDING_NAME = "cache[ab]"


class RecordingClient:
    """Records SCAN patterns and returns no keys."""

    def __init__(self):
        self.patterns: list[str] = []

    def scan(self, cursor=0, match=None, count=None):
        self.patterns.append(match)
        return 0, []

    def delete(self, *keys):  # pragma: no cover - no keys are ever returned
        raise AssertionError("delete() reached with no scan hits")


class AsyncRecordingClient(RecordingClient):
    async def scan(self, cursor=0, match=None, count=None):  # type: ignore[override]
        self.patterns.append(match)
        return 0, []


class StubIndex:
    """Only the attributes ``_route_pattern`` reads."""

    def __init__(self, prefix, key_separator=":"):
        self.prefix = prefix
        self.key_separator = key_separator


def test_cache_clear_escapes_name():
    cache = BaseCache(name=COLLIDING_NAME)
    client = RecordingClient()
    cache._get_redis_client = lambda: client  # type: ignore[method-assign]

    cache.clear()

    assert client.patterns == ["cache\\[ab]:*"]


@pytest.mark.asyncio
async def test_cache_aclear_escapes_name():
    cache = BaseCache(name=COLLIDING_NAME)
    client = AsyncRecordingClient()

    async def _get_client():
        return client

    cache._get_async_redis_client = _get_client  # type: ignore[method-assign]

    await cache.aclear()

    assert client.patterns == ["cache\\[ab]:*"]


@pytest.mark.parametrize(
    "prefix, route_name, expected",
    [
        ("router[ab]", "route", "router\\[ab]:route:*"),
        ("router", "route[ab]", "router:route\\[ab]:*"),
        ("", "route[ab]", "route\\[ab]:*"),
        ("router", "route", "router:route:*"),
    ],
)
def test_route_pattern_escapes_prefix_and_route(prefix, route_name, expected):
    assert SemanticRouter._route_pattern(StubIndex(prefix), route_name) == expected


def test_build_scan_match_patterns_escapes_prefixes():
    assert build_scan_match_patterns(["doc[ab]", "plain"], ":") == [
        "doc\\[ab]*",
        "plain*",
    ]


@pytest.mark.parametrize("flavour", ["sync", "async"])
def test_planner_sample_keys_escapes_prefix(flavour):
    """Asserts only the escaping; the two planners differ on the separator."""
    if flavour == "sync":
        from redisvl.migration.planner import MigrationPlanner

        planner = MigrationPlanner.__new__(MigrationPlanner)
        planner.key_sample_limit = 10
        client = RecordingClient()
        planner._sample_keys(client=client, prefixes=["doc[ab]"], key_separator=":")
    else:
        from redisvl.migration.async_planner import AsyncMigrationPlanner

        planner = AsyncMigrationPlanner.__new__(AsyncMigrationPlanner)
        planner.key_sample_limit = 10
        client = AsyncRecordingClient()
        asyncio.run(
            planner._async_sample_keys(
                client=client, prefixes=["doc[ab]"], key_separator=":"
            )
        )

    assert client.patterns == [f"doc\\[ab]{'' if flavour == 'sync' else ':'}*"]
