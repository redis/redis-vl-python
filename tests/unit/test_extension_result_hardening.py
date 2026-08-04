"""Defense-in-depth unit tests for extension result parsing.

On Redis 8.8+ a document that expires mid-search can be returned as a matched id
with a missing field payload. The core ``process_results`` parser already drops
such docs, but the extension consumers construct strict models from parsed
results and must independently tolerate an id-only / incomplete record rather
than raise. These tests exercise the extension processing methods directly.

The processing methods under test do not use ``self``, so they are invoked
unbound with a dummy ``self`` to avoid standing up a real Redis / vectorizer.
"""

import logging

import pytest

from redisvl.extensions.cache.llm.semantic import SemanticCache
from redisvl.extensions.message_history.base_history import BaseMessageHistory
from redisvl.extensions.router.semantic import SemanticRouter


def _valid_cache_hit():
    return {
        "id": "llmcache:key1",
        "entry_id": "e1",
        "prompt": "hello",
        "response": "world",
        "vector_distance": 0.1,
        "inserted_at": 1.0,
        "updated_at": 1.0,
    }


def test_process_cache_results_drops_id_only_hit(caplog):
    """An id-only cache hit fails CacheHit validation and is skipped, not raised."""
    with caplog.at_level(logging.WARNING):
        redis_keys, cache_hits = SemanticCache._process_cache_results(
            None, [{"id": "llmcache:expired"}], None
        )
    assert cache_hits == []
    # The expiring key is not returned for TTL refresh either.
    assert redis_keys == []
    assert any("missing field data" in r.message for r in caplog.records)


def test_process_cache_results_keeps_valid_hit_and_drops_bad():
    redis_keys, cache_hits = SemanticCache._process_cache_results(
        None, [_valid_cache_hit(), {"id": "llmcache:expired"}], None
    )
    assert len(cache_hits) == 1
    assert redis_keys == ["llmcache:key1"]
    assert cache_hits[0]["key"] == "llmcache:key1"


def test_process_cache_results_bad_then_valid_ordering():
    """A skipped hit's key must not leak into redis_keys before a later valid hit."""
    redis_keys, cache_hits = SemanticCache._process_cache_results(
        None, [{"id": "llmcache:expired"}, _valid_cache_hit()], None
    )
    assert len(cache_hits) == 1
    assert redis_keys == ["llmcache:key1"]  # only the valid hit's key


@pytest.mark.parametrize("as_text", [True, False])
def test_format_context_drops_id_only_message(as_text, caplog):
    """A message with 'id' but no entry_id/session_tag hits the KeyError arm."""
    messages = [
        {"id": "mh:expired"},  # generate_id -> KeyError on session_tag
        {"role": "user", "content": "hi", "session_tag": "s"},
    ]
    with caplog.at_level(logging.WARNING):
        out = BaseMessageHistory._format_context(None, messages, as_text=as_text)
    assert len(out) == 1  # only the healthy message survives
    assert any("missing field data" in r.message for r in caplog.records)


def test_format_context_drops_validation_error_message():
    """A message missing required role/content hits the ValidationError arm."""
    messages = [
        {"entry_id": "e1", "session_tag": "s"},  # no role/content -> ValidationError
        {"role": "user", "content": "hi", "session_tag": "s"},
    ]
    out = BaseMessageHistory._format_context(None, messages, as_text=False)
    assert len(out) == 1
    assert out[0]["content"] == "hi"


def test_process_route_drops_incomplete_row():
    """A route row missing route_name/distance returns None rather than KeyError.

    _process_route returns None silently; the aggregated warning is emitted by
    the caller (_get_route_matches), not per-row here.
    """
    assert SemanticRouter._process_route(None, []) is None
    assert SemanticRouter._process_route(None, ["route_name", "tech"]) is None


def test_process_route_keeps_valid_row():
    match = SemanticRouter._process_route(
        None, ["route_name", "tech", "distance", "0.1"]
    )
    assert match is not None
    assert match.name == "tech"
    assert match.distance == 0.1
