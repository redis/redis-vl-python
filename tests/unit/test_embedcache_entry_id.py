"""Unit tests for EmbeddingsCache entry IDs and the legacy-ID fallback checks."""

import pytest

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
from redisvl.redis.utils import hashify

CACHE = EmbeddingsCache()  # connects lazily, so no Redis is needed
make_id = CACHE._make_entry_id
make_legacy_id = CACHE._make_legacy_entry_id


@pytest.mark.parametrize(
    "first, second",
    [
        (("a:b", "c"), ("a", "b:c")),
        ((b"a", "m"), ("61", "m")),
        ((b"", "m"), ("", "m")),
    ],
    ids=["colon_in_content", "bytes_vs_hex_string", "empty"],
)
def test_distinct_inputs_get_distinct_ids(first, second):
    """Each pair shared one ID under the legacy format."""
    assert make_legacy_id(*first) == make_legacy_id(*second)
    assert make_id(*first) != make_id(*second)


def test_legacy_id_is_unchanged():
    """The fallback must keep finding keys written by RedisVL 0.32."""
    assert make_legacy_id("hi", "m") == hashify("hi:m")
    assert make_legacy_id(b"hi", "m") == hashify("6869:m")


@pytest.mark.parametrize(
    "stored, content, model_name, expected",
    [
        ({b"content": b"a:b", b"model_name": b"c"}, "a:b", "c", True),
        ({"content": "a:b", "model_name": "c"}, "a:b", "c", True),
        ({b"content": b"a:b", b"model_name": b"c"}, "a", "b:c", False),
        ({b"content": b"61", b"model_name": b"m"}, b"a", "m", False),
        ({b"content": b"\xff\x00", b"model_name": b"m"}, b"\xff\x00", "m", True),
    ],
    ids=["match", "decoded_reply", "collision", "bytes_collision", "binary"],
)
def test_legacy_entry_is_trusted_only_for_its_own_input(
    stored, content, model_name, expected
):
    assert (
        EmbeddingsCache._legacy_entry_matches(stored, content, model_name) is expected
    )


def test_legacy_key_that_expired_mid_lookup_is_not_migrated():
    """HGETALL can return data for a key whose PTTL then reports it gone."""
    data = {b"content": b"x", b"model_name": b"m"}
    replies = [data, -2, data, 5000]

    plan = CACHE._plan_legacy_migrations(["x", "x"], "m", [0, 1], replies)

    assert [(index, pttl) for index, *_, pttl in plan] == [(1, 5000)]
