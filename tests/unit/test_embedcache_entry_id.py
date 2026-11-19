"""Unit tests for EmbeddingsCache entry IDs."""

import pytest

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

make_id = EmbeddingsCache()._make_entry_id  # connects lazily, so no Redis needed


@pytest.mark.parametrize(
    "first, second",
    [
        (("a:b", "c"), ("a", "b:c")),
        ((b"a", "m"), ("61", "m")),
        ((b"", "m"), ("", "m")),
        (("x", "m", {"input_type": "query"}), ("x", "m", {"input_type": "document"})),
        (("x", "m", {"dimensions": 256}), ("x", "m", {"dimensions": "256"})),
        (("x", "m"), ("x", "m", {"input_type": None})),
    ],
    ids=[
        "colon_in_content",
        "bytes_vs_hex_string",
        "empty_bytes_vs_empty_string",
        "different_kwarg_value",
        "int_vs_string_kwarg",
        "no_kwargs_vs_none_valued_kwarg",
    ],
)
def test_distinct_inputs_get_distinct_ids(first, second):
    assert make_id(*first) != make_id(*second)


@pytest.mark.parametrize(
    "first, second",
    [
        (("x", "m"), ("x", "m", {})),
        (("x", "m", {"a": 1, "b": 2}), ("x", "m", {"b": 2, "a": 1})),
        (("x", "m", {"o": {"p": 1, "q": 2}}), ("x", "m", {"o": {"q": 2, "p": 1}})),
        (("x", "m", {"s": {"b", "a"}}), ("x", "m", {"s": {"a", "b"}})),
    ],
    ids=["none_vs_empty", "kwarg_order", "nested_order", "set_order"],
)
def test_equivalent_inputs_share_an_id(first, second):
    assert make_id(*first) == make_id(*second)
