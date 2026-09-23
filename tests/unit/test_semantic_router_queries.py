"""Hermetic tests for SemanticRouter's reference-id filter construction.

`_make_filter_queries` is a static method that touches neither Redis nor a
vectorizer, so its behaviour is testable without the router's integration
fixtures -- which matters because a defect here silently deletes data.
"""

import pytest

from redisvl.extensions.router.semantic import SemanticRouter


def test_reference_id_query_is_scoped_to_the_id():
    (query,) = SemanticRouter._make_filter_queries(["abc123"])
    assert str(query._filter_expression) == "@reference_id:{abc123}"


@pytest.mark.parametrize("empty", ["", None], ids=["empty_string", "none"])
def test_empty_reference_id_is_rejected_rather_than_matching_everything(empty):
    # `Tag("reference_id") == ""` renders as `*`. Callers read the first row of
    # each query's results, so a match-all here returns -- and, from
    # delete_route_references, deletes -- a reference nobody named.
    with pytest.raises(ValueError, match="non-empty"):
        SemanticRouter._make_filter_queries([empty])


def test_one_empty_id_rejects_the_whole_batch():
    with pytest.raises(ValueError, match="non-empty"):
        SemanticRouter._make_filter_queries(["abc123", "", "def456"])
