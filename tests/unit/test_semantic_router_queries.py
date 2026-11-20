"""Hermetic tests for SemanticRouter's reference-id filter construction.

`_make_filter_queries` is a static method that touches neither Redis nor a
vectorizer, so its behaviour is testable without the router's integration
fixtures -- which matters because a defect here silently deletes data.
"""

import re
from unittest.mock import Mock

import pytest

from redisvl.extensions.router.schema import (
    DistanceAggregationMethod,
    Route,
    RoutingConfig,
)
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


@pytest.mark.parametrize(
    "aggregation_method, filter_widened",
    [(DistanceAggregationMethod.min, True), (DistanceAggregationMethod.sum, False)],
)
def test_a_route_threshold_of_2_allows_for_rounding(aggregation_method, filter_widened):
    # Redis can report an exactly opposite vector a little above 2. The range
    # radius always allows for that; the FILTER does too, except for a sum,
    # which can legitimately pass 2.
    router = SemanticRouter.model_construct(
        name="router",
        routes=[Route(name="a", references=["x"], distance_threshold=2)],
        vectorizer=None,
        routing_config=RoutingConfig(),
    )
    router._index = Mock(aggregate=Mock(return_value=Mock(rows=[])))

    router(vector=[0.1, 0.2], aggregation_method=aggregation_method)

    request, params = router._index.aggregate.call_args.args
    (filter_bound,) = re.findall(r"@distance <= ([\d.]+)", str(request.build_args()))
    assert params["distance_threshold"] > 2.001
    assert (float(filter_bound) > 2.001) is filter_widened
