"""Hermetic tests that a route's distance threshold is validated on every path.

The threshold is interpolated into the router's aggregation FILTER, so a value
that slips past validation either breaks routing silently (NaN, out of range)
or rewrites the query (a string). The constructor has always enforced (0, 2];
these pin assignment and `update_route_thresholds` to the same rule.
"""

import math

import pytest
from pydantic import ValidationError

from redisvl.extensions.router.schema import Route
from redisvl.extensions.router.semantic import SemanticRouter

INVALID = [math.nan, True, "0.5", 3, 0, None]
INVALID_IDS = ["nan", "bool", "numeric_string", "above_range", "zero", "none"]


def _route(name="greeting", threshold=0.5):
    return Route(name=name, references=["hello"], distance_threshold=threshold)


def _router(*routes):
    # update_route_thresholds touches only `routes`, so skip the index setup.
    # Passing `vectorizer` keeps model_construct from building the default
    # HuggingFace one, which would download a model.
    return SemanticRouter.model_construct(routes=list(routes), vectorizer=None)


@pytest.mark.parametrize("value", INVALID, ids=INVALID_IDS)
def test_constructor_and_assignment_reject_the_same_values(value):
    with pytest.raises(ValidationError):
        _route(threshold=value)

    route = _route()
    with pytest.raises(ValidationError):
        route.distance_threshold = value
    assert route.distance_threshold == 0.5


@pytest.mark.parametrize("value", INVALID, ids=INVALID_IDS)
def test_update_route_thresholds_rejects_invalid_values(value):
    router = _router(_route())
    with pytest.raises(ValidationError):
        router.update_route_thresholds({"greeting": value})
    assert router.route_thresholds == {"greeting": 0.5}


def test_update_route_thresholds_applies_nothing_when_any_value_is_invalid():
    router = _router(_route("greeting"), _route("farewell"))
    with pytest.raises(ValidationError):
        router.update_route_thresholds({"greeting": 0.9, "farewell": math.nan})
    assert router.route_thresholds == {"greeting": 0.5, "farewell": 0.5}


def test_update_route_thresholds_accepts_upper_bound():
    # Passes before the fix too: it pins that the new up-front check keeps the
    # inclusive bound and still applies valid values.
    router = _router(_route("greeting"), _route("farewell"))
    router.update_route_thresholds({"greeting": 2, "farewell": 0.25})
    assert router.route_thresholds == {"greeting": 2.0, "farewell": 0.25}
