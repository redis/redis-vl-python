import pytest
from pydantic import ValidationError

from redisvl.extensions.router.schema import (
    DistanceAggregationMethod,
    Route,
    RouteMatch,
    RoutingConfig,
)


def test_route_valid():
    route = Route(
        name="Test Route",
        references=["reference1", "reference2"],
        metadata={"key": "value"},
        distance_threshold=0.3,
    )
    assert route.name == "Test Route"
    assert route.references == ["reference1", "reference2"]
    assert route.metadata == {"key": "value"}
    assert route.distance_threshold == 0.3


def test_route_empty_name():
    with pytest.raises(ValidationError) as excinfo:
        Route(
            name="",
            references=["reference1", "reference2"],
            metadata={"key": "value"},
            distance_threshold=0.3,
        )
    assert "Route name must not be empty" in str(excinfo.value)


def test_route_empty_references():
    with pytest.raises(ValidationError) as excinfo:
        Route(
            name="Test Route",
            references=[],
            metadata={"key": "value"},
            distance_threshold=0.3,
        )
    assert "References must not be empty" in str(excinfo.value)


def test_route_non_empty_references():
    with pytest.raises(ValidationError) as excinfo:
        Route(
            name="Test Route",
            references=["reference1", ""],
            metadata={"key": "value"},
            distance_threshold=0.3,
        )
    assert "All references must be non-empty strings" in str(excinfo.value)


def test_route_valid_no_threshold():
    route = Route(
        name="Test Route",
        references=["reference1", "reference2"],
        metadata={"key": "value"},
    )
    assert route.name == "Test Route"
    assert route.references == ["reference1", "reference2"]
    assert route.metadata == {"key": "value"}


def test_route_invalid_threshold_zero():
    with pytest.raises(ValidationError) as excinfo:
        Route(
            name="Test Route",
            references=["reference1", "reference2"],
            metadata={"key": "value"},
            distance_threshold=0,
        )
    assert "Input should be greater than 0" in str(excinfo.value)


def test_route_invalid_threshold_negative():
    with pytest.raises(ValidationError) as excinfo:
        Route(
            name="Test Route",
            references=["reference1", "reference2"],
            metadata={"key": "value"},
            distance_threshold=-0.1,
        )
    assert "Input should be greater than 0" in str(excinfo.value)


def test_route_match():
    route_match = RouteMatch(name="test", distance=0.25)
    assert route_match.name == "test"
    assert route_match.distance == 0.25


def test_route_match_no_route():
    route_match = RouteMatch()
    assert route_match.name is None
    assert route_match.distance is None


def test_distance_aggregation_method():
    assert DistanceAggregationMethod.avg == DistanceAggregationMethod("avg")
    assert DistanceAggregationMethod.min == DistanceAggregationMethod("min")
    assert DistanceAggregationMethod.sum == DistanceAggregationMethod("sum")


def test_routing_config_valid():
    config = RoutingConfig(aggregation_method=DistanceAggregationMethod.min, max_k=5)
    assert config.aggregation_method == DistanceAggregationMethod("min")
    assert config.max_k == 5


def test_routing_config_invalid_max_k():
    with pytest.raises(ValidationError) as excinfo:
        RoutingConfig(max_k=0)
    assert "Input should be greater than 0" in str(excinfo.value)
