import pytest
from pydantic.v1 import ValidationError
from redisvl.extensions.router.routes import Route, RoutingConfig, RouteSortingMethod


def test_route_creation():
    route = Route(
        name="test_route",
        references=["test reference 1", "test reference 2"],
        metadata={"priority": "1"}
    )
    assert route.name == "test_route"
    assert route.references == ["test reference 1", "test reference 2"]
    assert route.metadata == {"priority": "1"}


def test_route_name_empty():
    with pytest.raises(ValidationError):
        Route(name="", references=["test reference"])


def test_route_references_empty():
    with pytest.raises(ValidationError):
        Route(name="test_route", references=[])


def test_route_references_non_empty_strings():
    with pytest.raises(ValidationError):
        Route(name="test_route", references=["", "test reference"])


def test_routing_config_creation():
    config = RoutingConfig(
        distance_threshold=0.5,
        max_k=1,
        sort_by=RouteSortingMethod.avg_distance
    )
    assert config.distance_threshold == 0.5
    assert config.max_k == 1
    assert config.sort_by == RouteSortingMethod.avg_distance


def test_routing_config_invalid_max_k():
    with pytest.raises(ValidationError):
        RoutingConfig(distance_threshold=0.5, max_k=0)


def test_routing_config_invalid_distance_threshold():
    with pytest.raises(ValidationError):
        RoutingConfig(distance_threshold=-0.1, max_k=1)
    with pytest.raises(ValidationError):
        RoutingConfig(distance_threshold=1.1, max_k=1)
